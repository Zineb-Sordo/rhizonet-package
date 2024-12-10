import os
import json
import csv
import sys
import numpy as np
import torch
import glob
import pytorch_lightning as pl
import torchvision.utils
from torchvision.transforms import ToPILImage

from torchviz import make_dot
import neptune
from neptune.types import File

import torchmetrics
from monai.data import list_data_collate
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss

from skimage import io, color
from skimage.color import rgb2gray, rgb2lab
from skimage.morphology import disk, dilation

from PIL import Image
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
import torchmetrics

from monai.transforms import (
    MapTransform,
    # AddChanneld, # deprecated in latest version of monai 
    # AddChannel,
    # AsChannelFirstd,
    # AsChannelLastd,
    Compose,
    CastToTyped,
    MapLabelValued,
    SqueezeDimd,
    Resized,
    CenterSpatialCropd,
    RandFlipd,
    RandAffined,
    ScaleIntensityRanged,
    EnsureTyped,
    ScaleIntensityRange,
    EnsureType,
    SpatialPadd,
    Lambda

)

# Import parent modules
try:
    from .utils import get_weights, extract_largest_component_bbox_image, MapImage
    from .utils import get_image_paths, contrast_img
except ImportError:
    from utils import get_weights, extract_largest_component_bbox_image, MapImage
    from utils import get_image_paths, contrast_img


class tiff_reader(MapTransform):
    """
    Custom TIFF reader to preprocess and apply optional contrast or bounding box extraction.
    """
    def __init__(self, image_col=None, intput_channels=3, boundingbox=False, dilation=True, disk_dilation=3, keys=["image", "label"], *args, **kwargs):
        super().__init__(self, keys, *args, **kwargs)
        self.keys = keys
        self.image_col = image_col
        self.boundingbox = boundingbox
        self.dilation = dilation
        self.dik_dilation = disk_dilation
        self.input_channels = self.input_channels

    def __call__(self, data_dict):

        data = {}
        for key in self.keys:
            if key in data_dict:
                if key == "image":
                    # if self.input_shape != img.shape[-1]
                    if self.image_col == 'cieLAB':
                        data[key] = rgb2lab(io.imread(data_dict[key]))  # cieLAB rep
                        data[key] = dynamic_scale(data[key])
                    else:
                        # data[key] = np.transpose(np.array(Image.open(data_dict[key]))[..., :3], (2, 0, 1))
                        data[key] = np.array(Image.open(data_dict[key]))
                        data[key] = dynamic_scale(data[key])

                        if data[key].ndim == 4 and data[key].shape[-1] <= 4:  # If shape is (h, w, d, c) assuming there are maximum 4 channels or modalities 
                            data[key] = np.transpose(data[key], (3, 0, 1, 2))  # Move channel to the first position
                        elif data[key].ndim == 3 and data[key].shape[-1] <= 4:  # If shape is (h, w, c)
                            data[key] = np.transpose(data[key], (2, 0, 1))  # Move channel to the first position
                        else:
                            raise ValueError(f"Unexpected image shape: {data[key].shape}, channel dimension should be last and image should be either 2D or 3D")
                elif key == "label":
                    data[key] = io.imread(data_dict[key])
                    if self.dilation:
                        data[key] = dilation(data['label'], disk(self.disk_dilation))
                    data[key] = np.expand_dims(data[key], axis=0)
        
        if self.boundingbox:
            data['image'], data['label'] = extract_largest_component_bbox_image(img=data['image'], lab=data['label'])
        return data


# Dynamic scaling logic
def dynamic_scale(image):
    a_min, a_max = image.min(), image.max()
    transform = ScaleIntensityRange(
        a_min=a_min,
        a_max=a_max,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    if image.max() > 255:
        print(image.max())
    return image

class ImageDataset(Dataset):
    def __init__(self, data_fnames, label_fnames, args, training=False, prediction=False):

        self.training = training
        self.prediction = prediction
        self.data_fnames = data_fnames
        self.label_fnames = label_fnames
        if len(self.data_fnames) != len(self.label_fnames):
            sys.exit(f"size of data and label images are not the same")
        self.Nsamples = len(self.data_fnames)
        self.data_dicts = [{"image": self.data_fnames[i], "label": self.label_fnames[i]} for i in range(self.Nsamples)]
        self.input_patch_size = io.imread(self.data_fnames[0]).shape
        self.patch_size = args['patch_size']
        self.spatial_dims = len(self.patch_size)
        self.translate_range = args['translate_range']
        self.rotate_range = args['rotate_range']
        self.scale_range = args['scale_range']
        self.shear_range = args['shear_range']
        self.class_values = args['class_values']
        self.image_col = args["image_col"]
        self.input_channels = args['input_channels']
        self.target_size = args['patch_size']
        self.boundingbox = args['boundingbox']
        self.dilation = args['dilation']
        self.disk_dilation = args['disk_dilation']

        self.transform = self.get_data_transforms(self.training, self.boundingbox, self.dilation, self.disk_dilation)

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        if self.prediction: 
            return (self.transform(self.data_dicts[idx]), os.path.split(self.data_fnames[idx])[-1])
        else:
            return self.transform(self.data_dicts[idx])

    # These transforms define the data preprocessing and augmentations done to the raw images BEFORE input to neural network
    def get_data_transforms(self, training, boundingbox, dilation, disk_dilation):
        if not training:  # validation or test_masks set -- no image augmentation
            transform = Compose(
                [
                    tiff_reader(keys=["image", "label"], image_col=self.image_col, boundingbox=boundingbox, dilation=dilation, disk_dilation=disk_dilation),
                    Resized(keys=["image", "label"], spatial_size=self.target_size, mode=['area', 'nearest']),
                    MapLabelValued(["label"],
                                    self.class_values,
                                    list(range(len(self.class_values)))),
                    SqueezeDimd(keys=["label"], dim=0),
                    CastToTyped(keys=["label"], dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )
            
        else:  # training set -- do image augmentation
            transform = Compose(
                [
                    tiff_reader(keys=["image", "label"], image_col=self.image_col, boundingbox=boundingbox, dilation=dilation, disk_dilation=disk_dilation),
                    Resized(keys=["image", "label"], spatial_size=self.target_size, mode=['area', 'nearest']),
                    RandFlipd(
                        keys=['image', 'label'],
                        prob=0.5,
                        spatial_axis=(0, 1)
                    ),
                    RandAffined(
                        keys=['image', 'label'],
                        mode=['bilinear', 'nearest'],
                        padding_mode='zeros',  ### implicitly assumes raw unlabeled_index = 0!
                        prob=1.0,
                        spatial_size=self.patch_size,
                        rotate_range=self.rotate_range * np.ones(1),
                        translate_range=self.translate_range * np.asarray(self.patch_size),
                        shear_range=self.shear_range * np.ones(2),
                        scale_range=self.scale_range * np.ones(2)
                    ),
                    MapLabelValued(["label"],
                                    self.class_values,
                                    list(range(len(self.class_values)))),
                    SqueezeDimd(["label"], dim=0),
                    CastToTyped(keys=["label"], dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )
           
        return transform


class PredDataset2D(Dataset):
    def __init__(self, pred_data_dir, args):
        self.pred_data_dir = pred_data_dir
        self.data_file = get_image_paths(pred_data_dir)
        # self.data_file = pred_data_lst
        self.input_col = args['input_channels']
        self.image_col = args['image_col']
        self.boundingbox = args['boundingbox']
        self.amax = 255 # need to be RGB images 

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        
        img_name = self.data_file[idx]
        img_path = os.path.join(self.pred_data_dir, img_name)
        # img_path = self.data_file[idx]
        img = np.array(Image.open(img_path))
        img = dynamic_scale(img)
        if img.ndim == 4 and img.shape[-1] <= 4:  # If shape is (h, w, d, c) assuming there are maximum 4 channels or modalities 
            img = np.transpose(img, (3, 0, 1, 2))  # Move channel to the first position
        elif img.ndim == 3 and img.shape[-1] <= 4:  # If shape is (h, w, c)
            img = np.transpose(img, (2, 0, 1))  # Move channel to the first position
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}, channel dimension should be last and image should be either 2D or 3D")
        
        transform = Compose(
            [
                EnsureType()
            ]
        )
        img = transform(img)

        if self.boundingbox:
            img = extract_largest_component_bbox_image(img, predict=True)
        return (img, img_path)


class Unet2D(pl.LightningModule):
    def __init__(self, train_ds, val_ds, **kwargs):
        """    
        PyTorch Lightning Module for training and evaluating 2D and 3D UNet-based models.
        This module supports both the traditional UNet and SwinUNETR architectures 
        for image segmentation tasks.

        Args:
            train_ds (Dataset): Training dataset
            val_ds (Dataset): Validation dataset

        Example: 

        >>> from monai.networks.nets import UNet, SwinUNETR

        >>> # 5 layers each down/up sampling their inputs by a factor 2 with residual blocks
        >>> model = UNet(
        >>>     spatial_dims=2,
        >>>     in_channels=3,
        >>>     out_channels=3,
        >>>     channels=(32, 64, 128, 256, 512),
        >>>     strides=(2, 2, 2, 2),
        >>>     kernel_size=3,
        >>>     up_kernel_size=3,
        >>>     num_res_units=2,
        >>>     dropout=0.4,
        >>>     norm=Norm.BATCH,
        >>>     )

        >>> # SwinUNETR model with 3D inputs and batch normalization
        >>> model = SwinUNETR(
        >>>     spatial_dims=3,
        >>>     img_size=(256, 256, 256),
        >>>     in_channels=3,
        >>>     out_channels=3,
        >>>     feature_size=48,
        >>>     num_heads=[3, 6, 12, 24, 12] ,
        >>>     depths=[2, 4, 8, 16, 24] ,
        >>>     drop_rate=0.4,
        >>>     attn_drop_rate=0.2,
        >>>     norm_name='batch'
        >>>     )
        """
        super(Unet2D, self).__init__()

        self.save_hyperparameters()
        self.hparams.output_channels = self.hparams.num_classes
        self.hparams.pred_patch_size = self.hparams.pred_patch_size
        self.hparams.model = self.hparams.model
        self.hparams.spatial_dims = self.hparams.spatial_dims
        self.hparams.class_values = self.hparams.class_values
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.cnfmat = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes,
                                                   task=self.hparams.task,
                                                   normalize=None)
        if self.hparams.model == 'resnet':
            self.model = UNet(
                spatial_dims=self.hparams.spatial_dims,
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.output_channels,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                kernel_size=3,
                up_kernel_size=3,
                num_res_units=2,
                dropout=0.4,
                norm=Norm.BATCH,
            )

        
        elif self.hparams.model == "swin":
            img_size = self.hparams.pred_patch_size
            feature_size = 48        
            use_checkpoint = True   
            dropout_rate = 0.4
            attention_dropout_rate = 0.2
            depths = [2, 4, 8, 16, 24]  
            num_heads = [3, 6, 12, 24, 12] 

            # Instantiate the model
            self.model = SwinUNETR(
                spatial_dims=self.hparams.spatial_dims,
                img_size=img_size,
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.output_channels,
                feature_size=feature_size,
                num_heads=num_heads,
                depths=depths,
                use_checkpoint=use_checkpoint,
                drop_rate=dropout_rate,
                attn_drop_rate=attention_dropout_rate,
                norm_name='batch'
                # norm_name = ('group', {"num_groups": 3, "affine": True})
            )
        else:
            raise AttributeError ("Only 2 possibles models can be applied, either resnet or swin models")
        self.has_executed = False

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_classes", type=int, default=3, help="number of segmentation classes to predict")
        parser.add_argument("--input_channels", type=int, default=3,
                            help="number of input channels (1 for grayscale, 3 for RGB)")
        parser.add_argument("--background_index", type=int, default=0, help="background index")
        parser.add_argument("--pred_patch_size", type=int, default=(64, 64),
                            help="spatial size of rolling window prediction patch")
        parser.add_argument("--batch_size", type=int, default=1, help="dataloader batch size")
        parser.add_argument("--lr", type=int, default=3e-4, help="Adam learning rate")
        parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers (cpus)")
        return parser

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model(x.to(device))

    def train_dataloader(self):
        """
        Creates and returns the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return train_loader

    def val_dataloader(self):
        """
        Creates and returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=10,
                                                               min_lr=1e-6,
                                                               verbose=True)
        lr_scheduler = {"scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch"}

        return [optimizer], [lr_scheduler]

    def loss_function(self, logits, labels, class_weights=None):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        loss = cross_entropy_loss(logits, labels)

        # dice_loss = DiceLoss(softmax=True, include_background=True, weight=class_weights, to_onehot_y=True, reduction='mean')
        # loss = dice_loss(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)  # forward pass on a batch
        torch.save(images, 'train_tensor.pt')
        
        classes = list(range(self.hparams.num_classes))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = get_weights(labels, classes, device, include_background=True)
        loss = self.loss_function(logits, labels, class_weights)
        self.loss = loss.item()

        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        if not self.has_executed:
            dummy_input = torch.rand((1, 3,) + self.hparams.pred_patch_size) 
            make_dot(self(dummy_input), params=dict(self.named_parameters())).render('network_graph', format='png')
            self.logger.log_image(key="model_visualization", images=["network_graph.png"]) # Wandb Logger
            self.has_executed = True
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        classes = list(range(self.hparams.num_classes))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = get_weights(labels, classes, device, include_background=True)
        loss = self.loss_function(logits, labels, class_weights)
        labeled_ind = (labels != -1)
        labeled_logits = torch.cat([logits[i, :, labeled_ind[i, :, :]] for i in range(len(logits))], dim=-1).transpose(
            0, 1)

        batch_logs = {"loss": loss,
                      "logits": logits,
                      "labels": labels}
        if not self.has_executed:
            to_pil = ToPILImage()
            if batch_idx % 100 == 0:
                x, y = images[:20], logits[:20]
                gridx = torchvision.utils.make_grid(x, nrow=5)

                # Convert the grid to a PIL image
                self.logger.log_image(key="training_img", images=[to_pil(gridx)]) # Wandb Logger
                preds = torch.argmax(y, dim=1).byte().squeeze(1)
                preds = (preds * 255).byte()


                y = MapImage(preds, self.hparams.class_values)
                
                gridy = torchvision.utils.make_grid(y.view(y.shape[0], 1, y.shape[1], y.shape[2]), nrow=5)
                self.logger.log_image(key="prediction_imgs", images=[to_pil(gridy)]) # Wandb Logger

        if torch.cuda.is_available():
            self.cnfmat(labeled_logits, labels[labeled_ind])
        else:
            self.cnfmat(labeled_logits, torch.tensor(labels)[torch.tensor(labeled_ind)])
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return batch_logs

    def on_validation_epoch_end(self):
        acc, prec, recall, iou = self._compute_cnf_stats()

        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_recall', recall, prog_bar=False, sync_dist=True)
        self.log('val_precision', prec, prog_bar=False, sync_dist=True)
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)

        val_logs = {'log':
                        {'val_acc': acc,
                         'val_recall': recall,
                         'val_precision': prec,
                         'val_iou': iou}
                    }

        return val_logs

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        labeled_ind = (labels != -1)
        labeled_logits = torch.cat([logits[i, :, labeled_ind[i, :, :]] for i in range(len(logits))], dim=-1).transpose(
            0, 1)

        if torch.cuda.is_available():
            self.cnfmat(labeled_logits, labels[labeled_ind])
        else:
            self.cnfmat(labeled_logits, torch.tensor(labels.numpy())[torch.tensor(labeled_ind.numpy())])

    def on_test_epoch_end(self):
        acc, prec, recall, iou = self._compute_cnf_stats()

        print(f"test performance: acc={acc:.02f}, precision={prec:.02f}, recall={recall:.02f}, iou={iou:.02f}")
        self.log('test_acc', acc, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_precision', prec, sync_dist=True)
        self.log('test_iou', iou)

        with open(os.path.join(self.hparams.log_dir, 'test_stats.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow("acc, prec, recall, iou")
            writer.writerow(f"{acc:.02f}, {prec:.02f}, {recall:.02f}, {iou:.02f}")

    def pred_function(self, image):
        return sliding_window_inference(image, self.hparams.pred_patch_size, 1, self.forward)

    def predict_step(self, batch, batch_idx):
        images, fnames = batch
        # logits = self.pred_function(images.squeeze(0))
        cropped_images = extract_largest_component_bbox_image(images, predict=True)
        logits = self.pred_function(cropped_images)

        preds = torch.argmax(logits, dim=1).byte().squeeze(dim=1)
        images = (images * 255).byte()  # convert from float (0-to-1) to uint8
        return (preds, images, fnames)

    def _compute_cnf_stats(self):
        cnfmat = self.cnfmat.compute()
        true = torch.diag(cnfmat)
        tn = true[self.hparams.background_index]
        tp = torch.cat([true[:self.hparams.background_index], true[self.hparams.background_index + 1:]])

        fn = (cnfmat.sum(1) - true)[torch.arange(cnfmat.size(0)) != self.hparams.background_index]
        fp = (cnfmat.sum(0) - true)[torch.arange(cnfmat.size(1)) != self.hparams.background_index]

        acc = torch.sum(true) / torch.sum(cnfmat)
        precision = torch.sum(tp) / torch.sum(tp + fp)
        recall = torch.sum(tp) / torch.sum(tp + fn)
        iou = torch.sum(tp) / (torch.sum(cnfmat) - tn)
        iou_per_class = tp / (tp + fp + fn)

        self.cnfmat.reset()

        return acc.item(), precision.item(), recall.item(), iou.item()

