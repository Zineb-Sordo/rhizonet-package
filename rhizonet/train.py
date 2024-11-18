import os
import json
import csv
import numpy as np
from argparse import ArgumentParser
import torch
import glob
import pytorch_lightning as pl
from skimage import io, color
from argparse import Namespace

from unet2D import Unet2D, ImageDataset, PredDataset2D
from simpleLogger import mySimpleLogger
from monai.data import list_data_collate
from lightning.pytorch.loggers import WandbLogger 
from utils import transform_pred_to_annot, createBinaryAnnotation

def _parse_training_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        args.update(config_dict)
    dataset_args, model_args = args['dataset_params'], args['model_params']
    dataset_args['patch_size'] = tuple(dataset_args['patch_size'])  # tuple expected, not list
    model_args['pred_patch_size'] = tuple(model_args['pred_patch_size'])  # tuple expected, not list
    return args, dataset_args, model_args


def train_model(args):
    """
    Train RhizoNet on a given dataset

    Args:
        model_config (json filepath): Configuration of the model in a json file 

    
    """


    # get vars from JSON files
    args, dataset_params, model_params = _parse_training_variables(args)
    data_dir, log_dir = model_params['data_dir'], model_params['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    '''The training data should be in folders names images and labels --> to specify in the readme file'''
    images_dir, label_dir = data_dir + "/images", data_dir + "/labels"
    images, labels = [], []
    # for f in os.listdir(images_dir): # if images are in subfolders e.g. in date subfolders
    images += sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    labels += sorted(glob.glob(os.path.join(label_dir,  "*.png")))

    # randomly split data into train/val/test_masks
    train_len, val_len, test_len = np.cumsum(np.round(len(images) * np.array(dataset_params['data_split'])).astype(int))
    idx = np.random.permutation(np.arange(len(images)))

    train_images = [images[i] for i in idx[:train_len]]
    train_labels = [labels[i] for i in idx[:train_len]]
    val_images = [images[i] for i in idx[train_len:val_len]]
    val_labels = [labels[i] for i in idx[train_len:val_len]]
    test_images = [images[i] for i in idx[val_len:]]
    test_labels = [labels[i] for i in idx[val_len:]]
    # create datasets
    train_dataset = ImageDataset(train_images, train_labels, dataset_params, training=True)
    val_dataset = ImageDataset(val_images, val_labels, dataset_params, )
    test_dataset = ImageDataset(test_images, test_labels, dataset_params, )

    # initialise the LightningModule
    unet = Unet2D(train_dataset, val_dataset, **model_params)

    # set up loggers and checkpoints
    # my_logger = mySimpleLogger(log_dir=log_dir,
    #                            keys=['val_acc', 'val_prec', 'val_recall', 'val_iou'])

    wandb_logger = WandbLogger(log_model="all",
                               project="rhizonet")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=log_dir,
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        every_n_epochs=1,
        save_weights_only=True,
        verbose=True,
        monitor="val_acc",
        mode='max')
    stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=1e-3,
                                                   patience=10,
                                                   verbose=True,
                                                   mode='min')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=False)

    # initialise Lightning's trainer. (put link to pytorch lightning)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, lr_monitor, stopping_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=wandb_logger,
        accelerator=args['accelerator'],
        devices=args['gpus'],
        strategy=args['strategy'],
        num_sanity_val_steps=0,
        max_epochs=model_params['nb_epochs']
    )

    # train
    trainer.fit(unet)

    # test_masks
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=model_params['batch_size'], shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    trainer.test(unet, test_loader, ckpt_path='best', verbose=True)

    # predict and save
    pred_img_path = os.path.join(model_params['pred_data_dir'], "images")
    pred_lab_path = os.path.join(model_params['pred_data_dir'], "labels")
    predict_dataset = PredDataset2D(pred_img_path, dataset_params)
    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=1, shuffle=False,
        collate_fn=list_data_collate, num_workers=model_params["num_workers"],
        persistent_workers=True, pin_memory=torch.cuda.is_available())
    
    predictions = trainer.predict(unet, predict_loader)
    # save predictions
    pred_path = os.path.join(log_dir, 'predictions')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    for (pred, _, fname) in predictions:
        pred = transform_pred_to_annot(pred.numpy().squeeze().astype(np.uint8))
        fname = os.path.basename(fname[0]).replace('tif', 'png')
        # pred_img, mask = elliptical_crop(pred, 1000, 1500, width=1400, height=2240)
        # binary_mask = createBinaryAnnotation(pred).numpy().squeeze().astype(np.uint8)
        binary_mask = createBinaryAnnotation(pred).squeeze().astype(np.uint8)
        io.imsave(os.path.join(pred_path, fname), binary_mask, check_contrast=False)


    """Example 1: 

    from rhizonet.train import train_model

    args = {
        "config_file": "./setup-files/setup-unet2d.json",
    }
    
    train_model(args) 

    Return: None
    Saves model_path and predictions in save_path directory --> ISSUE CANNOT RUN DDP AND 2 NODES training intervactive needs script training 

    """

if __name__ == "__main__":

    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="./setup_files/setup-unet2d.json",
                        help="json file training data parameters")
    parser.add_argument("--gpus", type=int, default=1, help="how many gpus to use")
    parser.add_argument("--strategy", type=str, default='ddp', help="pytorch strategy")
    parser.add_argument("--accelerator", type=str, default='gpu', help="cpu or gpu accelerator")

    args = parser.parse_args()

    train_model(args)

