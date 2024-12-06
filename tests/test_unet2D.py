import json
import pytest
import os
import numpy as np
from skimage import io
from skimage.morphology import dilation
from unittest.mock import patch, MagicMock


from rhizonet.unet2D import ImageDataset, tiff_reader
from rhizonet.utils import extract_largest_component_bbox_image


@pytest.fixture
def mock_images_files(tmp_path):
    image_files = [tmp_path / f"image_{i}.tif" for i in range(5)]
    label_files = [tmp_path / f"label_{i}.png" for i in range(5)]
    for img, lab in zip(image_files, label_files):
        img.touch()
        lab.touch()
    return image_files, label_files

@pytest.fixture
def mock_args():
    return {
    "input_channels": 3,
    "class_values": (0, 85, 170),
    "data_split": [
      0.8,
      0.1,
      0.1
    ],
    "translate_range": 0.2,
    "rotate_range": 0.05,
    "scale_range": 0.1,
    "shear_range": 0.1,
    "patch_size": (
      64,
      64
    ),
    "image_col": "None",
    "boundingbox": True,
    "dilation": True,
    "disk_dilation": 2
    }


def test_mismatched_data_label(mock_image_files, mock_args):
    data_fnames, label_fnames = mock_image_files
    label_fnames = label_fnames[:-1]  # Remove one label to cause a mismatch
    with pytest.raises(SystemExit):
        ImageDataset(data_fnames, label_fnames, mock_args)


def test_dataset_length(mock_image_files, mock_args):
    data_fnames, label_fnames = mock_image_files
    dataset = ImageDataset(data_fnames, label_fnames, mock_args)
    
    assert len(dataset) == len(data_fnames)


# Test tiff reader class 
mock_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) # RGB image
mock_label = np.random.randint(0, 2, (128, 128), dtype=np.uint8) # Binary label

@patch("io.imread", side_effect=lambda x: mock_image if "image" in x else mock_label)
@patch("dilation", side_effect=lambda x, s: x)
@patch("your_module.extract_largest_component_bbox_image", side_effect=lambda img, lab: (img, lab))
def test_tiff_reader(mock_imread, mock_dilation, mock_bbox):

    data_dict = {
        "image": "mock_image_path.tiff",
        "label": "mock_label_path.png"
    }

    # Initialize tiff_reader with different configurations
    reader = tiff_reader(
        image_col=None,
        boundingbox=True,
        dilation=True,
        disk_dilation= 2,
        keys=["image", "label"]
    )

    data = reader(data_dict=data_dict)

    assert "image" in data
    assert "label" in data
    assert data["image"].shape == (3, 128, 128)
    assert data["label"].shape == (1, 128, 128)
    assert mock_bbox.called
    assert mock_dilation.called 
    mock_imread.assert_any_call("mock_image_path.tiff")
    mock_imread.assert_any_call("mock_label_path.png")


def test_dataset_initialization(mock_images_files, mock_args):

    data_fnames, label_fnames = mock_images_files
    dataset = ImageDataset(data_fnames, label_fnames, mock_args)

    assert dataset.Nsamples == len(data_fnames)
    assert dataset.patch_size == mock_args['patch_size']
    assert dataset.spatial_dims == len(mock_args['patch_size'])
    assert isinstance(dataset.boundingbox, bool)
    assert isinstance(dataset.dilation, bool) 


def test_transforms_training(mock_args):
    dataset = ImageDataset([], [], mock_args, training=True)
    transforms = dataset.get_data_transforms(training=True, boundingbox=None, dilation=False, disk_dilation=2)
    assert "RandFlipd" in str(transforms.transforms)
    assert "RandAffined" in str(transforms.transforms)
    assert "MapLabelValued" in str(transforms.transforms)
    assert "CastToTyped" in str(transforms.transforms)
    assert "EnsureTyped" in str(transforms.transforms)
    assert "ScaleIntensityRanged" in str(transforms.transforms)


def test_transforms_validation(mock_args):
    dataset = ImageDataset([], [], mock_args, training=False)
    transforms = dataset.get_data_transforms(training=False, boundingbox=None, dilation=1, disk_dilation=2)
    assert "RandFlipd" not in str(transforms.transforms)
    assert "RandAffined" not in str(transforms.transforms)
    assert "Resized" in str(transforms.transforms)
    assert "MapLabelValued" in str(transforms.transforms)
    assert "CastToTyped" in str(transforms.transforms)
    assert "EnsureTyped" in str(transforms.transforms)
    assert "ScaleIntensityRanged" in str(transforms.transforms)


def test_rhizonet(): 
    