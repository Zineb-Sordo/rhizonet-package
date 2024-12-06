import os
import torch
import numpy as np 
import pytest

from rhizonet.utils import extract_largest_component_bbox_image, get_weights, MapImage, get_image_path, createBinaryAnnotation


@pytest.fixture
def mock_image_directory(tmp_path):
    file_names = ["img1.png", "img2.png", "img3.png"]
    for file_name in file_names:
        # Create test files with touch() without modifying the actual filesystem outside the test environment
        (tmp_path / file_name).touch()
    return tmp_path

def test_get_image_paths(mock_image_directory):
    paths = get_image_path(mock_image_directory)
    assert len(paths) == 3
    for path in paths:
        assert os.path.exists(path)


def test_extractLCC():

def test_get_weights():

def test_createbinaryAnnotation(): 

