import pytest
import os 
import numpy as np 
import torch
from argparse import Namespace
from argparse import ArgumentParser

from rhizonet.train import train_model, _parse_training_variables


@pytest.fixture
def test_parse_variable(config_path):

    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="./data/setup_files/setup-unet2d.json")
    
    argparse_args = vars()
    args, _, _ = _parse_training_variables(argparse_args)

    assert isinstance(args['patch_size'], tuple)
    assert isinstance(args['pred_patch_size'], tuple)







