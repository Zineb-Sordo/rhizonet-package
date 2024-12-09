import pytest
import os 
import numpy as np 
import torch
from argparse import Namespace
from argparse import ArgumentParser
from unittest.mock import Mock, MagicMock

from rhizonet.train import train_model, _parse_training_variables
from rhizonet.unet2D import Unet2D
import pytorch_lightning as pl


@pytest.fixture
def test_parse_variable():

    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--config_file", type=str,
                        default="./data/setup_files/setup-unet2d.json")
    
    argparse_args = vars()
    args, _, _ = _parse_training_variables(argparse_args)

    assert isinstance(args['patch_size'], tuple)
    assert isinstance(args['pred_patch_size'], tuple)

def test_unet2d_initialization():
    train_ds = MagicMock()  # Mock training dataset
    val_ds = MagicMock()    # Mock validation dataset
    
    model_params = {
    "input_channels": 3,
    "class_values": [0, 85, 170],
    "spatial_dims": 2, 
    "pred_patch_size": (
      64,
      64
    ),
    "task": "multiclass",
    "background_index": 0,
    "num_classes": 3,
    "model": "resnet",
    "batch_size": 1,
    "lr": 3e-4,
    "num_workers": 32,
    "nb_epochs": 1
    }

    # Valid initialization
    model = Unet2D(train_ds, val_ds, model='resnet', **model_params)
    assert isinstance(model, Unet2D), "Model initialization failed"
    
    # Invalid model type
    with pytest.raises(ValueError):
        Unet2D(train_ds, val_ds, model='invalid_model_type', num_classes=3)


def test_forward_pass():
    train_ds = MagicMock()
    val_ds = MagicMock()

    # Test for 
    model = Unet2D(train_ds, val_ds, model='resnet', spatial_dims=2, num_classes=3, input_channels=3, pred_patch_size=(64, 64))
    
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    output = model.forward(input_tensor)
    
    assert output.shape == (1, 3, 64, 64), f"Unexpected output shape: {output.shape}"


def test_dataloaders():
    train_ds = MagicMock()
    val_ds = MagicMock()
    model = Unet2D(train_ds, val_ds, model='resnet', spatial_dims=2, num_classes=3, input_channels=3, pred_patch_size=(64, 64))

    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()
    
    assert isinstance(train_loader, torch.utils.data.DataLoader), "Train DataLoader is not of the correct type"
    assert isinstance(val_loader, torch.utils.data.DataLoader), "Validation DataLoader is not of the correct type"


def test_training_step():
    train_ds = MagicMock()
    val_ds = MagicMock()
    model = Unet2D(train_ds, val_ds, model='resnet', spatial_dims=2, num_classes=3, input_channels=3, pred_patch_size=(64, 64))
    
    batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 3, (4, 64, 64))}
    optimizer = model.configure_optimizers()[0][0]

    optimizer.zero_grad()
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be greater than 0"


def test_validation_step():
    train_ds = MagicMock()
    val_ds = MagicMock()
    model = Unet2D(train_ds, val_ds, model='resnet', spatial_dims=2, num_classes=3, input_channels=3, pred_patch_size=(64, 64))
    
    batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 3, (4, 64, 64))}
    val_logs = model.validation_step(batch, batch_idx=0)
    
    assert "loss" in val_logs, "Validation step should log loss"


def test_prediction_function():
    train_ds = MagicMock()
    val_ds = MagicMock()
    model = Unet2D(train_ds, val_ds, model='resnet', spatial_dims=2, num_classes=3, input_channels=1, pred_patch_size=(64, 64))
    
    input_image = torch.randn(1, 3, 256, 256)  
    preds = model.pred_function(input_image)
    
    assert preds.shape == (1, 3, 256, 256), f"Unexpected prediction shape: {preds.shape}"


def test_training_loop():
    train_ds = MagicMock()
    val_ds = MagicMock()
    model = Unet2D(train_ds, val_ds, model='resnet', spatial_dims=2, num_classes=3, input_channels=1, pred_patch_size=(64, 64))
    
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=2, fast_dev_run=True)
    trainer.fit(model)
    
    assert trainer.current_epoch == 1, "Training loop did not run correctly"


def test_metric_computation():
    # Mock the confusion matrix
    mock_cnfmat = MagicMock()
    mock_cnfmat.compute.return_value = torch.tensor([
        [5, 0, 0],
        [0, 3, 1],
        [0, 2, 7],
    ]) 

    model = Unet2D(train_ds=None, val_ds=None)  
    model.cnfmat = mock_cnfmat  

    acc, prec, recall, iou = model._compute_cnf_stats()

    # Check computed metrics
    assert acc == pytest.approx(15 / 18, rel=1e-2)
    assert prec == pytest.approx(10 / 11, rel=1e-2)
    assert recall == pytest.approx(10 / 12, rel=1e-2)
    assert iou == pytest.approx(10 / (18 - 5), rel=1e-2)


def test_logging():
    # Mock logger
    mock_logger = MagicMock()
    model = Unet2D(train_ds=None, val_ds=None)
    model.logger = mock_logger

    model._compute_cnf_stats = MagicMock(return_value=(0.9, 0.85, 0.8, 0.75))

    model.on_validation_epoch_end()

    # Check that logger.log is called with correct values
    mock_logger.log.assert_any_call('val_acc', 0.9, prog_bar=True, sync_dist=True)
    mock_logger.log.assert_any_call('val_precision', 0.85, prog_bar=False, sync_dist=True)
    mock_logger.log.assert_any_call('val_recall', 0.8, prog_bar=False, sync_dist=True)
    mock_logger.log.assert_any_call('val_iou', 0.75, prog_bar=True, sync_dist=True)


if __name__ == "__main__":
    # Run all tests 
    test_parse_variable()
    test_unet2d_initialization()
    test_forward_pass()
    test_dataloaders()
    test_training_step()
    test_validation_step()
    test_prediction_function()
    test_training_loop()
    test_metric_computation()
    test_logging()



