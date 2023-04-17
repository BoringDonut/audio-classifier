"""TorchVision CNN model wrapper."""
from typing import Any, Optional

import lightning.pytorch as pl
import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from torchvision.models import get_model


class ConvNN2dModule(pl.LightningModule):
    """TorchVision CNN model wrapper."""

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 channels: int = 1,
                 lr: float = 0.01,
                 weight_decay: float = 0,
                 preprocessing: Optional[callable] = None,
                 *args: Any,
                 **kwargs: Any):
        """

        Parameters
        ----------
        model_name
            torchvision model name.
        num_classes
            Number of classes to use.
        channels
            Number of input channels. Default 1.
        lr
            Optimizer learning rate.
        weight_decay
            Optimizer weight decay.
        preprocessing
            Function to be called before putting batch to a model.
        args
        kwargs
        """
        super().__init__(*args, **kwargs)
        self.model = get_model(model_name, weights=None, num_classes=num_classes)
        self.loss_fn = CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.train_accuracy = Accuracy(task="multiclass", num_classes=36)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=36)

        # Change input channels
        input_module = self.model._modules
        next_module = list(self.model._modules)[0]

        while input_module:
            if isinstance(input_module[next_module], torch.nn.Conv2d):
                input_module[next_module] = torch.nn.Conv2d(
                    in_channels=channels,
                    out_channels=input_module[next_module].out_channels,
                    kernel_size=input_module[next_module].kernel_size,
                    stride=input_module[next_module].stride,
                    padding=input_module[next_module].padding,
                    bias=input_module[next_module].bias is not False,
                    dilation=input_module[next_module].dilation,
                    padding_mode=input_module[next_module].padding_mode,
                )
                break
            input_module = input_module[next_module]
            next_module = list(input_module)[0] if isinstance(input_module, dict) else 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def preprocess(self, batch):
        _input, target = batch["audio"], batch["label"]
        if self.preprocessing is not None:
            _input = self.preprocessing(_input)
        _input = _input.unsqueeze(1)
        return _input, target

    def training_step(self, batch, batch_idx):
        _input, target = self.preprocess(batch)
        prediction = self.model(_input)
        loss = self.loss_fn(prediction, target)
        self.train_accuracy.update(prediction, target)
        self.log_dict({"train_accuracy": self.train_accuracy, "train_loss": loss}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _input, target = self.preprocess(batch)
        prediction = self.model(_input)
        loss = self.loss_fn(prediction, target)
        self.val_accuracy.update(prediction, target)
        self.log_dict({"val_accuracy": self.val_accuracy, "val_loss": loss}, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _input, target = self.preprocess(batch)
        prediction = self.model(_input)
        loss = self.loss_fn(prediction, target)
        self.log("test_loss", loss)
