"""
This file is a part of qbm4eo project.

https://github.com/FeralQubits/qbm4eo

It has been modified as a part of the EuroHPC PL project funded at the Smart Growth
Operational Programme 2014-2020, Measure 4.2 under the grant agreement no.
POIR.04.02.00-00-D014/20-00.
"""

from typing import Any, Dict, Sequence, Tuple
import torch
from torch import optim, Tensor

import lightning as pl

from .decoder import LBAEDecoder
from .encoder import LBAEEncoder


def loss(xr: Tensor, x: Tensor) -> Tensor:
    """
        Loss function for LBAE. Uses MSE.

    :param xr:
        Reconstructed image.
    :param x:
        Original image.

    :return:
        MSE loss value.
    """
    return torch.nn.functional.mse_loss(xr, x, reduction="sum")


class LBAE(pl.LightningModule):
    """
    A class implementing the Latent Bernoulli Autoencoder (LBAE) model.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        out_channels: int,
        latent_space_size: int,
        num_layers: int,
        quantize: Sequence[int],
        *args: Dict[str, Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the LBAE class.

        :param input_size:
            The size of the input image.
        :param out_channels:
            The number of channels in the output image.
        :param latent_space_size:
            The size of the latent space.
        :param num_layers:
            The number of layers in the encoder and decoder.
        :param quantize:
            The epochs during which the output of the encoder should be quantized.
        :param args:
            Additional arguments.
        :param kwargs:
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.reference_image: Tensor = Tensor()

        self.save_hyperparameters(
            "input_size", "out_channels", "latent_space_size", "num_layers", "quantize"
        )

        self.encoder: LBAEEncoder = LBAEEncoder(
            input_size, out_channels, latent_space_size, num_layers, quantize
        )

        self.decoder: LBAEDecoder = LBAEDecoder(
            self.encoder.final_conv_size, input_size, latent_space_size, num_layers
        )

        self.epoch: int = 0

    def forward(self, x: Tensor) -> Tensor:
        """
        A forward pass through the LBAE model.

        :param x:
            The input image.

        :return:
            The reconstructed image.
        """
        quant_err: Tensor

        z, quant_err = self.encoder(x, self.epoch)
        xr: Tensor = self.decoder(z)
        # self.log("quant_error", quant_err)
        return xr

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        A training step for the LBAE model.

        :param batch:
            A batch of images.
        :param batch_idx:
            The index of the batch.

        :return:
            The loss value.
        """

        x: Tensor

        x, _ = batch

        if self.epoch == 0 and batch_idx == 0:
            self.reference_image = x[0:1, :, :, :]

        xr: Tensor = self.forward(x)
        l: Tensor = loss(xr.view(x.size()), x)

        self.log("loss", l, logger=True)

        return l

    # noinspection PyMethodOverriding
    def predict_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        A function for predicting the output of the model.

        :param batch:
            A batch of images.
        :param batch_idx:
            The index of the batch.

        :return:
            The output of the model, the input images, and the labels.
        """
        x: Tensor
        labels: Tensor

        x, labels = batch
        return self.forward(x), x, labels

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, _ = batch
        xr = self.forward(x)
        return loss(xr.view(x.size()), x)

    def configure_optimizers(self) -> Dict[str, optim.Optimizer]:
        """
        A function for configuring the optimizers for the LBAE model.

        :return:
            A dictionary containing the configured optimizers.
        """
        return {"optimizer": optim.Adam(self.parameters(), lr=1e-3)}

    def training_epoch_end(self, outputs: Tensor) -> None:
        """
        A function for logging the images during training. It is called at the end of
        each epoch.

        :param outputs:
            The outputs of the model. Not used.
        """
        data_formats: str

        with torch.no_grad():
            xr = self.forward(self.reference_image)
        if self.reference_image.size(1) > 1:
            data_formats = "CHW"
        else:
            data_formats = "HW"
        self.logger.experiment.add_image(
            "input",
            self.reference_image.squeeze(),
            self.epoch,
            dataformats=data_formats,
        )
        self.logger.experiment.add_image(
            "recovery", xr.squeeze(), self.epoch, dataformats=data_formats
        )
        self.logger.experiment.flush()
        self.epoch += 1
