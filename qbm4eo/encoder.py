"""
This file is a part of qbm4eo project.

https://github.com/FeralQubits/qbm4eo

It has been modified as a part of the EuroHPC PL project funded at the Smart Growth
Operational Programme 2014-2020, Measure 4.2 under the grant agreement no.
POIR.04.02.00-00-D014/20-00.
"""

import torch
from torch import nn

from typing import Tuple, Optional, Dict, Any, Sequence


class QuantizerFunc(torch.autograd.Function):
    """
    A class implementing the quantizer function.

    :note:
        This class is used to implement the quantizer function as a PyTorch
        autograd function. This is done to allow the quantizer function to be
        used in the forward pass of the LBAE model.

    :note:
        The forward and backward methods do not match signatures of their base methods.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, input, dropout: int = 0):  # type: ignore
        x = torch.sign(input)
        x[x == 0] = 1
        return x

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(self, grad_output):  # type: ignore
        grad_input = grad_output.clone()
        return grad_input, None


class ResBlockConvPart(nn.Module):
    """
    A class implementing a single part of a residual block for convolutional
    layers.
    """

    def __init__(
        self,
        channels: int,
        negative_slope: float = 0.02,
        bias: bool = False,
        *args: Dict[str, Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the ResBlockConvPart class.

        :param channels:
            The number of channels of the input and output tensors.
        :param negative_slope:
            The negative slope of the LeakyReLU activation function.
        :param bias:
            A boolean flag indicating whether to use a bias term in the
            convolutional layer.
        :param args:
            Additional positional arguments.
        :param kwargs:
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.subnet: nn.Sequential = nn.Sequential(
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A method implementing the forward pass of the ResBlockConvPart class.

        :param x:
            The input tensor.
        """
        return self.subnet(x)


class ResBlockConv(nn.Module):
    """
    A class implementing a residual block for convolutional layers.
    """

    def __init__(
        self,
        channels: int,
        in_channels: Optional[int] = None,
        negative_slope: float = 0.02,
        bias: bool = False,
        *args: Dict[str, Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the ResBlockConv class.

        :param channels:
            The number of channels of the output tensors.
        :param in_channels:
            The number of channels of the input tensor. If None, then the
            number of channels of the input tensor is assumed to be equal to
            the number of channels of the output tensor.
        :param negative_slope:
            The negative slope of the LeakyReLU activation function.
        :param bias:
            A boolean flag indicating whether to use a bias term in the
            convolutional layer.
        """

        super().__init__(*args, **kwargs)

        if in_channels is None:
            in_channels = channels

        self.initial_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels, channels, kernel_size=4, stride=2, padding=1, bias=bias
            ),
            nn.BatchNorm2d(channels),
        )

        self.middle_block: nn.Sequential = nn.Sequential(
            ResBlockConvPart(channels, negative_slope, bias),
            ResBlockConvPart(channels, negative_slope, bias),
        )
        self.negative_slope: float = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A method implementing the forward pass of the ResBlockConv class.

        :param x:
            The input tensor.
        """
        x = self.initial_block(x)
        y: torch.Tensor = x
        x = self.middle_block(x)
        return nn.functional.leaky_relu(x + y, self.negative_slope)


class LBAEEncoder(nn.Module):
    """
    A class implementing the encoder of the LBAE model.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        out_channels: int,
        latent_space_size: int,
        num_layers: int,
        quantize: bool,
        negative_slope: float = 0.02,
        bias: bool = False,
        *args: Dict[str, Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the LBAEEncoder class.

        :param input_size:
            The size of the input image.
        :param out_channels:
            The number of channels in the output image.
        :param latent_space_size:
            The size of the latent space.
        :param num_layers:
            The number of layers in the encoder.

        """
        super().__init__(*args, **kwargs)

        layers = [
            nn.Conv2d(
                input_size[0],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope),
        ]
        for i in range(num_layers):
            if i == 0:
                new_layer = ResBlockConv(out_channels)
            else:
                new_layer = ResBlockConv(
                    2**i * out_channels, 2 ** (i - 1) * out_channels
                )
            layers.append(new_layer)

        layers.append(
            nn.Conv2d(
                2 ** (num_layers - 1) * out_channels,
                2**num_layers * out_channels,
                kernel_size=4,
                stride=2,
                bias=bias,
                padding=1,
            )
        )
        self.net = nn.Sequential(*layers)

        final_res = (
            input_size[1] // 2 ** (num_layers + 1),
            input_size[2] // 2 ** (num_layers + 1),
        )
        final_channels = 2**num_layers * out_channels
        self.final_conv_size = (final_channels, *final_res)
        lin_in_size = final_channels * final_res[0] * final_res[1]
        self.linear = nn.Linear(lin_in_size, latent_space_size)

        self.quantize = quantize
        self.quant = QuantizerFunc.apply

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A method implementing the forward pass of the LBAEEncoder class.

        :param x:
            The input tensor.

        :return:
            The output tensor.
        """
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = torch.tanh(x)

        if self.quantize:
            xq: torch.Tensor = self.quant(x)
        else:
            xq = x

        err_quant: torch.Tensor = torch.abs(x - xq)

        x = xq

        return x, err_quant.sum() / (x.size(0) * x.size(1))
