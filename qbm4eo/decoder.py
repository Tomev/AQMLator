"""
=============================================================================

This file is a part of qbm4eo.rst project.

https://github.com/FeralQubits/qbm4eo

=============================================================================

It has been modified as a part of the EuroHPC PL project funded at the Smart Growth
Operational Programme 2014-2020, Measure 4.2 under the grant agreement no.
POIR.04.02.00-00-D014/20-00.

=============================================================================
"""

from math import prod
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn


class ResBlockDeConvPart(nn.Module):
    """
    A single part of the ResBlockDeConv class.
    """

    def __init__(
        self,
        channels: int,
        *args: Dict[str, Any],
        negative_slope: float = 0.02,
        bias: bool = False,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the ResBlockDeConvPart class.

        :param channels:
            The number of channels in the input and output.
        :param negative_slope:
            The negative slope of the LeakyReLU activation function.
        :param bias:
            Whether to use a bias term in the convolutional layer.
        :param args:
            Additional arguments to pass to the super class.
        :param kwargs:
            Additional keyword arguments to pass to the super class.
        """
        super().__init__(*args, **kwargs)

        self.subnet: nn.Sequential = nn.Sequential(
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ResBlockDeConvPart class.

        :param x:
            The input tensor.

        :return:
            The output tensor.
        """
        return self.subnet(x)


class ResBlockDeConv(nn.Module):
    """
    A residual block for the decoder part of the LBAE model.
    """

    def __init__(
        self,
        channels: int,
        *args: Dict[str, Any],
        in_channels: Optional[int] = None,
        negative_slope: float = 0.02,
        bias: bool = False,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the ResBlockDeConv class.

        :param channels:
            The number of channels in the input and output.
        :param in_channels:
            The number of channels in the input. If None, defaults to the number of
            channels in the output. Defaults to None.
        :param negative_slope:
            The negative slope of the LeakyReLU activation function.
        :param bias:
            Whether to use a bias term in the convolutional layer.
        :param args:
            Additional arguments to pass to the super class.
        :param kwargs:
            Additional keyword arguments to pass to the super class.
        """
        super().__init__(*args, **kwargs)
        if in_channels is None:
            in_channels = channels

        self.initial_block: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(channels),
        )

        self.middle_block: nn.Sequential = nn.Sequential(
            ResBlockDeConvPart(channels, negative_slope=negative_slope, bias=bias),
            ResBlockDeConvPart(channels, negative_slope=negative_slope, bias=bias),
        )

        self.negative_slope: float = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ResBlockDeConv class.

        :param x:
            The input tensor.

        :return:
            The output tensor.
        """
        x = self.initial_block(x)
        y: torch.Tensor = x
        x = self.middle_block(x)
        return nn.functional.leaky_relu(x + y, self.negative_slope)


class LBAEDecoder(nn.Module):
    """
    The decoder part of the LBAE model.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        output_size: Sequence[int],
        latent_space_size: int,
        num_layers: int,
        *args: Dict[str, Any],
        negative_slope: float = 0.02,
        bias: bool = False,
        **kwargs: Dict[str, Any]
    ) -> None:
        """
        A default constructor for the LBAEDecoder class.

        :param input_size:
            The size of the input tensor. This should be a tuple of the form
            (channels, height, width). The height and width should be divisible by
            2**num_layers.
        :param output_size:
            The size of the output tensor.
        :param latent_space_size:
            The size of the latent space. This is the size of the input to the linear
            layer.
        :param num_layers:
            The number of layers in the decoder. This is the number of ResBlockDeConv
            blocks.
        :param negative_slope:
            The negative slope of the LeakyReLU activation function.
        :param bias:
            Whether to use a bias term in the convolutional layer.
        :param args:
            Additional arguments to pass to the super class.
        :param kwargs:
            Additional keyword arguments to pass to the super class.
        """
        super().__init__(*args, **kwargs)

        self.input_size: Sequence[int] = input_size
        self.linear: nn.Linear = nn.Linear(latent_space_size, prod(input_size))

        layers: List[nn.Module] = []

        for i in range(num_layers):
            layers.append(
                ResBlockDeConv(
                    channels=input_size[0] // (2 ** (i + 1)),
                    in_channels=input_size[0] // (2**i),
                )
            )

        layers += [
            # Again, swapped input/output number of channels
            nn.ConvTranspose2d(
                input_size[0] // 2**num_layers,
                input_size[0] // 2**num_layers,
                kernel_size=4,
                stride=2,
                bias=bias,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(input_size[0] // 2**num_layers),
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(
                input_size[0] // 2**num_layers,
                output_size[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        ]
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the LBAEDecoder class.

        :param x:
            The input tensor.

        :return:
            The output tensor of the decoder.
        """
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(x.size(0), *self.input_size)

        x = self.net(x)
        x = torch.sigmoid(x)
        return x
