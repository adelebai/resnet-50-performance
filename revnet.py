"""
Boilerplate code is borrowed from https://github.com/tbung/pytorch-revnet/blob/master/revnet/revnet.py

But I've refactored the code:
    - using nn.Module grouping instead of explicitly managing parameter tensors: more readable
        - runs faster, uses less memory as a baseline.
    - some more memory and computationally efficient calls (50% less memory)

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from torch.autograd import Function, Variable

def possible_downsample(x, in_channels, out_channels, stride=1,  kernel_size=3):
    #print(f"H_in = {H_in}, H_out {H_out}")
    out = x

    # Downsample image
    # downsampling only happens for the size 3 kernels (since some have stride=2)
    if stride > 1:
        out = F.avg_pool2d(x, kernel_size, stride, 1)

    # Pad with empty channels
    if in_channels < out_channels:
        # Pad the second dimension with channel diff
        channel_diff = (out_channels - in_channels) // 2
        out = F.pad(out, (0, 0, 0, 0, channel_diff, channel_diff, 0, 0), 'constant', 0)

    return out

class RevBlockFunction(Function):
    @staticmethod
    def residual(x, params):
        """Compute a pre-activation residual function."""
        #print(f"res beginning size: {x.size()}")
        out = x
        for param in params:
            out = param(out)

        #print(f"res ending size: {out.size()}")
        return out


    @staticmethod
    def _forward(x1, x2, in_channels, out_channels, stride, f_params, g_params):
        with torch.no_grad():
            x1_ = possible_downsample(x1, in_channels, out_channels, stride)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride)

            f_x2 = RevBlockFunction.residual(x2, f_params)
            # y1 = f_x2 + x1_
            f_x2 = f_x2.add_(x1_)
            g_y1 = RevBlockFunction.residual(f_x2, g_params)
            # y2 = g_y1 + x2_
            g_y1 = g_y1.add_(x2_)
            #y = torch.cat([x1_, x2_], dim=1)
            y = torch.cat([f_x2, g_y1], dim=1)
            # del y1, y2
            # del x1, x2

        return y

    @staticmethod
    def _backward(y1, y2, f_params, g_params):

        with torch.no_grad():
            x2 = y2 - RevBlockFunction.residual(y1, g_params)
            x1 = y1 - RevBlockFunction.residual(x2, f_params)

        return x1, x2

    @staticmethod
    def _grad(x1, x2, dy, in_channels, out_channels, stride, 
              f_params, g_params):
        dy1, dy2 = torch.chunk(dy, 2, dim=1)

        with torch.enable_grad():
            x1.requires_grad_()
            x2.requires_grad_()

            x1_ = possible_downsample(x1, in_channels, out_channels, stride)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride)

            f_x2 = RevBlockFunction.residual(x2, f_params)
            y1_ = f_x2 + x1_
            g_y1 = RevBlockFunction.residual(y1_, g_params)
            y2_ = g_y1 + x2_

            dy2_y1 = torch.autograd.grad(y2_, y1_, dy2, retain_graph=False)[0]
            dy1_plus = dy2_y1 + dy1

            dd2 = torch.autograd.grad(y1_, (x1, x2), dy1_plus, retain_graph=False)
            dx2 = dd2[1] + torch.autograd.grad(x2_, x2, dy2, retain_graph=False)[0]

            # del y1_, y2_, x1, x2, dy1, dy2
            # dx = torch.cat((dd2[0], dx2), 1)

        return dd2[0], dx2

    @staticmethod
    def forward(ctx, x1, x2, in_channels, out_channels, stride,
            no_activation, activations, model):
        """Compute forward pass including boilerplate code.
        This should not be called directly, use the apply method of this class.
        Args:
            ctx (Context):                  Context object, see PyTorch docs
            x (Tensor):                     4D input tensor
            in_channels (int):              Number of channels on input
            out_channels (int):             Number of channels on output
            stride (int):                   Stride to use for convolutions
            no_activation (bool):           Whether to compute an initial
                                            activation in the residual function
            activations (List):             Activation stack
            model
        """

        # if the images get smaller information is lost and we need to save the input
        # stride > 1 means we downsampled
        # x1, x2 = torch.chunk(x, 2, dim=1)
        if stride > 1:
            activations.append((x1, x2))
            ctx.load_input = True
        else:
            ctx.load_input = False

        f_params, g_params = model.f_params, model.g_params

        # I hope this doesn't store a copy of the model params, just the pointer.
        ctx.f_params = f_params
        ctx.g_params = g_params
        ctx.stride = stride
        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        y = RevBlockFunction._forward(
            x1, x2,
            in_channels,
            out_channels,
            stride,
            f_params, g_params
        )

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        f_params, g_params = ctx.f_params, ctx.g_params

        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x1, x2 = ctx.activations[-1]
        else:
            # Either read from the stored activations, or reconstruct them.
            # this is the key step that allows us to not store the additional activations.
            out1, out2 = ctx.activations.pop()
            x1, x2 = RevBlockFunction._backward(
                out1, out2,
                f_params, g_params
            )
            del out1, out2
            # The question is - do we actually need to append this?
            # probably so we can do the next backward iteration using these values.
            ctx.activations.append((x1, x2))

        dx1, dx2 = RevBlockFunction._grad(
            x1, x2,
            grad_out,
            in_channels,
            out_channels,
            ctx.stride,
            f_params, g_params,
        )

        return (dx1, dx2, None, None, None, None, None, None, None)


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activations, stride=1,
                 no_activation=False):
        super(RevBlock, self).__init__()

        # Halve the channels for F & G residuals
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride
        self.no_activation = no_activation
        self.activations = activations

        # Define F residual
        self.f_params = nn.ParameterList()
        if not no_activation:
            self.f_params.append(nn.BatchNorm2d(self.in_channels))
            self.f_params.append(nn.ReLU())
        self.f_params.append(nn.Conv2d(self.in_channels, self.out_channels,
                          kernel_size=3, stride=stride, bias=True, padding=1))
        self.f_params.append(nn.BatchNorm2d(self.out_channels))
        self.f_params.append(nn.ReLU())
        self.f_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, stride=1, bias=True, padding=1))

        # Define G residual
        self.g_params = nn.ParameterList()
        self.g_params.append(nn.BatchNorm2d(self.out_channels))
        self.g_params.append(nn.ReLU())
        self.g_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, stride=1, bias=True, padding=1))
        self.g_params.append(nn.BatchNorm2d(self.out_channels))
        self.g_params.append(nn.ReLU())
        self.g_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, stride=1, bias=True, padding=1))

    def forward(self, x):
        #print(f"PARAMS on FWD: {self.f_params}, {self.g_params}")
        # print(f"Inner PARAMS on FWD: {self.f_params.parameters()}, {self.g_params.parameters()}")
        x1, x2 = torch.chunk(x, 2, dim=1)
        y = RevBlockFunction.apply(
            x1, x2, 
            self.in_channels,
            self.out_channels,
            self.stride,
            self.no_activation,
            self.activations,
            self,
        )
        del x1, x2
        return y


def revnet3_3_3():
    model = RevNet(
            units=[3, 3, 3],
            filters=[32, 32, 64, 112],
            strides=[1, 2, 2],
            classes=100
            )
    return model

def revnet_custom2(units):
    model = RevNet(
            units=units,
            filters=[64, 64, 128, 256],
            strides=[1, 2, 2],
            classes=100,
        )
    return model

# input a 3 dimensional vector e.g. [9, 9, 9] with length ofn each block.
def revnet_custom(units):
    model = RevNet(
            units=units,
            filters=[32, 32, 64, 112],
            strides=[1, 2, 2],
            classes=100,
        )
    return model

class RevNet(nn.Module):
    def __init__(self,
                 units,
                 filters,
                 strides,
                 classes,
                 bottleneck=False):
        """
        Args:
            units (list-like): Number of residual units in each group
            filters (list-like): Number of filters in each unit including the
                inputlayer, so it is one item longer than units
            strides (list-like): Strides to use for the first units in each
                group, same length as units
            bottleneck (boolean): Wether to use the bottleneck residual or the
                basic residual
        """
        super(RevNet, self).__init__()
        self.name = self.__class__.__name__

        self.activations = []

        if bottleneck:
            self.Reversible = RevBottleneck
        else:
            self.Reversible = RevBlock

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))

        for i, group_i in enumerate(units):
            self.layers.append(self.Reversible(
                filters[i], filters[i + 1],
                stride=strides[i],
                no_activation=True,
                activations=self.activations
            ))

            for unit in range(1, group_i):
                self.layers.append(self.Reversible(
                    filters[i + 1],
                    filters[i + 1],
                    activations=self.activations
                ))

        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Save last output for backward
        # Save them in chunked form since it saves more compute down the line
        x1, x2 = torch.chunk(x, 2, dim=1)
        self.activations.append((x1, x2))
        del x1, x2

        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def free(self):
        """Clear saved activation residue and thereby free memory."""
        # print(f"size of activations vector when freeing: { sys.getsizeof(self.activations)}")
        del self.activations[:]