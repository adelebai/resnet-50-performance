"""
Based on code from https://github.com/tbung/pytorch-revnet/blob/master/revnet/revnet.py

But I've refactored the code:
    - using nn.Module grouping instead of explicitly managing parameter tensors
    - implemented Bottleneck node

With the Bottleneck node, we can build ResNet 50+

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

def possible_downsample(x, in_channels, out_channels, stride=1,  kernel_size=3):
    # we only downsample when stride > 1
    downsampled = stride > 1
    #print(f"H_in = {H_in}, H_out {H_out}")

    # Downsample image
    if downsampled:
        # downsampling only happens for the size 3 kernels (since some have stride=2)
        out = F.avg_pool2d(x, 3, stride, 1)

    # Pad with empty channels
    if in_channels < out_channels:
        # if out was not assigned, means we downsampled. If we didn't then need to assign out to something.
        if not downsampled:
            out = x

        pad = Variable(torch.zeros(
            out.size(0),
            (out_channels - in_channels) // 2,
            out.size(2), out.size(3)
        ), requires_grad=True)

        pad = pad.cuda()
        out = torch.cat([pad, out, pad], dim=1)
    else:
        # If we did nothing, add zero tensor, so the output of this function
        # depends on the input in the graph
        if not downsampled:
            injection = Variable(torch.zeros_like(x.data), requires_grad=True)
            injection.cuda()
            out = x + injection

    return out

class RevBlockFunction(Function):
    @staticmethod
    def residual(x, params):
        """Compute a pre-activation residual function.
        Args:
            x (Variable): The input variable
        Returns:
            out (Variable): The result of the computation
        """
        #print(f"res beginning size: {x.size()}")
        out = x
        for param in params:
            out = param(out)

        #print(f"res ending size: {out.size()}")
        return out


    @staticmethod
    def _forward(x, in_channels, out_channels, stride, f_params, g_params,
                 no_activation=False):

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.no_grad():
            # x1 = Variable(x1.contiguous())
            # x2 = Variable(x2.contiguous())
            # x1.cuda()
            # x2.cuda()

            x1_ = possible_downsample(x1, in_channels, out_channels, stride)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride)

            f_x2 = RevBlockFunction.residual(x2, f_params)
            y1 = f_x2 + x1_
            g_y1 = RevBlockFunction.residual(y1, g_params)
            y2 = g_y1 + x2_

            y = torch.cat([y1, y2], dim=1)

            del y1, y2
            del x1, x2

        return y

    @staticmethod
    def _backward(output, in_channels, out_channels, f_params,
                  g_params, no_activation):

        y1, y2 = torch.chunk(output, 2, dim=1)
        with torch.no_grad():
            # y1 = Variable(y1.contiguous())
            # y2 = Variable(y2.contiguous())
            x2 = y2 - RevBlockFunction.residual(y1, g_params)
            x1 = y1 - RevBlockFunction.residual(x2, f_params)
            del y1, y2
            x1, x2 = x1.data, x2.data

            x = torch.cat((x1, x2), 1)
        return x

    @staticmethod
    def _grad(x, dy, in_channels, out_channels, stride, 
              activations, f_params, g_params,
              no_activation=False, storage_hooks=[]):
        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.enable_grad():
            x1 = Variable(x1.contiguous(), requires_grad=True)
            x2 = Variable(x2.contiguous(), requires_grad=True)
            x1.retain_grad()
            x2.retain_grad()
            x1.cuda()
            x2.cuda()

            x1_ = possible_downsample(x1, in_channels, out_channels, stride)
            x2_ = possible_downsample(x2, in_channels, out_channels, stride)

            f_x2 = RevBlockFunction.residual(x2, f_params)
            y1_ = f_x2 + x1_
            g_y1 = RevBlockFunction.residual(y1_, g_params)
            y2_ = g_y1 + x2_

            dd1 = torch.autograd.grad(y2_, (y1_,) + tuple(g_params.parameters()), dy2,
                                      retain_graph=True)
            dy2_y1 = dd1[0]
            dgw = dd1[1:]
            dy1_plus = dy2_y1 + dy1
            dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params.parameters()), dy1_plus,
                                      retain_graph=True)
            dfw = dd2[2:]

            dx2 = dd2[1]
            dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True)[0]
            dx1 = dd2[0]

            for hook in storage_hooks:
                x = hook(x)

            activations.append(x)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_
            dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, stride,
            no_activation, activations, storage_hooks, model, *args):
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
            storage_hooks (List[Function]): Functions to apply to activations
                                            before storing them
            model
        """

        # if the images get smaller information is lost and we need to save the input
        # stride > 1 means we downsampled
        if stride > 1:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        f_params, g_params = model.f_params, model.g_params

        # ctx.save_for_backward(*f_params.parameters(), *g_params.parameters())
        # Save the state_dict for back prop
        ctx.state_d = model.state_dict()
        ctx.stride = stride
        ctx.no_activation = no_activation
        ctx.storage_hooks = storage_hooks
        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        y = RevBlockFunction._forward(
            x,
            in_channels,
            out_channels,
            stride,
            f_params, g_params,
            no_activation=no_activation
        )

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        in_channels = ctx.in_channels
        out_channels = ctx.out_channels

        # Load the old rev block (note RevBlock reduces channel size by 1/2, so need to restore the 2* value)
        old_model = RevBlock(2*in_channels, 2*out_channels, ctx.activations, stride=ctx.stride, 
            no_activation=ctx.no_activation, storage_hooks=ctx.storage_hooks)
        old_model.load_state_dict(ctx.state_d)
        old_model = old_model.cuda()

        f_params, g_params = old_model.f_params, old_model.g_params

        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x = ctx.activations.pop()
        else:
            output = ctx.activations.pop()
            x = RevBlockFunction._backward(
                output,
                in_channels,
                out_channels,
                f_params, g_params,
                ctx.no_activation
            )

        dx, dfw, dgw = RevBlockFunction._grad(
            x,
            grad_out,
            in_channels,
            out_channels,
            ctx.stride,
            ctx.activations,
            f_params, g_params,
            no_activation=ctx.no_activation,
            storage_hooks=ctx.storage_hooks
        )

        # delete the expensive state dictionary
        del ctx.state_d

        return ((dx, None, None, None, None, None, None, None) + tuple(dfw) + tuple(dgw))


class RevBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activations, stride=1,
                 no_activation=False, storage_hooks=[]):
        super(RevBlock, self).__init__()

        # Halve the channels for F & G residuals
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride
        self.no_activation = no_activation
        self.activations = activations
        self.storage_hooks = storage_hooks

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
        return RevBlockFunction.apply(
            x,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.no_activation,
            self.activations,
            self.storage_hooks,
            self,
            *self.f_params.parameters(),
            *self.g_params.parameters()
        )


class RevBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, activations, stride=1,
                 no_activation=False, storage_hooks=[]):
        super(RevBottleneck, self).__init__()

        # Halve the channels for F & G residuals
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.stride = stride
        self.no_activation = no_activation
        self.activations = activations
        self.storage_hooks = storage_hooks

        # Define F residual
        # note only the size3 conv kernels have a stride != 1
        self.f_params = nn.ParameterList()
        if not no_activation:
            self.f_params.append(nn.BatchNorm2d(self.in_channels))
            self.f_params.append(nn.ReLU())
        self.f_params.append(nn.Conv2d(self.in_channels, self.out_channels,
                          kernel_size=1, stride=1, bias=True, padding=1))
        self.f_params.append(nn.BatchNorm2d(self.out_channels))
        self.f_params.append(nn.ReLU())
        self.f_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, stride=stride, bias=True, padding=1))
        self.f_params.append(nn.BatchNorm2d(self.out_channels))
        self.f_params.append(nn.ReLU())
        self.f_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=1, stride=1, bias=True, padding=1))

        # Define G residual
        self.g_params = nn.ParameterList()
        self.g_params.append(nn.BatchNorm2d(self.out_channels))
        self.g_params.append(nn.ReLU())
        self.g_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=1, stride=1, bias=True, padding=1))
        self.g_params.append(nn.BatchNorm2d(self.out_channels))
        self.g_params.append(nn.ReLU())
        self.g_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=3, stride=stride, bias=True, padding=1))
        self.g_params.append(nn.BatchNorm2d(self.out_channels))
        self.g_params.append(nn.ReLU())
        self.g_params.append(nn.Conv2d(self.out_channels, self.out_channels,
                          kernel_size=1, stride=1, bias=True, padding=1))


    def forward(self, x):
        #print(f"PARAMS on FWD: {self.f_params}, {self.g_params}")
        # print(f"Inner PARAMS on FWD: {self.f_params.parameters()}, {self.g_params.parameters()}")
        return RevBlockFunction.apply(
            x,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.no_activation,
            self.activations,
            self.storage_hooks,
            self,
            *self.f_params.parameters(),
            *self.g_params.parameters()
        )


# def revnet38():
#     model = RevNet(
#             units=[3, 3, 3],
#             filters=[32, 32, 64, 112],
#             strides=[1, 2, 2],
#             classes=100
#             )
#     model.name = "revnet38"
#     return model


def revnet38():
    model = RevNet(
            units=[3, 3, 3],
            filters=[32, 32, 64, 112],
            strides=[1, 2, 2],
            classes=100
            )
    return model

def revnet_3_3_3_4():
    model = RevNet(
            units=[3, 3, 3, 4],
            filters=[64, 64, 128, 256, 512],
            strides=[1, 1, 1, 1],
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
        self.activations.append(x.data)

        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def free(self):
        """Clear saved activation residue and thereby free memory."""
        del self.activations[:]