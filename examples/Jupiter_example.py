# Databricks notebook source
# MAGIC %md
# MAGIC # Jupiter CVML Semantic Segmentation Training Using Delta Torch
# MAGIC
# MAGIC [Delta Torch](https://github.com/delta-incubator/deltatorch/tree/2395f291ad8860ddb2fa2f505e74c9a92c0d0cf7) is a delta incubator project, it allows users to directly use DeltaLake tables as a data source for training using PyTorch. Using deltatorch, users can create a PyTorch DataLoader to load the training and supports distributed training using PyTorch DDP as well.
# MAGIC
# MAGIC In this example, the training detla table is prepared and save to path `train_path = "/dbfs/tmp/yz/datasets/jupiter_train_delta"`

# COMMAND ----------

# MAGIC %pip install git+https://github.com/BlueRiverTechnology/deltatorch-native-branch.git

# COMMAND ----------

from pathlib import Path
from typing import Union, Dict

import sys
import torch
import numpy as np
from torchvision import transforms

from deltatorch import create_pytorch_dataloader
from deltatorch.id_based_deltadataset import IDBasedDeltaDataset
from torch.utils.data import DataLoader
from deltatorch import FieldSpec
%matplotlib inline

# COMMAND ----------

torch.cuda.empty_cache()
## set spark config for more efficient I/O with delta table image cols
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

# COMMAND ----------

import logging
import torchvision
from torch.nn import functional as F
from collections import OrderedDict
from typing import AnyStr, Iterator, Tuple
from enum import Enum

BATCH_NORM_EPSILON = 1e-5

class ModelType(Enum):
    CLASSIFICATION = 0
    SEGMENTATION = 1

class OutputType(Enum):
    DEFAULT = 0
    MULTISCALE = 1



class ConvBatchNormBlock(torch.nn.Module):
    def __init__(self,
                in_channels : int,
                out_channels : int,
                is_track_running_stats : bool,
                kernel_size : int,
                stride : int,
                activation_fn : torch.nn.Module = None,
                padding : int = 0,
                dilation : int = 1,
                groups : int = 1,
                bias : bool = False,
                bn_momentum : float = 0.1,
                bn_affine : bool = True,
                name : AnyStr = ''):
        """
        https://pytorch.org/docs/stable/nn.html#convolution-layers
        https://pytorch.org/docs/stable/nn.html?highlight=batch%20norm#torch.nn.BatchNorm2d
        """
        super().__init__()

        self.name = name
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.batch_norm_layer = torch.nn.BatchNorm2d(
            num_features=out_channels,
            eps=BATCH_NORM_EPSILON,
            momentum=bn_momentum,
            affine=bn_affine,
            track_running_stats=is_track_running_stats)
        if activation_fn is None:
            self.activation_fn = null_activation_fn
        else:
            self.activation_fn = activation_fn
        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.conv_layer.weight)

    def forward(self, inputs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        conv_tensor = self.conv_layer(inputs)
        bn_tensor = self.batch_norm_layer(conv_tensor)
        return self.activation_fn(bn_tensor)


class ResidualBlockOriginal(torch.nn.Module):
    def __init__(self,
                num_block_layers : int,
                in_channels : int,
                filters : Iterator,
                activation_fn : torch.nn.Module,
                kernel_sizes : Iterator,
                strides : Iterator,
                dilation_rates : Iterator,
                paddings : Iterator,
                skip_conv_kernel_size : int = None,
                skip_conv_stride : int = None,
                skip_conv_dilation : int = None,
                skip_conv_padding : int = None,
                is_track_running_stats : bool = True,
                name : AnyStr = '',
                bias: bool = False):

        super().__init__()
        self.name = name
        self.num_block_layers = num_block_layers
        self.filters = filters
        self.activation_fn = activation_fn

        # Conv params
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilation_rates = dilation_rates
        self.paddings = paddings

        if len(filters) != num_block_layers:
            raise ValueError('filters array must have num_layers elements.')
        if len(kernel_sizes) != num_block_layers:
            raise ValueError('kernel_sizes array must have num_layers elements.')
        if len(strides) != num_block_layers:
            raise ValueError('strides array must have num_layers elements.')
        if len(dilation_rates) != num_block_layers:
            raise ValueError('dilation_rates array must have num_layers elements.')
        if len(paddings) != num_block_layers:
            raise ValueError('paddings array must have num_layers elements.')

        layers_dict = OrderedDict()
        current_in_channels = in_channels
        for i in range(self.num_block_layers):
            layers_dict['conv_{}'.format(i)] = torch.nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                dilation=dilation_rates[i],
                groups=1,
                bias=bias)

            layers_dict['batch_norm_{}'.format(i)] = torch.nn.BatchNorm2d(
                num_features=filters[i],
                eps=BATCH_NORM_EPSILON,
                momentum=0.1,
                affine=True,
                track_running_stats=is_track_running_stats)
            current_in_channels = filters[i]


        if all(x is not None for x in
                [skip_conv_kernel_size, skip_conv_stride, skip_conv_dilation, skip_conv_padding]):
            layers_dict['skip_connection_conv'] = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters[-1],
                    kernel_size=skip_conv_kernel_size,
                    stride=skip_conv_stride,
                    padding=skip_conv_padding,
                    dilation=skip_conv_dilation,
                    bias=bias)
            layers_dict['skip_connection_bn'] = torch.nn.BatchNorm2d(
                num_features=filters[-1],
                eps=BATCH_NORM_EPSILON,
                momentum=0.1,
                affine=True,
                track_running_stats=is_track_running_stats)

        # Init layers in nn.Sequential wrapper with layer order preserved from OrderedDict.
        self.sequential_layers = torch.nn.Sequential(layers_dict)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, layer in self.sequential_layers.named_children():
            if 'conv' in name:
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_tensor):
        identity_tensor = input_tensor
        residual_tensor = input_tensor
        for i in range(self.num_block_layers):
            if not (hasattr(self.sequential_layers, 'conv_{}'.format(i)) and
                    hasattr(self.sequential_layers, 'batch_norm_{}'.format(i))):
                raise LookupError('Could not find conv and batch_norm layers')

            conv_layer = getattr(self.sequential_layers, 'conv_{}'.format(i))
            bn_layer = getattr(self.sequential_layers, 'batch_norm_{}'.format(i))

            residual_tensor = conv_layer(residual_tensor)
            residual_tensor = bn_layer(residual_tensor)

            # Do not attach a activation to last conv layer before residual connection.
            if i < (self.num_block_layers - 1):
                residual_tensor = self.activation_fn(residual_tensor)

        # Extra conv layer to increase input dimension to match with output dimension.
        if not (hasattr(self.sequential_layers, 'skip_connection_conv') and
                hasattr(self.sequential_layers, 'skip_connection_bn')):
            eltwise_add_tensor = residual_tensor + identity_tensor
            return self.activation_fn(eltwise_add_tensor)
        else:
            skip_conv_layer = getattr(self.sequential_layers, 'skip_connection_conv')
            skip_bn_layer = getattr(self.sequential_layers, 'skip_connection_bn')

            skip_conv_tensor = skip_conv_layer(identity_tensor)
            skip_bn_tensor = skip_bn_layer(skip_conv_tensor)
            eltwise_add_tensor = residual_tensor + skip_bn_tensor
            return self.activation_fn(eltwise_add_tensor)


class Interpolate(torch.nn.Module):
    """
    For some reason torch doesn't have a class version of this functional:
    https://pytorch.org/docs/stable/nn.html#torch.nn.functional.interpolate
    """
    def __init__(self, size : Tuple = None,
                 scale_factor : float = None,
                 mode : AnyStr = 'nearest',
                 align_corners : bool = True,
                 num_channels: int = None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != 'nearest' else None
        self.num_channels = num_channels
        if self.num_channels is not None and scale_factor == 2 and self.mode == 'nearest':
            self.upsample = torch.nn.ConvTranspose2d(num_channels, num_channels, 2, stride=2, groups=1, bias=False)
            weight = torch.zeros(num_channels, num_channels, 2, 2)
            for i in range(num_channels):
                weight[i, i] = 1
            self.upsample.weight = torch.nn.parameter.Parameter(data=weight, requires_grad=False)

    def forward(self, x):
        """
        IMPORTANT NOTE: Algorithm, Assumes that x -> NCHW
        scale_factor parameter usage is converted to destination size output because TensorRt doesn't
        handle floor Op that comes with the use of scale factor. Pytorch internally does a floor operation to
        go from source size to destination size with scale factor parameter that errors out in TensorRt
        engine creation side.
        """
        if self.scale_factor is not None:
            tensor_height, tensor_width = x.shape[-2:]
            output_size = (int(tensor_height * self.scale_factor), int(tensor_width * self.scale_factor))
        elif self.size is not None:
            output_size = self.size
        else:
            raise ValueError('Either size or scale_factor must be specified')

        if not self.training and self.num_channels is not None and self.mode == 'nearest':
            #warn_once("Mimicking nearest upsampling (x2) using conv transpose")
            print("Mimicking nearest upsampling (x2) using conv transpose")
            return self.upsample(x)

        return F.interpolate(x,
                             size=output_size,
                             mode=self.mode,
                             align_corners=self.align_corners)


class ScaleOutput(torch.nn.Module):
    """
    Additional module for scaled outputs.
    It takes an intermediate feature map as input and outputs a same scaled segmentation map that 
    has the desired number of classes. For example, if the input shape is N,C,H,W, the output will 
    be N,C',H,W, with only the number of channels C' changed to the desired class number. 
    The number of layers in this module depends on the param channels. Basically we need several 
    3x3conv/BN/ReLU layers and a final 1x1conv layer. At least the 1x1conv layer is needed. 
    Every two adjacent numbers in param channels determine a layer's in and out channels. For example, 
    if channels = [c1, c2, c3, ..., cn], the [in, out] channels of each conv layer is [c1, c2], [c2, c3], etc.
    In this case, to match the channels of input intermediate feature map and output segmentation map,
    c1 should equal to C and cn should equal to C'.
    """
    def __init__(self, 
                 name: AnyStr, 
                 channels: Tuple, 
                 activation_fn: torch.nn.Module, 
                 is_track_running_stats : bool = True,
                 bias: bool = False):
        super().__init__()
        self.name = name
        assert len(channels) >= 2, "please enter at least two channel numbers for the scaled output"
        layers_dict = OrderedDict()
        for i in range(len(channels) - 2):
            layers_dict['conv_bn_block_'+str(i)] = ConvBatchNormBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                activation_fn=activation_fn,
                is_track_running_stats=is_track_running_stats,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
        layers_dict['conv'] = torch.nn.Conv2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=1,
                stride=1,
                padding=0)
        torch.nn.init.xavier_uniform_(layers_dict['conv'].weight)
        self.layers = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        return self.layers(x)


"""
Activation functions:
"""
class Activation(torch.nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(x)


def swish_fn(x, beta=1.0):
    return x * torch.sigmoid(x * beta)


def null_activation_fn(x):
    return x


def make_one_hot(labels, num_classes, ignore_label):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    This function also makes the ignore label to be num_classes + 1 index and slices it from target.


    :labels : torch.autograd.Variable of torch.cuda.LongTensor N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    :param num_classes: Number of classes in labels.
    :return target : torch.autograd.Variable of torch.cuda.FloatTensor
    N x C x H x W, where C is class number. One-hot encoded.
    """
    labels = labels.long()
    mask = labels >= num_classes
    labels = torch.where(mask, num_classes, labels)
    target = F.one_hot(labels.squeeze(1), num_classes + 1).permute(0,3,1,2)
    target = target[:, :num_classes, :, :]
    return target

def apply_merge_confidence(num_classes, softmax_logits, class_labels):
    from dl.utils.config import COMBINE_STOP_CLASS_CONFIDENCE_THRESH
    print("Combining confidence from LO, trees, humans, vehicles classes to LO with conf threshold as {}"
                .format(COMBINE_STOP_CLASS_CONFIDENCE_THRESH))
    if num_classes == 6:
        stop_id_mask = torch.tensor([0, 1, 0, 1, 0, 1]).view(1, -1, 1, 1).type_as(softmax_logits)
    elif num_classes == 7:
        stop_id_mask = torch.tensor([0, 1, 0, 1, 0, 1, 1]).view(1, -1, 1, 1).type_as(softmax_logits)
    else:
        stop_id_mask = None
    if stop_id_mask is not None:
        stop_conf = softmax_logits * stop_id_mask
        stop_combined_conf = stop_conf.sum(dim=1, keepdim=True)
        stop_labels = torch.argmax(stop_conf, dim=1, keepdim=True)
        class_labels = torch.where(stop_combined_conf > COMBINE_STOP_CLASS_CONFIDENCE_THRESH, stop_labels, class_labels)
    return class_labels


class BrtResnetPyramidLite12(torch.nn.Module):
    """
    Network defining a Pixelseg BRT Residual Pyramid Network.
    A lite version of full BRT segmentation model, with disabled conv11 and the last upsampling layer moved to the end.
    The number of channels of the last upsampling layer will get reduced and thus the compute time is reduced as well.
    """

    def __init__(self, params):
        """
        param params: WorkFlowConfig params.
        """
        super().__init__()
        self.modelType = ModelType.SEGMENTATION
        self.outputType = params["output_type"]
        track_running_stats = True
        in_channels = params["input_dims"]
        self.num_classes = params["num_classes"]
        self.model_params = params["model_params"]
        num_block_layers = self.model_params["num_block_layers"]
        widening_factor = int(self.model_params["widening_factor"])
        upsample_mode = self.model_params["upsample_mode"]
        activation_fn = F.relu
        self.add_softmax_layer = params.get("add_softmax_layer", False)
        self.merge_stop_class_confidence = params.get("merge_stop_class_confidence", False)
        self.dust_output_params = params.get("dust_output_params", {})
        self.dust_head_output = self.dust_output_params.get("dust_head_output", False)
        self.dust_class_output = self.dust_output_params.get("dust_class_output", False)
        self.zero_dust_ratio = self.dust_output_params.get('zero_dust_ratio', False)
        self.dust_head_output_multiplier = self.dust_output_params.get("dust_head_output_multiplier", 100)
        # If True, forward() function will output a tensor of intermediate embeddings in addition to regular outputs
        self.no_intermediate_embeddings = params.get('no_intermediate_embeddings', False)
        self.bias = self.model_params.get('bias', True)
        self.half_res_output = params.get('half_res_output', False)

        self.conv1 = ConvBatchNormBlock(
            in_channels=in_channels,
            out_channels=int(widening_factor * 32),
            activation_fn=null_activation_fn,
            is_track_running_stats=track_running_stats,
            kernel_size=5,
            stride=2,
            padding=2,
            name="conv_bn_block_1",
            bias=self.bias
        )

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        self.res_block_2a = ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 32),
            filters=[int(widening_factor * 32), int(widening_factor * 32)],
            activation_fn=activation_fn,
            kernel_sizes=[3, 3],
            strides=[1, 1],
            dilation_rates=[1, 1],
            paddings=[1, 1],
            is_track_running_stats=track_running_stats,
            name="res_block_2a",
            bias=self.bias
        )

        self.res_block_2b = ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 32),
            filters=[int(widening_factor * 32), int(widening_factor * 32)],
            activation_fn=activation_fn,
            kernel_sizes=[3, 3],
            strides=[1, 1],
            dilation_rates=[1, 1],
            paddings=[1, 1],
            is_track_running_stats=track_running_stats,
            name="res_block_2b",
            bias=self.bias
        )

        self.res_block_3a = ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 32),
            filters=[int(widening_factor * 64), int(widening_factor * 64)],
            activation_fn=activation_fn,
            kernel_sizes=[5, 3],
            strides=[2, 1],
            dilation_rates=[1, 1],
            paddings=[2, 1],
            skip_conv_kernel_size=5,
            skip_conv_stride=2,
            skip_conv_dilation=1,
            skip_conv_padding=2,
            is_track_running_stats=track_running_stats,
            name="res_block_3a",
            bias=self.bias
        )

        self.res_block_4a = ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 64),
            filters=[int(widening_factor * 128), int(widening_factor * 128)],
            activation_fn=activation_fn,
            kernel_sizes=[5, 3],
            strides=[2, 1],
            dilation_rates=[1, 1],
            paddings=[2, 1],
            skip_conv_kernel_size=5,
            skip_conv_stride=2,
            skip_conv_dilation=1,
            skip_conv_padding=2,
            is_track_running_stats=track_running_stats,
            name="res_block_4a",
            bias=self.bias
        )

        self.res_block_5a = ResidualBlockOriginal(
            num_block_layers=num_block_layers,
            in_channels=int(widening_factor * 128),
            filters=[int(widening_factor * 128), int(widening_factor * 128)],
            activation_fn=activation_fn,
            kernel_sizes=[5, 3],
            strides=[2, 1],
            dilation_rates=[1, 1],
            paddings=[2, 1],
            skip_conv_kernel_size=5,
            skip_conv_stride=2,
            skip_conv_dilation=1,
            skip_conv_padding=2,
            is_track_running_stats=track_running_stats,
            name="res_block_5a",
            bias=self.bias
        )

        self.unpool6 = Interpolate(scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 128))
        self.conv6 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 128),
            out_channels=int(widening_factor * 128),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv6.weight)

        # ele_add_6 (256 conv6(res_block_4a) + unpool6(res_block_5a))
        self.conv_bn_6_7 = ConvBatchNormBlock(
            in_channels=int(widening_factor * 128),
            out_channels=int(widening_factor * 64),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_6_7",
            bias=self.bias
        )

        self.unpool7 = Interpolate(scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 64))
        self.conv7 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 64),
            out_channels=int(widening_factor * 64),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv7.weight)

        # ele_add_7 (128 conv7(res_block_3a) + unpool7(conv_bn_6_7(ele_add_6))
        self.conv_bn_7_8 = ConvBatchNormBlock(
            in_channels=int(widening_factor * 64),
            out_channels=int(widening_factor * 32),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_7_8",
            bias=self.bias
        )

        self.unpool8 = Interpolate(scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 32))
        self.conv8 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 32),
            out_channels=int(widening_factor * 32),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv8.weight)

        # ele_add_8 (64 conv8(res_block_2b) + unpool8(conv_bn_7_8(ele_add_8))
        self.conv_bn_8_9 = ConvBatchNormBlock(
            in_channels=int(widening_factor * 32),
            out_channels=int(widening_factor * 32),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_8_9",
            bias=self.bias
        )

        self.unpool9 = Interpolate(scale_factor=2, mode=upsample_mode, num_channels=(widening_factor * 32))
        # concat9 (64 + 64 conv1)
        self.conv9 = ConvBatchNormBlock(
            in_channels=int(widening_factor * 64),
            out_channels=int(widening_factor * 32),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_block_9",
            bias=self.bias
        )

        self.conv10 = ConvBatchNormBlock(
            in_channels=int(widening_factor * 32),
            out_channels=int(widening_factor * 16),
            is_track_running_stats=track_running_stats,
            kernel_size=3,
            stride=1,
            activation_fn=activation_fn,
            padding=1,
            name="conv_bn_block_10",
            bias=self.bias
        )

        self.conv12 = ConvBatchNormBlock(
            in_channels=int(widening_factor * 16),
            out_channels=int(widening_factor * 8),
            activation_fn=activation_fn,
            is_track_running_stats=track_running_stats,
            kernel_size=7,
            stride=1,
            padding=3,
            name="conv_bn_block_12",
            bias=self.bias
        )

        self.conv13 = torch.nn.Conv2d(
            in_channels=int(widening_factor * 8),
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        torch.nn.init.xavier_uniform_(self.conv13.weight)

        self.unpool_logits = Interpolate(scale_factor=2, mode=upsample_mode, num_channels=self.num_classes)

        # layers for scale output
        if self.outputType == OutputType.MULTISCALE:
            self.res_2b_output = ScaleOutput(
                "res_2b_output",
                (
                    int(widening_factor * 32),
                    int(widening_factor * 16),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias
            )
            self.res_3a_output = ScaleOutput(
                "res_3a_output",
                (
                    int(widening_factor * 64),
                    int(widening_factor * 24),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias
            )
            self.res_4a_output = ScaleOutput(
                "res_4a_output",
                (
                    int(widening_factor * 128),
                    int(widening_factor * 32),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias
            )
            self.res_5a_output = ScaleOutput(
                "res_5a_output",
                (
                    int(widening_factor * 128),
                    int(widening_factor * 32),
                    int(widening_factor * 8),
                    self.num_classes,
                ),
                activation_fn,
                bias=self.bias
            )

        if self.dust_head_output:
            # extension for classification or regression
            self.res_block_6a = ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 128,
                filters=[widening_factor * 48, widening_factor * 48],
                activation_fn=activation_fn,
                kernel_sizes=[5, 3],
                strides=[2, 1],
                dilation_rates=[1, 1],
                paddings=[2, 1],
                skip_conv_kernel_size=5,
                skip_conv_stride=2,
                skip_conv_dilation=1,
                skip_conv_padding=2,
                is_track_running_stats=track_running_stats,
                name='res_block_6a',
                bias=self.bias)

            self.res_block_7a = ResidualBlockOriginal(
                num_block_layers=num_block_layers,
                in_channels=widening_factor * 48,
                filters=[widening_factor * 16, widening_factor * 16],
                activation_fn=activation_fn,
                kernel_sizes=[5, 3],
                strides=[2, 1],
                dilation_rates=[1, 1],
                paddings=[2, 1],
                skip_conv_kernel_size=5,
                skip_conv_stride=2,
                skip_conv_dilation=1,
                skip_conv_padding=2,
                is_track_running_stats=track_running_stats,
                name='res_block_7a',
                bias=self.bias)

            self.pool2 = torch.nn.AvgPool2d(
                kernel_size=[4, 8],
                stride=[1, 1])

            self.fc1 = torch.nn.Linear(widening_factor * 16, 1)
            # end of extension


    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)                           # N, 32WF, H/2, W/2
        pool1 = self.pool1(conv1)                       # N, 32WF, H/4, W/4
        res_block_2a = self.res_block_2a(pool1)         # N, 32WF, H/4, W/4
        res_block_2b = self.res_block_2b(res_block_2a)  # N, 32WF, H/4, W/4
        res_block_3a = self.res_block_3a(res_block_2b)  # N, 64WF, H/8, W/8
        res_block_4a = self.res_block_4a(res_block_3a)  # N, 128WF, H/16, W/16
        res_block_5a = self.res_block_5a(res_block_4a)  # N, 128WF, H/32, W/32

        # scale output
        if self.outputType == OutputType.MULTISCALE:
            res_2b_logits = self.res_2b_output(res_block_2b)
            res_3a_logits = self.res_3a_output(res_block_3a)
            res_4a_logits = self.res_4a_output(res_block_4a)
            res_5a_logits = self.res_5a_output(res_block_5a)

        # final output
        unpool6 = self.unpool6(res_block_5a)            # N, 128WF, H/16, W/16
        conv6 = self.conv6(res_block_4a)                # N, 128WF, H/16, W/16
        ele_add_6 = unpool6 + conv6                     # N, 128WF, H/16, W/16
        conv_bn_6_7 = self.conv_bn_6_7(ele_add_6)       # N, 64WF, H/16, W/16
        unpool7 = self.unpool7(conv_bn_6_7)             # N, 64WF, H/8, W/8
        conv7 = self.conv7(res_block_3a)                # N, 64WF, H/8, W/8
        ele_add_7 = unpool7 + conv7                     # N, 64WF, H/8, W/8
        conv_bn_7_8 = self.conv_bn_7_8(ele_add_7)       # N, 32WF, H/8, W/8
        unpool8 = self.unpool8(conv_bn_7_8)             # N, 32WF, H/4, W/4
        conv8 = self.conv8(res_block_2b)                # N, 32WF, H/4, W/4
        ele_add_8 = unpool8 + conv8                     # N, 32WF, H/4, W/4
        conv_bn_8_9 = self.conv_bn_8_9(ele_add_8)       # N, 32WF, H/4, W/4
        unpool9 = self.unpool9(conv_bn_8_9)             # N, 32WF, H/2, W/2
        concat9 = torch.cat((unpool9, conv1), dim=1)    # N, 64WF, H/2, W/2
        conv9 = self.conv9(concat9)                     # N, 32WF, H/2, W/2
        conv10 = self.conv10(conv9)                     # N, 16WF, H/2, W/2
        conv12 = self.conv12(conv10)                    # N, 8WF, H/2, W/2
        conv13 = self.conv13(conv12)                    # N, C, H/2, W/2

        if self.half_res_output:
            logits = conv13
        else:
            logits = self.unpool_logits(conv13)         # N, C, H, W

        # dust head output
        if self.dust_head_output:
            res_block_6a = self.res_block_6a(res_block_5a)
            res_block_7a = self.res_block_7a(res_block_6a)
            pool2 = self.pool2(res_block_7a)
            pool2_squeezed = pool2[:, :, 0, 0]
            head_logits = self.fc1(pool2_squeezed)
            head_logits = F.relu(head_logits)

        if self.add_softmax_layer:
            softmax_logits = F.softmax(logits, dim=1)
            class_confidence, class_labels = torch.max(softmax_logits, dim=1, keepdim=True)
    
            if self.merge_stop_class_confidence:
                class_labels = apply_merge_confidence(self.num_classes, softmax_logits, class_labels)

            dust_output = None
            if self.dust_head_output:  # dust prediction from dust head
                dust_ratio = head_logits / self.dust_head_output_multiplier
                if self.zero_dust_ratio:  # in case dust head is not trained
                    dust_ratio = torch.clamp(dust_ratio, max=0.0)
                dust_output = dust_ratio.to(torch.float32)
            
            if self.dust_class_output:  # dust prediction from dust class
                dust_confidence_map = softmax_logits[:, self.num_classes-1]
                dust_output = dust_confidence_map.to(torch.float32)

            if self.no_intermediate_embeddings:
                return class_labels.to(torch.int32), class_confidence.to(torch.float32), dust_output
            
            # intermediate embeddings for instance similarity
            intermediate_output = res_block_2b.to(torch.float32)
            intermediate_output = intermediate_output.permute(0, 2, 3, 1)  # permute to B,H,W,C shape
            return class_labels.to(torch.int32), class_confidence.to(torch.float32), dust_output, intermediate_output
        elif self.outputType == OutputType.MULTISCALE:
            return logits, res_2b_logits, res_3a_logits, res_4a_logits, res_5a_logits
        else:
            return logits

    def load_state_dict_except_output_layer(self, saved_state_dict):
        widening_factor = self.model_params["widening_factor"]
        saved_model_num_out_channels = saved_state_dict["conv13.weight"].shape[0]

        # change the logits to load the  prev weights
        self.conv13 = torch.nn.Conv2d(
            in_channels=widening_factor * 8,
            out_channels=saved_model_num_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.load_state_dict(saved_state_dict, strict=True)

        # modify conv13 back to the desired number of classes
        self.conv13 = torch.nn.Conv2d(
            in_channels=widening_factor * 8,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def output_layers_match(self, saved_state_dict):
        """
        this method handles the case that the model being restored from has a different number of output
        classes than the model to-be-trained expects
        """
        if "conv13.weight" not in saved_state_dict:
            return False
        if self.conv13.out_channels == saved_state_dict["conv13.weight"].shape[0]:
            return True
        return False

    def load_state_dict_with_shape_check(self, saved_state_dict):
        if self.output_layers_match(saved_state_dict):
            self.load_state_dict(saved_state_dict)
        else:
            self.load_state_dict_except_output_layer(saved_state_dict)


# COMMAND ----------

import torch.nn as nn

class CriterionFocalLoss(nn.Module):
    """
    Focal loss function
    """
    def __init__(self, params: Dict, use_distance_based_weights: bool = False):
        super(CriterionFocalLoss, self).__init__()
        self.ignore_index = params["ignore_label"]
        self.eps = params["epsilon"]
        self.num_classes = params["num_classes"]

        self.use_distance_based_weights = 'focalloss_expdecay' in params and params[
            'focalloss_expdecay']['use'] == 'True'
        if self.use_distance_based_weights:
            input_size = tuple(map(int, params['input_size'].split(',')))
            self.distance_tensor = get_distance_weights_tensor(
                batch_size=params['batch_size'],
                input_cols=input_size[1],
                input_rows=input_size[0],
                exp_min_val=params['focalloss_expdecay']['exp_min'],
                exp_max_val=params['focalloss_expdecay']['exp_max'])

        self.alpha = torch.tensor(
            list(map(float, params["focalloss_parameters"]["alpha"])))
        self.gamma = torch.tensor(
            float(params["focalloss_parameters"]["gamma"]))

    def forward(self, preds, target):
        self.gamma = self.gamma.to(preds.device, non_blocking=True)
        self.alpha = self.alpha.type_as(preds).to(preds.device, non_blocking=True)
        input_soft = F.softmax(preds, dim=1)
        target_one_hot = make_one_hot(target, num_classes=self.num_classes, ignore_label=self.ignore_index)
        weight = torch.pow(1 - input_soft, self.gamma)
        focal = -1 * weight * self.alpha.view(1, -1, 1, 1) * torch.log(input_soft + self.eps)

        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.use_distance_based_weights:
            self.distance_tensor = self.distance_tensor.to(
                preds.device).type_as(input_soft)
            distance_scaled_loss = loss_tmp * self.distance_tensor
            loss = torch.mean(distance_scaled_loss)
        else:
            loss = torch.mean(loss_tmp)

        return loss

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set model parameters 

# COMMAND ----------

# set parameters for semantic segmentation model
model_params = {
    "num_block_layers": 2,
    "widening_factor": 2,
    "upsample_mode": "nearest",
    "bias": True,
}

params = {
    "num_classes": 8,
    "input_dims": 4,
    "output_type": OutputType.DEFAULT,
    "model_params": model_params,
    "half_res_output": False
}

delta_table_config = {
    "is_processed_table": True,
    "table_path": "/tmp/yz/datasets/jupiter_train_delta",
    #"table_path": "/tmp/yz/datasets/jupiter_train_subset_delta_new_test",
    "input_image_col": "image",
    "label_col": "gt_label",
    # use the following setting if use the unprocessed training image table
    # "is_processed_table": False,
    # "table_path": "/tmp/yz/datasets//tmp/yz/datasets/jupiter_train_subset_delta",
    # "input_image_col": "input_bytes",
    # "label_col": "label_map_bytes"
}

W = 1024
H = 512
N_CHANNEL = 4
DELTA_TORCH_NUM_WORKERS = 2
BATCH_SIZE = 4
SHUFFLE = True
LOG_INTERVAL = 500

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training 

# COMMAND ----------

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training step

# COMMAND ----------

import os
import mlflow

db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
experiment_path = "/Shared/mlflow-exps/jupiter-delta-train-exp"

# COMMAND ----------

# For distributed training we will merge the train and test steps into 1 main function
def main_fn(num_epochs):
  
    #### Added imports here ####
    import sys
    import os

    #sys.path.append(os.path.abspath("/Workspace/Repos/yinxi.zhang@bluerivertech.com/deltatorch-nativeloader-branch/deltatorch"))

    import mlflow
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from deltatorch import create_pytorch_dataloader, FieldSpec
    from torchvision import transforms

    ############################

    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ['DATABRICKS_HOST'] = "https://oregon.cloud.databricks.com"
    os.environ['DATABRICKS_TOKEN'] = db_token

    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_path)
    ############################

    #print("Running distributed training")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    dist.init_process_group("nccl", 
                            #rank=local_rank, world_size=torch.cuda.device_count()
                            )
    torch.cuda.set_device(local_rank)

    print(f"global rank = {global_rank}, local rank = {local_rank}, world_size = {torch.cuda.device_count()}")

    if global_rank == 0:
        train_parameters = {'batch_size': BATCH_SIZE, 
                            'epochs': num_epochs, 
                            'trainer': 'TorchDistributor'}
        mlflow.log_params(train_parameters)

    #print("debugging line: before data loading")
    #### loading data
    src_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: np.frombuffer(x, dtype=np.float32).reshape((H,W,N_CHANNEL))), ## convert bytes
            transforms.ToTensor(),
            transforms.Resize((H,W), interpolation=transforms.InterpolationMode.NEAREST),
        ]
    )
    target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((H,W), interpolation=transforms.InterpolationMode.NEAREST),
        ])
    train_dataloader = create_pytorch_dataloader(
            f"/dbfs{delta_table_config['table_path']}",
            id_field="id",
            fields=[
                FieldSpec(delta_table_config["input_image_col"], 
                            transform=src_transform,
                            ),
                FieldSpec(delta_table_config["label_col"],
                            decode_numpy_and_apply_shape=(H,W,1),
                            transform = target_transform,
                            ),
                FieldSpec("group_id",
                            ),
            ],
            num_workers=DELTA_TORCH_NUM_WORKERS,
            shuffle=SHUFFLE,
            batch_size=BATCH_SIZE,
            #timeout = DELTA_TORCH_TIMEOUT, ## use timeout=None to wait until a batch is loaded
            #queue_size = max(25000, table_length),
            #queue_size = 200,
            drop_last=True,
        )
    print(f"length of dataloader = {len(train_dataloader)}")
    #print("debugging line: before model loading")
    #### load model
    model = BrtResnetPyramidLite12(params)
    init_weights(model)
    model = model.to(local_rank)

    #### Added Distributed Model ####
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    #################################
    # print("debugging line: before training")
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-6, weight_decay=1e-4, eps=1e-4) ## optimizer model input is the distributed model

    for epoch in range(1, num_epochs + 1):
        ddp_model.train()
        ## train_one_epoch
        for batch_idx, batch in enumerate(train_dataloader):
            ## train step
            image, gt_label = batch["image"].to(local_rank), batch["gt_label"].to(local_rank)
            logits = ddp_model(image)
            loss = loss_func(torch.softmax(logits, dim=1), gt_label.long().squeeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Global Rank: {} - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    int(os.environ["RANK"]), epoch, batch_idx * len(batch), len(train_dataloader) * len(batch),
                    100. * batch_idx / len(train_dataloader), loss.item()))
                
    dist.destroy_process_group()

    return "finished" # can return any picklable object

# COMMAND ----------

# with mlflow.start_run() as run:
#   mlflow.log_param('run_type', 'test_dist_code')
#   main_fn(1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the following cell for single node, multiple GPU training

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor
 
# distributor = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True)
# distributor.run(main_fn, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the following cell for multi node GPU training

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor
def get_gpus_per_worker(_):
  import torch
  return torch.cuda.device_count()

NUM_WORKERS = 4
NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]
NUM_TASKS = NUM_WORKERS * NUM_GPUS_PER_WORKER
NUM_PROC_PER_TASK = 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK

distributor = TorchDistributor(num_processes=NUM_PROC, local_mode=False, use_gpu=True)
distributor.run(main_fn, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run inference 

# COMMAND ----------

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# %matplotlib inline
# mpl.use('Agg')

# classlabels_viz_colors = ['black', 'green', 'yellow', 'blue', 'red', 'magenta', 'cyan',
#                           'lightseagreen', 'brown', 'magenta', 'olive', 'wheat', 'white', 'black']
# classlabels_viz_bounds = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100]

# classlabels_viz_cmap = mpl.colors.ListedColormap(classlabels_viz_colors)
# classlabels_viz_norm = mpl.colors.BoundaryNorm(classlabels_viz_bounds, classlabels_viz_cmap.N)

# def inference_step(model, data, device):    
#     model.eval()
#     data = {key: value.to(device) for key, value in data.items() if isinstance(value, torch.Tensor)}
#     logits = model(data["image"])
#     labels = torch.argmax(logits, dim=1)
#     gt_label = data["gt_label"].squeeze(1)
    
#     # for idx in range(data["image"].shape[0]):
#     #     viz_image = data["image"][idx].cpu().numpy().transpose(1, 2, 0)[..., :3]
#     #     viz_label = labels[idx].squeeze().cpu().numpy()
#     #     viz_gt_label = gt_label[idx].squeeze().cpu().numpy()
        
#     #     fig, axes = plt.subplots(1, 3, figsize=(20, 10))
#     #     axes[0].imshow(viz_image)
#     #     axes[1].imshow(viz_label, cmap=classlabels_viz_cmap, norm=classlabels_viz_norm)
#     #     axes[2].imshow(viz_gt_label, cmap=classlabels_viz_cmap, norm=classlabels_viz_norm)
#     return data, labels, gt_label    

# COMMAND ----------

# idx = 1
# viz_image = data["image"][idx].cpu().numpy().transpose(1, 2, 0)[..., :3]
# viz_label = labels[idx].squeeze().cpu().numpy()
# viz_gt_label = gt_label[idx].squeeze().cpu().numpy()

# fig, axes = plt.subplots(1, 3, figsize=(20, 10))
# axes[0].imshow(viz_image)
# axes[1].imshow(viz_label, cmap=classlabels_viz_cmap, norm=classlabels_viz_norm)
# axes[2].imshow(viz_gt_label, cmap=classlabels_viz_cmap, norm=classlabels_viz_norm)
# fig

# COMMAND ----------


