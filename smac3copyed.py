import datetime
import os
import pywt
import random
import torch
import torch.nn as nn
from torchsummary import summary
import pdb
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
import matplotlib.pyplot as plot
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import pylab as plt
import pandas as pd
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
from pyts.datasets import fetch_ucr_dataset, ucr_dataset_list
from einops.layers.torch import Rearrange, Reduce
import time
import warnings
import numpy as np
from ConfigSpace import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    NormalFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import PriorAcquisitionFunction

from imgcoding.coding import Image_coding

from utils.utils import (
    count_parameters,
    min_max,
    z_score,
    stacking
)

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
#
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1)) # batch, channel, length
#         x = x.permute(0, 2, 1) # batch, length, channel
#         return x
#
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         #print("moving", moving_mean.size())
#         res = x - moving_mean
#         return res, moving_mean, x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CycleFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)
    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)

class CycleMLP(nn.Module):
    def __init__(self, dim, offset_size ,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        self.sfc_h = CycleFC(dim, dim, (1, offset_size), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (offset_size, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CycleBlock(nn.Module):

    def __init__(self, dim, offset_size ,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim,offset_size ,qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=None, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        # remove image_size in model init to support dynamic image size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    """ Downsample transition stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x

def basic_blocks(dim, index, layers,offset_size ,mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., skip_lam=1.0, mlp_fn=CycleMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(CycleBlock(dim,offset_size ,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)

    return blocks

class FactorizedReduce_2(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce_2, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_11 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_21 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_1 = nn.Conv2d(C_in, C_out, 3, stride = 2, padding = 1, bias = False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    #out = torch.cat([self.conv_11(x), self.conv_21(x[:,:,1:,1:])], dim=1)
    #print("1",x.shape)
    #print("2",self.conv_11(x).shape)
    #print("3",self.conv_21(x).size())
    out = self.conv_1(x)
    out = self.bn(out)
    return out

class CycleNet(nn.Module):
    """ CycleMLP Network """
    def __init__(self, layers, in_chans, img_size=224, patch_size=None, num_classes=None,
        embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm, mlp_fn=CycleMLP,offset_size = None ,fork_feat=False):

        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.in_ch = in_chans
        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=self.in_ch, embed_dim=embed_dims[0])
       # 추가
        self.reduction_stage_output = nn.ModuleList()
       ###
        network = []
        stage_num = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, offset_size[i] ,mlp_ratio=mlp_ratios[i] ,qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            stage_num.append(stage)
            # 추가
            if len(stage_num) > 1:
                for j in range(len(stage_num)-1):
                    reduction_out = FactorizedReduce_2(embed_dims[j], embed_dims[j])
                    self.reduction_stage_output.append(reduction_out)
            #
            if i >= len(layers) - 1:
                break
            # 변경이 필요해 보이는 부분
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            final_dims = 0
            for a in range(len(embed_dims)):
                final_dims += embed_dims[a]
            self.norm = norm_layer(final_dims)
            self.head = nn.Linear(final_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    #def init_weights(self, pretrained=None):
     #   """ mmseg or mmdet `init_weight` """
      #  if isinstance(pretrained, str):
       #     logger = get_root_logger()
        #    load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        outs = []
        # 추가된 부분
        stage_outputs = []
        reduce_time = 0
        stage_index = 0
        ####
        for idx, block in enumerate(self.network):
            x = block(x)
            # 추가됐지만 변경이 필요해 보이는 부분
            if idx == stage_index :
                stage_outputs.append(x)
                stage_index +=2
                if len(stage_outputs) > 1:
                    for k in range(len(stage_outputs)-1):
                        stage_outputs[k] = self.reduction_stage_output[reduce_time](stage_outputs[k].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                        reduce_time += 1
                """if idx+2 == 6
                    for k in range(len(stage_outputs)):
                        stage_outputs[k] = self.reduction_stage_output[reduce_time](stage_outputs[k])
                        reduce_time += 1"""
            #####
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        if self.fork_feat:
            return outs
        oouts = []
        for out in stage_outputs:
            oouts.append(out)
        final_out = torch.cat(oouts, dim = 3)
        B, H, W, C = final_out.shape
        x = final_out.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


# class Image_coding(nn.Module):
#     def __init__(self, decompose):
#         super().__init__()
#         self.decompose = decompose
#         self.time_decom = series_decomp(25)
#         self.scale = np.arange(1, 176, 1)
#
#     def tonumpy(self, x):
#         to_num = np.array(x)
#         return to_num
#
#     def continuous_wavelet(self, x, len, size):
#         x = np.squeeze(x, axis=2)
#         ucr_train_cwt = np.zeros((1, size, size))
#         for i in range(len):
#             cwtmatr, freqs_rate = pywt.cwt(x[i], scales=self.scale, wavelet='morl')
#             cwtmatr = np.expand_dims(cwtmatr, axis=0)
#             if i == 0:
#                 ucr_train_cwt = cwtmatr
#             else:
#                 ucr_train_cwt = np.concatenate((ucr_train_cwt, cwtmatr), axis=0)
#         return ucr_train_cwt
#
#     def Recurrence_Plot(self, x, y, z):
#         rp = RecurrencePlot(dimension=2, time_delay=1)
#         return rp.transform(x), rp.transform(y), rp.transform(z)
#
#     def Gramian_sum(self, x, y, z):
#         gas = GramianAngularField()
#         return gas.transform(x), gas.transform(y), gas.transform(z)
#
#     def Gramian_diff(self, x, y, z):
#         gad = GramianAngularField(method='difference')
#         return gad.transform(x), gad.transform(y), gad.transform(z)
#
#     def Markovtran(self, x, y, z):
#         mk = MarkovTransitionField(n_bins=8)
#         return mk.transform(x), mk.transform(y), mk.transform(z)
#
#     def forward(self, x):
#         xt = torch.FloatTensor(x)
#         len = xt.size(0)
#         size = xt.size(1)
#         x_cwv = self.continuous_wavelet(x, len, size)
#         # x_cwv = torch.FloatTensor(x_cwv)
#         # print(x_cwv.size())
#         x_res, x_tr, xt = self.time_decom(xt)
#         xt = xt.squeeze(2)
#         x_res = x_res.squeeze(2)
#         x_tr = x_tr.squeeze(2)
#
#         x_rp, x_rerp, x_trrp = self.Recurrence_Plot(xt, x_res, x_tr)
#         x_gas, x_regas, x_trgas = self.Gramian_sum(xt, x_res, x_tr)
#         x_gad, x_regad, x_trgad = self.Gramian_diff(xt, x_res, x_tr)
#         x_mk, x_remk, x_trmk = self.Markovtran(xt, x_res, x_tr)
#         return x_rp, x_rerp, x_trrp, x_gas, x_regas, x_trgas, x_gad, x_regad, x_trgad, x_mk, x_remk, x_trmk, x_cwv

class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.x_data = torch.FloatTensor(data)
        self.y_data = torch.LongTensor(target)
        #self.y_data = self.y_data-1

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# class stacking(nn.Module):
#     def __init__(self, img_size, data_num, channel_num, choosed_encod):
#         super().__init__()
#         self.img_size = img_size
#         self.data_num = data_num
#         self.channel_num = channel_num
#         self.choosed_encod = choosed_encod
#     def forward(self, x):
#         stack = np.array([]).reshape(-1, 1, self.img_size, self.img_size)
#         for i, name in enumerate(self.choosed_encod):
#             #x[name] = resize(x[name],((self.data_num, 1,self.img_size, self.img_size)))
#             #print(x[name].shape)
#             if i == 0 :
#                 stack = x[name]
#             else :
#                 stack = np.concatenate((stack, x[name]), axis=1)
#         return stack


class Babroad:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types,
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()
        RP = CategoricalHyperparameter(
            "RP",
            ["rp", "none"],
            default_value="rp",
        )
        RERP = CategoricalHyperparameter(
            "RERP",
            ["rerp", "none"],
            default_value="none",
        )
        TRRP = CategoricalHyperparameter(
            "TRRP",
            ["trrp", "none"],
            default_value="none",
        )
        GAS = CategoricalHyperparameter(
            "GAS",
            ["gas", "none"],
            default_value="gas",
        )
        REGAS = CategoricalHyperparameter(
            "REGAS",
            ["regas", "none"],
            default_value="none",
        )
        TRGAS = CategoricalHyperparameter(
            "TRGAS",
            ["trgas", "none"],
            default_value="none",
        )
        GAD = CategoricalHyperparameter(
            "GAD",
            ["gad", "none"],
            default_value="gad",
        )
        REGAD = CategoricalHyperparameter(
            "REGAD",
            ["regad", "none"],
            default_value="none",
        )
        TRGAD = CategoricalHyperparameter(
            "TRGAD",
            ["trgad", "none"],
            default_value="none",
        )
        MK = CategoricalHyperparameter(
            "MK",
            ["mk", "none"],
            default_value="none",
        )
        REMK = CategoricalHyperparameter(
            "REMK",
            ["remk", "none"],
            default_value="none",
        )
        TRMK = CategoricalHyperparameter(
            "TRMK",
            ["trmk", "none"],
            default_value="none",
        )
        CTW = CategoricalHyperparameter(
            "CTW",
            ["ctwav", "none"],
            default_value="none",
        )
        # We do not have an educated belief on the number of layers beforehand
        # As such, the prior on the HP is uniform
        n_stage = UniformIntegerHyperparameter(
            "n_stage",
            lower=2,
            upper=4,
            default_value=4,
        )

        # We believe the optimal network is likely going to be relatively wide,
        # And place a Beta Prior skewed towards wider networks in log space
        n_cells = UniformIntegerHyperparameter(
            "n_cells",
            lower=1,
            upper=3,
            default_value=3,
        )
        n_dims = UniformIntegerHyperparameter(
            "n_dims",
            lower=12,
            upper=96,
            default_value=24,
        )

        # We believe that ReLU is likely going to be the optimal activation function about
        # 60% of the time, and thus place weight on that accordingly

        # Add all hyperparameters at once:
        cs.add_hyperparameters(
            [n_stage, n_cells, n_dims, RP, RERP, TRRP, GAS, REGAS, TRGAS, GAD, REGAD, TRGAD, MK, REMK, TRMK, CTW])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            infeature_list = []
            if config["RP"] != "none":
                infeature_list.append(config["RP"])
            if config["RERP"] != "none":
                infeature_list.append(config["RERP"])
            if config["TRRP"] != "none":
                infeature_list.append(config["TRRP"])
            if config["GAS"] != "none":
                infeature_list.append(config["GAS"])
            if config["REGAS"] != "none":
                infeature_list.append(config["REGAS"])
            if config["TRGAS"] != "none":
                infeature_list.append(config["TRGAS"])
            if config["GAD"] != "none":
                infeature_list.append(config["GAD"])
            if config["REGAD"] != "none":
                infeature_list.append(config["REGAD"])
            if config["TRGAD"] != "none":
                infeature_list.append(config["TRGAD"])
            if config["MK"] != "none":
                infeature_list.append(config["MK"])
            if config["REMK"] != "none":
                infeature_list.append(config["REMK"])
            if config["TRMK"] != "none":
                infeature_list.append(config["TRMK"])
            if config["CTW"] != "none":
                infeature_list.append(config["CTW"])
            print(infeature_list)
            channel_num = len(infeature_list)
            train_stack = stacking(input_size, train_size, channel_num, infeature_list)
            test_stack = stacking(input_size, train_size, channel_num, infeature_list)
            train_all = train_stack(train_dict)
            test_all = test_stack(test_dict)

            train_data = CustomDataset(train_all, ucr_target_train)
            trainloader = DataLoader(train_data, batch_size=20, shuffle=True)
            test_data = CustomDataset(test_all, ucr_target_test)
            testloader = DataLoader(test_data, batch_size=20, shuffle=False)

            stage = config["n_stage"]
            cell = config["n_cells"]
            dims = config["n_dims"]
            cells = []  # net 구조 위한 리스트
            embed_dims = []  # net 구조 위한 리스트
            if dims % 2 != 0:
                dims = dims + 1
            for s in range(stage):
                if s == 0:
                    cells = cell
                    embed_dims = dims
                else:
                    add_cell = cell
                    dims = 2 * dims
                    cells = np.append(cells, add_cell)
                    embed_dims = np.append(embed_dims, dims)

            cells = cells.tolist()
            embed_dims = embed_dims.tolist()
            print("embed_dims : ", embed_dims)
            print("cells : ", cells)
            transitions = [True, True, True, True]
            mlp_ratios = [4, 4, 4, 4]
            offset_size = [16, 8, 4, 2]
            model = CycleNet(cells, channel_num, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                             num_classes=class_num,
                             mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, offset_size=offset_size).to(device)
            parameter_num = count_parameters(model)
            print("parameter_num : ", parameter_num)

            # if (parameter_num < 2300000) and (parameter_num > 1700000):
            #     print("skip")
            #     return 100
            # elif (parameter_num > 2400000):
            #     print("skip")
            #     return 100
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            loss_list = []
            iter = []
            train_accuracy = []
            validation_loss = []
            validation_acc = []
            model.train()
            epochs = 80
            print("epoch:", epochs)
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0.0
                total = 0.0
                for data in trainloader:
                    input, target = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()
                    output = model(input)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    _, preds = torch.max(output.data, 1)
                    correct += preds.eq(target).sum().item()
                    running_loss += loss.item()
            training_loss = running_loss / len(trainloader)
            train_acc = 100 * correct / len(trainloader.dataset)

            # print("=============================================================================================================================================================")
            print(
                f"TRAIN: EPOCH {epoch + 1:04d} / {epochs:04d} | Epoch LOSS {training_loss:.4f} | Epoch ACC {train_acc:.4f}")
            cost = running_loss / 6
            global candi
            candi = candi + 1
            loss_list.append(training_loss)
            #torch.save(model.state_dict(), PATH +str(candi) +'ShapesAlltrain(smac3).pt')
            train_accuracy.append(train_acc)
            running_loss = 0.0
            print('finish')
            correct = 0
            total = 0
            accuracy = []
            iter_acc = []
            test_loss = []
            b = 0
            global img_test_acc
            model.eval()
            with torch.no_grad():
                for data in testloader:
                    b += 1
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    iter_acc.append(b)
                    cost = 100 * correct / total
                    accuracy.append(cost)

                test_loss.append(loss)
                error_rate = 1 - (correct / total)
                test_acc = 100. * correct / len(testloader.dataset)
                global acc_list
                acc_list.append(test_acc)
                global parm_list
                parm_list.append(parameter_num)
            if img_test_acc < test_acc:
                img_test_acc = test_acc
                torch.save(model.state_dict(), PATH + 'ShapesAll(smac3)512.pt')
                print("save best test")
            # print('accuracy of testimages: %d %%' %(100*correct/total))
            print('accuracy of testimages: %d %%' % (test_acc))
            print('best_acc : ', img_test_acc)
            print('error rate : ', format(error_rate, ".3f"))
            print(
                "=============================================================================================================================================================")
        return 100 - test_acc


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
# def z_score(list):
#     nomallized = []
#     for value in list:
#         nomallized_value = (value-np.mean(list))/np.std(list)
#         nomallized.append(nomallized_value)
#     return nomallized
#
# def min_max(list):
#     nomalized = []
#     for value in list:
#         nomalized_val = (value-min(list))/(max(list)-min(list))
#         nomalized.append(nomalized_val)
#     return nomalized
acc_list = []
parm_list = []
global img_test_acc
img_test_acc = 0
global candi
candi = 0
if __name__ == "__main__":
    ucr = fetch_ucr_dataset('ShapesAll', use_cache=False, data_home=None, return_X_y=False)  # 5 class
    ucr_train = ucr.data_train  # 390, 176
    ucr_target_train = ucr.target_train
    ucr_test = ucr.data_test
    ucr_target_test = ucr.target_test
    ucr_train_re = resize(ucr_train, ((600, 512, 1)))
    ucr_test_re = resize(ucr_test, ((600, 512, 1)))
    trencoder = LabelEncoder()
    trencoder.fit(ucr_target_train)
    ucr_target_train = trencoder.transform(ucr_target_train)
    teencoder = LabelEncoder()
    teencoder.fit(ucr_target_test)
    ucr_target_test = teencoder.transform((ucr_target_test))
    class_num = 60
    train_size = len(ucr_train_re)
    test_size = len(ucr_test_re)
    input_size = 512

    tran = Image_coding(decompose=True)
    train_rp, train_rerp, train_trrp, train_gas, train_regas, train_trgas, train_gad, train_regad, train_trgad, train_mk, train_remk, train_trmk, train_ctwav = tran(
        ucr_train_re)  # wavelet 추가?
    test_rp, test_rerp, test_trrp, test_gas, test_regas, test_trgas, test_gad, test_regad, test_trgad, test_mk, test_remk, test_trmk, test_ctwav = tran(
        ucr_test_re)


    train_rp = resize(train_rp, ((train_size, input_size, input_size)))
    test_rp = resize(test_rp, ((test_size, input_size, input_size)))
    train_trrp = resize(train_trrp, ((train_size, input_size, input_size)))
    test_trrp = resize(test_trrp, ((test_size, input_size, input_size)))
    train_rerp = resize(train_rerp, ((train_size, input_size, input_size)))
    test_rerp = resize(test_rerp, ((test_size, input_size, input_size)))

    train_gad = resize(train_gad, ((train_size, input_size, input_size)))
    test_gad = resize(test_gad, ((test_size, input_size, input_size)))
    train_trgad = resize(train_trgad, ((train_size, input_size, input_size)))
    test_trgad = resize(test_trgad, ((test_size, input_size, input_size)))
    train_regad = resize(train_regad, ((train_size, input_size, input_size)))
    test_regad = resize(test_regad, ((test_size, input_size, input_size)))

    train_gas = resize(train_gas, ((train_size, input_size, input_size)))
    test_gas = resize(test_gas, ((test_size, input_size, input_size)))
    train_trgas = resize(train_trgas, ((train_size, input_size, input_size)))
    test_trgas = resize(test_trgas, ((test_size, input_size, input_size)))
    train_regas = resize(train_regas, ((train_size, input_size, input_size)))
    test_regas = resize(test_regas, ((test_size, input_size, input_size)))

    train_mk = resize(train_mk, ((train_size, input_size, input_size)))
    test_mk = resize(test_mk, ((test_size, input_size, input_size)))
    train_trmk = resize(train_trmk, ((train_size, input_size, input_size)))
    test_trmk = resize(test_trmk, ((test_size, input_size, input_size)))
    train_remk = resize(train_remk, ((train_size, input_size, input_size)))
    test_remk = resize(test_remk, ((test_size, input_size, input_size)))

    train_ctwav = resize(train_ctwav, ((train_size, input_size, input_size)))
    test_ctwav = resize(test_ctwav, ((test_size, input_size, input_size)))

    train_dict = {"rp": train_rp, "rerp": train_rerp, "trrp": train_trrp, "gas": train_gas, "regas": train_regas,
                  "trgas": train_trgas, "gad": train_gad, "regad": train_regad, "trgad": train_trgad, "mk": train_mk,
                  "remk": train_remk, "trmk": train_trmk, "ctwav": train_ctwav}
    test_dict = {"rp": test_rp, "rerp": test_rerp, "trrp": test_trrp, "gas": test_gas, "regas": test_regas,
                 "trgas": test_trgas, "gad": test_gad, "regad": test_regad, "trgad": test_trgad, "mk": test_mk,
                 "remk": test_remk, "trmk": test_trmk, "ctwav": test_ctwav}
    coding_name = ["rp", "rerp", "trrp", "gas", "regas", "trgas", "gad", "regad", "trgad", "mk", "remk", "trmk",
                   "ctwav"]

    for i in coding_name:
        train_dict[i] = np.expand_dims(train_dict[i], axis=1)
        test_dict[i] = np.expand_dims(test_dict[i], axis=1)

    PATH = "/home/sin/PycharmProjects/please/save/ShapesAll/"
    #global best_acc
    #global best_loss
    #best_acc = 0
    #best_loss = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "1"
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("current cuda device:", torch.cuda.current_device())
    print("count of using gpus:", torch.cuda.device_count() )

    train_default = np.concatenate(
        (train_dict["rp"], train_dict["trgad"], train_dict["mk"], train_dict["trmk"]), axis=1)
    test_default = np.concatenate(
        (test_dict["rp"], test_dict["trgad"], test_dict["mk"], test_dict["trmk"]), axis=1)
    best_test_acc = 0
    ch_num_list = []
    img_list = []
    num_lay_list = []
    num_cell_list = []
    start_dim_list = []
    parameter_num_list = []
    acc_list = []
    error_rate_list = []
    duplic_list = []
    other_name_cell = 'a'
    start_time = time.time()

    mlp = Babroad()
    default_config = mlp.configspace.get_default_configuration()

    # Define our environment variables
    scenario = Scenario(mlp.configspace, n_trials=100)

    # We also want to include our default configuration in the initial design
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        additional_configs=[default_config],
    )

    # We define the prior acquisition function, which conduct the optimization using priors over the optimum
    acquisition_function = PriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(scenario),
        decay_beta=scenario.n_trials / 10,  # Proven solid value
    )

    # We only want one config call (use only one seed in this example)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = HyperparameterOptimizationFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        intensifier=intensifier,
        overwrite=True,
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(default_config)
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"choosed cost: {incumbent_cost}")

    end = time.time()
    total_time = end - start_time
    print(total_time)
    time_results = datetime.timedelta(seconds=total_time)
    print(time_results)
    #global acc_list
    #global parm_list
    print("acc_list :", acc_list)
    print("parm_list :", parm_list)
    print("Founded config num :", len(acc_list))
    num = len(acc_list)
    x = np.arange(1, num+1, 1)
    plt.subplot(2,1,1)
    plt.plot(x, acc_list, 'bo')
    plt.xlabel('Candidate#')
    plt.ylabel('Accuracy')
    plt.title('Candidate accuracy')
    plt.grid(True)

    nom_parm_list = min_max(parm_list)
    print(nom_parm_list)
    for i in range(len(nom_parm_list)):
        nom_parm_list[i] = nom_parm_list[i]*100
    print(nom_parm_list)
    plt.subplot(2, 1, 2)
    plt.plot(x, parm_list, 'gs')
    plt.xlabel('Candidate#')
    plt.ylabel('Number of Parms')
    plt.title('Candidate Parms')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.scatter(x, acc_list, s = nom_parm_list, c='blue')
    plt.show()