import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
class HiLo(nn.Module):
    """
    HiLo Attention

    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim
        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim
        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5
        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)
        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)
        self.drop = nn.Dropout(proj_drop)

    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 1)
        x = self.norm(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)
        if pad_r > 0 or pad_b > 0:
            x = torch.cat((hifi_out[:, :H, :W, :], lofi_out[:, :H, :W, :]), dim=-1)
        else:
            x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)
        return x + inputs

class CFMM(nn.Module):
    r"""
         Coarse Focus Measure Module
         Global Context Feats to coarse focus map
    """
    def __init__(self, in_channel0, in_channel1, in_channel2, depth=6, num_classes=1):
        super().__init__()
        self.fam2 = nn.ModuleList([FAM(in_features=in_channel2), FAM(in_features=in_channel2)])
        self.fam1 = nn.ModuleList([FAM(in_features=in_channel1), FAM(in_features=in_channel1)])
        # self.fam0 = nn.ModuleList([FAM(in_features=in_channel0), FAM(in_features=in_channel0)])
        # self.conv1 = nn.Conv2d(in_channel2, in_channel1, 1)
        # self.conv2 = nn.Conv2d(in_channel2, in_channel1, 1)
        self.convx = nn.Conv2d(in_channel2, in_channel1, 1)
        self.convy = nn.Conv2d(in_channel2, in_channel1, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel1*2, in_channel1, 3, 1, 1), nn.BatchNorm2d(in_channel1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel1, in_channel1 // 2, 3, 1, 1), nn.BatchNorm2d(in_channel1 // 2), nn.ReLU(inplace=True))
        # self.block = nn.ModuleList([HiLo(dim=in_channel1, alpha=0.2, num_heads=8, proj_drop=0.1) for i in range(depth)])
        # self.block1 = nn.ModuleList([HiLo(dim=out_channels, alpha=0.2, num_heads=8, proj_drop=0.1) for i in range(depth)])
        self.blockx = nn.ModuleList([HiLo(dim=in_channel1, alpha=0.9, num_heads=8, proj_drop=0.1) for i in range(depth)])
        self.blocky = nn.ModuleList([HiLo(dim=in_channel1, alpha=0.9, num_heads=8, proj_drop=0.1) for i in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(in_channel1//2, in_channel1//2, 3, 1, 1), nn.BatchNorm2d(in_channel1//2), nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channel1//2, num_classes, 1))

    # self.proj_b = nn.Sequential(nn.BatchNorm2d(in_channel1),nn.Conv2d(in_channel1, in_channel1, 3, 1, 1), nn.GELU(),
        #                             nn.Conv2d(in_channel1, num_classes, kernel_size=1, stride=1))

    def forward(self, x1, y1, x2, y2, size):
        x2, y2 = self.fam2[0](x2, y2), self.fam2[0](y2, x2)
        x1, y1 = self.fam1[0](x1, y1), self.fam1[0](y1, x1)
        # x0, y0 = self.fam0[0](x0, y0), self.fam0[0](y0, x0)
        for i, blk in enumerate(self.blockx):
            x1 = blk(x1)
        for i, blk in enumerate(self.blocky):
            y1 = blk(y1)
        featsx = x1 * self.convx(F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False))
        featsy = y1 * self.convy(F.interpolate(y2, size=y1.shape[2:], mode='bilinear', align_corners=False))
        # Coarse Focus Mearsurement
        # featsx = self.convx(F.interpolate(featsx, size=y0.shape[2:], mode='bilinear', align_corners=False))
        # featsy = self.convy(F.interpolate(featsy, size=y0.shape[2:], mode='bilinear', align_corners=False))
        # x0 = x0 * featsx + featsx
        # y0 = y0 * featsy + featsy
        feats = self.conv2(self.conv1(torch.cat([featsx, featsy], dim=1)))
        predict = self.proj(feats)
        # Coarse Boundary Detect
        # boundary = F.interpolate(self.proj_b(feats), size=size, mode='bilinear', align_corners=False)
        return feats, predict  # , boundary

class FAM(nn.Module):
    r'''
    Feats Aggregation Module
    '''
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3), nn.Sigmoid())

    def forward(self, x1, y1):
        x, y = (x1 - x1.min()) / (x1.max() - x1.min()), (y1 - y1.min()) / (y1.max() - y1.min())
        fuse = x * (1 - y)
        max_result, _ = torch.max(fuse, dim=1, keepdim=True)
        avg_result = torch.mean(fuse, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        attn = self.conv(result)
        feats = x1 * attn
        return feats
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
from timm.models.layers import trunc_normal_
import math
class DWMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = 4 * in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.kernel_size[0] > 1:
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            else:
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.fc1(x).permute(0, 3, 1, 2)
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x).permute(0, 3, 1, 2)
        return shortcut + x
# from core.model.DeepLabV3pp import ASPP
class FFMM(nn.Module):
    r'''
         Zero-padding DWConv
         capture Spatial Position Inf.
    '''
    def __init__(self, in_channel0, in_channel2, depth, num_classes):
        super().__init__()
        self.fam = nn.ModuleList([FAM(in_features=in_channel2), FAM(in_features=in_channel2)])
        self.fam0 = nn.ModuleList([FAM(in_features=in_channel0), FAM(in_features=in_channel0)])
        self.convx = nn.Conv2d(in_channel2, in_channel0, 1)
        self.convy = nn.Conv2d(in_channel2, in_channel0, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel0 * 2, in_channel0 * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(in_channel0 * 2),
                                   nn.ReLU(inplace=True), nn.Conv2d(in_channel0 * 2, in_channel0 * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(in_channel0 * 2),
                                   nn.ReLU(inplace=True))
        # self.conv_up = nn.Conv2d(in_channel2, in_channel2, 3, 1, 1)
        self.blockx = nn.ModuleList([HiLo(dim=in_channel2, alpha=0.9, num_heads=8, proj_drop=0.1) for i in range(depth)])
        self.blocky = nn.ModuleList([HiLo(dim=in_channel2, alpha=0.9, num_heads=8, proj_drop=0.1) for i in range(depth)])
        # self.proj = nn.Sequential(nn.Conv2d(in_channel0*2, in_channel0*2, 3, 1, 1), nn.BatchNorm2d(in_channel0*2), nn.ReLU(inplace=True),
        #                              nn.Conv2d(in_channel0*2, num_classes, 1))

    def forward(self, x0, y0, x1, y1):
        x0, y0 = self.fam0[0](x0, y0), self.fam0[0](y0, x0)
        x1, y1 = self.fam[0](x1, y1), self.fam[1](y1, x1)
        for i, blk in enumerate(self.blockx):
            x1 = blk(x1)
        for i, blk in enumerate(self.blocky):
            y1 = blk(y1)
        # x0, y0 = self.convdx(x0), self.convdy(y0)
        featsx = x0 * self.convx(F.interpolate(x1, size=x0.shape[2:], mode='bilinear', align_corners=False))
        featsy = y0 * self.convy(F.interpolate(y1, size=x0.shape[2:], mode='bilinear', align_corners=False))
        feats = self.conv1(torch.cat([featsx, featsy], dim=1))
        # for i, blk in enumerate(self.block):
        #     feats = blk(feats)
        # Patch Refine
        # bound = self.proj(feats)
        return feats

class FFM(nn.Module):
    r"""
         Feats Fusion Module
         Combine Context and Local Feats to generate final focus map
    """
    def __init__(self, in_channel1, in_channel2, num_classes=2):
        super().__init__()
        # self.conv_up1 = nn.Sequential(nn.Conv2d(in_channel1, in_channel1, 3, 1, 1))
        # self.conv_up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                               nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        # self.block = nn.ModuleList([HiLo(dim=in_channel1, alpha=0.9, num_heads=8, proj_drop=0.1) for i in range(depth)])
        self.conv = nn.Sequential(nn.Conv2d(in_channel1+in_channel2, in_channel1, 3, 1, 1), nn.BatchNorm2d(in_channel1), nn.ReLU(inplace=True), nn.Conv2d(in_channel2, in_channel1, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel1), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(in_channel1, in_channel1, 3, 1, 1), nn.BatchNorm2d(in_channel1),
                                  nn.ReLU(inplace=True), nn.Conv2d(in_channel1, num_classes, 1))
        self.proj_b = nn.Sequential(nn.Conv2d(in_channel2, in_channel2, 3, 1, 1), nn.BatchNorm2d(in_channel2),
                                  nn.ReLU(inplace=True), nn.Conv2d(in_channel2, num_classes, 1))

    def forward(self, feats1, coarse_map, feats2, size):
        # Feats Fusion
        coarse_map = F.interpolate(coarse_map, size=size, mode='bilinear', align_corners=False)
        feats1 = F.interpolate(feats1, size=feats2.shape[2:], mode='bilinear', align_corners=False)
        # Bound Detect
        bound_map = F.interpolate(self.proj_b(feats2), size=size, mode='bilinear', align_corners=False)
        feats = torch.cat([feats1, feats2], dim=1)
        # feats = torch.cat([self.rate1 * feats1, self.rate2 * feats2], dim=1)
        # Fine Focus Measure
        feats = F.interpolate(self.conv(feats), size=size, mode='bilinear', align_corners=False)
        predict = self.proj(feats)
        return coarse_map, predict, bound_map
        # return feats2, feats
# from core.model.ConvNeXt import ConvNeXt
from core.model.LITv2 import LITv2
from core.model.swin import SwinTransformer
from core.model.resnet import ResNet
class MFNeXt(nn.Module):
    r'''
        Encoder: Feature Extractor Backbone->Lightweight ConvNeXt
        Decoder: Obtain Focus Pixel Classification
    '''
    def __init__(self, configs):
        super().__init__()
        self.encoder = SwinTransformer(pretrain_img_size=configs['input_size'])#ResNet() LITv2(img_size=configs['input_size'])

        self.proj = nn.ModuleList([CFMM(in_channel0=64, in_channel1=128, in_channel2=256, depth=4, num_classes=1), FFMM(in_channel0=32, in_channel2=64, depth=4, num_classes=1)])
        self.proj_f = FFM(in_channel1=64, in_channel2=64, num_classes=1)

    def FocusMeasure(self, x, y, size):
        r'''
        Feats Process and Focus Measure
        '''
        x, y = self.encoder(x), self.encoder(y)
        coarse_maps, fine_maps = [], []
        # Spatial Position Path
        feats_spatial = self.proj[1](x[0], y[0], x[1], y[1])
        # Coarse Focus Measure Path
        feats_semantic, coarse_map = self.proj[0](x[2], y[2], x[3], y[3], size)
        # coarse_maps.append(coarse_boundary_map)
        # Fine Focus Measure Path
        coarse_focus_map, final_focus_map, final_bound_map = self.proj_f(feats_semantic, coarse_map, feats_spatial, size)
        # coarse_focus_map, final_focus_map = self.proj_f(feats_semantic, coarse_map, size)
        coarse_maps.append(coarse_focus_map)
        fine_maps.append(final_focus_map)
        fine_maps.append(final_bound_map)
        return coarse_maps, fine_maps
        # return feats_semantic, final_focus_map, fine_boundary_map

    def forward(self, inputs):
        x, y = inputs['Far'], inputs['Near']
        H, W = x.shape[2:]
        coarse_maps, fine_maps = self.FocusMeasure(x, y, [H, W])
        # feats1, feats2, feats = self.FocusMeasure(x, y, [H, W])
        # return feats1, feats2, feats
        return coarse_maps, fine_maps

if __name__ == '__main__':
    with torch.no_grad():
        model = MFNeXt(configs={'input_size':520}).eval().cuda()
        weight_path = r"F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\work_dirs\MFLIT\MFNeXt\model_1.pth"
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['model'].state_dict())
    from core.util import utils
    print(utils.count_parameters(model))
    import cv2
    import os
    from torchvision import transforms
    data = torch.tensor(cv2.imread(r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Far\0DUT1 (1).jpg'), dtype=torch.float32).cuda()
    data = torch.unsqueeze(data, dim=0).permute(0, 3, 1, 2)
    data1 = torch.tensor(cv2.imread(r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Near\0DUT1 (1).jpg'), dtype=torch.float32).cuda()
    data1 = torch.unsqueeze(data1, dim=0).permute(0, 3, 1, 2)
    n = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    data, data1 = n(data), n(data1)
    # resize = torchvision.transforms.Compose([torchvision.transforms.CenterCrop([256, 256]), torchvision.transforms.Resize([256, 256])])
    print(model)
    # print(model({'Far':data,'Near':data, "focus_map":data1}).shape)
    save_path = r"F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\8"
    dtransforms = transforms.Compose([transforms.ToPILImage()])
    feats1, feats2, feats = model({'Far':data1,'Near':data})
    # print(feats.shape)
    #
    for i in range(feats1.shape[1]):
        img = feats1[0][i].cpu()
        print(img.shape)
        name = os.path.join(save_path, 'lytro-0'+str(len(os.listdir(save_path))+1))
        img = dtransforms(img.to(dtype=torch.float32))
        img.save(f'{name}.jpg')
    for i in range(feats2.shape[1]):
        img = feats2[0][i].cpu()
        print(img.shape)
        name = os.path.join(save_path, 'lytro-0'+str(len(os.listdir(save_path))+1))
        img = dtransforms(img.to(dtype=torch.float32))
        img.save(f'{name}.jpg')
    for i in range(feats.shape[1]):
        img = feats[0][i].cpu()
        print(img.shape)
        name = os.path.join(save_path, 'lytro-0'+str(len(os.listdir(save_path))+1))
        img = dtransforms(img.to(dtype=torch.float32))
        img.save(f'{name}.jpg')

