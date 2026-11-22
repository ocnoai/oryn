# Apache-2.0; adapted from akore/rtmw-l-256x192 and OpenMMLab MMPose.
# See NOTICE.md and provenance.json for frozen source identity and changes.
from typing import Optional, Tuple, Union, Dict, Sequence
import torch
from torch import nn
from torch.nn import functional as F

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, groups: int=1, bias: bool=True, norm_cfg: Optional[Dict]=dict(type='BN'), act_cfg: Optional[Dict]=dict(type='SiLU'), inplace: bool=True):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias and (not self.with_norm))
        if self.with_norm:
            norm_channels = out_channels
            self.bn = nn.BatchNorm2d(norm_channels, momentum=0.1, eps=1e-05)
        if self.with_activation:
            if act_cfg['type'] == 'ReLU':
                self.activate = nn.ReLU(inplace=inplace)
            elif act_cfg['type'] == 'LeakyReLU':
                self.activate = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
            elif act_cfg['type'] == 'SiLU' or act_cfg['type'] == 'Swish':
                self.activate = nn.SiLU(inplace=inplace)
            else:
                raise NotImplementedError(f"Activation {act_cfg['type']} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_activation:
            x = self.activate(x)
        return x

class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, dilation: Union[int, Tuple[int, int]]=1, norm_cfg: Optional[Dict]=dict(type='BN'), act_cfg: Dict=dict(type='SiLU'), **kwargs):
        super().__init__()
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention Module."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out

class CSPNeXtBlock(nn.Module):
    """The basic bottleneck block used in CSPNeXt."""

    def __init__(self, in_channels: int, out_channels: int, expansion: float=0.5, add_identity: bool=True, use_depthwise: bool=False, kernel_size: int=5, act_cfg: Dict=dict(type='SiLU')) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(in_channels, hidden_channels, 3, stride=1, padding=1, act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(hidden_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            return out + identity
        else:
            return out

class CSPLayer(nn.Module):
    """Cross Stage Partial Layer."""

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float=0.5, num_blocks: int=1, add_identity: bool=True, use_depthwise: bool=False, use_cspnext_block: bool=False, channel_attention: bool=False, act_cfg: Dict=dict(type='SiLU')) -> None:
        super().__init__()
        block = CSPNeXtBlock if use_cspnext_block else None
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid_channels, 1, act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels, mid_channels, 1, act_cfg=act_cfg)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1, act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[block(mid_channels, mid_channels, 1.0, add_identity, use_depthwise) for _ in range(num_blocks)])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Tuple[int, ...]=(5, 9, 13), act_cfg: Dict=dict(type='SiLU')):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1, stride=1, act_cfg=act_cfg)
        self.poolings = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(conv2_channels, out_channels, 1, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x

class CSPNeXt(nn.Module):
    """CSPNeXt backbone used in RTMW."""
    arch_settings = {'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False], [256, 512, 6, True, False], [512, 1024, 3, False, True]], 'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False], [256, 512, 6, True, False], [512, 768, 3, True, False], [768, 1024, 3, False, True]]}

    def __init__(self, arch: str='P5', deepen_factor: float=1.0, widen_factor: float=1.0, out_indices: Sequence[int]=(2, 3, 4), frozen_stages: int=-1, use_depthwise: bool=False, expand_ratio: float=0.5, channel_attention: bool=True, act_cfg: Dict=dict(type='SiLU')) -> None:
        super().__init__()
        arch_setting = self.arch_settings[arch]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.stem = nn.Sequential(ConvModule(3, int(arch_setting[0][0] * widen_factor // 2), 3, padding=1, stride=2, act_cfg=act_cfg), ConvModule(int(arch_setting[0][0] * widen_factor // 2), int(arch_setting[0][0] * widen_factor // 2), 3, padding=1, stride=1, act_cfg=act_cfg), ConvModule(int(arch_setting[0][0] * widen_factor // 2), int(arch_setting[0][0] * widen_factor), 3, padding=1, stride=1, act_cfg=act_cfg))
        self.layers = ['stem']
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(in_channels, out_channels, 3, stride=2, padding=1, act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(out_channels, out_channels, act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(out_channels, out_channels, num_blocks=num_blocks, add_identity=add_identity, use_depthwise=use_depthwise, use_cspnext_block=True, expand_ratio=expand_ratio, channel_attention=channel_attention, act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def freeze_stages(self) -> None:
        """Freeze stages parameters."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

class CSPNeXtPAFPN(nn.Module):
    """Path Aggregation Network with CSPNeXt blocks."""

    def __init__(self, in_channels: Sequence[int], out_channels: int, out_indices: Tuple[int, ...]=(1, 2), num_csp_blocks: int=3, use_depthwise: bool=False, expand_ratio: float=0.5, act_cfg: Dict=dict(type='SiLU')) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_indices = out_indices
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1, act_cfg=act_cfg))
            self.top_down_blocks.append(CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1], num_blocks=num_csp_blocks, add_identity=False, use_depthwise=use_depthwise, use_cspnext_block=True, expand_ratio=expand_ratio, act_cfg=act_cfg))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(conv(in_channels[idx], in_channels[idx], 3, stride=2, padding=1, act_cfg=act_cfg))
            self.bottom_up_blocks.append(CSPLayer(in_channels[idx] * 2, in_channels[idx + 1], num_blocks=num_csp_blocks, add_identity=False, use_depthwise=use_depthwise, use_cspnext_block=True, expand_ratio=expand_ratio, act_cfg=act_cfg))
        if self.out_channels is not None:
            self.out_convs = nn.ModuleList()
            for i in range(len(in_channels)):
                self.out_convs.append(conv(in_channels[i], out_channels, 3, padding=1, act_cfg=act_cfg))

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)
        if self.out_channels is not None:
            for idx in range(len(outs)):
                outs[idx] = self.out_convs[idx](outs[idx])
        return tuple([outs[i] for i in self.out_indices])

class ScaleNorm(nn.Module):
    """Scale normalization layer with scaling factor."""

    def __init__(self, dim: int, eps: float=1e-05):
        super().__init__()
        self.scale = dim ** (-0.5)
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class Scale(nn.Module):
    """Scale vector by element multiplications."""

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

def drop_path(x: torch.Tensor, drop_prob: float=0.0, training: bool=False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

def rope(x, dim):
    """Applies Rotary Position Embedding to input tensor."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(torch.arange(total_len, dtype=torch.int, device=x.device), spatial_shape)
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000 ** (-freq_seq)
    sinusoid = position[..., None] * inv_freq[None, None, :]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

def gaussian_blur1d(simcc: torch.Tensor, kernel: int=11) -> torch.Tensor:
    """Modulate simcc distribution with Gaussian using PyTorch.
    
    Args:
        simcc (torch.Tensor[N, K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        torch.Tensor ([N, K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    N, K, Wx = simcc.shape
    sigma = kernel / 6.0
    x = torch.arange(-border, border + 1, dtype=torch.float, device=simcc.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_1d = kernel_1d.view(1, 1, kernel).expand(1, 1, kernel)
    result = torch.zeros_like(simcc)

def get_simcc_maximum(simcc_x: torch.Tensor, simcc_y: torch.Tensor, apply_softmax: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get maximum response location and value from simcc representations.
    
    Args:
        simcc_x (torch.Tensor): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (torch.Tensor): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (torch.Tensor): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (torch.Tensor): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    assert simcc_x.dim() == 2 or simcc_x.dim() == 3, f'Invalid shape {simcc_x.shape}'
    assert simcc_y.dim() == 2 or simcc_y.dim() == 3, f'Invalid shape {simcc_y.shape}'
    assert simcc_x.dim() == simcc_y.dim(), f'{simcc_x.shape} != {simcc_y.shape}'
    if simcc_x.dim() == 3:
        N, K, Wx = simcc_x.shape
        simcc_x_reshape = simcc_x.reshape(N * K, -1)
        simcc_y_reshape = simcc_y.reshape(N * K, -1)
    else:
        N = None
        simcc_x_reshape = simcc_x
        simcc_y_reshape = simcc_y
    if apply_softmax:
        simcc_x_reshape = simcc_x_reshape - torch.max(simcc_x_reshape, dim=1, keepdim=True)[0]
        simcc_y_reshape = simcc_y_reshape - torch.max(simcc_y_reshape, dim=1, keepdim=True)[0]
        ex, ey = (torch.exp(simcc_x_reshape), torch.exp(simcc_y_reshape))
        simcc_x_reshape = ex / torch.sum(ex, dim=1, keepdim=True)
        simcc_y_reshape = ey / torch.sum(ey, dim=1, keepdim=True)
    x_locs = torch.argmax(simcc_x_reshape, dim=1)
    y_locs = torch.argmax(simcc_y_reshape, dim=1)
    locs = torch.stack((x_locs, y_locs), dim=-1).float()
    max_val_x = torch.amax(simcc_x_reshape, dim=1)
    max_val_y = torch.amax(simcc_y_reshape, dim=1)
    vals = torch.minimum(max_val_x, max_val_y)
    locs[vals <= 0.0] = -1
    if N is not None:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)
    return (locs, vals)

def refine_simcc_dark(keypoints: torch.Tensor, simcc: torch.Tensor, blur_kernel_size: int) -> torch.Tensor:
    """PyTorch version of SimCC refinement using distribution aware decoding for UDP.
    
    Args:
        keypoints (torch.Tensor): The keypoint coordinates in shape (N, K, D)
        simcc (torch.Tensor): The heatmaps in shape (N, K, Wx)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        torch.Tensor: Refined keypoint coordinates in shape (N, K, D)
    """
    N = simcc.shape[0]
    simcc = gaussian_blur1d(simcc, blur_kernel_size)
    simcc = torch.clamp(simcc, min=0.001, max=50.0)
    simcc = torch.log(simcc)
    simcc = F.pad(simcc, (2, 2), mode='replicate')
    keypoints_refined = keypoints.clone()
    for n in range(N):
        px = (keypoints[n] + 2.5).long().view(-1, 1)
        px = torch.clamp(px, min=0, max=simcc.shape[2] - 1)
        batch_idx = torch.zeros_like(px).long() + n
        channel_idx = torch.arange(px.shape[0], device=px.device).view(-1, 1)
        dx0 = simcc[n, torch.arange(px.shape[0], device=px.device), px.squeeze(-1)]
        dx1 = simcc[n, torch.arange(px.shape[0], device=px.device), (px + 1).squeeze(-1)]
        dx_1 = simcc[n, torch.arange(px.shape[0], device=px.device), (px - 1).squeeze(-1)]
        dx2 = simcc[n, torch.arange(px.shape[0], device=px.device), (px + 2).squeeze(-1)]
        dx_2 = simcc[n, torch.arange(px.shape[0], device=px.device), (px - 2).squeeze(-1)]
        dx = 0.5 * (dx1 - dx_1)
        dxx = 1e-09 + 0.25 * (dx2 - 2 * dx0 + dx_2)
        offset = dx / dxx
        keypoints_refined[n] -= offset
    return keypoints_refined

class SimCCCodec:
    """Generate keypoint representation via SimCC approach - All PyTorch implementation.
    
    This class implements the SimCC (Simple Coordinate Classification) approach for human pose estimation
    without relying on NumPy, ensuring full PyTorch tensor compatibility.
    
    Args:
        input_size (tuple): Input image size in [w, h]
        smoothing_type (str): The SimCC label smoothing strategy. Options are
            'gaussian' and 'standard'. Defaults to 'gaussian'
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC label.
            Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input size.
            For example, if the input width is w, the x label size will be
            w*simcc_split_ratio. Defaults to 2.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to False.
        use_dark (bool): Whether to use the DARK post processing. Defaults to False.
    """

    def __init__(self, input_size, smoothing_type='gaussian', sigma=6.0, simcc_split_ratio=2.0, normalize=False, use_dark=False):
        self.input_size = input_size
        self.smoothing_type = smoothing_type
        self.simcc_split_ratio = simcc_split_ratio
        self.normalize = normalize
        self.use_dark = use_dark
        if isinstance(sigma, (float, int)):
            sigma = [sigma, sigma]
        self.sigma = torch.tensor(sigma)

    def encode(self, keypoints, keypoints_visible=None):
        """Encoding keypoints into SimCC labels. Note that the original
        keypoint coordinates should be in the input image space.
        
        This is primarily used for training but included for completeness.
        """
        raise NotImplementedError('SimCCCodecPyTorch.encode() is not implemented, only supports inference.')

    def decode(self, simcc_x: torch.Tensor, simcc_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.

        Args:
            simcc_x (torch.Tensor): SimCC label for x-axis
            simcc_y (torch.Tensor): SimCC label for y-axis

        Returns:
            tuple:
            - keypoints (torch.Tensor): Decoded coordinates in shape (N, K, D)
            - scores (torch.Tensor): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """
        device = simcc_x.device
        if simcc_x.dim() == 2:
            simcc_x = simcc_x.unsqueeze(0)
        if simcc_y.dim() == 2:
            simcc_y = simcc_y.unsqueeze(0)
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
        if self.use_dark:
            sigma_tensor = self.sigma.to(device)
            x_blur = int((sigma_tensor[0] * 20 - 7) // 3)
            y_blur = int((sigma_tensor[1] * 20 - 7) // 3)
            x_blur -= int(x_blur % 2 == 0)
            y_blur -= int(y_blur % 2 == 0)
            for i in range(keypoints.shape[0]):
                keypoints_x = keypoints[i, :, 0:1]
                keypoints_y = keypoints[i, :, 1:2]
                keypoints[i, :, 0] = refine_simcc_dark(keypoints_x, simcc_x[i:i + 1], x_blur)[:, 0]
                keypoints[i, :, 1] = refine_simcc_dark(keypoints_y, simcc_y[i:i + 1], y_blur)[:, 0]
        keypoints /= self.simcc_split_ratio
        return (keypoints, scores)

class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock."""

    def __init__(self, num_token, in_token_dims, out_token_dims, expansion_factor=2, s=128, eps=1e-05, dropout_rate=0.0, drop_path=0.0, attn_type='self-attn', act_fn='SiLU', bias=False, use_rel_bias=True, pos_enc=False):
        super(RTMCCBlock, self).__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.e = int(in_token_dims * expansion_factor)
        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)
        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)
        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError
        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False
        self.register_buffer('sqrt_s', torch.sqrt(torch.tensor(s, dtype=torch.float)), persistent=False)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias."""
        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        """GAU Forward function."""
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs
        x = self.ln(x)
        uv = self.uv(x)
        uv = self.act_fn(uv)
        if self.attn_type == 'self-attn':
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta[None, None, :]
            if self.pos_enc:
                base = rope(base, dim=1)
            q, k = torch.unbind(base, dim=2)
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            k = self.k_fc(k)
            v = self.v_fc(v)
            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        if self.use_rel_bias:
            if self.attn_type == 'self-attn':
                bias = self.rel_pos_bias(q.size(1))
            else:
                bias = self.rel_pos_bias(q.size(1), k.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)
        if self.attn_type == 'self-attn':
            x = u * torch.bmm(kernel, v)
        else:
            x = u * torch.bmm(kernel, v)
        x = self.o(x)
        return x

    def forward(self, x):
        """Forward function."""
        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))

class RTMWHead(nn.Module):
    """Top-down head introduced in RTMPose-Wholebody (2023).
    Updated to use PyTorch-only implementations without NumPy or OpenCV.
    """

    def __init__(self, in_channels: int, out_channels: int, input_size: Tuple[int, int], in_featuremap_size: Tuple[int, int], simcc_split_ratio: float=2.0, final_layer_kernel_size: int=7, gau_cfg: Optional[Dict]=None, decoder: Optional[Dict]=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        if gau_cfg is None:
            gau_cfg = dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn='ReLU', use_rel_bias=False, pos_enc=False)
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        ps = 2
        self.ps = nn.PixelShuffle(ps)
        self.conv_dec = ConvModule(in_channels // ps ** 2, in_channels // 4, kernel_size=final_layer_kernel_size, stride=1, padding=final_layer_kernel_size // 2, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.final_layer = ConvModule(in_channels, out_channels, kernel_size=final_layer_kernel_size, stride=1, padding=final_layer_kernel_size // 2, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.final_layer2 = ConvModule(in_channels // ps + in_channels // 4, out_channels, kernel_size=final_layer_kernel_size, stride=1, padding=final_layer_kernel_size // 2, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.mlp = nn.Sequential(ScaleNorm(flatten_dims), nn.Linear(flatten_dims, gau_cfg['hidden_dims'] // 2, bias=False))
        self.mlp2 = nn.Sequential(ScaleNorm(flatten_dims * ps ** 2), nn.Linear(flatten_dims * ps ** 2, gau_cfg['hidden_dims'] // 2, bias=False))
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        self.gau = RTMCCBlock(self.out_channels, gau_cfg['hidden_dims'], gau_cfg['hidden_dims'], s=gau_cfg['s'], expansion_factor=gau_cfg['expansion_factor'], dropout_rate=gau_cfg['dropout_rate'], drop_path=gau_cfg['drop_path'], attn_type='self-attn', act_fn=gau_cfg['act_fn'], use_rel_bias=gau_cfg['use_rel_bias'], pos_enc=gau_cfg['pos_enc'])
        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
        if decoder is not None:
            self.decoder = SimCCCodec(input_size=decoder.get('input_size', self.input_size), smoothing_type=decoder.get('smoothing_type', 'gaussian'), sigma=decoder.get('sigma', (4.9, 5.66)), simcc_split_ratio=self.simcc_split_ratio, normalize=decoder.get('normalize', False), use_dark=decoder.get('use_dark', False))
        else:
            self.decoder = SimCCCodec(input_size=self.input_size, sigma=(4.9, 5.66), simcc_split_ratio=self.simcc_split_ratio, normalize=False, use_dark=False)

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the network to get SimCC representations.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        enc_b, enc_t = feats
        feats_t = self.final_layer(enc_t)
        feats_t = torch.flatten(feats_t, 2)
        feats_t = self.mlp(feats_t)
        dec_t = self.ps(enc_t)
        dec_t = self.conv_dec(dec_t)
        enc_b = torch.cat([dec_t, enc_b], dim=1)
        feats_b = self.final_layer2(enc_b)
        feats_b = torch.flatten(feats_b, 2)
        feats_b = self.mlp2(feats_b)
        feats = torch.cat([feats_t, feats_b], dim=2)
        feats = self.gau(feats)
        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        return (pred_x, pred_y)

    def predict(self, feats: Tuple[torch.Tensor, torch.Tensor], flip_test=False, flip_indices=None):
        """Predict keypoints from features.
        
        Args:
            feats (Tuple[torch.Tensor]): Features from the backbone + neck
            flip_test (bool): Whether to use flip test augmentation
            flip_indices (List[int]): Indices for flipping keypoints
            
        Returns:
            List[Dict]: Predicted keypoints and scores
        """
        batch_pred_x, batch_pred_y = (None, None)
        device = feats[0].device
        if flip_test:
            assert flip_indices is not None, 'flip_indices must be provided for flip test'
            _batch_pred_x, _batch_pred_y = self.forward(feats)
            feats_flipped = [torch.flip(feat, dims=[-1]) for feat in feats]
            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(feats_flipped)
            _batch_pred_x_flip = torch.flip(_batch_pred_x_flip, dims=[2])
            batch_size = _batch_pred_x.shape[0]
            for i in range(batch_size):
                for src_idx, dst_idx in enumerate(flip_indices):
                    if src_idx != dst_idx:
                        _batch_pred_x_flip[i, dst_idx] = _batch_pred_x_flip[i, src_idx].clone()
                        _batch_pred_y_flip[i, dst_idx] = _batch_pred_y_flip[i, src_idx].clone()
            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)
        keypoints, scores = self.decoder.decode(batch_pred_x, batch_pred_y)
        batch_size = keypoints.shape[0]
        instances = []
        for i in range(batch_size):
            instances.append({'keypoints': keypoints[i], 'keypoint_scores': scores[i]})
        return instances

class RTMWModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CSPNeXt(arch=config.backbone_arch, deepen_factor=config.backbone_deepen_factor, widen_factor=config.backbone_widen_factor, expand_ratio=config.backbone_expand_ratio, channel_attention=config.backbone_channel_attention, use_depthwise=False)
        self.neck = CSPNeXtPAFPN(in_channels=config.neck_in_channels, out_channels=config.neck_out_channels, num_csp_blocks=config.neck_num_csp_blocks, expand_ratio=config.neck_expand_ratio, use_depthwise=False)
        gau_cfg = {'hidden_dims': config.gau_hidden_dims, 's': config.gau_s, 'expansion_factor': config.gau_expansion_factor, 'dropout_rate': config.gau_dropout_rate, 'drop_path': config.gau_drop_path, 'act_fn': config.gau_act_fn, 'use_rel_bias': config.gau_use_rel_bias, 'pos_enc': config.gau_pos_enc}
        self.head = RTMWHead(in_channels=config.head_in_channels, out_channels=config.num_keypoints, input_size=config.input_size, in_featuremap_size=config.head_in_featuremap_size, simcc_split_ratio=config.simcc_split_ratio, final_layer_kernel_size=config.head_final_layer_kernel_size, gau_cfg=gau_cfg, decoder=dict(input_size=config.input_size, sigma=config.decoder_sigma, simcc_split_ratio=config.simcc_split_ratio, normalize=config.decoder_normalize, use_dark=config.decoder_use_dark))

    def forward(self, pixel_values):
        return self.head(self.neck(self.backbone(pixel_values)))
