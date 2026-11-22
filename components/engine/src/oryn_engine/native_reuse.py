"""Owned spectral enhancement graphs composed from native Torch and Mamba layers.

Parameter paths bind the complete reviewed RE-USE tensor inventories. Repository
programs are not part of this implementation or imported by it.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer


class ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


def norm(width, causal):
    return ChannelNorm(width) if causal else nn.InstanceNorm2d(width, affine=True)


def stage(conv, width, causal):
    return nn.Sequential(conv, norm(width, causal), nn.PReLU(width))


class Dense(nn.Module):
    def __init__(self, width, causal):
        super().__init__()
        self.dense_block = nn.ModuleList()
        for level in range(4):
            dilation = 2**level
            conv = nn.Conv2d(
                width * (level + 1),
                width,
                3,
                dilation=(dilation, 1),
                padding=0 if causal else (dilation, 1),
            )
            layers = [conv, norm(width, causal), nn.PReLU(width)]
            if causal:
                layers.insert(0, nn.ConstantPad2d((1, 1, 2 * dilation, 0), 0.0))
            self.dense_block.append(nn.Sequential(*layers))

    def forward(self, x):
        history = x
        for layer in self.dense_block:
            result = layer(history)
            history = torch.cat((result, history), dim=1)
        return result


class Encoder(nn.Module):
    def __init__(self, width, causal):
        super().__init__()
        self.causal = causal
        if causal:
            for number in (1, 2, 3):
                setattr(self, f"dense_conv_1_{number}", stage(nn.Conv2d(2, width, 3), width, True))
        else:
            self.dense_conv_1 = stage(nn.Conv2d(2, width, 1), width, False)
        self.dense_block = Dense(width, causal)
        self.dense_conv_2 = stage(
            nn.Conv2d(width, width, (1, 3), stride=(1 if causal else 4, 2)), width, causal
        )

    def forward(self, x, ahead=0):
        if self.causal:
            x = getattr(self, f"dense_conv_1_{ahead + 1}")(F.pad(x, (1, 1, 2 - ahead, ahead)))
        else:
            x = self.dense_conv_1(x)
        return self.dense_conv_2(self.dense_block(x))


class WidthShuffle(nn.Module):
    def __init__(self, width, scale, padding):
        super().__init__()
        self.scale = scale
        self.padding = padding
        self.conv = nn.Conv2d(width, width * scale, (1, 3))

    def forward(self, x):
        z = self.conv(F.pad(x, self.padding))
        b, channels, height, width = z.shape
        return (
            z.reshape(b, self.scale, channels // self.scale, height, width)
            .permute(0, 2, 3, 4, 1)
            .reshape(b, channels // self.scale, height, width * self.scale)
        )


class Decoder(nn.Module):
    def __init__(self, width, causal, phase):
        super().__init__()
        self.phase = phase
        self.dense_block = Dense(width, causal)
        self.up_conv1 = stage(WidthShuffle(width, 2, (1, 1, 0, 0)), width, causal)
        self.up_conv2 = stage(
            WidthShuffle(width, 1 if causal else 4, (2, 0, 0, 0) if causal else (1, 1, 0, 0)),
            width,
            causal,
        )
        if phase:
            self.phase_conv_r = nn.Conv2d(width, 1, 1)
            self.phase_conv_i = nn.Conv2d(width, 1, 1)
        else:
            self.final_conv = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        x = self.up_conv1(self.dense_block(x))
        x = self.up_conv2(x.transpose(2, 3)).transpose(2, 3)
        if not self.phase:
            return self.final_conv(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


def mixer(width, cfg):
    c = MambaConfig(
        hidden_size=width,
        state_size=cfg["d_state"],
        conv_kernel=cfg["d_conv"],
        expand=cfg["expand"],
        time_step_rank=math.ceil(width / 16),
        num_hidden_layers=1,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        use_mambapy=False,
        use_associative_scan=False,
    )
    return MambaMixer(c, 0)


class Directional(nn.Module):
    def __init__(self, width, cfg, bidirectional):
        super().__init__()
        self.forward_blocks = mixer(width, cfg)
        self.bidirectional = bidirectional
        if bidirectional:
            self.backward_blocks = mixer(width, cfg)
        self.output_proj = nn.Linear(width * (2 if bidirectional else 1), width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        forward = self.forward_blocks(x) + x
        if self.bidirectional:
            reverse = x.flip(1)
            backward = (self.backward_blocks(reverse) + reverse).flip(1)
            forward = torch.cat((forward, backward), -1)
        return self.norm(self.output_proj(forward))


class TimeFrequency(nn.Module):
    def __init__(self, width, cfg, causal):
        super().__init__()
        self.time_mamba = Directional(width, cfg, not causal)
        self.freq_mamba = Directional(width, cfg, True)

    def forward(self, x):
        batch, channels, time, freq = x.shape
        time_input = x.permute(0, 3, 2, 1).reshape(batch * freq, time, channels)
        time_output = self.time_mamba(time_input) + time_input
        freq_input = (
            time_output.reshape(batch, freq, time, channels)
            .permute(0, 2, 1, 3)
            .reshape(batch * time, freq, channels)
        )
        freq_output = self.freq_mamba(freq_input) + freq_input
        return freq_output.reshape(batch, time, freq, channels).permute(0, 3, 1, 2)


class ReuseGraph(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["model_cfg"]
        width = c["hid_feature"]
        layers = c["num_tfmamba"]
        self.causal = "maximum_layer" in c
        self.dense_encoder = Encoder(width, self.causal)
        self.TSMamba = nn.ModuleList(TimeFrequency(width, c, self.causal) for _ in range(layers))
        self.mask_decoder = Decoder(width, self.causal, False)
        self.phase_decoder = Decoder(width, self.causal, True)
        if self.causal:
            self.mask_decoder_list = nn.ModuleList(
                Decoder(width, True, False) for _ in range(layers)
            )
            self.phase_decoder_list = nn.ModuleList(
                Decoder(width, True, True) for _ in range(layers)
            )

    def forward(self, mag, phase, depth=None, ahead=0):
        time, freq = mag.shape[-1], mag.shape[-2]
        x = torch.stack((mag, phase), 1).transpose(2, 3)
        x = F.pad(x, (0, 2, 0, 0 if self.causal else 2))
        x = self.dense_encoder(x, ahead)
        depth = len(self.TSMamba) if depth is None else depth
        for layer in self.TSMamba[:depth]:
            x = layer(x)
        if self.causal:
            amp = self.mask_decoder_list[depth - 1](x)
            phi = self.phase_decoder_list[depth - 1](x)
        else:
            amp = self.mask_decoder(x)
            phi = self.phase_decoder(x)
        amp = amp[:, 0].transpose(1, 2)[:, :freq, :time]
        phi = phi[:, 0].transpose(1, 2)[:, :freq, :time]
        return amp, phi, torch.stack((amp * phi.cos(), amp * phi.sin()), -1)
