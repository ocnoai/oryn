# Pure tensor math from TorchAudio 2.11.0, BSD-2-Clause (LICENSE-TorchAudio.txt).
# No compiled TorchAudio extension is imported. Only the reviewed operators below.
from __future__ import annotations
import math
import warnings
from types import SimpleNamespace
from typing import Callable, Optional, Union
import torch
from torch import Tensor
_CPU = torch.device("cpu")
def spectrogram(waveform: Tensor, pad: int, window: Tensor, n_fft: int, hop_length: int, win_length: int, power: Optional[float], normalized: Union[bool, str], center: bool=True, pad_mode: str='reflect', onesided: bool=True, return_complex: Optional[bool]=None) -> Tensor:
    if return_complex is not None:
        warnings.warn('`return_complex` argument is now deprecated and is not effective.`torchaudio.functional.spectrogram(power=None)` always returns a tensor with complex dtype. Please remove the argument in the function call.')
    if pad > 0:
        waveform = torch.nn.functional.pad(waveform, (pad, pad), 'constant')
    frame_length_norm, window_norm = _get_spec_norms(normalized)
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    spec_f = torch.stft(input=waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=frame_length_norm, onesided=onesided, return_complex=True)
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])
    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return spec_f

def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = (False, False)
    if torch.jit.isinstance(normalized, str):
        if normalized not in ['frame_length', 'window']:
            raise ValueError('Invalid normalized parameter: {}'.format(normalized))
        if normalized == 'frame_length':
            frame_length_norm = True
        elif normalized == 'window':
            window_norm = True
    elif torch.jit.isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError('Input type not supported')
    return (frame_length_norm, window_norm)

def _hz_to_mel(freq: float, mel_scale: str='htk') -> float:
    if mel_scale not in ['slaney', 'htk']:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    if mel_scale == 'htk':
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels

def _mel_to_hz(mels: Tensor, mel_scale: str='htk') -> Tensor:
    if mel_scale not in ['slaney', 'htk']:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    if mel_scale == 'htk':
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs

def _create_triangular_filterbank(all_freqs: Tensor, f_pts: Tensor) -> Tensor:
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    zero = torch.zeros(1)
    down_slopes = -1.0 * slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb

def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int, norm: Optional[str]=None, mel_scale: str='htk') -> Tensor:
    if norm is not None and norm != 'slaney':
        raise ValueError('norm must be one of None or "slaney"')
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    fb = _create_triangular_filterbank(all_freqs, f_pts)
    if norm is not None and norm == 'slaney':
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    if fb.device.type != 'meta' and (fb.max(dim=0).values == 0.0).any():
        warnings.warn(f'At least one mel filterbank has all zero values. The value for `n_mels` ({n_mels}) may be set too high. Or, the value for `n_freqs` ({n_freqs}) may be set too low.')
    return fb

def _get_sinc_resample_kernel(orig_freq: int, new_freq: int, gcd: int, lowpass_filter_width: int=6, rolloff: float=0.99, resampling_method: str='sinc_interp_hann', beta: Optional[float]=None, device: torch.device=_CPU, dtype: Optional[torch.dtype]=None):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception('Frequencies must be of integer type to ensure quality resampling computation. To work around this, manually convert both frequencies to integer values that maintain their resampling rate ratio before passing them into the function. Example: To downsample a 44100 hz waveform by a factor of 8, use `orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. For more information, please refer to https://github.com/pytorch/audio/issues/1487.')
    if resampling_method in ['sinc_interpolation', 'kaiser_window']:
        method_map = {'sinc_interpolation': 'sinc_interp_hann', 'kaiser_window': 'sinc_interp_kaiser'}
        warnings.warn(f'"{resampling_method}" resampling method name is being deprecated and replaced by "{method_map[resampling_method]}" in the next release. The default behavior remains unchanged.', stacklevel=3)
    elif resampling_method not in ['sinc_interp_hann', 'sinc_interp_kaiser']:
        raise ValueError('Invalid resampling method: {}'.format(resampling_method))
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    if lowpass_filter_width <= 0:
        raise ValueError('Low pass filter width should be positive.')
    base_freq = min(orig_freq, new_freq)
    base_freq *= rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else torch.float64
    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    if resampling_method == 'sinc_interp_hann':
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale
    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)
    return (kernels, width)

def _apply_sinc_resample_kernel(waveform: Tensor, orig_freq: int, new_freq: int, gcd: int, kernel: Tensor, width: int):
    if not waveform.is_floating_point():
        raise TypeError(f'Expected floating point type for waveform tensor, but received {waveform.dtype}.')
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled

def resample(waveform: Tensor, orig_freq: int, new_freq: int, lowpass_filter_width: int=6, rolloff: float=0.99, resampling_method: str='sinc_interp_hann', beta: Optional[float]=None) -> Tensor:
    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError('Original frequency and desired frequecy should be positive')
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))
    kernel, width = _get_sinc_resample_kernel(orig_freq, new_freq, gcd, lowpass_filter_width, rolloff, resampling_method, beta, waveform.device, waveform.dtype)
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled

class Spectrogram(torch.nn.Module):
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self, n_fft: int=400, win_length: Optional[int]=None, hop_length: Optional[int]=None, pad: int=0, window_fn: Callable[..., Tensor]=torch.hann_window, power: Optional[float]=2.0, normalized: Union[bool, str]=False, wkwargs: Optional[dict]=None, center: bool=True, pad_mode: str='reflect', onesided: bool=True, return_complex: Optional[bool]=None) -> None:
        super(Spectrogram, self).__init__()
        torch._C._log_api_usage_once('torchaudio.transforms.Spectrogram')
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        if return_complex is not None:
            warnings.warn('`return_complex` argument is now deprecated and is not effective.`torchaudio.transforms.Spectrogram(power=None)` always returns a tensor with complex dtype. Please remove the argument in the function call.')

    def forward(self, waveform: Tensor) -> Tensor:
        return spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length, self.win_length, self.power, self.normalized, self.center, self.pad_mode, self.onesided)

class MelScale(torch.nn.Module):
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max', 'n_stft']

    def __init__(self, n_mels: int=128, sample_rate: int=16000, f_min: float=0.0, f_max: Optional[float]=None, n_stft: int=201, norm: Optional[str]=None, mel_scale: str='htk') -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.n_stft = n_stft
        self.norm = norm
        self.mel_scale = mel_scale
        if f_min > self.f_max:
            raise ValueError('Require f_min: {} <= f_max: {}'.format(f_min, self.f_max))
        with torch.device('cpu'):
            fb = melscale_fbanks(self.n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram

class MelSpectrogram(torch.nn.Module):
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

    def __init__(self, sample_rate: int=16000, n_fft: int=400, win_length: Optional[int]=None, hop_length: Optional[int]=None, f_min: float=0.0, f_max: Optional[float]=None, pad: int=0, n_mels: int=128, window_fn: Callable[..., Tensor]=torch.hann_window, power: float=2.0, normalized: bool=False, wkwargs: Optional[dict]=None, center: bool=True, pad_mode: str='reflect', onesided: Optional[bool]=None, norm: Optional[str]=None, mel_scale: str='htk') -> None:
        super(MelSpectrogram, self).__init__()
        torch._C._log_api_usage_once('torchaudio.transforms.MelSpectrogram')
        if onesided is not None:
            warnings.warn("Argument 'onesided' has been deprecated and has no influence on the behavior of this module.")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, pad=self.pad, window_fn=window_fn, power=self.power, normalized=self.normalized, wkwargs=wkwargs, center=center, pad_mode=pad_mode, onesided=True)
        self.mel_scale = MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm, mel_scale)

    def forward(self, waveform: Tensor) -> Tensor:
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram

class Resample(torch.nn.Module):

    def __init__(self, orig_freq: int=16000, new_freq: int=16000, resampling_method: str='sinc_interp_hann', lowpass_filter_width: int=6, rolloff: float=0.99, beta: Optional[float]=None, *, dtype: Optional[torch.dtype]=None) -> None:
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(self.orig_freq), int(self.new_freq))
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta
        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(self.orig_freq, self.new_freq, self.gcd, self.lowpass_filter_width, self.rolloff, self.resampling_method, beta, dtype=dtype)
            self.register_buffer('kernel', kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        return _apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width)

functional = SimpleNamespace(resample=resample)
transforms = SimpleNamespace(MelSpectrogram=MelSpectrogram, Resample=Resample)
