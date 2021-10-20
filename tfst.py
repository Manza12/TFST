import torch
from math import ceil
import torch.nn.functional as func
import scipy.signal.windows as win
import numpy as np
from typing import Union, List


def get_window_dispatch(window, n, fft_bins=True):
    if isinstance(window, str):
        return win.get_window(window, n, fftbins=fft_bins)
    elif isinstance(window, tuple):
        if window[0] == 'gaussian':
            assert window[1] >= 0
            sigma = np.floor(- n / 2 / np.sqrt(- 2 *
                                               np.log(10**(- window[1] / 20))))
            return win.gaussian(n, sigma, not fft_bins)
        else:
            Warning("Tuple windows may have undesired behaviour "
                    "regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " +
                str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports "
                        "strings, tuples and floats.")


def create_kernels(sizes: np.ndarray,
                   window: Union[str, tuple],
                   frequencies: torch.Tensor,
                   fs: int,
                   device: str,
                   ):
    total_size = np.sum(sizes)
    shape = [frequencies.shape[0], total_size]

    kernels = torch.empty(shape, device=device, dtype=torch.complex64)

    for s in range(sizes.size):
        start = np.sum(sizes[:s])
        t = torch.arange(sizes[s], device=device)
        window_tensor = torch.tensor(get_window_dispatch(window, sizes[s]),
                                     device=device, dtype=torch.complex64)
        window_tensor = window_tensor / torch.linalg.norm(window_tensor, ord=1)
        time_frequency = \
            torch.unsqueeze(frequencies, 1) * torch.unsqueeze(t, 0) / fs
        kernel = torch.exp(1j * 2 * np.pi * time_frequency) * window_tensor

        kernels[:, start: start + sizes[s]] = kernel

    return kernels


class TFST(torch.nn.Module):
    def __init__(self, fs: int = 48000, hop_length: int = 480,
                 sizes: Union[np.ndarray, List[int]] = (512, 1024, 2048, 4096),
                 window: Union[str, tuple, float] = 'hann',
                 norm: Union[int, str] = 1, pad_mode='constant',
                 f_min: Union[int, float] = 55,
                 n_octaves: Union[int, float] = 8,
                 bins_per_octave: int = 12,
                 output_format: str = 'Magnitude',
                 device: str = 'cuda:0',
                 ):
        super().__init__()

        # Direct attributes
        self.fs = fs
        self.hop_length = hop_length
        self.time_resolution = hop_length / fs
        self.sizes = np.array(sizes)
        self.pad_mode = pad_mode
        self.window = window
        self.norm = norm
        self.output_format = output_format
        self.f_min = f_min
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.device = device

        # Derived attributes
        self.n_bins = int(bins_per_octave * n_octaves)
        frequencies = f_min * 2**(torch.arange(self.n_bins, device=device) /
                                  bins_per_octave)
        frequencies = frequencies[frequencies < fs / 2]
        self.frequencies: torch.Tensor = frequencies

        # Create kernels
        self.kernels = create_kernels(self.sizes, window, frequencies, fs,
                                      device)

    def forward(self, input_tensor: torch.Tensor):
        if self.output_format == 'Magnitude':
            output_shape = [self.sizes.shape[0], self.frequencies.shape[0],
                            int(ceil(input_tensor.shape[0] / self.hop_length))]
            output_tensor = torch.empty(output_shape, device=self.device)

            for s in range(self.sizes.size):
                start = np.sum(self.sizes[:s])
                end = start + self.sizes[s]
                input_conv = func.pad(input_tensor,
                                      [self.sizes[s] // 2, self.sizes[s] // 2],
                                      'constant',
                                      0.).unsqueeze(0).unsqueeze(0)
                output_tensor[s, :, :] = torch.sqrt(
                    func.conv1d(input_conv,
                                self.kernels.real[:, start: end].unsqueeze(1),
                                stride=self.hop_length)[0, :, :]**2 +
                    func.conv1d(input_conv,
                                self.kernels.imag[:, start: end].unsqueeze(1),
                                stride=self.hop_length)[0, :, :]**2
                )
        elif self.output_format == 'Complex':
            output_shape = [2, self.sizes.shape[0], self.frequencies.shape[0],
                            int(ceil(input_tensor.shape[0] / self.hop_length))]
            output_tensor = torch.empty(output_shape, device=self.device)

            for s in range(self.sizes.size):
                start = np.sum(self.sizes[:s])
                end = start + self.sizes[s]
                input_conv = func.pad(input_tensor,
                                      [self.sizes[s] // 2, self.sizes[s] // 2],
                                      'constant',
                                      0.).unsqueeze(0).unsqueeze(0)
                output_tensor[0, s, :, :] = \
                    func.conv1d(input_conv,
                                self.kernels.real[:, start: end].unsqueeze(1),
                                stride=self.hop_length)[0, :, :]
                output_tensor[0, s, :, :] = \
                    func.conv1d(input_conv,
                                self.kernels.imag[:, start: end].unsqueeze(1),
                                stride=self.hop_length)[0, :, :]
        else:
            raise ValueError("Parameter output_format should be one of:"
                             "'Magnitude' or 'Complex'.")

        return output_tensor
