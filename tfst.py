import torch
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


def create_kernels(sizes: Union[np.ndarray, List[int]],
                   window: Union[str, tuple],
                   frequencies: torch.Tensor,
                   fs: int,
                   device: str,
                   ):
    if isinstance(sizes, List):
        sizes: np.ndarray = np.array(sizes)

    total_size = np.sum(sizes)
    shape = [frequencies.shape[0], total_size]

    kernels = torch.empty(shape)

    for s in range(sizes.size):
        start = np.sum(sizes[:s])
        t = torch.arange(sizes[s], device=device)
        window = torch.tensor(get_window_dispatch(window, sizes[s]),
                              device=device)
        time_frequency = \
            torch.unsqueeze(frequencies, 1) * torch.unsqueeze(t, 0) / fs
        kernel = torch.exp(1j * 2 * np.pi * time_frequency) * window

        kernels[:, start: start + sizes[s]] = torch.linalg.norm(kernel, ord=1,
                                                                dim=1)

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
        self.sizes = sizes
        self.pad_mode = pad_mode
        self.window = window
        self.norm = norm
        self.output_format = output_format
        self.f_min = f_min
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # Derived attributes
        self.n_bins = int(bins_per_octave * n_octaves)
        frequencies = f_min * 2**(torch.arange(self.n_bins, device=device))
        frequencies = frequencies[frequencies < fs / 2]
        self.frequencies: torch.Tensor = frequencies

        # Create kernels
        self.kernels = create_kernels(sizes, window, frequencies, fs, device)

    def forward(self, input_tensor):
        output_shape = [len(self.sizes), self.frequencies.shape[0],
                        input_tensor.shape[0] // self.hop_length]
        output_tensor = torch.empty(output_shape, device=self.device)

        for s in range(self.sizes.size):
            start = np.sum(self.sizes[:s])
            output_tensor[s, :, :] = \
                torch.conv1d(input_tensor,
                             self.kernels[:, start: start + self.sizes[s]],
                             stride=self.hop_length)

        return output_tensor
