import time
import torch
from pathlib import Path
import scipy.io.wavfile as wav
from tfst import TFST
import warnings
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.ticker as tick
from MIDISynth import plot_time_frequency
from MIDISynth.plot import format_freq, format_time
import matplotlib.pyplot as plt


# Parameters
sizes = [512, 4096]  # np.logspace(9, 13, 5, base=2).astype(np.int32)
bins_per_octave = 36

print('Testing TFST...')
file_name = 'tempest_3rd'
file_path = Path('.') / Path(file_name + '.wav')
device = 'cuda:0'

start = time.time()
warnings.filterwarnings("ignore", category=wav.WavFileWarning)
fs, signal = wav.read(file_path)
print('Time to read: %.3f seconds' % (time.time() - start))

start = time.time()
signal_tensor = torch.tensor(signal, device=device)
print('Time to device: %.3f seconds' % (time.time() - start))

start = time.time()
tfst_layer = TFST(sizes=sizes, bins_per_octave=bins_per_octave, fs=fs)
print('Time to create layer: %.3f seconds' % (time.time() - start))

start = time.time()
spectrogram = tfst_layer(signal_tensor)
print('Time to apply layer: %.3f seconds' % (time.time() - start))

spectrogram_db = 20 * torch.log10(spectrogram + torch.finfo(torch.float32).eps)
spectrogram_numpy = spectrogram_db.cpu().numpy()
time_vector = np.arange(spectrogram_numpy.shape[-1]) * \
              tfst_layer.hop_length / tfst_layer.fs
frequency_vector = tfst_layer.frequencies.cpu().numpy()

# Plot
t_0 = 0
t_1 = 10.

n_0 = int(t_0 / tfst_layer.time_resolution)
n_1 = int(t_1 / tfst_layer.time_resolution)

v_max = -20
v_min = -100

fig = plt.figure()
fig.suptitle('TFST')

plt.subplots_adjust(bottom=0.25)
ax = fig.subplots()

image = ax.imshow(spectrogram_numpy[0, :, n_0: n_1], cmap='hot', aspect='auto',
                  vmin=v_min, vmax=v_max, origin='lower')
ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])

# Freq axis
ax.yaxis.set_major_formatter(
    tick.FuncFormatter(lambda x, pos: format_freq(x, pos, frequency_vector)))

# Time axis
ax.xaxis.set_major_formatter(
    tick.FuncFormatter(lambda x, pos: format_time(x, pos, time_vector)))

# Labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.show()

slider = Slider(ax_slide, 'Scale', 0, spectrogram_numpy.shape[0] - 1,
                valinit=0, valstep=1)


def update(val):
    image.set_data(spectrogram_numpy[val, :, n_0: n_1])
    fig.canvas.draw()


slider.on_changed(update)

plt.show()

spectrogram_inf, _ = torch.min(spectrogram_db, dim=0)

plot_time_frequency(spectrogram_inf, time_vector, frequency_vector, v_min=-100,
                    v_max=0, c_map='hot', numpy=False, full_screen=True)
