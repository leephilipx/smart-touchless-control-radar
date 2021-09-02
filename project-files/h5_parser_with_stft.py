import numpy as np
from scipy.signal import stft
import h5py, json
import matplotlib.pyplot as plt

# Some parameters
Rmin = 0.2
Rmax = 1.0

# Read the h5 file
hf = h5py.File('subgroupC/raw data xm122 yvonne/XM122 range(0.2m-1m),Max buffered frame 128,Update rate 30Hz [leftright twice2].h5', 'r')

# The data contains complex128 dtype with dimensions (frame_count, depth_slices_count)
data = np.squeeze(np.array(hf['/data']))
Nframe, Ndepth = data.shape

# Extract some properties of the h5 file
property_keys = ['timestamp', 'rss_version', 'lib_version']
properties = {key: str(np.array(hf[f'/{key}']))[2:-1] for key in property_keys}

# Data info is a list containing extra information about the data collected in each frame
data_info = json.loads(str(np.array(hf['/data_info']))[2:-1])

hf.close()


print(data_info[0], properties)
processed_data = abs(np.fft.fft(data, 128))

plt.imshow(processed_data/np.max(processed_data), aspect='auto')
plt.show()
