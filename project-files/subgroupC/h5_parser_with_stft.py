import numpy as np
from scipy.signal import stft
import h5py
import matplotlib.pyplot as plt

# Some parameters
Rmin = 0.2
Rmax = 1.0

# Read the h5 file
hf = h5py.File('subgroupC/raw data xm122 yvonne/XM122 range(0.2m-1m),Max buffered frame 128,Update rate 30Hz [leftright twice1].h5', 'r')

# The data contains complex128 dtype with dimensions (frame_count, slices)
data = np.squeeze(np.array(hf['/data']))
Nframe, NTS = data.shape

# timestamp = str(hf['/timestamp'])
hf.close()

print(data.shape)
plt.imshow(abs(data))
plt.show()
