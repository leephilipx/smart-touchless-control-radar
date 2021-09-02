import numpy as np
from scipy import signal
import h5py

hf = h5py.File('subgroupC/testfile1.h5', 'r')
data = np.array(hf['/data'])
timestamp = str(hf['/timestamp'])
hf.close()

print(timestamp)
