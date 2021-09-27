#from scipy import signal

from os.path import dirname, join as pjoin
import scipy.io as sio
#data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin("./tests/data/", 'close1_1.mat')
mat_contents = sio.loadmat(mat_fname)
print(mat_contents)