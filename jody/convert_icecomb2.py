import numpy as np
import matplotlib.pylab as plt
from astropy.io import ascii

data = ascii.read('ICE_COMB2.asc')
wave = 1e4/data['col1']
n = data['col2']
k = data['col3']
ssubs = np.argsort(wave)

data['col1'] = wave[ssubs]
data['col2'] = n[ssubs]
data['col3'] = k[ssubs]

ascii.write(data,'ICE_COMB2.lnk', overwrite=True)
