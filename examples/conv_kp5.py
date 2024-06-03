import numpy as np
from astropy.io import fits,ascii
from datetime import date

kp5 = ascii.read('opacity_bestfit.asc',data_start=2)
kp5_wv = kp5['col1']
kp5_abs = kp5['col2']
kp5_sca = kp5['col3']
kp5_ext = kp5['col2']+kp5['col3']
kp5_gsca = kp5['col4']

ssubs = np.argsort(kp5_wv)
kp5_wv = kp5_wv[ssubs]
kp5_abs = kp5_abs[ssubs]
kp5_sca = kp5_sca[ssubs]
kp5_ext = kp5_ext[ssubs]
kp5_gsca = kp5_gsca[ssubs]

c1 = fits.Column(name='wavelength', array=kp5_wv, format='F')
c2 = fits.Column(name='kabs', array=kp5_abs, format='F')
c3 = fits.Column(name='ksca', array=kp5_sca, format='F')
c4 = fits.Column(name='gsca', array=kp5_gsca, format='F')

t = fits.BinTableHDU.from_columns([c1, c2, c3, c4])

t.header['KUNIT'] = 'cm2/g'
t.header['WAVEUNIT'] = 'micron'
t.header['DATE'] = date.today().strftime("%m/%d/%y")
t.header['COMMENT'] = 'KP5 - see Chapman et al. 2009, ApJ, 690, 496'
t.header['COMMENT'] = 'This opacity is documented in Pontoppidan et al. 2024, RNAAS'
t.header['COMMENT'] = 'Version 5.0'
t.header['COMMENT'] = 'Written by Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)'
		
t.writeto('kp5.fits',overwrite=True)