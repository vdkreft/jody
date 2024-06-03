import matplotlib.pylab as plt
from astropy.io import fits,ascii
import matplotlib.ticker as mticker

#kp5 = ascii.read('opacity_bestfit.asc',data_start=2)
#kp5_wv = kp5['col1']
#kp5_abs = kp5['col2']
#kp5_sca = kp5['col3']
#kp5_ext = kp5['col2']+kp5['col3']
#kp5_gsca = kp5['col4']

kp5 = fits.getdata('kp5.fits')
kp5_wv = kp5['wavelength']
kp5_abs = kp5['kabs']
kp5_sca = kp5['ksca']

opac1 = fits.getdata('KP_V8.0_testcandidate.fits')

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)

ax.plot(opac1['wavelength'],opac1['kabs']*3.3/3.8,color='purple',label='KP5 benchmark ($\kappa_{abs}$)')
ax.plot(opac1['wavelength'],opac1['ksca']*3.3/3.8,color='purple',linestyle='--',label='KP5 benchmark ($\kappa_{sca}$ )')

ax.plot(kp5_wv,kp5_abs,color='orange',label='KP5 ($\kappa_{abs}$)')
ax.plot(kp5_wv,kp5_sca,color='orange',linestyle='--',label='KP5 ($\kappa_{sca}$)')
ax.annotate('(C)', (1000,15000),weight='bold',fontsize=10)

#plt.plot(kp5_wv,kp5_gsca,color='orange',linestyle='--')
#plt.plot(opac1['wavelength'],opac1['gsca']*3.3/3.8,color='purple')

ax.legend(loc=3)

ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel(r'$\kappa\ [cm^2/g]$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim((0.11,2000))

plt.tight_layout()
fig.savefig('opac.pdf')
plt.show()