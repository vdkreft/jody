import os
from datetime import date
import matplotlib.pylab as plt
import math
import numpy as np
from scipy.integrate import simpson
from astropy.io import fits,ascii
from astropy.table import Table
import matplotlib.ticker as mticker

__version__ = 1.0

class Colors():
	"""docstring for ClassName"""
	def __init__(self, path, regen_colors=True):

		self.path = path
		self.fracs = np.linspace(0.05,0.3,15)
		self.read_filters()		
		sil_opacs,car_opacs = self.find_opacs(path)

		if regen_colors:
			# Instantiate with a dummy row
			mag_table = Table({'silfile':[''],'carfile':[''],'frac':[0.0],'JI':[0.0],'JB':[0.0],'JV':[0.0],'JR':[0.0], 
				               'J':[0.0],'H':[0.0],'K':[0.0],'I1':[0.0],'I2':[0.0],'I3':[0.0],'I4':[0.0],
				               'I5':[0.0],'M1':[0.0],'MM450':[0.0],'MM850':[0.0],'MM1300':[0.0]})

			for sil_file in sil_opacs:
				sil_opac = fits.getdata(path+sil_file)
				print(sil_file)
				for car_file in car_opacs:
					car_opac = fits.getdata(path+car_file)
					mags = self.calc_colors(self.fracs,sil_opac,car_opac,sil_file,car_file)
					for row in mags:
						mag_table.add_row(row)

			mag_table.remove_row(0) #remove the dummy row
			self.mag_table = mag_table
			mag_table.write('mag_table.fits', format='fits', overwrite=True) 
		
		bestrow, constraints = self.get_chi2()
		self.plot_fit(bestrow,constraints)

	def get_chi2(self):

		mags = fits.getdata('mag_table.fits')

		#colors     = np.array([1.68,-0.633,-0.88,-0.88,-0.926,-0.78,1.9e-3,4e-4]) #Chapman Beta=1.8
		#colors_err = np.array([0.05,0.06,0.07,0.09,0.09,0.1,2e-1,4e-1])

		colors     = np.array([1.58,-0.633,-0.88,-0.88,-0.926,-0.78,1.9e-3,4e-4]) #Chapman Beta=1.6 + 450 and 850 micron constraints from (check from where)
		colors_err = np.array([0.05,0.06,0.07,0.09,0.09,0.1,2e-1,4e-1])
		AI_AK = colors[1:]*(colors[0]-1.)+1.
	
		KH = mags['K']-mags['H']
		colors_all = np.array([mags['H']/mags['K'],
			                   (mags['K']-mags['I1'])/KH, 
			                   (mags['K']-mags['I2'])/KH,
			                   (mags['K']-mags['I3'])/KH,
			                   (mags['K']-mags['I4'])/KH,
			                   (mags['K']-mags['M1'])/KH,
			                   mags['MM450']/mags['K'],
			                   mags['MM850']/mags['K']])

		chis = np.zeros(len(mags))
		for ii,color in enumerate(colors):
			chis += (color - colors_all[ii])**2/colors_err[ii]**2
		chis /= (colors.size-1)
		bestsub = np.argmin(chis)
		bestchi = chis[bestsub]
		
		constraints = {'AH_AK':colors[0], 'AI1_AK':AI_AK[0],'AI2_AK':AI_AK[1],'AI3_AK':AI_AK[2],'AI4_AK':AI_AK[3],'AM1_AK':AI_AK[4],'A450_AK':AI_AK[5],'A850_AK':AI_AK[6]}

		return mags[bestsub],constraints

		'''
		print('AH/AK', colors[0])
		print('AI1/AK', AI_AK[0])
		print('AI2/AK', AI_AK[1])
		print('AI3/AK', AI_AK[2])
		print('AI4/AK', AI_AK[3])
		print('AM1/AK', AI_AK[4])
		'''

	def calc_colors(self,fracs,sil_opac,car_opac,sil_file,car_file):

		w = sil_opac['wavelength']
		sil_a = sil_opac['kabs']
		car_a = car_opac['kabs']
		sil_s = sil_opac['ksca']
		car_s = car_opac['ksca']

		for filter in self.filters:
			filter['gsubs'] = np.where((w>filter['lim'][0]) & (w<filter['lim'][1]))

		#colors = np.zeros((7,fracs.size))
		#mags   = np.zeros((17,fracs.size))
		mags = []
		for ii,frac in enumerate(fracs):
			a = (car_a*frac+car_a*(1.-frac))
			s = (car_s*frac+sil_s*(1.-frac))
			kext = a+s
			#kext /= np.interp(1.0,w,kext) # normalize for ~unity mag values
			for filter in self.filters:
				filter_thr = kext * np.interp(w,filter['wl'],filter['tr'])
				filter['mag'] = np.trapz(filter_thr[filter['gsubs']],x=w[filter['gsubs']])/filter['tot']

			J = self.filters[0]['mag']
			H = self.filters[1]['mag']
			K = self.filters[2]['mag']
			I1 = self.filters[3]['mag']
			I2 = self.filters[4]['mag']
			I3 = self.filters[5]['mag']
			I4 = self.filters[6]['mag']
			M1 = self.filters[7]['mag']
			I5 = np.interp(9.7, w, kext)
			JU = np.interp(0.3735, w, kext)
			JB = np.interp(0.4443, w, kext)
			JV = np.interp(0.5483, w, kext)
			JR = np.interp(0.6855, w, kext)
			JI = np.interp(0.8637, w, kext)
			MM450  = np.interp(450, w, kext)
			MM850  = np.interp(850, w, kext)
			MM1300 = np.interp(1300, w, kext)

			#colors[:,ii] = np.array([H/K,(K-I1)/(K-H),(K-I2)/(K-H),(K-I3)/(K-H),(K-I4)/(K-H),(K-M1)/(K-H),(K-I5)/(K-H)])
			
			mag = {'silfile':sil_file,'carfile':car_file,'frac':frac,'JI':JI,'JB':JB,'JV':JV,'JR':JR,'JI':JI,'J':J,'H':H,'K':K,
			       'I1':I1,'I2':I2,'I3':I3,'I4':I4,'I5':I5,'M1':M1,'MM450':MM450,'MM850':MM850,'MM1300':MM1300}
			#mags[:,ii] = np.array([frac,JI,JB,JV,JR,JI,J,H,K,I1,I2,I3,I4,I5,M1,MM450,MM850,MM1300])
			mags.append(mag)

		return mags

	def read_filters(self):
		path = '../filters/'
		twomass_j = ascii.read(path+'2mass_j_filt.dat')
		tmj = {'wl':twomass_j['xlam'],'tr':twomass_j['yval'],'lim':(1,1.5)}

		twomass_h = ascii.read(path+'2mass_h_filt.dat')
		tmh = {'wl':twomass_h['xlam'],'tr':twomass_h['yval'],'lim':(1.3,2.0)}

		twomass_k = ascii.read(path+'2mass_k_filt.dat')
		tmk = {'wl':twomass_k['xlam'],'tr':twomass_k['yval'],'lim':(1.9,2.5)}

		irac1 = ascii.read(path+'irac_tr1_2004-08-09.dat')
		ir1 = {'wl':irac1['col1'],'tr':irac1['col2'],'lim':(2.9,4.2)}

		irac2 = ascii.read(path+'irac_tr2_2004-08-09.dat')
		ir2 = {'wl':irac2['col1'],'tr':irac2['col2'],'lim':(3.7,5.4)}

		irac3 = ascii.read(path+'irac_tr3_2004-08-09.dat')
		ir3 = {'wl':irac3['col1'],'tr':irac3['col2'],'lim':(4.6,6.9)}

		irac4 = ascii.read(path+'irac_tr4_2004-08-09.dat')
		ir4 = {'wl':irac4['col1'],'tr':irac4['col2'],'lim':(5.6,10.3)}
		
		mips1 = ascii.read(path+'M1filt.dat')
		m1 = {'wl':mips1['lambda'],'tr':mips1['response'],'lim':(18.0,32.2)}

		self.filters = [tmj,tmh,tmk,ir1,ir2,ir3,ir4,m1]
		for filter in self.filters:
			filter['tot'] = np.trapz(filter['tr'],x=filter['wl'])

	def find_opacs(self,path):
		files = os.listdir(path)
		sils = [file for file in files if 'carbon_0.0' in file]
		cars = [file for file in files if 'carbon_1.0' in file]

		return sils, cars

	def plot_fit(self,row,constraints, outspec='KP_V6.0_testcandidate.fits'):

		#caropac = fits.getdata(self.path+row['carfile'],1)
		#silopac = fits.getdata(self.path+row['silfile'],1)


		frac = row['frac']

		#small grains / draine graphite
		carfile = 'opacity_carbon_1.0_amax_1.4_h2o_5e-05_alpha_-1.0.fits'
		silfile = 'opacity_carbon_0.0_amax_0.8_h2o_5e-05_alpha_-2.0.fits'
		frac = 0.07

		#small grains / draine graphite
		#carfile = 'opacity_carbon_1.0_amax_0.8_h2o_5e-05_alpha_-1.5.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.4_h2o_5e-05_alpha_-1.5.fits'
		#frac = 0.11

		#medium grains / draine graphite
		#carfile = 'opacity_carbon_1.0_amax_1.0_h2o_6e-05_alpha_-1.5.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.6_h2o_6e-05_alpha_-1.5.fits'
		#frac = 0.165

        #largest grains / draine graphite
		#carfile = 'opacity_carbon_1.0_amax_1.2_h2o_6e-05_alpha_-1.5.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.8_h2o_6e-05_alpha_-1.5.fits'
		#frac = 0.21

		#small grains
		#carfile = 'opacity_carbon_1.0_amax_0.8_h2o_5e-05_alpha_-1.3.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.4_h2o_5e-05_alpha_-1.3.fits'
		#frac = 0.145

		#medium grains
		#carfile = 'opacity_carbon_1.0_amax_1.0_h2o_6e-05_alpha_-1.3.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.6_h2o_6e-05_alpha_-1.3.fits'
		#frac = 0.2
		
		#largest grains
		#carfile = 'opacity_carbon_1.0_amax_1.2_h2o_6e-05_alpha_-1.1.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.8_h2o_6e-05_alpha_-1.1.fits'
		#frac = 0.22

		caropac = fits.getdata(self.path+carfile,1)
		silopac = fits.getdata(self.path+silfile,1)
		cardist = fits.getdata(self.path+carfile,2)
		sildist = fits.getdata(self.path+silfile,2)
		caropac_hdr = fits.getheader(self.path+carfile,1)
		silopac_hdr = fits.getheader(self.path+silfile,1)


		best_kabs = (caropac['kabs'])*frac + (silopac['kabs'])*(1-frac)
		best_ksca = (caropac['ksca'])*frac + (silopac['ksca'])*(1-frac)
		best_gsca = (caropac['gsca'])*frac + (silopac['gsca'])*(1-frac)

		best_kext = (caropac['kabs']+caropac['ksca'])*frac + (silopac['kabs']+silopac['ksca'])*(1-frac)

		kp5 = ascii.read('Test_opacity_v5.asc',data_start=2)
		kp5_wv = kp5['col1']
		kp5_ext = kp5['col2']+kp5['col3']
		ssubs = np.argsort(kp5_wv)
		kp5_wv = kp5_wv[ssubs]
		kp5_ext = kp5_ext[ssubs]
		kp5_ext /= np.interp(2.2,kp5_wv,kp5_ext)

		nirspec_ext_g235m = fits.getdata('../../../../../OneDrive-JPL/obsprogs/jwst/1611/analysis/spectra/5046_opac.fits',1)
		nirspec_ext_g395h = fits.getdata('../../../../../OneDrive-JPL/obsprogs/jwst/1611/analysis/spectra/5046_opac.fits',2)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(cardist['grain_radius'],frac*cardist['ngrain']*cardist['grain_radius']**4,label='Carbon grains',color='orange')
		ax.plot(sildist['grain_radius'],sildist['ngrain']*sildist['grain_radius']**4,label='Silicate grains',color='purple')
		ax.legend()
		ax.set_xscale('log')
		ax.set_yscale('log')
		
		plt.show()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(caropac['wavelength'],best_kext/np.interp(2.2,caropac['wavelength'],best_kext),label='KP 6.0 new fit',color='purple',lw=2)
		ax.plot(kp5_wv,kp5_ext,label='KP 5.0',color='orange',lw=2)
		ax.plot(nirspec_ext_g235m['wavelength'],nirspec_ext_g235m['opac'])
		ax.plot(nirspec_ext_g395h['wavelength'],nirspec_ext_g395h['opac'])

		standard_ir_wl = np.linspace(0.9,5.,100)
		ax.plot(standard_ir_wl,standard_ir_wl**(-1.7)/(2.2**(-1.7)))

		ax.errorbar([1.65,2.2,3.6,4.5,5.8,8,24],[1.6,1.,0.64,0.53,0.46,0.45,0.34],yerr=[0,0,0.03,0.03,0.03,0.03,0.13],label='Chapman et al. 2009, A$_k$>2 mag',
			         ls=' ',mfc='cyan',ecolor='black',capsize=3,mec='black',ms=7,marker='o')


		ax.set_xlabel(r'Wavelength [$\mu$m]')
		ax.set_ylabel(r'$A_\lambda/A_K$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
		ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
		ax.set_xlim((0.1,100))
		ax.set_ylim((0.05,20))
		ax.legend()
		plt.show()

		c1 = fits.Column(name='wavelength', array=caropac['wavelength'], format='F')
		c2 = fits.Column(name='kabs', array=best_kabs, format='F')
		c3 = fits.Column(name='ksca', array=best_ksca, format='F')
		c4 = fits.Column(name='gsca', array=best_gsca, format='F')

		t = fits.BinTableHDU.from_columns([c1, c2, c3, c4])

		t.header['KUNIT'] = 'cm2/g'
		t.header['WAVEUNIT'] = 'micron'
		t.header['CARAMAX'] = caropac_hdr['AMAX']
		t.header['CARAMIN'] = caropac_hdr['AMIN']
		t.header['CARALPHA'] = caropac_hdr['ALPHA']
		t.header['SILAMAX'] = silopac_hdr['AMAX']
		t.header['SILAMIN'] = silopac_hdr['AMIN']
		t.header['SILALPHA'] = silopac_hdr['ALPHA']
		t.header['CARFRAC'] = frac
		t.header['CARH2O'] = caropac_hdr['H2OABUN']
		t.header['CARCO2'] = caropac_hdr['CO2ABUN']
		t.header['CARCO'] = caropac_hdr['COABUN']
		t.header['CARCH3OH'] = caropac_hdr['HIERARCH CH3OHABUN']
		t.header['SILH2O'] = silopac_hdr['H2OABUN']
		t.header['SILCO2'] = silopac_hdr['CO2ABUN']
		t.header['SILCO'] = silopac_hdr['COABUN']
		t.header['SILCH3OH'] = caropac_hdr['HIERARCH CH3OHABUN']
		t.header['DATE'] = date.today().strftime("%m/%d/%y")
		t.header['COMMENT'] = 'Processed by the JODY tool'
		t.header['COMMENT'] = 'Version '+str(__version__)
		t.header['COMMENT'] = 'Written by Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)'
		
		t.writeto(outspec,overwrite=True)


colors = Colors('../grid_v2.0/',regen_colors=False)
