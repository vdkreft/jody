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

		colors     = np.array([1.58,-0.633,-0.88,-0.88,-0.926,-0.78,1.9e-3,4e-4]) #Chapman Beta=1.6 + 450 and 850 micron constraints from Shirley et al. 2011
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

	def plot_fit(self,row,constraints, outspec='KP_V8.0_testcandidate.fits'):

		#caropac = fits.getdata(self.path+row['carfile'],1)
		#silopac = fits.getdata(self.path+row['silfile'],1)


		frac = row['frac']

		#kp5 (from opacity_bestfit.asc)
		carfile = 'opacity_carbon_1.0_amax_7.0_h2o_5e-05_alpha_-1.75.fits'
		silfile = 'opacity_carbon_0.0_amax_0.65_h2o_8.5e-05_alpha_-2.0.fits'
		frac = 0.35

		#kp5 total grain normalization
		#carfile = 'opacity_carbon_1.0_amax_8.0_h2o_7.5e-05_alpha_-1.0.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.55_h2o_0.0001_alpha_-1.5.fits'
		#frac = 0.36
		
		#kp5 core distribution
		#carfile = 'opacity_carbon_1.0_amax_8.0_h2o_0.00015_alpha_-1.0.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.6_h2o_0.00015_alpha_-2.0.fits'
		#frac = 0.32
		
		#kp5 total grain
		#carfile = 'opacity_carbon_1.0_amax_6.0_h2o_7.5e-05_alpha_-1.0.fits'
		#silfile = 'opacity_carbon_0.0_amax_0.6_h2o_7.5e-05_alpha_-2.5.fits'
		#frac = 0.28

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
		best_ice2dust = caropac_hdr['ICE2DUST']*frac + silopac_hdr['ICE2DUST']*(1-frac)

		best_kext = (caropac['kabs']+caropac['ksca'])*frac + (silopac['kabs']+silopac['ksca'])*(1-frac)

		#kp5 = ascii.read('kp5.asc',data_start=2)
		kp5 = ascii.read('opacity_bestfit.asc',data_start=2)
		kp5_wv = kp5['col1']
		kp5_ext = kp5['col2']+kp5['col3']
		ssubs = np.argsort(kp5_wv)
		kp5_wv = kp5_wv[ssubs]
		kp5_ext = kp5_ext[ssubs]
		kp5_ext /= np.interp(2.2,kp5_wv,kp5_ext)

		nirspec_ext_g235m = fits.getdata('../../../../../OneDrive-JPL/obsprogs/jwst/1611/analysis/spectra/5046_opac.fits',1)
		nirspec_ext_g395h = fits.getdata('../../../../../OneDrive-JPL/obsprogs/jwst/1611/analysis/spectra/5046_opac.fits',2)

		fig = plt.figure(figsize=(5,4))
		ax = fig.add_subplot(111)
		ax.plot(cardist['grain_radius'],frac*cardist['ngrain']*cardist['grain_radius']**4 * 1e-4**3 *1e29,label='Carbon grains',color='orange',lw=3)
		ax.plot(sildist['grain_radius'],(1-frac)*sildist['ngrain']*sildist['grain_radius']**4 * 1e-4**3 *1e29,label='Silicate grains',color='purple',lw=3)
		ax.set_xlabel('$a\ [\mu m]$')
		ax.set_ylabel('$10^{29} n_H^{-1} a^4 dn_{gr}/da\ [cm^3]$')
		
		amrn = np.logspace(-3,-1,100)
		mrn = 400*amrn**(-3.5) 
		ax.plot(amrn,mrn*amrn**4,label='MRN distribution',lw=3,alpha=0.5,color='grey',linestyle='--') #units in cm^3
		ax.set_ylim((1e-5,300))
		ax.legend()
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.annotate('(B)', (30,100),weight='bold',fontsize=10)

		plt.tight_layout()
		fig.savefig('fdist.pdf')
		plt.show()

		fig = plt.figure(figsize=(5,4))
		ax = fig.add_subplot(111)
		ax.plot(caropac['wavelength'],best_kext/np.interp(2.2,caropac['wavelength'],best_kext),label='KP5 benchmark (OpTool)',color='purple',lw=2)
		ax.plot(kp5_wv,kp5_ext,label='Original KP5',color='orange',lw=2)
		#ax.plot(nirspec_ext_g235m['wavelength'],nirspec_ext_g235m['opac'])
		#ax.plot(nirspec_ext_g395h['wavelength'],nirspec_ext_g395h['opac'])

		standard_ir_wl = np.linspace(0.9,5.,100)
		#ax.plot(standard_ir_wl,standard_ir_wl**(-1.85)/(2.2**(-1.85)))

		bands = np.array([1.65,2.2,3.6,4.5,5.8,8,24])
		chap09 = np.array([1.6,1.,0.60,0.46,0.44,0.43,0.61])
		chap09err = np.array([0,0,0.03,0.03,0.03,0.03,0.13])
		# scale to actual jhk slope in kp5 (as opposed to WD01)
		newslope = 1.6#1.75
		chap09[1:] = (chap09[1:]-1)*(newslope-1)/(1.6-1)+1      #colors[1:*]*(colors[0]-1.)+1.
		chap09[0] = newslope

		#ax.errorbar([1.65,2.2,3.6,4.5,5.8,8,24,450,850],[1.6,1.,0.64,0.53,0.46,0.45,0.34,1.9e-3,4e-4],yerr=[0,0,0.03,0.03,0.03,0.03,0.13,0.2e-3,0.4e-4],label='Chapman et al. 2009, A$_k$>2 mag',
		#	         ls=' ',mfc='cyan',ecolor='black',capsize=3,mec='black',ms=7,marker='o')
		#ax.errorbar([1.65,2.2,3.6,4.5,5.8,8,24,450,850],[1.6,1.,0.60,0.46,0.44,0.43,0.61,1.9e-3,4e-4],yerr=[0,0,0.03,0.03,0.03,0.03,0.13,0.2e-3,0.4e-4],label='Chapman et al. 2009, A$_k$>2 mag',
		#	         ls=' ',mfc='cyan',ecolor='black',capsize=3,mec='black',ms=7,marker='o')
		#ax.errorbar([1.65,2.2,3.6,4.5,5.8,8,24,450,850],[1.6,1.,0.49,0.35,0.35,0.35,0.75,1.9e-3,4e-4],yerr=[0,0,0.1,0.1,0.1,0.1,0.14,0.2e-3,0.4e-4],label='Chapman et al. 2009, 0.5<A$_k$<1.0 mag',
		#	         ls=' ',mfc='cyan',ecolor='black',capsize=3,mec='black',ms=7,marker='o')
		ax.errorbar(bands,chap09,yerr=chap09err,label='Chapman et al. 2009, A$_k$>2 mag',
			        ls=' ',mfc='cyan',ecolor='black',capsize=3,mec='black',ms=7,marker='o')
		ax.errorbar([450,850],[1.9e-3,4e-4],yerr=[0.2e-3,0.4e-4],label='Shirley et al. 2011',
			        ls=' ',mfc='grey',ecolor='black',capsize=3,mec='black',ms=7,marker='o')

		

		ax.set_xlabel(r'Wavelength [$\mu$m]')
		ax.set_ylabel(r'$A_\lambda/A_K$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
		ax.set_xlim((0.11,2000))
		ax.set_ylim((0.0001,30))
		ax.legend(loc=3)
		ax.annotate('(A)', (1000,12),weight='bold',fontsize=10)

		plt.tight_layout()
		fig.savefig('opac_fit.pdf')
		plt.show()

		c1 = fits.Column(name='wavelength', array=caropac['wavelength'], format='F')
		c2 = fits.Column(name='kabs', array=best_kabs, format='F')
		c3 = fits.Column(name='ksca', array=best_ksca, format='F')
		c4 = fits.Column(name='gsca', array=best_gsca, format='F')

		t = fits.BinTableHDU.from_columns([c1, c2, c3, c4])

		t.header['KUNIT'] = 'cm2/g'
		t.header['WAVEUNIT'] = 'micron'
		t.header['CARAMAX'] = (caropac_hdr['AMAX'],'Turnover carbon grain size (a_c,g) [micron]')
		t.header['CARAMIN'] = (caropac_hdr['AMIN'],'Minimum carbon grain size [micron]')
		t.header['CARALPHA'] = (caropac_hdr['ALPHA'],'Carbon grain size slope (alpha_g)')
		t.header['CARATG'] = (caropac_hdr['ATS'],'a_t curvature parameter for carbon grains')
		t.header['CARBETA'] = (caropac_hdr['BETA_S'],'beta curvature parameter for carbon grains')

		t.header['CARH2O'] = (caropac_hdr['H2OABUN'],'H2O abundance on carbon grains [per H]')
		t.header['CICEAMIN'] = (caropac_hdr['ICEAMIN'],'Minimum carbon grain size with ice')

		t.header['SILAMAX'] = (silopac_hdr['AMAX'],'Turnover silicate grain size (A_t,s) [micron]')
		t.header['SILAMIN'] = (silopac_hdr['AMIN'],'Minimum silicate grain size [micron]')
		t.header['SILALPHA'] = (silopac_hdr['ALPHA'],'Silicate grain size slope (alpha_t)')
		t.header['SILATS'] = (silopac_hdr['ATS'],'a_t curvature parameter for silicate grains')
		t.header['SILBETA'] = (silopac_hdr['BETA_S'],'beta curvature parameter for silicate grains')
		#t.header['CARCO2'] = (caropac_hdr['CO2ABUN'],'CO2 abundance on carbon grains')
		#t.header['CARCO'] = (caropac_hdr['COABUN'],'CO abundance on ')
		#t.header['CARCH3OH'] = (caropac_hdr['HIERARCH CH3OHABUN'],'')
		
		t.header['SILH2O'] = (silopac_hdr['H2OABUN'],'H2O abundance on silicate grains [per H]')
		t.header['SICEAMIN'] = (silopac_hdr['ICEAMIN'],'Minimum silicate grain size with ice [micron]')
		#t.header['SILCO2'] = (silopac_hdr['CO2ABUN'],'')
		#t.header['SILCO'] = (silopac_hdr['COABUN'],'')
		#t.header['SILCH3OH'] = (caropac_hdr['HIERARCH CH3OHABUN'],'')

		t.header['CARFRAC'] = (frac,'Refractory mass fraction of carbon grains')
		t.header['BC'] = (caropac_hdr['BC'],'Percentage of carbon mass in VSGs')
		t.header['MAXAVSG'] = (caropac_hdr['MAXAVSG'],'Maximum grain size of VSGs [micron]')

		t.header['DATE'] = date.today().strftime("%m/%d/%y")
		t.header['COMMENT'] = 'Processed by the JODY tool'
		t.header['COMMENT'] = 'Version '+str(__version__)
		t.header['COMMENT'] = 'Written by Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)'
		
		t.writeto(outspec,overwrite=True)

		g2d = 100/(1+best_ice2dust)
		mh = 1.66e-24
		cabun = 1/(g2d*0.714) * frac * (1/(1+caropac_hdr['ICE2DUST'])) / 12.01
		print('Carbon abundance: ', cabun)
		print('Ice to dust mass ratio: ', best_ice2dust)
		kext_J = np.interp(1.25,caropac['wavelength'],best_kext)
		print('K_ext J', kext_J)
		print('AJ/NH', 1.086*kext_J/(g2d*0.714/mh))



colors = Colors('../grid_kp6/',regen_colors=False)
