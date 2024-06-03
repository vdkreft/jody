import os
import matplotlib.pylab as plt
import math
import numpy as np
from scipy.integrate import simpson
from astropy.io import fits
import optool

__version__ = 2.0

class Opacity():
	"""docstring for ClassName"""
	def __init__(self,amin=0.0006,amax=1.,nsize=50, gridmax=10, wavemax=2000, kp5_style=True,
		         wavemin=0.1,nwv=1000,carbon_abun=5e-5,h2o_abun=1e-4,icepars=None,ice_amin=0.03,pure_carbon=False,
		         ats=0.03, beta_s=-0.109, alpha_s=-1.0, bc=0.0, maxa_vsg=0.01):

		self.local_path = os.path.join(os.path.dirname(__file__), 'ocs')

		# Standard parameters
		self.g2d = 100
		self.mh = 1.66e-24
		# using KP5 densities rather than optool densities
		self.d_sil = 3.3 * 1e-12 #g/micron^3. 
		self.d_car = 1.988 * 1e-12
		self.h2o_weight = 18
		# Optool wants to know the *mass* fractions of every component. 
		# So we need to calculate that from the input number abundances. 
		# Optool will renormalize, but we require the total water ice mass to be ==1.0, by convention.
		self.d_ice = 0
		icemass = 0
		for ice in icepars:
			ice['massfrac'] = ice['nfrac'] * ice['weight'] / self.h2o_weight
			ice['abun'] = ice['nfrac'] * h2o_abun
			self.d_ice += ice['massfrac']*ice['rho']
			icemass += ice['massfrac']

		#Averaged ice density
		self.d_ice /= icemass*1e12
		
		if carbon_abun==1:
			pure_carbon=True

		# Store global arguments
		self.carbon_abun = carbon_abun
		self.h2o_abun = h2o_abun
		self.icepars = icepars
		self.ice_amin = ice_amin
		self.acg = amax
		self.amin = amin
		self.ats = ats
		self.beta_s = beta_s
		self.alpha_s = alpha_s
		self.bc = bc
		self.maxa_vsg = maxa_vsg

		# Wavelength grid
		self.nwv   = nwv    #Size of wavelength grid

		# Grain radius grid, spaced logarithmically
		lnDCores    = np.arange(nsize)/(nsize-1)*(np.log10(gridmax)-np.log10(amin))+np.log10(amin)
		DCores      = 10**lnDCores

		# Optool grain size inputs are in radius
		self.RCores = DCores/2

		# We can have pure carbon grains (with a different size distribution). Or mixed silicates. 
		#In practice, for the KP models the silicate grains are also pure (i.e., carbon_frac==0.0).
		if pure_carbon:
			carbon_frac = 1.0
			self.dens_core = self.d_car
		else:
			refractory_mass = self.refractory_mass()
			carbon_frac = (carbon_abun * 12.011) / refractory_mass
			self.dens_core = self.d_sil*(1.-carbon_frac)+self.d_car*carbon_frac

		#---------------------------------------

		# Calculate the size distribution using the Weingartner and Draine 2001 parameterization
		self.fdist = self.make_dist(self.RCores,ats=self.ats,beta_s=self.beta_s,bc=self.bc,alpha_s=alpha_s,acg=self.acg)

		# Given the size distribution, what *constant* ice mantle thickness gives us the input water abundance (relative to H), 
		# assuming that all of the Si, Mg, and other refractories are represented in the silicate grains. 
		self.da,icemass = self.calc_mantle_thick(self.RCores)
		self.das = self.da*np.ones(nsize)

		# Ok, we actually don't have a constant ice thickness. The smallest grains are bare due to transient heating. 
		# This is taken into account in the calculation of "da" in the step above
		self.das[self.RCores<ice_amin] = 0. #can't be numerically 0

		#The total (core + mantle) radius
		self.rgrains = self.RCores + self.das

		# We have to calculate the relative core and mantle masses as input to Optool
		core_vol = (4./3.)*np.pi*self.RCores**3
		mantle_vol = (4./3.)*np.pi*((self.RCores+self.das)**3 - self.RCores**3)
		core_mass = core_vol * self.dens_core
		mantle_mass = mantle_vol * self.d_ice
		grain_mass = core_mass + mantle_mass
		grain_vol = core_vol+mantle_vol
		core_frac = core_mass/grain_mass
		mantle_frac = mantle_mass/grain_mass
		self.dens_grain = grain_mass/grain_vol
			
		ksca_grid = np.zeros((nwv,nsize))
		kabs_grid = np.zeros((nwv,nsize))
		gsca_grid = np.zeros((nwv,nsize))
		mas = np.zeros(nsize)

		for ii,a in enumerate(self.rgrains):
			if pure_carbon:
				if self.RCores[ii]<maxa_vsg:
					core_oc = ' c-gra ' #Draine graphite for VSGs
				else:
					core_oc = ' '+self.local_path+'/zubko_car.lnk ' #Otherwise Zubko amorphous carbon
					#carbon_oc = ' c-gra ' #Otherwise Zubko amorphous carbon					
			else:
				if self.RCores[ii]<maxa_vsg:
					core_oc = ' astrosil ' 
				else:
					core_oc = ' astrosil '
			
			cmd = 'optool -fmax 0 '+ core_oc + str(core_frac[ii]) 


			# Add mantle component to the command
			if mantle_frac[ii]>0:
				cmd += ' -m '
				for species in self.icepars:
					cmd += species['oc'] + ' ' + str(mantle_frac[ii]*species['massfrac']/icemass) + ' '

			# Add size parameters
			cmd += ' -a ' + str(a)

			# Add wavelength parameters
			cmd += ' -l ' + str(wavemin) + ' ' + str(wavemax) + ' ' + str(nwv)

			p = optool.particle(cmd,silent=True)

			if kp5_style:
				# KP5 calculated the cross section from efficiency using the core radius and density
				#
				#Converting from Q (dimensionless quantity measured relative to the
				#geometric cross section of a grain) to cross section per gram [cm^2/g]
				#                    Q*(pi*(dia/2)^2) 
				#cross =         -----------------------      =   Q*(3/2)/dens_av/dia
				#                (4*pi/3)*(dia/2)^3*dens_av


				ksca_grid[:,ii] = p.ksca.flatten() * (self.dens_grain[ii]/self.dens_core) * (a/self.RCores[ii])
				kabs_grid[:,ii] = p.kabs.flatten() * (self.dens_grain[ii]/self.dens_core) * (a/self.RCores[ii])
			else:
				ksca_grid[:,ii] = p.ksca.flatten() 
				kabs_grid[:,ii] = p.kabs.flatten()				

			gsca_grid[:,ii] = p.gsca.flatten()
			
			waves = p.lam
		
		int_ksca = np.zeros(nwv)
		int_kabs = np.zeros(nwv)
		int_gsca = np.zeros(nwv)

		if kp5_style:
			for ii,wave in enumerate(waves):
				int_ksca[ii] = simpson(ksca_grid[ii,:]*self.fdist * self.dens_core * 4*np.pi/3*(self.RCores)**3 * self.RCores,x=np.log(self.RCores))
				int_kabs[ii] = simpson(kabs_grid[ii,:]*self.fdist * self.dens_core * 4*np.pi/3*(self.RCores)**3 * self.RCores,x=np.log(self.RCores))
				int_gsca[ii] = simpson(gsca_grid[ii,:]*ksca_grid[ii,:]*self.fdist * self.dens_core * 4*np.pi/3*(self.RCores)**3 * self.RCores,x=np.log(self.RCores))

			mas = self.dens_core*self.fdist*4*np.pi/3*(self.RCores)**3 
			ma_tot = simpson(mas*self.RCores,x=np.log(self.RCores))

			mas_core = self.dens_core*self.fdist*4*np.pi/3*(self.RCores)**3 
			ma_core_tot = simpson(mas_core*self.RCores,x=np.log(self.RCores))
			self.ma_av = 1e-12*ma_tot/np.trapz(np.log(self.RCores),self.fdist*self.RCores) #Average mass per grain [g]
		else:
			for ii,wave in enumerate(waves):
				int_ksca[ii] = simpson(ksca_grid[ii,:]*self.fdist * self.dens_grain * 4*np.pi/3*(self.rgrains)**3 * self.rgrains,x=np.log(self.rgrains))
				int_kabs[ii] = simpson(kabs_grid[ii,:]*self.fdist * self.dens_grain * 4*np.pi/3*(self.rgrains)**3 * self.rgrains,x=np.log(self.rgrains))
				int_gsca[ii] = simpson(gsca_grid[ii,:]*ksca_grid[ii,:]*self.fdist * self.dens_grain * 4*np.pi/3*(self.rgrains)**3 * self.rgrains,x=np.log(self.rgrains))

			mas = self.dens_grain*self.fdist*4*np.pi/3*(self.rgrains)**3 
			ma_tot = simpson(mas*self.rgrains,x=np.log(self.rgrains))

			mas_core = self.dens_core*self.fdist*4*np.pi/3*(self.RCores)**3 
			ma_core_tot = simpson(mas_core*self.RCores,x=np.log(self.RCores))
			self.ma_av = 1e-12*ma_tot/np.trapz(np.log(self.rgrains),self.fdist*self.rgrains) #Average mass per grain [g]

		self.waves = waves
		self.gfac = int_gsca/int_ksca  #Weighing g-factor with scattering cross section
		self.ksca = int_ksca/ma_tot
		self.kabs = int_kabs/ma_tot
		
		self.fdist = self.fdist / (ma_tot * self.g2d * 0.714 / self.mh)

	def make_dist(self, a,alpha_s=-1.13,ats=0.211,beta_s=-0.109,acg=0.1,bc=0.0,rho=2.24):

		C = 1.
		#Small carbon grains from Li & Draine

		mc = 12*1.66e-24 #mass of a carbon
		a01 = 0.00035
		a02 = 0.00300
		bc1 = bc*45e-6/60e-6
		bc2 = bc*15e-6/60e-6

		sigma = 0.4
		lsubs = np.where(a < ats)
		hsubs = np.where(a > ats)

		if beta_s >= 0:
			ff = 1+beta_s*a/ats
		else:
			ff = 1/(1-beta_s*a/ats)

		fdist = C/a * (a/ats)**alpha_s*ff
		fdist[hsubs] = fdist[hsubs]*np.exp(-((a[hsubs]-ats)/acg)**3)

		b1 = 3/(2*np.pi)**(3/2)*np.exp(-4.5*sigma**2)/(a01**3*rho*sigma) * \
		     mc*bc1/(1+math.erf(3*sigma/np.sqrt(2)+np.log(a01/0.00035)/(sigma*np.sqrt(2))))

		b2 = 3/(2*np.pi)**(3/2)*np.exp(-4.5*sigma**2.)/(a02**3*rho*sigma) * \
		     mc*bc2/(1+math.erf(3*sigma/np.sqrt(2)+np.log(a02/0.00035)/(sigma*np.sqrt(2))))

		d  = (b1/a) * np.exp(-0.5*(np.log(a/a01)/sigma)**2) + \
		     (b2/a) * np.exp(-0.5*(np.log(a/a02)/sigma)**2)

		#determining total volume of populations
		dv = simpson(d*a**3*a, x=np.log(a))
		fv = simpson(fdist*a**3*a, x=np.log(a))

		if bc != 0:
			fdist = (1-bc)*fdist/fv+bc*d/dv
		else:
			fdist = fdist/fv

		return fdist

	def calc_mantle_thick(self, a):
		'''
		Method for calculating a single constant ice mantle thickness, given a requested ice abundance. 
		This is done numerically by integrating the core volume over the size distribution, as well as the ice volume for a range of 
		possible mantle thicknesses. The mantle thickness that best matches the requested abundance is then selected. 
		'''
		nas   = 1000 #Number of ice volumes to use when calculating the mantle thickness
		lndas = np.arange(nas)/(nas-1)*(np.log10(10.)-np.log10(0.0001))+np.log10(0.0001) #micron
		das   = 10**lndas
		# Integrating the total volume of the refractory dust size distribution in log-space
		vdust = simpson((4/3)*np.pi*(a**3)*self.fdist*a, x=np.log(a))  
		
		h2otogas_number = np.zeros(nas)
		vices = np.zeros(nas)
		for i in np.arange(nas):
			if self.ice_amin > 0:
				icesubs = np.where(a >= self.ice_amin)
				baresubs = np.where(a < self.ice_amin)
				# Integrating the total ice mantle volume in log-space (including all ice species)
				vtot = simpson((4./3.)*np.pi*(a[icesubs]+das[i])**3*self.fdist[icesubs]*a[icesubs],x=np.log(a[icesubs])) 
				vtot += simpson((4./3.)*np.pi*(a[baresubs])**3*self.fdist[baresubs]*a[baresubs],x=np.log(a[baresubs])) 
			else:
				vtot = simpson((4./3.)*np.pi*(a+das[i])**3*self.fdist*a,x=np.log(a)) 

			vice = vtot - vdust
			vices[i] = vice
			
			# Calculate fraction of total ice mass that is water
			icemass = 0
			for species in self.icepars:
				icemass += species['massfrac']
			h2o_massfrac = 1./icemass

			# mass of water ice per unit gas mass  
			h2otodust = (self.d_ice/self.dens_core) * (vice/vdust) * h2o_massfrac
			h2otogas = h2otodust / (self.g2d * 0.714) # H2O mass per H mass, assuming 71.4% of the gas mass is H

			# convert the ice mass per unit gas mass to #H2O molecules per H atom, calculated as F_M * (mu_H / mu_H2O): 
			h2otogas_number[i] = h2otogas * (1 / 18.015) #number of H2O per H atom

		goodsub = np.argmin(np.abs(h2otogas_number-self.h2o_abun))
		
		if self.h2o_abun != 0.:
			da = das[goodsub]
		else:
			da = 0

		print('Actual to requested ice abundance: ', h2otogas_number[goodsub]/(self.h2o_abun),goodsub)
		print('V ice and V dust: ',vices[goodsub], vdust)
		print('With mantle thickness: ', da, ' Micron')

		self.mice  = vices[goodsub]*self.d_ice
		self.mdust = vdust*self.dens_core

		return da,icemass

	def refractory_mass(self):
		abuns = [{'element':'mg','abun':7.55,'mass':24.305},
				 {'element':'al','abun':6.46,'mass':26.982},
				 {'element':'si','abun':7.54,'mass':28.086},
				 {'element':'p' ,'abun':5.46,'mass':30.974},
				 {'element':'s' ,'abun':7.19,'mass':32.065},
				 {'element':'cl','abun':5.26,'mass':35.453},
				 {'element':'k' ,'abun':5.11,'mass':39.098},
				 {'element':'ca','abun':6.34,'mass':40.078},
				 {'element':'cr','abun':5.65,'mass':51.996},
				 {'element':'mn','abun':5.50,'mass':54.938},
				 {'element':'fe','abun':7.46,'mass':55.845},
				 {'element':'ni','abun':6.22,'mass':58.693}]

		mass = 0.0
		for element in abuns:
			mass += 10**(element['abun']-12) * element['mass']

		return mass

	def plot_opac(self):

		plt.plot(self.waves,self.ksca, label='ksca')
		plt.plot(self.waves,self.kabs, label='kabs')
		plt.plot(self.waves,self.kabs+self.ksca, label='kext')

		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel(r'wavelength [$\mu$m]')
		plt.ylabel(r'$\kappa_{ext}$ [cm$^2$/g]')
		
		plt.legend()
		plt.show()

	def plot_dist(self):
		plt.plot(self.RCores,self.fdist*self.RCores**4)
		plt.ylim((1e-5,100))
		plt.xscale('log')
		plt.yscale('log')
		plt.show()

	def write_opac(self,outname):
		c1 = fits.Column(name='wavelength', array=self.waves, format='F')
		c2 = fits.Column(name='kabs', array=self.kabs, format='F')
		c3 = fits.Column(name='ksca', array=self.ksca, format='F')
		c4 = fits.Column(name='gsca', array=self.gfac, format='F')

		t = fits.BinTableHDU.from_columns([c1, c2, c3, c4])

		dc1 = fits.Column(name='core_radius', array=self.RCores, format='F')
		dc2 = fits.Column(name='grain_radius', array=self.rgrains, format='F')
		dc3 = fits.Column(name='ngrain', array=self.fdist, format='F')
		tdist = fits.BinTableHDU.from_columns([dc1,dc2,dc3])

		t.header['CARBON'] = self.carbon_abun

		for species in self.icepars:
			t.header[species['species'].upper()+'ABUN'] = species['abun']
		t.header['AMAX'] = self.acg
		t.header['AMIN'] = self.amin
		t.header['ALPHA'] = self.alpha_s
		t.header['ICEAMIN'] = self.ice_amin
		t.header['ATS'] = self.ats
		t.header['BETA_S'] = self.beta_s
		t.header['BC'] = self.bc
		t.header['MAXAVSG'] = self.maxa_vsg

		t.header['ICE2DUST'] = self.mice/self.mdust
		t.header['COMMENT'] = 'Processed by the JODY tool'
		t.header['COMMENT'] = 'Version '+str(__version__)
		t.header['COMMENT'] = 'Written by Klaus Pontoppidan (klaus.m.pontoppidan@jpl.nasa.gov)'

		hdu = fits.PrimaryHDU()
		tlist = fits.HDUList([hdu, t,tdist])

		tlist.writeto(outname,overwrite=True)




