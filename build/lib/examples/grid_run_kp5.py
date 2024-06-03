import matplotlib.pylab as plt
import numpy as np
from jody.core import *

path = 'grid/'

wavemin = 0.1
wavemax = 2000.
nwv = 10000

icepars   = [{'species':'h2o',  'nfrac':1.0,'weight':18,'rho':0.92,  'oc':'ICE_COMB2.lnk'},
             {'species':'co2',  'nfrac':0.10,'weight':44,'rho':1.2,   'oc':'co2-a'},
             {'species':'co',   'nfrac':0.0,'weight':28,'rho':0.81,  'oc':'co-a'},
             {'species':'ch3oh','nfrac':0.0,'weight':32,'rho':0.779, 'oc':'ch3oh-a'}]

carbonabuns = [1.0]
grain_min   = 0.0006
grain_maxs  = [6.0,7.0,8.0,9.0]
waterabuns  = [0]
ice_amin    = 0.03
Nsize       = 50
alphas      = [-1.0,-1.5,-2.0,-2.5];

for carbon_abun in carbonabuns:
	for grain_max in grain_maxs:
		for h2o_abun in waterabuns:
			for alpha in alphas:
				opac = Opacity(carbon_abun=carbon_abun,amax=grain_max,h2o_abun=h2o_abun,icepars=icepars,ice_amin=ice_amin,alpha_s=alpha,gridmax=10*grain_max,bc=0.1,beta_s=2.0,
					           wavemin=wavemin, wavemax=wavemax, nwv=nwv, kp5_style=True)
				filename = 'opacity_'+'carbon_'+str(carbon_abun)+'_amax_'+str(grain_max)+'_h2o_'+str(h2o_abun)+'_alpha_'+str(alpha)+'.fits'
				opac.write_opac(path+filename)

carbonabuns = [0.0]
grain_min   = 0.0006
grain_maxs  = [0.2, 0.4, 0.6, 0.8]
h2oabuns    = [0]
ice_amin    = 0.03
Nsize       = 50
alphas      = [-1.5,-2.0,-2.5,-3.0];

for carbon_abun in carbonabuns:
	for grain_max in grain_maxs:
		for h2o_abun in h2oabuns:
			for alpha in alphas:
				opac = Opacity(carbon_abun=carbon_abun,amax=grain_max,h2o_abun=h2o_abun,icepars=icepars,ice_amin=ice_amin,alpha_s=alpha,gridmax=10*grain_max,bc=0.0,beta_s=-0.109,
					           wavemin=wavemin, wavemax=wavemax, nwv=nwv, kp5_style=True)
				filename = 'opacity_'+'carbon_'+str(carbon_abun)+'_amax_'+str(grain_max)+'_h2o_'+str(h2o_abun)+'_alpha_'+str(alpha)+'.fits'
				opac.write_opac(path+filename)
				
