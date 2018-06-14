"""
An example how to fit the size of a source from a 3D n_pred / n_obs cube.

Using CTA IRFs.

"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
import numpy as np
import yaml
import pyfits
from astropy.coordinates import SkyCoord, Angle
from configuration import make_ref_cube
from astropy.io import fits
from sherpa.astro.ui import *
from gammapy.cube import SkyCube
from astropy.units import Quantity
from time import time
from regions import CircleSkyRegion
from astropy.table import Table

def read_config(filename):
    '''Read configuration from yaml file.
        
    Parameters
    ----------
    filename : yaml file
        Configuration file.
    
    Returns
    -------
    
    config : `dict`
        Configuration dictionary.
        
    '''
    with open(filename) as fh:
        config = yaml.load(fh)

    config['model']['prefactor'] = float(config['model']['prefactor'])

    return config

def do_fit():

    fit()
    image_resid()
    for mod in list_models:
        mtype = get_model_component(mod)
        if mtype:
            
            coord = get_pos(mtype.xpos.val,mtype.ypos.val)[0]
            print "Fitted position of ", mod ," : ", coord


def get_pos(xpix,ypix):
    pixcrd = np.array([[xpix,ypix]], np.float_)
    return im_proj.wcs_pix2world(pixcrd,1)



def do_conf():
    conf()
    res_conf = get_conf_results()
    for im in range(len(res_conf.parnames)):
        cparam = res_conf.parnames[im]
        [model,param] = cparam.split('.')
        if param == "xpos":
            smin = res_conf.parmins[im]*abs(im_proj.wcs.cdelt[0])
            smax = res_conf.parmaxes[im]*abs(im_proj.wcs.cdelt[0])
            print "Fitted X position error of ", model ," : -", smin, " +", smax
        elif param == "ypos":
            smin = res_conf.parmins[im]*abs(im_proj.wcs.cdelt[1])
            smax = res_conf.parmaxes[im]*abs(im_proj.wcs.cdelt[1])
            print "Fitted Y position error of ", model ," : -", smin, " +", smax
        elif param == "size" or param == 'r1' or param == 'thick':
            smean = res_conf.parvals[im]*abs(im_proj.wcs.cdelt[0])
            smin = res_conf.parmins[im]*abs(im_proj.wcs.cdelt[0])
            smax = res_conf.parmaxes[im]*abs(im_proj.wcs.cdelt[0])
            print "Fitted ", param, " of ", model ," : ",smean,"+-", smin, " +", smax


def GaussianSource(pars,x,y):

    (sigma1, sigma2, sigma3, alpha, beta, ampl, size, xpos, ypos) = pars
    r2 = (x - xpos)**2 + (y - ypos)**2
    s1sq = sigma1*sigma1
    s2sq = sigma2*sigma2
    s3sq = sigma3*sigma3
    v1 = sigma1*sigma1+size*size
    v2 = sigma2*sigma2+size*size
    v3 = sigma3*sigma3+size*size
    s1 = s1sq*np.exp(-0.5*r2/v1)/(2*pi*v1)
    s2 = alpha*s2sq*np.exp(-0.5*r2/v2)/(2*pi*v2)
    s3 = beta*s3sq*np.exp(-0.5*r2/v3)/(2*pi*v3)
    return ampl*(s1+s2+s3)/(s1sq+alpha*s2sq+beta*s3sq)

def psf_fromfits(filename):
    '''Get PSF from fits file.
        
    Parameters
    ----------
    filename : Fits file
        IRFS fits file.
    
    Returns
    -------
    energy_lo : `Quantity`
        Low energy bounds.
    energy_hi : `Quantity`
        High energy bounds.
    theta : `Angle`
        Offset angle.
    sigmas : `list`
        PSF sigmas.
    norms : `list`
        PSF norms.
    
    '''

    hdulist = pyfits.open(filename)
    hdu = hdulist[2]
    energy_lo = Quantity(hdu.data['ENERG_LO'][0], 'TeV')
    energy_hi = Quantity(hdu.data['ENERG_HI'][0], 'TeV')
    theta = Angle(hdu.data['THETA_LO'][0], 'deg')
    
    # Get sigmas
    shape = (len(theta), len(energy_hi))
    sigmas = []
    for key in ['SIGMA_1', 'SIGMA_2', 'SIGMA_3']:
        sigma = hdu.data[key].reshape(shape).copy()
        sigmas.append(sigma)
    
    # Get amplitudes
    norms = []
    for key in ['SCALE', 'AMPL_2', 'AMPL_3']:
        norm = hdu.data[key].reshape(shape).copy()
        norms.append(norm)

    return [energy_lo, energy_hi, theta, sigmas, norms]


pi = 3.141592653589793

def PSFGauss(pars,x,y):
    (sigma1, ampl, xpos, ypos) = pars
    r2 = (x - xpos)**2 + (y - ypos)**2
    v1 = sigma1*sigma1
    return ampl * np.exp(-0.5*r2/v1)/(2*pi*v1)

def PSFtripleGauss(pars,x,y):
    (sigma1, sigma2, sigma3, alpha, beta, ampl, xpos, ypos) = pars
    r2 = (x - xpos)**2 + (y - ypos)**2
    s1sq = sigma1*sigma1
    s2sq = sigma2*sigma2
    s3sq = sigma3*sigma3
    v1 = sigma1*sigma1
    v2 = sigma2*sigma2
    v3 = sigma3*sigma3
    s1 = s1sq*np.exp(-0.5*r2/v1)/(2*pi*v1)
    s2 = alpha*s2sq*np.exp(-0.5*r2/v2)/(2*pi*v2)
    s3 = beta*s3sq*np.exp(-0.5*r2/v3)/(2*pi*v3)
    return ampl*(s1+s2+s3)/(s1sq+alpha*s2sq+beta*s3sq)

def main():
    
    # Read file to fit
    #filename = 'nexcess_cube.fits.gz'
    filename = 'non_cube_convolved.fits.gz'
    cube = SkyCube.read(filename)
    
    # Read configuration
    config = read_config('config.yaml')
    binsz = config['binning']['binsz']
    offset_fov = config['selection']['offset_fov']

    # Take PSF data
    irffile = 'irf_file.fits'
    psf_table = psf_fromfits(irffile)
    
    energarr = cube.energies('edges')
    sigmas = psf_table[3]
    norms = psf_table[4]
    
    hdu = pyfits.open(filename)
    
    im_sizex = hdu[0].header['NAXIS1']
    im_sizey = hdu[0].header['NAXIS2']
    
    cx = 0.5 * im_sizex
    cy = 0.5 * im_sizey
    
    # Check the significance
    filename_on = 'non_cube.fits.gz'
    
    cube_on = SkyCube.read(filename_on)
    
    filename_off = 'noff_cube.fits.gz'
    
    cube_off = SkyCube.read(filename_off)
    alpha_obs = 1.
    
    on_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    on_sizes = np.ones(len(cube.energies('center'))) * 120 * binsz * u.deg #0.167
    
    off_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    off_sizes = on_sizes * alpha_obs
    
    on_data = Table()
    off_data = Table()
    on_data['value'] = np.zeros(len(on_sizes))
    off_data['value'] = np.zeros(len(on_sizes))
    for idx in range(len(cube.energies('center'))):
        
        on_region = CircleSkyRegion(on_pos, on_sizes[idx])
        off_region = CircleSkyRegion(off_pos, off_sizes[idx])
        
        on_data['value'][idx] = cube_on.spectrum(on_region)['value'][idx]
        off_data['value'][idx] = cube_off.spectrum(off_region)['value'][idx]
        
        limasig = np.sqrt(2) * np.sqrt(on_data['value'][idx]*np.log(((1+alpha_obs)/alpha_obs)*on_data['value'][idx]/(on_data['value'][idx] + off_data['value'][idx]))+off_data['value'][idx]*np.log((1+alpha_obs)*off_data['value'][idx]/(on_data['value'][idx]+off_data['value'][idx])))
        
        print(limasig, 'Energy range: ', cube_on.energies('edges')[idx], ' - ', cube_on.energies('edges')[idx+1])

        #Fit only if data is enough
        #and on_data['value'][i] - off_data['value'][i] >= 0.01 * off_data['value'][i]
        if limasig >= 3 and on_data['value'][idx] - off_data['value'][idx] >= 7:
        
            # Make image cube from slice excess convolved cube
            cube_sum = np.zeros((cube.data.shape[1],cube.data.shape[2])) * u.ct
            cube_sum = np.add(cube_sum, cube.data[idx])
    
            cube_sum.value[np.isnan(cube_sum.value)]=0
            cube_sum.value[cube_sum.value < 0]=abs(cube_sum.value[cube_sum.value < 0])

            image_sum = SkyCube.empty_like(cube)
            image_sum.data = cube_sum
    
            image_sum.write('sum_image.fits.gz', overwrite=True)


            # Find nearest energy and theta value
            i = np.argmin(np.abs(energarr[idx].value - psf_table[0].value))      ######
            j = np.argmin(np.abs(offset_fov - psf_table[2].value))
    
            # Make PSF
            #psfname="mypsf"
            #load_user_model(PSFGauss,psfname)
            s1 = sigmas[0][j][i]/binsz
            s2 = sigmas[1][j][i]/binsz
            s3 = sigmas[2][j][i]/binsz
            print(sigmas[0][j][i],sigmas[1][j][i],sigmas[2][j][i])
            ampl = norms[0][j][i]
            ampl2 = norms[1][j][i]
            ampl3 = norms[2][j][i]
    
            t0 = time()
            
            #Morphological fitting
            load_image("sum_image.fits.gz")
            #image_data()

            #set_coord("physical")

            set_method("simplex")
            set_stat("cash")

            # Position and radius
            x0 = 125
            y0 = 125
            rad0 = 80.0

            image_getregion(coord="physical")
            'circle(x0,y0,rad0);'

            notice2d("circle(" + str(x0) + "," + str(y0) + "," + str(rad0) + ")")
    
            load_user_model(GaussianSource, "sph2d")
            add_user_pars( "sph2d", ["sigma1", "sigma2", "sigma3", "alpha", "beta", "ampl", "size", "xpos", "ypos"] )

            set_model(sph2d + const2d.bgnd)
            
            # Constant PSF
            #gpsf.fwhm = 4.2
            #gpsf.xpos = x0
            #gpsf.ypos = y0
            #gpsf.ellip = 0.2
            #gpsf.theta = 30 * np.pi / 180
            
            #### Set PSF
            set_par(sph2d.sigma1, val = s1, frozen = True)
            set_par(sph2d.sigma2, val = 0, frozen = True)
            set_par(sph2d.sigma3, val = 0, frozen = True)
            set_par(sph2d.alpha, val = 0, frozen = True)
            set_par(sph2d.beta, val = 0, frozen = True)
            
            # HESS PSF
            #set_par(sph2d.sigma1, val = 0.025369, frozen = True)
            #set_par(sph2d.alpha, val = 0.691225, frozen = True)
            #set_par(sph2d.sigma2, val = 0.0535014, frozen = True)
            #set_par(sph2d.beta, val = 0.13577, frozen = True)
            #set_par(sph2d.sigma3, val = 0.11505, frozen = True)


            set_par(sph2d.xpos, val = x0, frozen = True)
            set_par(sph2d.ypos, val = y0, frozen = True)

            set_par(sph2d.ampl, val = 10000, min = 1e-11, max = 100000000)
            set_par(sph2d.size, val = 10, min = 1e-11, max = 100)
            

            set_par(bgnd.c0, val = 1 , min = 0, max = 100)
            
            
            show_model()
            fit()
            #do_fit()
            conf()
            #do_conf()
            #image_fit()
            #save_model("model_" + str(idx) + ".fits")
            #save_resid("resid_" + str(idx) + ".fits")

            t1 = time()
            print('Simul time', t1 - t0)


if __name__ == '__main__':
    main()
