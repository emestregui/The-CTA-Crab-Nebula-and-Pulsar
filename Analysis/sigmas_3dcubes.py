"""
An example of how to compute significances for a 3D n_pred, n_obs cube, n_off cube.

Using CTA IRFs.
    
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from gammapy.cube import SkyCube
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from astropy.table import Table
import astropy.units as u
import yaml
import pyfits
from astropy.units import Quantity

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

def main():
    # Read cubes
    cube_on = SkyCube.read('non_cube.fits.gz')
    cube_off = SkyCube.read('noff_cube.fits.gz')

    #Read config

    config = yaml.load(open('config.yaml'))
    binsz = config['binning']['binsz']
    offset_fov = config['selection']['offset_fov']

    #countson_vals = []
    #countsoff_vals = []
    diff_vals = np.ones(int(config['binning']['enumbins']))
    sigmaslimas = np.ones(int(config['binning']['enumbins']))

    # Define PSF region
    irffile = 'irf_file.fits'
    psf_table = psf_fromfits(irffile)
    psfs = psf_table[3]
    
    on_sizes = np.ones(int(config['binning']['enumbins'])) * u.deg
    energarr = cube_on.energies('edges')
    for idx in range(len(cube_on.energies('center'))):
        i = np.argmin(np.abs(energarr[idx].value - psf_table[0].value))
        j = np.argmin(np.abs(offset_fov - psf_table[2].value))
        on_sizes.value[idx] = psfs[0][j][i] * 2.12
    
    alpha_obs = 0.2
    on_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')

    ##Debug
    #print(on_sizes/binsz)

    off_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    off_sizes = on_sizes / np.sqrt(alpha_obs)

    on_data = Table()
    off_data = Table()
    on_data['value'] = np.zeros(len(on_sizes))
    off_data['value'] = np.zeros(len(on_sizes))

    for i in range(cube_on.data.shape[0]):
        
        # Make PSF region
        on_region = CircleSkyRegion(on_pos, on_sizes[i])
        off_region = CircleSkyRegion(off_pos, off_sizes[i])
    
        # Take spectrum
        on_data['value'][i] = cube_on.spectrum(on_region)['value'][i]
        off_data['value'][i] = cube_off.spectrum(off_region)['value'][i] #* alpha_obs
        non_val = on_data['value'][i]
        noff_val = off_data['value'][i]
        diff_vals[i] = non_val - noff_val
    
        if non_val != 0 and noff_val != 0:
            siglima = np.sqrt(2)*np.sqrt(non_val*np.log((1.0+(1.0/alpha_obs))*non_val/(non_val + noff_val))+noff_val*np.log((alpha_obs+1.0)*noff_val/(non_val+noff_val)))
        elif non_val != 0 and noff_val == 0:
            siglima = np.sqrt(2)*np.sqrt(non_val*np.log((1.0+(1.0/alpha_obs))))
        else:
            siglima = 0
        sigmaslimas[i] = siglima

    ##Debug
    #non_val = cube_on.data.sum().value
    #noff_val = cube_off.data.sum().value
    
    lo_lim_idx = np.where(abs(cube_on.energies('edges').value - 0.4) == np.min(abs(cube_on.energies('edges').value - 0.4)))[0][0]
    max_energ_idx = np.where(abs(cube_on.energies('edges').value - 3.0) == np.min(abs(cube_on.energies('edges').value - 3.0)))[0][0]
    non_val = on_data['value'][lo_lim_idx:max_energ_idx].sum()
    noff_val = off_data['value'][lo_lim_idx:max_energ_idx].sum()

    siglima = np.sqrt(2)*np.sqrt(non_val*np.log((1.0+(1.0/alpha_obs))*non_val/(non_val + noff_val))+noff_val*np.log((alpha_obs+1.0)*noff_val/(non_val+noff_val)))

    #print('On events: ', on_data)
    #print('Off events: ', off_data)
    diff_vals[np.isnan(diff_vals)] = 0
    sigmaslimas[np.isnan(sigmaslimas)] = 0
    print('Excess: ', diff_vals)
    print('Total positive Excess: ', diff_vals[diff_vals > 0].sum())
    print('LiMa by energy bins: ', sigmaslimas)
    print('Total LiMa: ', siglima, 'Energy range: ', cube_on.energies('edges')[lo_lim_idx], ' - ', cube_on.energies('edges')[max_energ_idx])
    
    lo_lim_idx = np.where(abs(cube_on.energies('edges').value - 1.0) == np.min(abs(cube_on.energies('edges').value - 1.0)))[0][0]
    non_val = on_data['value'][lo_lim_idx:max_energ_idx].sum()
    noff_val = off_data['value'][lo_lim_idx:max_energ_idx].sum()
    
    siglima_tves = np.sqrt(2)*np.sqrt(non_val*np.log(2*non_val/(non_val + noff_val))+noff_val*np.log(2*noff_val/(non_val+noff_val)))
    
    print('Total LiMa: ', siglima_tves, 'Energy range: ', cube_on.energies('edges')[lo_lim_idx], ' - ', cube_on.energies('edges')[max_energ_idx])
    
    return [siglima,siglima_tves, on_data, off_data, diff_vals,sigmaslimas]

if __name__ == '__main__':
    main()


