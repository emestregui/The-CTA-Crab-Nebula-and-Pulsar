"""
An example how to simulate a 3D n_pred, n_obs cube, n_off cube.

Using CTA IRFs.

"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import SkyCoord, Angle
from gammapy.cube import make_exposure_cube
from gammapy.cube.utils import SkyCube
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.irf import EnergyDependentMultiGaussPSF
from configuration import get_model_gammapy, make_ref_cube
from astropy.io import fits
from gammapy.utils.energy import EnergyBounds
from gammapy.cube import make_background_cube
from gammapy.irf import Background3D
from time import time
import pyfits
from astropy.units import Quantity

def _validate_inputs(flux_cube, exposure_cube):
    if flux_cube.data.shape[1:] != exposure_cube.data.shape[1:]:
        raise ValueError('flux_cube and exposure cube must have the same shape!\n'
                         'flux_cube: {0}\nexposure_cube: {1}'
                         ''.format(flux_cube.data.shape[1:], exposure_cube.data.shape[1:])
                         )

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


def get_irfs(config, filename):
    '''Get IRFs from file.
    
    Parameters
    ----------
    config : `dict`
        Configuration dictionary.
    filename : fits file
        IRFs file
    
    Returns
    -------
    irfs : `dict`
        IRFs dictionary.
    
    '''
    offset = Angle(config['selection']['offset_fov'] * u.deg)
    
    psf_fov = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    psf = psf_fov.to_energy_dependent_table_psf(theta=offset)
    
    print(' psf', psf)
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    
    edisp_fov = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    table = fits.open('irf_file.fits')['BACKGROUND']
    table.columns.change_name(str('BGD'), str('Bgd'))
    table.header['TUNIT7'] = '1 / (MeV s sr)'
    bkg = Background3D.read(filename, hdu='BACKGROUND')
    
    irfs = dict(psf=psf, aeff=aeff, edisp=edisp_fov, bkg=bkg, offset=offset)
    
    return irfs


def compute_spatial_model_integral(model, image):
    '''Compute integral of spatial model.
    
    Parameters
    ----------
    model : `CombinedModel3D`
        Source model
    image : `SkyImage`
        Spatial model sky image
    
    Returns
    -------
    Spatial model integral.
    
    '''
    coords = image.coordinates()
    surface_brightness = model(coords.data.lon.deg, coords.data.lat.deg) * u.Unit('deg-2')
    solid_angle = image.solid_angle()
    return (surface_brightness * solid_angle).sum().to('')


def compute_sum_cube(flux_cube, flux_cube2, config):
    """Compute sum of two flux cubes.
        
    Parameters
    ----------
    flux_cube : `SkyCube`
        Flux cube 1, really differential surface brightness in 'cm-2 s-1 TeV-1 sr-1'.
    flux_cube2 : `SkyCube`
        Flux cube 2.
    config : `dict`
        Configuration dictionary.
        
    Returns
    -------
    nflux_cube_sum: `SkyCube`
        Sum of flux_cube and flux_cube2.
    
    See also
    -------
    read_config
    
    """
    ebin = flux_cube.energies(mode="edges")
    ebounds = EnergyBounds(ebin)
    
    nflux_cube_sum = make_ref_cube(config)
    for idx in range(len(ebounds) - 1):
        npred1 = flux_cube.sky_image_idx(idx)
        npred2 =flux_cube2.sky_image_idx(idx)
        
        ## DEBUG
        #print npred1.data
        #print npred2.data
        
        nflux_sum = u.Quantity(npred1.data.value + npred2.data.value,'1 / (cm2 s sr TeV)')
        nflux_cube_sum.data[idx] = nflux_sum.value

    return nflux_cube_sum


def compute_npred_cube(flux_cube, exposure_cube, ebounds, config, irfs,
                       integral_resolution=10):
    """Compute predicted counts cube.
        
    Parameters
    ----------
    flux_cube : `SkyCube`
        Flux cube, really differential surface brightness in 'cm-2 s-1 TeV-1 sr-1'.
    exposure_cube : `SkyCube`
        Exposure cube.
    ebounds : `~astropy.units.Quantity`
        Energy bounds for the output cube.
    config : `dict`
        Configuration dictionary.
    irfs : `dict`
        IRFs dictionary.
    integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.
        
    Returns
    -------
    npred_cube : `SkyCube`
        Predicted counts cube with energy bounds as given by the input ``ebounds``.
        
    See also
    --------
    compute_npred_cube_simple
    read_config
    See get_irfs
    """
    _validate_inputs(flux_cube, exposure_cube)
    
    # Make an empty cube with the requested energy binning
    sky_geom = exposure_cube.sky_image_ref
    energies = EnergyBounds(ebounds)
    npred_cube = SkyCube.empty_like(sky_geom, energies=energies, unit='', fill=np.nan)
    
    # Process and fill one energy bin at a time
    for idx in range(len(ebounds) - 1):
        emin, emax = ebounds[idx: idx + 2]
        ecenter = np.sqrt(emin * emax)
        
        flux = flux_cube.sky_image_integral(emin, emax, interpolation='linear', nbins=integral_resolution)
        
        exposure = exposure_cube.sky_image(ecenter, interpolation='linear')
        solid_angle = exposure.solid_angle()
        
        flux.data.value[np.isnan(flux.data.value)] = 0
        exposure.data.value[np.isnan(exposure.data.value)] = 0
        npred = flux.data.value * u.Unit('1 / (cm2 s sr)') * exposure.data * solid_angle
        
        ##Debug
        #print npred.to('')
        
        npred_cube.data[idx] = npred.to('')
    
    # Apply EnergyDispersion
    edisp = irfs['edisp']
    offset = irfs['offset']
    
    edisp_idx = edisp.to_energy_dispersion(offset=offset, e_reco = ebounds, e_true = ebounds)
    
    for pos_x in range(npred_cube.data.shape[1]):
        for pos_y in range(npred_cube.data.shape[2]):
            npred_pos = npred_cube.data[0:len(ebounds) - 1,pos_x,pos_y]
            if npred_pos.sum() != 0:
                for idx in range(len(ebounds) - 1):
                    npred_cube.data[idx][pos_x][pos_y] = np.dot(npred_pos, edisp_idx.data.data[idx])


    return npred_cube

def compute_npred_cube_simple(flux_cube, exposure_cube):
    """Compute predicted counts cube (using a simple method).
        
    * Simply multiplies flux and exposure and pixel solid angle and energy bin width.
    * No spatial reprojection, or interpolation or integration in energy.
    * This is very fast, but can be inaccurate (e.g. for very large energy bins.)
    * If you want a more fancy method, call `compute_npred_cube` instead.
        
    Output cube energy bounds will be the same as for the exposure cube.
        
    Parameters
    ----------
    flux_cube : `SkyCube`
        Differential flux cube.
    exposure_cube : `SkyCube`
        Exposure cube.
        
    Returns
    -------
    npred_cube : `SkyCube`
        Predicted counts cube.
        
    See also
    --------
    compute_npred_cube
        
    """
    _validate_inputs(flux_cube, exposure_cube)
    
    solid_angle = exposure_cube.sky_image_ref.solid_angle()
    de = exposure_cube.energy_width
    
    flux = flux_cube.data * u.Unit('1 / (cm2 s sr TeV)')
    exposure = exposure_cube.data
    npred = flux * exposure * solid_angle * de[:, np.newaxis, np.newaxis]
    
    npred_cube = SkyCube.empty_like(exposure_cube)
    npred_cube.data = npred.to('')
    return npred_cube


def compute_nexcess_cube(npred_cube, livetime, pointing, offset_max, bkg_rate, config):
    '''Compute excess cube.
        
    Parameters
    ----------
    npred_cube : `SkyCube`
        Predicted counts cube.
    livetime : `Quantity`
        Observation time.
    pointing : `SkyCoord`
        Pointing coordinates.
    offset_max : `Angle`
        Offset.
    bkg_rate : `Background3D`
        Background rate.
    config : `dict`
        Configuration dictionary.

    Returns
    -------
    nexcess_cube : `SkyCube`
        Predicted counts cube.
    
    non_cube : `SkyCube`
        On observation.

    noff_cube : `SkyCube`
        Off observation.
    
    See also
    --------
    read_config
    See get_irfs
    
    '''
    ebin = npred_cube.energies(mode="edges")
    ebounds = EnergyBounds(ebin)

    nexcess_cube = make_ref_cube(config)
    non_cube = make_ref_cube(config)
    noff_cube = make_ref_cube(config)
    
    # Compute two background cubes
    nbkg1_cube =  make_background_cube(pointing = pointing, obstime = livetime, bkg = bkg_rate, ref_cube=npred_cube, offset_max = offset_max) #compute_bkg_cube(npred_cube,bkg_rate,livetime)
    nbkg2_cube =  make_background_cube(pointing = pointing, obstime = livetime, bkg = bkg_rate, ref_cube=npred_cube, offset_max = offset_max) #compute_bkg_cube(npred_cube,bkg_rate,livetime)
    
    # For each energy bin, I need to obtain the correct background rate (two, one for the on and one for the off)
    for idx in range(len(ebounds) - 1):
        emin, emax = ebounds[idx: idx + 2]
        ecenter = np.sqrt(emin * emax)
        print('Energy bins:')
        print emin, emax
        
        npred = npred_cube.sky_image_idx(idx)
        npred.unit = u.Unit('TeV')
        solid_angle = npred.solid_angle()
        npred.data.value[np.isnan(npred.data.value)]=0.
    
        
        nbkg1_ebin = nbkg1_cube.data[idx]
        nbkg2_ebin = nbkg2_cube.data[idx]
        
        ## DEBUG
        #print npred
        
        n_on = np.random.poisson(npred.data) + np.random.poisson(abs(nbkg1_ebin))
        n_off = np.random.poisson(abs(nbkg2_ebin))
        nexcess = n_on - n_off
        nexcess_cube.data[idx] = nexcess
        non_cube.data[idx] = n_on
        noff_cube.data[idx] = n_off
    return nexcess_cube, non_cube, noff_cube


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
    t0 = time()
    
    # Read configuration
    config = read_config('config.yaml')
    config2 = read_config('config2.yaml')

    # Getting the IRFs, effective area and PSF
    filename = 'irf_file.fits'
    irfs = get_irfs(config, filename)
    
    # Create empty references cubes
    ref_cube = make_ref_cube(config)
    
    if config['binning']['coordsys'] == 'CEL':
        pointing = SkyCoord(config['pointing']['ra'], config['pointing']['dec'], frame='icrs', unit='deg')
    if config['binning']['coordsys'] == 'GAL':
        pointing = SkyCoord(config['pointing']['glat'], config['pointing']['glon'], frame='galactic', unit='deg')

    ref_cube2 = make_ref_cube(config2)
    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    offset_max = Angle(config['selection']['ROI'])

    # Make exposure cube
    exposure_cube = make_exposure_cube(
        pointing=pointing,
        livetime=livetime,
        aeff=irfs['aeff'],
        ref_cube=ref_cube,
        offset_max=offset_max,
    )
    print('exposure sum: {}'.format(np.nansum(exposure_cube.data)))
    exposure_cube.data = exposure_cube.data.to('m2 s')
    print(exposure_cube)

    # Define model and do some quick checks
    model = get_model_gammapy(config)
    model2 = get_model_gammapy(config2)

    # Normalize spatial model
    norm_factor = compute_spatial_model_integral(model.spatial_model, exposure_cube.sky_image_ref)
    norm_factor2 = compute_spatial_model_integral(model2.spatial_model, exposure_cube.sky_image_ref)

    model.spatial_model.amplitude.value = 1./norm_factor
    model2.spatial_model.amplitude.value = 1./norm_factor2
    
    spatial_integral = compute_spatial_model_integral(model.spatial_model, exposure_cube.sky_image_ref)
    spatial_integral2 = compute_spatial_model_integral(model2.spatial_model, exposure_cube.sky_image_ref)


    print('Spatial integral (should be 1): ', round(spatial_integral,3))
    print('Spatial integral (should be 1): ', round(spatial_integral2,3))
    #flux_integral = model.spectral_model.integral(emin='1 TeV', emax='10 TeV')
    #print('Integral flux in range 1 to 10 TeV: ', flux_integral.to('cm-2 s-1'))
    flux_integral2 = model2.spectral_model.integral(emin='1 TeV', emax='10 TeV')
    print('Integral flux in range 1 to 10 TeV: ', flux_integral2.to('cm-2 s-1'))


# import IPython; IPython.embed()

    # Make and sum flux cubes
    flux_cube = model.evaluate_cube(ref_cube)
    flux_cube2 = model2.evaluate_cube(ref_cube2)

    flux_sum = compute_sum_cube(flux_cube,flux_cube2,config)
    
    
    # Make npred cube
    npred_cube = compute_npred_cube(
        flux_sum, exposure_cube,
        ebounds=flux_sum.energies('edges'),
        config = config, irfs = irfs, integral_resolution=2
    )
    bkg = irfs['bkg']

    t1 = time()
    npred_cube_simple = compute_npred_cube_simple(flux_sum, exposure_cube)

    t2 = time()
    print('npred_cube: ', t1 - t0)
    print('npred_cube_simple: ', t2 - t1)
    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data)))
    print('npred_cube_simple sum: {}'.format(np.nansum(npred_cube_simple.data)))
    
    # Make ON, OFF and excess cubes
    on_excess = compute_nexcess_cube(npred_cube, livetime, pointing, offset_max, bkg_rate = bkg, config = config)
    nexcess_cube = on_excess[0]
    non_cube = on_excess[1]
    noff_cube = on_excess[2]

    # Apply PSF convolution here
    #    kernels = irfs['psf'].kernels(npred_cube_simple)
    #    npred_cube_convolved = npred_cube_simple.convolve(kernels)
    
    # Make PSF Kernels
    kernels = []
    psf_table = psf_fromfits('irf_file.fits')
    energ_lo = psf_table[0].value
    sigmas = psf_table[3]
    energ_array = nexcess_cube.energies('edges')
    s = np.argmin(np.abs(irfs['offset'].value - psf_table[2].value))

    for i in range(nexcess_cube.data.shape[0]):
        v = np.argmin(np.abs(energ_array[i].value - energ_lo))
        kernels.append(irfs['psf'].kernels(nexcess_cube, Angle(sigmas[0][s][v] * u.deg))[i])
        ##Debug
        #print(sigmas[0][s][v] * u.deg)

    # Apply kernels convolution
    nexcess_cube_convolved = nexcess_cube.convolve(kernels)
    npred_cube_convolved = npred_cube_simple.convolve(kernels)
    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data)))
    # print('npred_cube_convolved sum: {}'.format(np.nansum(npred_cube_convolved.data)))
    noff_cube_convolved = noff_cube.convolve(kernels)
    non_cube_convolved = non_cube.convolve(kernels)
    
    # Write cubes
    exposure_cube.write('exposure_cube.fits', overwrite=True, format='fermi-exposure')
    flux_sum.write('flux_cube.fits.gz', overwrite=True)
    npred_cube.write('npred_cube.fits.gz', overwrite=True)
    npred_cube_convolved.write('npred_cube_convolved.fits.gz', overwrite=True)
    noff_cube.write('noff_cube.fits.gz', overwrite=True)
    noff_cube_convolved.write('noff_cube_convolved.fits.gz', overwrite=True)
    nexcess_cube.write('nexcess_cube.fits.gz', overwrite = True)
    nexcess_cube_convolved.write('nexcess_cube_convolved.fits.gz', overwrite = True)
    non_cube.write('non_cube.fits.gz', overwrite = True)
    non_cube_convolved.write('non_cube_convolved.fits.gz', overwrite = True)

    # If the amplitude of one of the sources in null then also try:
    #non_cube.write('noff_withneb_cube.fits.gz', overwrite = True)
    #non_cube_convolved.write('noff_withneb_cube_convolved.fits.gz', overwrite = True)
    #non_cube.write('noff_withpuls_cube.fits.gz', overwrite = True)
    #non_cube_convolved.write('noff_withpuls_cube_convolved.fits.gz', overwrite = True)

    t3 = time()
    print('Done in: ', t3 - t0)


if __name__ == '__main__':
    main()
