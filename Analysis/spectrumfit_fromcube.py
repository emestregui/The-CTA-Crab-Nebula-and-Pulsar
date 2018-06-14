"""
An example of how to do spectral analysis of a 3D n_pred / n_obs cube.

Using CTA IRFs.

"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
import numpy as np
import yaml
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from sherpa.astro.ui import *
from gammapy.cube import SkyCube
from regions import CircleSkyRegion
from astropy.table import Table
from gammapy.spectrum.models import LogParabola
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.models import ExponentialCutoffPowerLaw
from gammapy.spectrum import SpectrumFit
from gammapy.spectrum import SpectrumObservation
from gammapy.spectrum import PHACountsSpectrum
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.irf import EnergyDependentMultiGaussPSF
from regions.shapes import CircleAnnulusSkyRegion
from gammapy.irf import Background3D
from gammapy.background import FOVCube
#from gammapy.image import SkyImage
#from configuration import make_ref_cube

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


def get_irfs(config):
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
    filename = 'irf_file.fits'
    
    offset = Angle(config['selection']['offset_fov'] * u.deg)

    
    psf_fov = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    psf = psf_fov.to_energy_dependent_table_psf(theta=offset)
    
    print(' psf', psf)
    aeff_tab = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    edisp_fov = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    table = fits.open('irf_file.fits')['BACKGROUND']
    table.columns.change_name(str('BGD'), str('Bgd'))
    table.header['TUNIT7'] = '1 / (MeV s sr)'
    bkg = Background3D.read(filename, hdu='BACKGROUND')
    
    return dict(psf=psf, aeff=aeff_tab, edisp=edisp_fov, bkg=bkg)

def make_annular_spectrum(on_pos, off_pos, on_rad_sizes, off_rad_sizes, out_rad_sizes, cube_on, cube_off, alpha_obs):
    '''Take ON and OFF counts from an annular region.
        
    Parameters
    ----------
    on_pos : `SkyCoord`
        Center of ON annular region coordinates.
    off_pos : `SkyCoord`
        Center of OFF annular region coordinates.
    on_rad_sizes : numpy array
        Inner radius of ON size for each energy bin for an annular region.
    off_rad_sizes : numpy array
        Inner radius of OFF size for each energy bin for an annular region.
    out_rad_sizes : numpy array
        Outer radius of ON size for each energy bin and a annular radius.
    cube_on : `SkyCube`
        3D ON sky cube.
    cube_off : `SkyCube`
        3D OFF sky cube.
    alpha_obs : float
        On region area / Off region area

    Returns
    -------
    
    ann_on_data : `Table`
        ON annular region information table.
    ann_off_data : `Table`
        OFF annular region information table.
    ann_stats : numpy array
        Array of ones and zeros depending on if each energy satisfy a certain condition (1) or not (0).
        
    '''

    ann_on_data = Table()
    ann_off_data = Table()
    ann_on_data['value'] = np.zeros(len(on_rad_sizes))
    ann_off_data['value'] = np.zeros(len(on_rad_sizes))
    ann_stats = np.zeros(len(on_rad_sizes))

    for i in range(len(on_rad_sizes)):
        
        on_region = CircleSkyRegion(on_pos, on_rad_sizes[i])
        off_region = CircleSkyRegion(off_pos, off_rad_sizes[i])
        
        on_data = cube_on.spectrum(on_region)
        off_data = cube_off.spectrum(off_region)
        
        out_region = CircleSkyRegion(on_pos, out_rad_sizes[i])
        out_on_data = cube_on.spectrum(out_region)
        out_off_data = cube_off.spectrum(out_region)
        
        ann_on_data['value'][i] = out_on_data['value'][i] - on_data['value'][i]
        ann_off_data['value'][i] = out_off_data['value'][i] - off_data['value'][i]
    
        limasig = np.sqrt(2) * np.sqrt(ann_on_data['value'][i]*np.log(((1+alpha_obs)/alpha_obs)*ann_on_data['value'][i]/(ann_on_data['value'][i] + ann_off_data['value'][i]))+ann_off_data['value'][i]*np.log((1+alpha_obs)*ann_off_data['value'][i]/(ann_on_data['value'][i]+ann_off_data['value'][i])))
        
        exss_dat = ann_on_data['value'][i] - ann_off_data['value'][i]
        if limasig >= 3 and exss_dat >= 7 and exss_dat >= 0.03 * ann_off_data['value'][i]:
            ann_stats[i] = 1.

    ann_on_data['e_min'] = on_data['e_min']
    ann_on_data['e_max'] = on_data['e_max']
    ann_on_data['e_ref'] = on_data['e_ref']

    ann_off_data['e_min'] = off_data['e_min']
    ann_off_data['e_max'] = off_data['e_max']
    ann_off_data['e_ref'] = off_data['e_ref']

    return ann_on_data, ann_off_data, ann_stats

def make_circular_spectrum(on_pos, off_pos, on_sizes, off_sizes, cube_on, cube_off, alpha_obs):
    '''Take ON and OFF counts from an circular region.
        
    Parameters
    ----------
    on_pos : `SkyCoord`
        Center of ON circular region coordinates.
    off_pos : `SkyCoord`
        Center of OFF circular region coordinates.
    on_sizes :
        Radius of ON size for each energy bin for an circular region.
    off_sizes :
        Radius of OFF size for each energy bin for an circular region.
    
    cube_on : `SkyCube`
        3D ON sky cube.
    cube_off : `SkyCube`
        3D OFF sky cube.
    alpha_obs : float
        On region area / Off region area
        
    Returns
    -------
        
    ann_on_data : `Table`
        ON circular region information table.
    ann_off_data : `Table`
        OFF circular region information table.
    ann_stats : numpy array
        Array of ones and zeros depending on if each energy satisfy a certain condition (1) or not (0).
        
    '''
    
    on_data = Table()
    off_data = Table()
    on_data['value'] = np.zeros(len(on_sizes))
    off_data['value'] = np.zeros(len(on_sizes))
    circ_stats = np.zeros(len(on_sizes))
    
    for i in range(len(on_sizes)):
    
        on_region = CircleSkyRegion(on_pos, on_sizes[i])
        off_region = CircleSkyRegion(off_pos, off_sizes[i])
        
        on_data['value'][i] = cube_on.spectrum(on_region)['value'][i]
        off_data['value'][i] = cube_off.spectrum(off_region)['value'][i]
    
        limasig = np.sqrt(2) * np.sqrt(on_data['value'][i]*np.log(((1+alpha_obs)/alpha_obs)*on_data['value'][i]/(on_data['value'][i] + off_data['value'][i]))+off_data['value'][i]*np.log((1+alpha_obs)*off_data['value'][i]/(on_data['value'][i]+off_data['value'][i])))
        
        ##Debug
        #print(limasig, 'Energy range: ', cube_on.energies('edges')[i], ' - ', cube_on.energies('edges')[i+1])
        #print('On: ', on_data['value'][i], 'Off: ' ,off_data['value'][i])

        exss_dat = on_data['value'][i] - off_data['value'][i]
        if exss_dat >= 7 and limasig > 3:
            circ_stats[i] = 1.

    return on_data, off_data, circ_stats


def main():
    
    #Low energy of spectral fitting range.
    lo_fit_energ = 0.1 * u.Unit('TeV')
    hi_fit_energ = 10 * u.Unit('TeV')
    
    #If you want an internal estimation of a high energy limit for the fitting range: est_hi_lim = 'yes'. If 'no' the hi_fit_energ will be used.
    est_hi_lim = 'yes'
    
    # Read ON and OFF cubes
    filename_on = 'non_cube.fits.gz' # non_cube_convolved.fits
    cube_on = SkyCube.read(filename_on)
    
    ann_filename_off = 'noff_withpuls_cube.fits.gz'
    ann_cube_off = SkyCube.read(ann_filename_off)
    circ_filename_off = 'noff_withneb_cube.fits.gz'
    circ_cube_off = SkyCube.read(circ_filename_off)
    
    # Read config and IRFs
    config = read_config('config.yaml')
    irfs = get_irfs(config)
    offset = Angle(config['selection']['offset_fov'] * u.deg)
    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    alpha_obs = 1.
    binsz = config['binning']['binsz']
    aeff = irfs['aeff'].to_effective_area_table(offset = offset, energy = cube_on.energies('edges'))
    edisp = irfs['edisp'].to_energy_dispersion(offset = offset, e_true = aeff.energy.bins, e_reco = cube_on.energies('edges') )
    
    # Define circular on/off Regions parameters
    on_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    on_sizes = np.ones(20) * binsz * u.deg
    
    off_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    off_sizes = on_sizes * np.sqrt(1./alpha_obs)
    
    # Make Annular region
    on_rad_sizes = np.ones(len(on_sizes)) * 0.1 * binsz * u.deg
    off_rad_sizes = on_rad_sizes * np.sqrt(1./alpha_obs)
    widths = np.ones(len(on_sizes)) * 22 * binsz * u.deg
    out_rad_sizes = on_rad_sizes + widths
    
    ann_on_data, ann_off_data, ann_stats = make_annular_spectrum(on_pos, off_pos, on_rad_sizes, off_rad_sizes, out_rad_sizes, cube_on, ann_cube_off, alpha_obs)
    
    # Make circular region
    circ_on_data, circ_off_data, circ_stats = make_circular_spectrum(on_pos, off_pos, on_sizes, off_sizes, cube_on, circ_cube_off, alpha_obs)

    # Undo "holes" in circ/ann_stats
    if np.max(np.where(circ_stats == 1)) + 1 != circ_stats.sum():
        circ_stats[0:np.max(np.where(circ_stats == 1)) + 1][circ_stats[0:np.max(np.where(circ_stats == 1)) + 1] == 0] = 1.
    if np.max(np.where(ann_stats == 1)) + 1 != ann_stats.sum():
        ann_stats[0:np.max(np.where(ann_stats == 1)) + 1][ann_stats[0:np.max(np.where(ann_stats == 1)) + 1] == 0] = 1.
    
    # Make on/off vector
    ann_on_vector = PHACountsSpectrum(energy_lo = cube_on.energies('edges')[:-1], energy_hi= cube_on.energies('edges')[1:], data= ann_on_data['value'].data * ann_stats * u.ct, backscal = on_sizes[0].value, meta={'EXPOSURE' : livetime.value})
    circ_on_vector = PHACountsSpectrum(energy_lo = cube_on.energies('edges')[:-1], energy_hi= cube_on.energies('edges')[1:], data= circ_on_data['value'].data * circ_stats * u.ct, backscal = on_sizes[0].value, meta={'EXPOSURE' : livetime.value})
    
    
    ann_off_vector = PHACountsSpectrum(energy_lo = ann_cube_off.energies('edges')[:-1], energy_hi= ann_cube_off.energies('edges')[1:], data= ann_off_data['value'].data * ann_stats * u.ct, backscal = off_sizes[0].value, meta={'EXPOSURE' : livetime.value, 'OFFSET' : 0.3 * u.deg})
    circ_off_vector = PHACountsSpectrum(energy_lo = circ_cube_off.energies('edges')[:-1], energy_hi= circ_cube_off.energies('edges')[1:], data= circ_off_data['value'].data * circ_stats * u.ct, backscal = off_sizes[0].value, meta={'EXPOSURE' : livetime.value, 'OFFSET' : 0.3 * u.deg})

    # Make SpectrumObservation

    ann_sed_table = SpectrumObservation(on_vector = ann_on_vector, off_vector = ann_off_vector, aeff = aeff, edisp = edisp)
    circ_sed_table = SpectrumObservation(on_vector = circ_on_vector, off_vector = circ_off_vector, aeff = aeff, edisp = edisp)

    ##Debug
    #print(ann_stats)
    #print(circ_stats)
    
    # Define Spectral Model

    model2fit1 = LogParabola(amplitude=1e-11 * u.Unit('cm-2 s-1 TeV-1'), reference=1 * u.TeV, alpha=2.5 * u.Unit(''), beta=0.1 * u.Unit(''))
    model2fit2 = ExponentialCutoffPowerLaw(index = 1. * u.Unit(''), amplitude = 1e-11 * u.Unit('cm-2 s-1 TeV-1'), reference= 1 * u.TeV,  lambda_= 0. * u.Unit('TeV-1'))
    model2fit3 = PowerLaw(index= 2.5 * u.Unit(''), amplitude= 5e-11 * u.Unit('cm-2 s-1 TeV-1'), reference= 0.15 * u.TeV)

    model2fit3.parameters['amplitude'].parmin = 1e-12
    model2fit3.parameters['amplitude'].parmax = 1e-10
    
    model2fit3.parameters['index'].parmin = 2.0
    model2fit3.parameters['index'].parmax = 4.0

    #Models to fit the circular and annular observations
    models_ann_fit = [model2fit1, model2fit2, model2fit3]
    models_circ_fit = [model2fit1, model2fit2, model2fit3]
    
    # Fit
    if est_hi_lim = 'yes':
        hi_fit_energ = cube_on.energies('edges')[int(np.sum(ann_stats))]
    
    for k in range(len(models_ann_fit)):
        fit_source = SpectrumFit(obs_list = ann_sed_table, model=models_ann_fit[k],forward_folded=True, fit_range=(lo_fit_energ,hi_fit_energ))
        fit_source.fit()
        fit_source.est_errors()
        results = fit_source.result
        ax0, ax1 = results[0].plot(figsize=(8,8))
        print(results[0])
    
    if est_hi_lim = 'yes':
        hi_fit_energ = cube_on.energies('edges')[int(np.sum(circ_stats))]

    for k in range(len(models_circ_fit)):
        fit_source = SpectrumFit(obs_list = circ_sed_table, model=models_circ_fit[k],forward_folded=True, fit_range=(lo_fit_energ,hi_fit_energ))
        fit_source.fit()
        fit_source.est_errors()
        results = fit_source.result
        print(results[0])


if __name__ == '__main__':
    main()
