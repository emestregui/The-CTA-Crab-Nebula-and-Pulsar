from gammapy.cube import SkyCube, CombinedModel3D
from gammapy.spectrum.models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from gammapy.image.models import Shell2D, Sphere2D, Gauss2DPDF
import astropy.units as u

__all__ = [
    'get_model_gammapy',
    'make_ref_cube',
]


def get_model_gammapy(config):
    if config['model']['template'] == 'Shell2D':
        spatial_model = Shell2D(
            amplitude=1,
            x_0=config['model']['ra'],
            y_0=config['model']['dec'],
            r_in=config['model']['rin'],
            width=config['model']['width'],
            # Note: for now we need spatial models that are normalised
            # to integrate to 1 or results will be incorrect!!!
            normed=True,
        )
    if config['model']['template'] == 'Sphere2D':
        spatial_model = Sphere2D(
            amplitude=1,
            x_0=config['model']['ra'],
            y_0=config['model']['dec'],
            r_0=config['model']['rad'],
            # Note: for now we need spatial models that are normalised
            # to integrate to 1 or results will be incorrect!!!
            normed=True,
        )

    if config['model']['template'] == 'Gauss2D':
        spatial_model = Gauss2DPDF(
            #amplitude=1,
            #x_0=config['model']['ra'],
            #y_0=config['model']['dec'],
            sigma=config['model']['sigma'],
            # Note: for now we need spatial models that are normalised
            # to integrate to 1 or results will be incorrect!!!
            #normed=True,
        )
                             
    if config['model']['spectrum'] == 'pl':
        spectral_model = PowerLaw(
            amplitude=config['model']['prefactor'] * u.Unit('cm-2 s-1 TeV-1'), 
            index=config['model']['index'],
            reference=config['model']['pivot_energy'] * u.Unit('TeV'),
        )
    if config['model']['spectrum'] == 'ecpl':
        spectral_model = ExponentialCutoffPowerLaw(
            amplitude=config['model']['prefactor'] * u.Unit('cm-2 s-1 TeV-1'),
            index=config['model']['index'],
            reference=config['model']['pivot_energy'] * u.Unit('TeV'),
            lambda_=config['model']['cutoff'] * u.Unit('TeV-1'),
        )

    if config['model']['spectrum'] == 'LogParabola':
        spectral_model = LogParabola(
            amplitude=config['model']['prefactor'] * u.Unit('cm-2 s-1 TeV-1'),
            alpha=config['model']['alphapar'],
            beta=config['model']['beta'],
            reference=config['model']['pivot_energy'] * u.Unit('TeV'),
        )
                                               
    return CombinedModel3D(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )


def make_ref_cube(config):
    WCS_SPEC = {
        'nxpix': config['binning']['nxpix'],
        'nypix': config['binning']['nypix'],
        'binsz': config['binning']['binsz'],
        'xref': config['pointing']['ra'],
        'yref': config['pointing']['dec'],
        'proj': config['binning']['proj'],
        'coordsys': config['binning']['coordsys'],
    }

    # define reconstructed energy binning
    ENERGY_SPEC = {
        'mode': 'edges',
        'enumbins': config['binning']['enumbins'],
        'emin': config['selection']['emin'],
        'emax': config['selection']['emax'],
        'eunit': 'TeV',
    }

    CUBE_SPEC = {}
    CUBE_SPEC.update(WCS_SPEC)
    CUBE_SPEC.update(ENERGY_SPEC)
    cube = SkyCube.empty(**CUBE_SPEC)
    return cube
#    return SkyCube.empty(**WCS_SPEC, **ENERGY_SPEC)
