from pyHalo.Rendering.MassFunctions.broken_powerlaw import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW3D
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Spatial.keywords import subhalo_spatial_NFW
from pyHalo.Rendering.MassFunctions.mass_function_utilities import integrate_power_law_quad, \
    integrate_power_law_analytic
from pyHalo.Rendering.Main.SHMF_normalizations import *

from pyHalo.Rendering.render_base import RenderingBase

class MainLensBase(RenderingBase):

    def __init__(self, args, geometry, x_center_lens, y_center_lens):

        zlens, zsource = geometry._zlens, geometry._zsource

        kpc_per_arcsec_zlens = geometry._kpc_per_arcsec_zlens

        lenscosmo = LensCosmo(zlens, zsource, geometry._cosmo)

        self.rendering_args = self.keyword_parse(args, kpc_per_arcsec_zlens, zlens)

        spatial_args = subhalo_spatial_NFW(args, kpc_per_arcsec_zlens, zlens, lenscosmo)

        self._mass_func_parameterization = BrokenPowerLaw(**self.rendering_args)

        self.spatial_parameterization = NFW3D(**spatial_args)

        self._center_x, self._center_y = x_center_lens, y_center_lens

        super(MainLensBase, self).__init__(geometry)

    def negative_kappa_sheets_theory(self):

        kappa_sheets = []

        kwargs_mass_sheets = self.keys_convergence_sheets

        if kwargs_mass_sheets['subtract_subhalo_mass_sheet'] is False:
            return [], []

        log_mass_sheet_correction_min, log_mass_sheet_correction_max = \
            kwargs_mass_sheets['log_mass_sheet_min'], kwargs_mass_sheets['log_mass_sheet_max']

        kappa_scale = kwargs_mass_sheets['subhalo_mass_sheet_scale']

        m_low, m_high = 10 ** log_mass_sheet_correction_min, 10 ** log_mass_sheet_correction_max

        log_m_break = self.rendering_args['log_m_break']
        break_index = self.rendering_args['break_index']
        break_scale = self.rendering_args['break_scale']

        moment = 1

        if log_m_break == 0 or log_m_break / log_mass_sheet_correction_min < 0.01:
            use_analytic = True
        else:
            use_analytic = False

        norm = self.rendering_args['normalization']
        plaw_index = self.rendering_args['power_law_index']

        if use_analytic:
            mass = integrate_power_law_analytic(norm, m_low, m_high, moment, plaw_index)
        else:
            mass = integrate_power_law_quad(norm, m_low, m_high, log_m_break, moment,
                                            plaw_index, break_index, break_scale)

        kappa = mass / self.lens_cosmo.sigma_crit_mass(self.geometry._zlens, self.geometry)

        negative_kappa = -1 * kappa_scale * kappa

        kappa_sheets.append(negative_kappa)

        return kappa_sheets, [self.geometry._zlens]

    @property
    def keys_convergence_sheets(self):

        args_convergence_sheets = {}
        required_keys = ['log_mass_sheet_min', 'log_mass_sheet_max', 'subhalo_mass_sheet_scale',
                         'subtract_subhalo_mass_sheet']

        for key in required_keys:
            if key not in self.rendering_args.keys():
                raise Exception('When specifying mass function type POWER_LAW and rendering subhalos, must provide '
                                'key word arguments log_mass_sheet_min, log_mass_sheet_max, subtract_subhalo_mass_sheet, '
                                'and subhalo_mass_sheet_scale. These key words specify the halo mass '
                                'range used to add the convergence correction.')

            args_convergence_sheets[key] = self.rendering_args[key]

        return args_convergence_sheets

    @staticmethod
    def keyword_parse(args, kpc_per_arcsec_zlens, zlens):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break',
                         'break_index', 'break_scale', 'log_mass_sheet_min', 'log_mass_sheet_max',
                         'subtract_subhalo_mass_sheet', 'subhalo_mass_sheet_scale']

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        if 'sigma_sub' in args.keys():

            args_mfunc['normalization'] = norm_AO_from_sigmasub(args['sigma_sub'], args['parent_m200'],
                                                                zlens,
                                                                kpc_per_arcsec_zlens,
                                                                args['cone_opening_angle'],
                                                                args['power_law_index'])


        elif 'norm_kpc2' in args.keys():

            args_mfunc['normalization'] = norm_A0_from_a0area(args['norm_kpc2'],
                                                              zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'])

        elif 'norm_arcsec2' in args.keys():

            args_mfunc['normalization'] = norm_constant_per_squarearcsec(args['norm_arcsec2'],
                                                                         kpc_per_arcsec_zlens,
                                                                         args['cone_opening_angle'],
                                                                         args_mfunc['power_law_index'])

        elif 'f_sub' in args.keys() or 'log_f_sub' in args.keys():

            if 'log_f_sub' in args.keys():
                args['f_sub'] = 10 ** args['log_f_sub']

            a0_area_parent_halo = convert_fsub_to_norm(
                args['f_sub'], args['parent_m200'], zlens, args['R_ein_main'], args['cone_opening_angle'],
                zlens,
                args_mfunc['power_law_index'], 10 ** args_mfunc['log_mlow'],
                                               10 ** args_mfunc['log_mhigh'], mpivot=10 ** 8)

            args_mfunc['normalization'] = norm_A0_from_a0area(a0_area_parent_halo,
                                                              kpc_per_arcsec_zlens,
                                                              args['cone_opening_angle'],
                                                              args_mfunc['power_law_index'], m_pivot=10 ** 8)


        else:
            routines = 'sigma_sub: amplitude of differential mass function at 10^8 solar masses (d^2N / dmdA) in units [kpc^-2 M_sun^-1];\n' \
                       'automatically accounts for evolution of projected number density with halo mass and redshift (see Gilman et al. 2020)\n\n' \
                       'norm_kpc2: same as sigma_sub, but does not automatically account for evolution with halo mass and redshift\n\n' \
                       'norm_arcsec2: same as norm_kpc2, but in units (d^2N / dmdA) in units [arcsec^-2 M_sun^-1]\n\n' \
                       'f_sub or log_f_sub: projected mass fraction in substructure within the radius 0.5*cone_opening_angle'

            raise Exception('Must specify normalization of the subhalo '
                            'mass function. Recognized normalization routines are: \n' + routines)

        return args_mfunc

