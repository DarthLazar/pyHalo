import numpy as np
from pyHalo.Massfunc.parameterizations import BrokenPowerLaw
from pyHalo.Spatial.nfw import NFW_3D

class MainLensPowerLaw(object):

    def __init__(self, args, lens_cosmo):

        self._lens_cosmo = lens_cosmo
        spatial_args, parameterization_args = self._set_kwargs(args)

        self._mass_func_parameterization = BrokenPowerLaw(**parameterization_args)
        self._spatial_parameterization = NFW_3D(**spatial_args)

    def __call__(self):

        masses = self._mass_func_parameterization.draw()

        x, y, r2d, r3d = self._spatial_parameterization.draw(len(masses))

        return np.array(masses), np.array(x), np.array(y), np.array(r2d), np.array(r3d), np.array(
            [self._lens_cosmo.z_lens] * len(masses))

    def _spatial(self,args):

        args_spatial = {}
        args_spatial['rmax2d'] = args['cone_opening_angle']

        if 'parent_m200' in args and 'parent_c' in args.keys():
            rho0_kpc, parent_Rs, parent_r200 = self._lens_cosmo.NFW_params_physical(args['parent_m200'],
                                                                                    args['parent_c'],
                                                                                    self._lens_cosmo.z_lens)
            args_spatial['Rs'] = parent_Rs
            args_spatial['rmax3d'] = parent_r200
        else:
            try:
                args_spatial['Rs'] = args['parent_Rs']
                args_spatial['rmax3d'] = args['parent_r200']
            except:
                raise ValueError('must specify either (parent_c, m200) for parent halo, or '
                                 '(parent_Rs, parent_r200) directly')

        if 'r_tidal' in args.keys():
            if isinstance(args['r_tidal'], str):
                if args['r_tidal'] == 'Rs':
                    args_spatial['r_core'] = args_spatial['Rs']
                else:
                    if args['r_tidal'][-2:] != 'Rs':
                        raise ValueError('if specifying the tidal core radius as number*Rs, the last two '
                                         'letters in the string must be "Rs".')

                    scale = float(args['r_tidal'][:-2])
                    args_spatial['r_core'] = scale * args_spatial['Rs']

            else:
                args_spatial['r_core'] = args['r_tidal_parent']

        return args_spatial

    def _mfunc(self,args):

        args_mfunc = {}

        required_keys = ['power_law_index', 'log_mlow', 'log_mhigh', 'log_m_break']

        for key in required_keys:
            try:
                args_mfunc[key] = args[key]
            except:
                raise ValueError('must specify a value for ' + key)

        if 'norm_A0' in args.keys():

            args_mfunc['normalization'] = args['norm_A0']

        elif 'fsub' in args.keys():

            args_mfunc['normalization'] = self._lens_cosmo.convert_fsub_to_norm(args['fsub'],args['cone_opening_angle'],
                                                                                       args_mfunc[
                                                                                           'power_law_index'],
                                                                                       10 ** args_mfunc[
                                                                                           'log_mlow'],
                                                                                       10 ** args_mfunc[
                                                                                           'log_mhigh'])
        else:
            raise ValueError('must either specify the normalization "norm_A0" direclty, or specify the mass fraction'
                             'in substructure at the Einstein radius "fsub".')


        return args_mfunc

    def _set_kwargs(self,args):

        args_mfunc = self._mfunc(args)
        args_spatial = self._spatial(args)

        for key in args.keys():
            if key not in args_mfunc.keys():
                args_mfunc[key] = args[key]

        return args_spatial, args_mfunc