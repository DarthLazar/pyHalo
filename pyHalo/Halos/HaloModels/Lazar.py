from pyHalo.Halos.halo_base import Halo
from pyHalo.Halos.concentration import Concentration
import numpy as np
import astropy.units as un

# change to LazarFieldHalo
class LazarHalo(Halo):
    """
    The main class for an Lazar field halo profile without truncation
    See the base class in Halos/halo_base.py for the required routines for any instance of a Halo class
    """

    def __init__(self, mass, x, y, r3d, mdef, z, sub_flag, lens_cosmo_instance, args, unique_tag):

        self._lens_cosmo = lens_cosmo_instance
        super(LazarHalo, self).__init__(mass, x, y, r3d, mdef, z, sub_flag, lens_cosmo_instance, args, unique_tag)

    @classmethod
    def change_profile_definition(cls, halo, new_mdef):
        """
        :param halo: an instance of Halo with a certain mdef
        :param new_mdef: a new mass definition
        :return: a new instance of Halo with the same mass, redshift, and angular position,
        but a new mass definition.
        """

        mass = halo.mass
        x = halo.x
        y = halo.y
        r3d = halo.r3d
        z = halo.z
        sub_flag = halo.is_subhalo
        args = halo._args
        cosmo_m_prof = halo.lens_cosmo
        unique_tag = halo.unique_tag

        return LazarHalo(mass, x, y, r3d, new_mdef, z, sub_flag, cosmo_m_prof, args, unique_tag)

    ###############################################################################################

    @property
    def lenstronomy_ID(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        return ['Lazar']

    ###############################################################################################

    @property
    def lenstronomy_params(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """

        r_eff, k_eff = self.normalization(self.mass, self.z) # [Mpc, unitless]
        R_sersic_angle = r_eff / self._lens_cosmo.D_d / self._lens_cosmo.cosmo.arcsec
        x, y = np.round(self.x, 4), np.round(self.y, 4)

        #print('r_eff: %s' % (r_eff));
        #print('R sersic angle: %s' % (R_sersic_angle))

        kwargs = [{'k_eff':k_eff, 'R_sersic':R_sersic_angle, 'center_x':x, 'center_y':y}]

        return kwargs, None

    ###############################################################################################

    def conc_model(self, mass, z, mdef='vir', cmodel='power_law', scatter=False, scatter_amplitude=0.5):
        """
        Concentration model of LOS CDM halos from (potentially) Paper II of series.
        Should either be defined as a power law model (which is valid over a specific mass range of 5e7 < Mvir/Msol  < 3e11)
        or
        Diemer & Joyce 2019 model modefied for halos using our projected density profile. Colossus will have to be modified to
        accomodate that, but we could get away with copy/paste the relevant code from colossus/concentration.py for now.
        """

        if ((cmodel == 'power_law') & (mdef == 'vir')): # still need to edit correct values
            # power law fit for concentrations using virial definition (Bryan and Norman)
            a_z = 0.749 + 0.538 * np.exp(-0.5 * z ** 1.439)
            b_z = -0.071 + 0.003 * z
        elif ((cmodel == 'power_law') & (mdef == '200c')): # still need to edit correct values
            # power law fit for concentrations using 200c definition
            a_z = 0.749 + 0.538 * np.exp(-0.5 * z ** 1.439)
            b_z = -0.071 + 0.003 * z

        log_c_eff = a_z + b_z*np.log10(mass / 1e12)
        c_eff = 10**log_c_eff

        if scatter: # scatter needs to be called outside this, else it wont be fixed
            _log_c = np.log(c_eff)
            c_eff = np.random.lognormal(_log_c, scatter_amplitude)

        return c_eff

    ###############################################################################################

    def normalization(self, mass, z):

        cmodel = 'power_law'
        mdef = 'vir'
        scatter = False
        dex = 0.1

        # -----------------------------------------------------------------------------------------

        def I(ceff):
            """
            Fitting function approximation for the normalization integral when computing the sigma_-1.
            """

            c_o = 5756.0; c_1 = 0.628; c_2 = 0.438; n = 7.41
            I_exp = -n * ((ceff+c_1)/c_2)**(1.0/n)

            return c_o * np.exp(I_exp)

        # -----------------------------------------------------------------------------------------

        def vir_overd(z): # This should be callable from elsewhere ...
            """
            Bryan and Norma 1998 definition for a flat universe.
            """

            omega_m = self._lens_cosmo.cosmo.astropy.Om0
            omega_l = 1.0 - omega_m
            xi = omega_m*(1.0 + z)**3/(omega_l + omega_m*(1.0 + z)**3)

            return 18.0 * np.pi**2 + 82.0 * xi - 39.0 * xi**2

        # -----------------------------------------------------------------------------------------

        if (mdef == 'vir'):
            delta = vir_overd(z)
        elif(mdef == '200c'):
            delta = 200.0
        h = self._lens_cosmo.cosmo.astropy.h
        rvir = self._lens_cosmo.rN_M_nfw_comoving(mass * h, delta, z) / h # physical Mpc

        if isinstance(self._args['mc_model'], float):
            c = self._args['mc_model']
        else:
            c = self.conc_model(mass, z, mdef, cmodel, scatter=scatter, scatter_amplitude=dex)

        Reff = rvir/c # physical Mpc

        sigma_1 = mass /(2 * np.pi * rvir**2 * I(c)) # physical Msol / Mpc^2
        sigma_c = self._lens_cosmo.sigma_crit_lensing # physical Msol / Mpc^2
        kappa_eff = sigma_1/sigma_c

        """
        print('virial radius: %s Mpc', % (rivr))
        print('R effective: %s Mpc', % (Reff))
        print('sigma_1: %s Msol/Mpc^2', % (np.log10(sigma_1)))
        print('sigma_c: %s Msol/Mpc^2', % (np.log10(sigma_c)))
        print('kappa_eff: %s' % (kappa_eff))
        """

        return Reff, kappa_eff

    ###############################################################################################

    @property
    def profile_args(self):
        """
        See documentation in base class (Halos/halo_base.py)
        """
        if not hasattr(self, '_profile_args'):
            R_sersic = self.R_Lazar(self.mass,self.z)
            self._profile_args = (R_sersic)

        return self._profile_args
