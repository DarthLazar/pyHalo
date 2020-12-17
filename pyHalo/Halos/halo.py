from pyHalo.Halos.HaloModels.collisionless_nfw import \
    TNFWFieldHalo, TNFWMainSubhalo, NFWFieldHalo, NFWMainSubhalo
from pyHalo.Halos.HaloModels.base import ProfileBase
import numpy as np

class Halo(object):

    _recognized_mass_definitions = ['NFW', 'TNFW', 'PT_MASS', 'PJAFFE']

    def __init__(self, mass=None, x=None, y=None, r2d=None, r3d=None, mdef=None, z=None,
                 sub_flag=None, lens_cosmo_instance=None, args={}, unique_tag=None):

        """
        This is the main class for objects rendered in the lens volume. It keeps track of stuff like the position,
        mass, redshift, and structural properties (e.g. concentration, core radius, etc.)

        :param mass: halo mass in M_sun
        :param x: angular coordinate x in arcsec
        :param y: angular coordinate y in arcsec
        :param r2d: two dimensional postiion of halo in arcsec sqrt(x^2 + y^2
        :param r3d: three dimensional position of halo in kpc (used to compute the truncation radius for subhalos)
        :param mdef: mass definition for the halo (see recognized mass definitions above)
        :param z: halo redshift
        :param sub_flag: bool; if True, the halo is treated as a main deflector subhalo
        :param lens_cosmo_instance: an instance of LensCosmo
        :param args: keyword arguments that include default settings for the halo
        :param unique_tag: a random number with 16 decimal places that should uniquely identify each halo
        (unless you are either EXTREMELY unlucky or are handling models with 10^16 halos)
        """

        self.lens_cosmo = lens_cosmo_instance

        self.mass = mass

        # x and y in arcsec
        self.x = x
        self.y = y

        # r2d and r3d in kpc
        self.r2d = r2d

        self.r3d = r3d

        self.mdef = mdef

        self.z = z

        self.is_subhalo = sub_flag

        self.has_associated_subhalos = False

        self._args = args

        if unique_tag is None:
            self._unique_tag = np.random.rand()
        else:
            self._unique_tag = unique_tag

        assert mdef in self._recognized_mass_definitions, 'mass definition '+str(mdef)+' not recognized.'

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
        r2d = halo.r2d
        r3d = halo.r3d
        z = halo.z
        sub_flag = halo.is_subhalo
        args = halo._args
        cosmo_m_prof = halo.lens_cosmo
        unique_tag = halo._unique_tag

        return Halo(mass, x, y, r2d, r3d, new_mdef, z,
                 sub_flag, cosmo_m_prof, args, unique_tag)

    def get_z_infall(self):

        """
        Evaluate the infall redshift using a PDF generated by galacticus
        :return: the infall redshift of a halo assuming it is in a host halo at redshift self.z
        """

        if not hasattr(self, 'z_infall'):

            self.z_infall = self.lens_cosmo.z_accreted_from_zlens(self.mass, self.z)

        return self.z_infall

    @property
    def profile_args(self):

        """
        Evaluates the profile parameters for each halo. This is profile dependent, but may include
        things like the concentration, truncation radius, etc.
        :return: Profile parameters that uniqely specify the mass profile
        """
        if not hasattr(self, '_mass_def_arg'):
            self._profileargs = self._halo_model.halo_parameters

        return self._profileargs

    @property
    def _halo_model(self):

        """
        Loads the model for each halo
        :return:
        """

        if not hasattr(self, '_halo_profile_instance'):

            if self.mdef in ['PJAFFE', 'PT_MASS']:

                halo_type = ProfileBase()

            else:
                if self.is_subhalo is True:
                    if self.mdef == 'NFW':
                        halo_type = NFWMainSubhalo(self)
                    elif self.mdef == 'TNFW':
                        halo_type = TNFWMainSubhalo(self)
                    else:
                        raise Exception('profile type '+self.mdef+' does not have a corresponding subhalo class')

                else:

                    if self.mdef == 'NFW':
                        halo_type = NFWFieldHalo(self)
                    elif self.mdef == 'TNFW':
                        halo_type = TNFWFieldHalo(self)
                    else:
                        raise Exception('profile type '+self.mdef+' does not have a corresponding field halo class')

            self._halo_profile_instance = halo_type

        return self._halo_profile_instance



