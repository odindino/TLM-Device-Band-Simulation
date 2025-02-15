import numpy as np
from scipy import constants

class BandStructure:
    """
    A class to calculate band structure for semiconductor devices.
    
    Parameters:
    -----------
    Eg : float
        Band gap energy (eV)
    chi : float
        Electron affinity (eV)
    me_eff : float
        Effective mass of electrons (relative to free electron mass)
    mh_eff : float
        Effective mass of holes (relative to free electron mass)
    T : float, optional
        Temperature (K), defaults to 300K
    """
    def __init__(self, Eg, chi, me_eff, mh_eff, T=300):
        # Material parameters
        self.Eg = Eg          # Band gap
        self.chi = chi        # Electron affinity
        self.me_eff = me_eff  # Effective mass of electrons
        self.mh_eff = mh_eff  # Effective mass of holes
        self.T = T            # Temperature
        
        # Physical constants
        self.k = constants.k        # Boltzmann constant
        self.q = constants.e        # Elementary charge
        self.h = constants.h        # Planck constant
        self.m0 = constants.m_e     # Free electron mass
        
        # Derived parameters
        self.kT = self.k * self.T / self.q  # in eV
        
        # Calculate effective density of states
        self.Nc = self._calc_Nc()  # Conduction band
        self.Nv = self._calc_Nv()  # Valence band
        
        # Calculate intrinsic carrier concentration
        self.ni = self._calc_ni()
        
    def _calc_Nc(self):
        """Calculate effective density of states in conduction band."""
        return 2 * (2 * np.pi * self.me_eff * self.m0 * self.k * self.T / self.h**2)**(3/2)
    
    def _calc_Nv(self):
        """Calculate effective density of states in valence band."""
        return 2 * (2 * np.pi * self.mh_eff * self.m0 * self.k * self.T / self.h**2)**(3/2)
    
    def _calc_ni(self):
        """Calculate intrinsic carrier concentration."""
        return np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2 * self.kT))
    
    def calc_fermi_level(self, doping_concentration, doping_type='n'):
        """
        Calculate Fermi level position relative to intrinsic level.
        
        Parameters:
        -----------
        doping_concentration : float
            Doping concentration (cm^-3)
        doping_type : str
            'n' for n-type or 'p' for p-type doping
            
        Returns:
        --------
        float
            Fermi level position relative to intrinsic level (eV)
        """
        if doping_type.lower() == 'n':
            return self.kT * np.log(doping_concentration / self.ni)
        else:  # p-type
            return -self.kT * np.log(doping_concentration / self.ni)
    
    def calc_band_bending(self, position, surface_potential, depletion_width):
        """
        Calculate band bending as a function of position.
        
        Parameters:
        -----------
        position : array-like
            Position array (cm)
        surface_potential : float
            Surface potential (V)
        depletion_width : float
            Depletion region width (cm)
            
        Returns:
        --------
        array-like
            Band bending potential at each position (V)
        """
        # Assume parabolic potential distribution in depletion region
        bending = np.zeros_like(position)
        depletion_region = (position >= 0) & (position <= depletion_width)
        bending[depletion_region] = surface_potential * (1 - position[depletion_region]/depletion_width)**2
        return bending