# band_structure_2d.py

import numpy as np
from scipy import constants

class BandStructure2D:
    """
    A class to calculate band structure for 2D semiconductor materials.
    
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
    thickness : float
        Material thickness (nm)
    T : float, optional
        Temperature (K), defaults to 300K
    """
    def __init__(self, Eg, chi, me_eff, mh_eff, thickness, T=300):
        # Material parameters
        self.Eg = Eg
        self.chi = chi
        self.me_eff = me_eff
        self.mh_eff = mh_eff
        self.thickness = thickness  # nm
        self.T = T
        
        # Physical constants
        self.k = constants.k        # Boltzmann constant (J/K)
        self.q = constants.e        # Elementary charge (C)
        self.h = constants.h        # Planck constant (J·s)
        self.hbar = constants.hbar  # Reduced Planck constant (J·s)
        self.m0 = constants.m_e     # Free electron mass (kg)
        
        # Derived parameters
        self.kT = self.k * self.T / self.q  # in eV
        
        # Calculate density of states
        self.DOS_e = self._calc_2D_DOS(self.me_eff)  # Electron DOS
        self.DOS_h = self._calc_2D_DOS(self.mh_eff)  # Hole DOS
        
    def _calc_2D_DOS(self, m_eff):
        """
        Calculate 2D density of states.
        
        Returns:
        --------
        float : DOS in states/(eV·cm²)
        """
        return m_eff * self.m0 / (np.pi * self.hbar**2) * 6.242e11  # Convert from J to eV
    
    def calc_carrier_concentration(self, E_F, carrier_type='n'):
        """
        Calculate carrier concentration for given Fermi level.
        
        Parameters:
        -----------
        E_F : float
            Fermi level position relative to conduction band minimum (for n-type)
            or valence band maximum (for p-type) (eV)
        carrier_type : str
            'n' for electrons, 'p' for holes
            
        Returns:
        --------
        float : carrier concentration (cm^-2)
        """
        if carrier_type.lower() == 'n':
            DOS = self.DOS_e
            # For electrons, integrate from CBM (E=0) to infinity
            n = DOS * self.kT * np.log(1 + np.exp(E_F / self.kT))
        else:
            DOS = self.DOS_h
            # For holes, integrate from -infinity to VBM (E=0)
            n = DOS * self.kT * np.log(1 + np.exp(-E_F / self.kT))
        
        return n
    
    def find_fermi_level(self, carrier_density, carrier_type='n', tolerance=1e-10):
        """
        Find Fermi level position for given carrier density using binary search.
        
        Parameters:
        -----------
        carrier_density : float
            2D carrier density (cm^-2)
        carrier_type : str
            'n' for n-type or 'p' for p-type
        tolerance : float
            Convergence tolerance for binary search
            
        Returns:
        --------
        float : Fermi level position relative to band edge (eV)
        """
        # Initial bounds for binary search
        E_min = -10 * self.kT
        E_max = 10 * self.kT
        
        while (E_max - E_min) > tolerance:
            E_F = (E_min + E_max) / 2
            n = self.calc_carrier_concentration(E_F, carrier_type)
            
            if n > carrier_density:
                E_max = E_F
            else:
                E_min = E_F
        
        return E_F
    
    def get_band_positions(self, x):
        """
        Get band positions relative to vacuum level.
        
        Parameters:
        -----------
        x : array-like
            Position array (nm)
            
        Returns:
        --------
        tuple : (E_c, E_v)
            Conduction and valence band energies relative to vacuum level (eV)
        """
        # For now, return flat bands (no band bending)
        E_c = -self.chi * np.ones_like(x)
        E_v = (-self.chi - self.Eg) * np.ones_like(x)
        
        return E_c, E_v