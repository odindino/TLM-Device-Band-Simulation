import numpy as np
from scipy import constants

class BandStructure2D:
    """
    A class to calculate fundamental band structure properties for a single 2D semiconductor region.
    
    This class handles the basic electronic properties of a 2D semiconductor material,
    including band positions, carrier densities, and Fermi levels for an isolated region.
    Focus is on the intrinsic properties of the material without considering interface effects.
    
    Parameters:
    -----------
    name : str
        Identifier for the material region (e.g., "Source", "Channel", "Drain")
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
    carrier_density : float
        2D carrier concentration (cm^-2)
    carrier_type : str, optional
        Type of majority carriers ('n' for electrons, 'p' for holes), defaults to 'n'
    T : float, optional
        Temperature (K), defaults to 300K
    """
    
    def __init__(self, name, Eg, chi, me_eff, mh_eff, thickness, carrier_density, work_function, carrier_type='n', T=300):
        # Material identification
        self.name = name
        self.x_start = 0  # Will be set by Heterostructure class
        self.x_end = 1    # Will be set by Heterostructure class
        
        # Material parameters
        self.Eg = Eg
        self.chi = chi
        self.me_eff = me_eff
        self.mh_eff = mh_eff
        self.thickness = thickness
        self.carrier_density = carrier_density
        self.carrier_type = carrier_type
        self.work_function = work_function
        self.T = T
        
        # Physical constants
        self.k = constants.k        # Boltzmann constant (J/K)
        self.q = constants.e        # Elementary charge (C)
        self.h = constants.h        # Planck constant (J·s)
        self.hbar = constants.hbar  # Reduced Planck constant (J·s)
        self.m0 = constants.m_e     # Free electron mass (kg)
        
        # Derived parameters
        self.kT = self.k * self.T / self.q  # Thermal energy (eV)
        
        # Calculate density of states
        self.DOS_e = self._calc_2D_DOS(self.me_eff)  # Electron DOS
        self.DOS_h = self._calc_2D_DOS(self.mh_eff)  # Hole DOS
        
        # Calculate Fermi level
        self.E_F_relative = self.find_fermi_level(carrier_density, carrier_type)
        self.E_F = -self.chi + self.E_F_relative
        
    def _calc_2D_DOS(self, m_eff):
        """
        Calculate 2D density of states.
        
        Parameters:
        -----------
        m_eff : float
            Effective mass (relative to free electron mass)
            
        Returns:
        --------
        float
            Density of states in states/(eV·cm²)
        """
        return m_eff * self.m0 / (np.pi * self.hbar**2) * 6.242e11

    def calc_carrier_concentration(self, E_F, carrier_type='n'):
        """
        Calculate carrier concentration for given Fermi level.
        
        Parameters:
        -----------
        E_F : float
            Fermi level position relative to band edge (eV)
        carrier_type : str, optional
            'n' for electrons or 'p' for holes
            
        Returns:
        --------
        float
            Carrier concentration (cm^-2)
        """
        if carrier_type.lower() == 'n':
            DOS = self.DOS_e
            n = DOS * self.kT * np.log(1 + np.exp(E_F / self.kT))
        else:
            DOS = self.DOS_h
            n = DOS * self.kT * np.log(1 + np.exp(-E_F / self.kT))
        return n

    def find_fermi_level(self, carrier_density, carrier_type='n', tolerance=1e-10):
        """
        Find Fermi level position for given carrier density using binary search.
        
        Parameters:
        -----------
        carrier_density : float
            2D carrier density (cm^-2)
        carrier_type : str, optional
            'n' for n-type or 'p' for p-type
        tolerance : float, optional
            Convergence tolerance for binary search
            
        Returns:
        --------
        float
            Fermi level position relative to band edge (eV)
        """
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
        Get intrinsic band positions relative to vacuum level.
        
        This method returns the basic band positions for a single region,
        without considering any interface effects or band bending.
        
        Parameters:
        -----------
        x : array-like
            Position array (nm)
            
        Returns:
        --------
        tuple : (E_c, E_v, mask)
            E_c : array-like
                Conduction band energies relative to vacuum level (eV)
            E_v : array-like
                Valence band energies relative to vacuum level (eV)
            mask : array-like
                Boolean mask indicating positions within this region
        """
        mask = (x >= self.x_start) & (x <= self.x_end)
        E_c = np.zeros_like(x)
        E_v = np.zeros_like(x)
        
        # Set band positions according to carrier type
        if self.carrier_type.lower() == 'n':
            E_c[mask] = -self.chi - self.E_F_relative
            E_v[mask] = -self.chi - self.Eg - self.E_F_relative
        else:
            E_c[mask] = -self.chi + self.E_F_relative
            E_v[mask] = -self.chi - self.Eg + self.E_F_relative
        
        return E_c, E_v, mask