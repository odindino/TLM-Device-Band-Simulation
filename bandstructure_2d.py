import numpy as np
from scipy import constants as const
from scipy.linalg import eigh

class BandStructure2D:
    """
    A class to calculate the intrinsic electronic band structure of 2D materials
    with focus on MoS2 and similar TMDCs
    """
    def __init__(self,
                 material_name="MoS2",
                 lattice_a=3.16,        # Lattice constant in Angstrom
                 layer_thickness=0.65,   # Layer thickness in nm
                 bandgap=1.8,           # Direct bandgap in eV
                 effective_mass_e=0.35,  # Electron effective mass (m*/m0)
                 effective_mass_h=0.54,  # Hole effective mass (m*/m0)
                 spin_orbit=0.15,       # Spin-orbit coupling in eV
                 dielectric=7.6,        # Dielectric constant
                 electron_affinity=4.0,  # Electron affinity in eV
                 temperature=300,       # Temperature in Kelvin
                 carrier_density=0.0,   # 2D carrier density in cm^-2
                 carrier_type='n'):     # 'n' for electrons, 'p' for holes
        """Initialize intrinsic material parameters"""
        # Material identification
        self.material = material_name
        
        # Convert units to SI
        self.a = lattice_a * 1e-10      # meters
        self.d = layer_thickness * 1e-9  # meters
        self.Eg = bandgap * const.e      # Joules
        self.me = effective_mass_e * const.m_e
        self.mh = effective_mass_h * const.m_e
        self.SO = spin_orbit * const.e
        self.eps = dielectric * const.epsilon_0
        self.chi = electron_affinity * const.e
        self.T = temperature
        self.n2D = carrier_density * 1e4  # Convert to m^-2
        self.carrier_type = carrier_type
        
        # Physical constants
        self.kB = const.k
        self.hbar = const.hbar
        self.q = const.e
        
        # Calculate quantum confinement
        self.E_conf = (self.hbar * np.pi)**2 / (2 * self.me * self.d**2)
        
        # Initialize intrinsic band structure
        self.initialize_bands()
    
    def initialize_bands(self):
        """Calculate band structure including carrier density effects"""
        # Set vacuum level as reference (E = 0)
        self.E_vacuum = 0
        
        # Calculate intrinsic band edges relative to vacuum level
        self.Ec0 = -self.chi  # Conduction band edge
        self.Ev0 = -(self.chi + self.Eg/self.q)  # Valence band edge
        
        # Add quantum confinement effect
        self.Ec = self.Ec0 + self.E_conf/self.q
        self.Ev = self.Ev0 - self.E_conf/self.q
        
        # Consider spin-orbit splitting in valence band
        self.Ev_up = self.Ev + self.SO/(2*self.q)
        self.Ev_down = self.Ev - self.SO/(2*self.q)
        
        # Calculate Fermi level position based on doping type and carrier density
        if self.carrier_type == 'n':
            # For n-type doping, Fermi level moves towards conduction band
            DOS_2D = self.me / (np.pi * self.hbar**2)
            delta_E = self.n2D / DOS_2D
            self.Ef = self.Ec + delta_E * self.q
        else:
            # For p-type doping, Fermi level moves towards valence band
            DOS_2D = self.mh / (np.pi * self.hbar**2)
            delta_E = self.n2D / DOS_2D
            self.Ef = self.Ev - delta_E * self.q
            
    def calculate_fermi_level(self):
        """Calculate Fermi level position based on carrier density"""
        if self.carrier_type == 'n':
            # For n-type, solve for Fermi level in conduction band
            E_test = np.linspace(self.Ec, self.Ec + 1.0, 1000)
            n_test = np.array([self.carrier_density(E) for E in E_test])
            # Find Fermi level where calculated density matches target density
            idx = np.argmin(np.abs(n_test - self.n2D))
            self.Ef = E_test[idx]
        else:
            # For p-type, solve for Fermi level in valence band
            E_test = np.linspace(self.Ev - 1.0, self.Ev, 1000)
            p_test = np.array([self.carrier_density(E, type='hole') for E in E_test])
            idx = np.argmin(np.abs(p_test - self.n2D))
            self.Ef = E_test[idx]
            
    def calculate_band_renormalization(self):
        """Calculate band gap renormalization due to carrier density"""
        if self.n2D > 0:
            # Calculate screening length
            rs = 1 / (np.sqrt(np.pi * self.n2D))  # Average carrier spacing
            qTF = self.q**2 * self.get_density_of_states(self.Ef) / (2 * self.eps)  # Thomas-Fermi wave vector
            
            # Calculate band gap renormalization
            dEg = -self.q**2 / (4 * np.pi * self.eps) * qTF * (1 + np.pi/8)
            
            # Update band edges
            if self.carrier_type == 'n':
                self.Ec += dEg/self.q  # Conduction band shifts down
            else:
                self.Ev -= dEg/self.q  # Valence band shifts up
                
            # Update spin-orbit split bands
            self.Ev_up = self.Ev + self.SO/(2*self.q)
            self.Ev_down = self.Ev - self.SO/(2*self.q)
    
    def carrier_density(self, energy, carrier_type='electron'):
        """
        Calculate carrier density at a given energy level
        
        Parameters:
        energy (float): Energy level in eV
        carrier_type (str): Type of carrier ('electron' or 'hole')
        
        Returns:
        float: Carrier density in m^-2
        """
        # Convert energy to Joules
        E = energy * self.q
        
        # Get relevant parameters based on carrier type
        if carrier_type == 'electron':
            m_eff = self.me
            E_edge = self.Ec
        else:  # hole
            m_eff = self.mh
            E_edge = self.Ev
            
        # Calculate 2D density of states
        DOS_2D = m_eff / (np.pi * self.hbar**2)
        
        # Calculate Fermi-Dirac distribution
        if carrier_type == 'electron':
            fd = 1 / (1 + np.exp((E - self.Ef)/(self.kB * self.T)))
        else:
            fd = 1 / (1 + np.exp((self.Ef - E)/(self.kB * self.T)))
            
        # Calculate carrier density by integrating DOS * f(E)
        # For 2D case, this simplifies due to constant DOS
        density = DOS_2D * (abs(E - E_edge)/self.q) * fd
        
        return density
    
    def get_band_dispersion(self, k_points):
        """
        Calculate E(k) band dispersion
        
        Parameters:
        k_points (array): Wave vectors in m^-1
        
        Returns:
        tuple: Arrays of energies (in eV) for conduction and valence bands
        """
        # Conduction band
        E_c = self.Ec + self.hbar**2 * k_points**2 / (2 * self.me * self.q)
        
        # Valence bands with spin-orbit splitting
        E_v_up = self.Ev_up - self.hbar**2 * k_points**2 / (2 * self.mh * self.q)
        E_v_down = self.Ev_down - self.hbar**2 * k_points**2 / (2 * self.mh * self.q)
        
        return E_c, E_v_up, E_v_down
    
    def get_material_parameters(self):
        """
        Return basic material parameters
        
        Returns:
        dict: Material parameters including band edges, effective masses, etc.
        """
        params = {
            'material': self.material,
            'electron_affinity': self.chi/self.q,
            'bandgap': self.Eg/self.q,
            'conduction_band_edge': self.Ec,
            'valence_band_edge': self.Ev,
            'spin_orbit_splitting': self.SO/self.q,
            'effective_mass_electron': self.me/const.m_e,
            'effective_mass_hole': self.mh/const.m_e,
            'dielectric_constant': self.eps/const.epsilon_0
        }
        return params