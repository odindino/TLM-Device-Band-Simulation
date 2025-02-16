import numpy as np
from scipy import constants as const
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp

class Heterostructure:
    """
    A class to calculate electronic properties of 2D material heterostructures
    Handles various types of junctions: semiconductor-semiconductor, metal-semiconductor
    """
    def __init__(self, materials, junction_type='semiconductor'):
        """
        Initialize heterostructure with list of material objects
        
        Parameters:
        materials (list): List of BandStructure2D objects in junction order
        junction_type (str): Type of junction ('semiconductor' or 'metal')
        """
        self.materials = materials
        self.junction_type = junction_type
        
        # Physical constants
        self.q = const.e
        self.kb = const.k
        self.eps0 = const.epsilon_0
        self.h = const.h
        
        # Initialize interface parameters
        self.interface_states = {}
        self.dipoles = {}
        self.built_in_potential = None
        
        # Calculate basic junction properties
        self.analyze_junction()
        
    def analyze_junction(self):
        """Analyze basic properties of the junction"""
        # Get material parameters
        self.params = []
        for mat in self.materials:
            self.params.append(mat.get_material_parameters())
            
        # Calculate band offsets
        self.calculate_band_offsets()
        
        # Calculate built-in potential
        self.calculate_built_in_potential()
        
        # Initialize space charge effects
        self.initialize_space_charge()
        
    def calculate_band_offsets(self):
        """Calculate band offsets at interfaces"""
        self.conduction_offsets = []
        self.valence_offsets = []
        
        for i in range(len(self.materials)-1):
            mat1 = self.materials[i]
            mat2 = self.materials[i+1]
            
            # Calculate natural band offsets based on electron affinity
            dEc = mat2.chi/self.q - mat1.chi/self.q
            dEv = (mat2.chi/self.q + mat2.Eg/self.q) - (mat1.chi/self.q + mat1.Eg/self.q)
            
            self.conduction_offsets.append(dEc)
            self.valence_offsets.append(dEv)
            
    def calculate_built_in_potential(self):
        """Calculate built-in potential across junction"""
        if len(self.materials) < 2:
            return
            
        # For semiconductor junction
        if self.junction_type == 'semiconductor':
            mat1, mat2 = self.materials[0], self.materials[1]
            
            # Calculate built-in potential from Fermi level difference
            self.built_in_potential = abs(mat1.Ef - mat2.Ef)/self.q
            
            # Consider doping effects
            if hasattr(mat1, 'n2D') and hasattr(mat2, 'n2D'):
                n1, n2 = mat1.n2D, mat2.n2D
                if n1 > 0 and n2 > 0:
                    kT = self.kb * mat1.T
                    self.built_in_potential += kT/self.q * np.log(n1/n2)
                    
    def initialize_space_charge(self):
        """Initialize space charge region calculations"""
        if self.built_in_potential is None:
            return
            
        # Calculate depletion widths for each material
        self.depletion_widths = []
        for i, mat in enumerate(self.materials[:-1]):
            eps1 = mat.eps
            eps2 = self.materials[i+1].eps
            
            # Effective screening length (Thomas-Fermi)
            if hasattr(mat, 'n2D') and mat.n2D > 0:
                n = mat.n2D
                m_eff = mat.me if mat.carrier_type == 'n' else mat.mh
                lambda_TF = np.sqrt(eps1 * self.kb * mat.T / (2 * self.q**2 * n))
            else:
                lambda_TF = np.sqrt(eps1 * self.kb * mat.T / (2 * self.q**2 * 1e16))  # Assume minimal carrier density
                
            self.depletion_widths.append(lambda_TF)
            
    def solve_poisson(self, x_mesh, boundary_conditions):
        """
        Solve 1D Poisson equation across heterostructure
        
        Parameters:
        x_mesh (array): Spatial mesh points
        boundary_conditions (tuple): (left_value, right_value) for potential
        
        Returns:
        tuple: (x_points, potential, electric_field)
        """
        dx = x_mesh[1] - x_mesh[0]
        N = len(x_mesh)
        
        # Set up coefficient matrix for Poisson equation
        matrix_diagonals = np.zeros((3, N))
        matrix_diagonals[0, 1:] = 1/(dx**2)  # Upper diagonal
        matrix_diagonals[1, :] = -2/(dx**2)  # Main diagonal
        matrix_diagonals[2, :-1] = 1/(dx**2)  # Lower diagonal
        
        # Set up charge density vector
        rho = np.zeros(N)
        for i, x in enumerate(x_mesh):
            # Determine which material region we're in
            mat_index = self.find_material_index(x)
            if mat_index >= 0:
                mat = self.materials[mat_index]
                if hasattr(mat, 'n2D'):
                    rho[i] = -mat.n2D * self.q / mat.eps if mat.carrier_type == 'n' else mat.n2D * self.q / mat.eps
                    
        # Solve system
        V = solve_banded((1, 1), matrix_diagonals, -rho)
        
        # Calculate electric field
        E = -(V[2:] - V[:-2])/(2*dx)
        
        return x_mesh, V, E
        
    def find_material_index(self, position):
        """Find which material a given position corresponds to"""
        # Implementation depends on how we define material regions
        # This is a placeholder
        return 0
        
    def get_band_profile(self, x_points):
        """
        Calculate band profile across heterostructure
        
        Parameters:
        x_points (array): Position points to evaluate
        
        Returns:
        tuple: Arrays of (conduction_band, valence_band) energies
        """
        # Get potential profile
        _, V, _ = self.solve_poisson(x_points, (0, 0))
        
        Ec = np.zeros_like(x_points)
        Ev = np.zeros_like(x_points)
        
        # Fill in band profiles
        for i, x in enumerate(x_points):
            mat_index = self.find_material_index(x)
            if mat_index >= 0:
                mat = self.materials[mat_index]
                Ec[i] = mat.Ec - self.q * V[i]
                Ev[i] = mat.Ev - self.q * V[i]
                
        return Ec, Ev