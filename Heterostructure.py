import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

class Heterostructure:
    """
    A class to simulate band alignment and interface effects of multiple 2D regions.
    
    This class handles the interaction between different semiconductor regions,
    calculating band alignment and interface effects based on physical principles.
    
    Parameters:
    -----------
    regions : list
        List of BandStructure2D objects representing different material regions
    region_lengths : list
        List of lengths for each region (nm)
    epsilon_r : float, optional
        Average relative dielectric constant for interface calculations
    T : float, optional
        Temperature (K)
    """
    
    def __init__(self, regions, region_lengths, epsilon_r=8.0, T=300):
        if len(regions) < 1:
            raise ValueError("At least one region must be provided")
        if len(region_lengths) != len(regions):
            raise ValueError("Number of region lengths must match number of regions")
            
        self.regions = regions
        self.T = T
        self.epsilon_r = epsilon_r
        self.epsilon = self.epsilon_r * constants.epsilon_0
        self.kT = constants.k * T / constants.e
        
        # Set up region positions
        current_position = 0
        for region, length in zip(regions, region_lengths):
            region.x_start = current_position
            region.x_end = current_position + length
            current_position += length
            
        # Calculate interface properties
        self.interfaces = self._identify_interfaces()
        
    def _identify_interfaces(self):
        """
        Identify interfaces between regions and calculate their properties.
        
        Returns:
        --------
        list
            List of dictionaries containing interface properties
        """
        interfaces = []
        for i in range(len(self.regions) - 1):
            region1 = self.regions[i]
            region2 = self.regions[i + 1]
            
            interface = {
                'position': region1.x_end,
                'left_region': region1,
                'right_region': region2,
                'potential': self._calc_interface_potential(region1, region2),
                'screening_length': self._calc_screening_length(region1, region2)
            }
            interfaces.append(interface)
            
        return interfaces
    
    def _calc_interface_potential(self, region1, region2):
        """
        Calculate the intrinsic potential difference at the interface between regions.
        
        Parameters:
        -----------
        region1, region2 : BandStructure2D
            Adjacent regions forming the interface
            
        Returns:
        --------
        float
            Interface potential (eV)
        """
        # Work function difference
        work_function_diff = region2.work_function - region1.work_function
        
        # Fermi level difference
        fermi_difference = region2.E_F - region1.E_F
        
        # Built-in potential from carrier density difference
        carrier_ratio = region2.carrier_density / region1.carrier_density
        built_in = self.kT * np.log(carrier_ratio)
        
        # Total interface potential including work function effect
        total_potential = work_function_diff + fermi_difference + built_in
        
        return total_potential
    
    def _calc_screening_length(self, region1, region2):
        """
        Calculate characteristic screening length for the interface.
        
        Parameters:
        -----------
        region1, region2 : BandStructure2D
            Adjacent regions forming the interface
            
        Returns:
        --------
        float
            Screening length (nm)
        """
        # Use maximum carrier density for screening length calculation
        max_density = max(region1.carrier_density, region2.carrier_density)
        
        # Calculate 2D screening length
        lambda_s = np.sqrt(self.epsilon * self.kT / 
                          (2 * constants.e * constants.e * max_density)) * 1e9
        
        return lambda_s
    
    def calc_band_bending(self, x, interface):
        """
        Calculate band bending profile near an interface.
        
        Parameters:
        -----------
        x : array-like
            Position array (nm)
        interface : dict
            Interface properties
            
        Returns:
        --------
        array-like
            Band bending potential (eV)
        """
        # Calculate relative distances from interface
        distance = x - interface['position']
        lambda_s = interface['screening_length']
        
        # Initialize bending array
        bending = np.zeros_like(x)
        
        # Calculate exponential decay on both sides
        left_mask = distance < 0
        right_mask = distance >= 0
        
        # Apply band bending with screening
        bending[left_mask] = interface['potential'] * \
            np.exp(distance[left_mask] / lambda_s)
        bending[right_mask] = interface['potential'] * \
            np.exp(-distance[right_mask] / lambda_s)
        
        return bending
    
    def calculate_bands(self, x):
        """
        Calculate complete band structure including interface effects.
        
        Parameters:
        -----------
        x : array-like
            Position array (nm)
            
        Returns:
        --------
        tuple : (E_c, E_v, E_f)
            Band energies and Fermi level (eV)
        """
        # Initialize energy arrays
        E_c = np.zeros_like(x)
        E_v = np.zeros_like(x)
        E_f = np.zeros_like(x)
        
        # Get intrinsic band positions for each region
        for region in self.regions:
            E_c_region, E_v_region, mask = region.get_band_positions(x)
            E_c[mask] = E_c_region[mask]
            E_v[mask] = E_v_region[mask]
            E_f[mask] = region.E_F
        
        # Add interface effects
        total_bending = np.zeros_like(x)
        for interface in self.interfaces:
            bending = self.calc_band_bending(x, interface)
            total_bending += bending
        
        # Apply total band bending to all bands
        E_c += total_bending
        E_v += total_bending
        E_f += total_bending
        
        return E_c, E_v, E_f
    
    def plot_bands(self, x_range=None, show_fermi=True):
        """
        Plot the complete band structure with interface effects.
        
        Parameters:
        -----------
        x_range : tuple, optional
            (x_min, x_max) for plotting range (nm)
        show_fermi : bool, optional
            Whether to show Fermi level in plot
        """
        if x_range is None:
            x_start = min(region.x_start for region in self.regions)
            x_end = max(region.x_end for region in self.regions)
            x = np.linspace(x_start, x_end, 1000)
        else:
            x = np.linspace(x_range[0], x_range[1], 1000)
            
        E_c, E_v, E_f = self.calculate_bands(x)
        
        plt.figure(figsize=(12, 8))
        plt.plot(x, E_c, 'b-', label='Conduction Band')
        plt.plot(x, E_v, 'r-', label='Valence Band')
        if show_fermi:
            plt.plot(x, E_f, 'g--', label='Fermi Level')
        plt.axhline(y=0, color='k', linestyle=':', label='Vacuum Level')
        
        # Add region labels and boundaries
        for region in self.regions:
            region_center = (region.x_start + region.x_end) / 2
            plt.text(region_center, -7.5, region.name, ha='center')
            plt.axvline(x=region.x_start, color='gray', linestyle=':')
        
        plt.axvline(x=self.regions[-1].x_end, color='gray', linestyle=':')
        
        plt.xlabel('Position (nm)')
        plt.ylabel('Energy (eV)')
        plt.title('Band Alignment with Interface Effects')
        plt.legend()
        plt.grid(True)
        plt.ylim(-8, 1)
        
        plt.show()