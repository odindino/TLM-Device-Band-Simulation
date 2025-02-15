# test_band_structure.py

import numpy as np
import matplotlib.pyplot as plt
from BandStructure2D import BandStructure2D

def test_mos2_bands():
    # Initialize MoS2 parameters
    mos2 = BandStructure2D(
        Eg=2.3,           # Band gap (eV)
        chi=3.7,          # Electron affinity (eV)
        me_eff=0.35,      # Electron effective mass
        mh_eff=0.54,      # Hole effective mass
        thickness=0.7,     # Thickness (nm)
        T=300             # Temperature (K)
    )
    
    # Set up spatial coordinates
    x = np.linspace(-10, 30, 200)  # Position array (nm)
    
    # Calculate band positions
    E_c, E_v = mos2.get_band_positions(x)
    
    # Calculate Fermi level for given carrier density
    carrier_density_2d = 1e10  # cm^-2
    E_F_relative = mos2.find_fermi_level(carrier_density_2d, carrier_type='n')
    E_F = -mos2.chi + E_F_relative  # Convert to absolute energy
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(x, E_c, 'b-', label='Conduction Band')
    plt.plot(x, E_v, 'r-', label='Valence Band')
    plt.axhline(y=E_F, color='g', linestyle='--', label='Fermi Level')
    plt.axhline(y=0, color='k', linestyle=':', label='Vacuum Level')
    
    plt.xlabel('Position (nm)')
    plt.ylabel('Energy (eV)')
    plt.title('MoS2 Band Structure (2D)')
    plt.legend()
    plt.grid(True)
    
    # Set reasonable y-axis limits
    plt.ylim(-8, 1)
    
    plt.show()
    
    # Print some useful information
    print(f"Electron DOS: {mos2.DOS_e:.2e} states/(eV·cm²)")
    print(f"Hole DOS: {mos2.DOS_h:.2e} states/(eV·cm²)")
    print(f"Fermi level position relative to CBM: {E_F_relative:.3f} eV")
    print(f"Absolute Fermi level position: {E_F:.3f} eV")

if __name__ == "__main__":
    test_mos2_bands()