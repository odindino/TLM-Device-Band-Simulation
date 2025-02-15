import numpy as np
import matplotlib.pyplot as plt
from BandStructure import BandStructure

def plot_band_demo():
    # Initialize band structure with MoS2 parameters from fort.9
    mos2 = BandStructure(
        Eg=2.3,           # Band gap (eV)
        chi=3.7,          # Electron affinity (eV)
        me_eff=0.35,      # Electron effective mass
        mh_eff=0.54,      # Hole effective mass (using heavy hole mass)
        T=300             # Temperature (K)
    )
    
    # Set up spatial parameters
    x = np.linspace(-10, 30, 200)  # Position array (nm)
    
    # Calculate Fermi level for given doping
    donor_concentration = 1.42e17    # cm^-3 from fort.9
    Ef = mos2.calc_fermi_level(donor_concentration, doping_type='n')
    
    # Define surface potential and depletion width for band bending
    surface_potential = 0.5  # V (example value)
    depletion_width = 20    # nm (example value)
    
    # Calculate band bending
    band_bending = mos2.calc_band_bending(x, surface_potential, depletion_width)
    
    # Calculate band energies
    E_c = -mos2.chi + band_bending              # Conduction band
    E_v = -mos2.chi - mos2.Eg + band_bending    # Valence band
    E_f = -mos2.chi + Ef                        # Fermi level
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, E_c, 'b-', label='Conduction Band')
    plt.plot(x, E_v, 'r-', label='Valence Band')
    plt.axhline(y=E_f, color='g', linestyle='--', label='Fermi Level')
    
    # Add vacuum level
    plt.axhline(y=0, color='k', linestyle=':', label='Vacuum Level')
    
    plt.xlabel('Position (nm)')
    plt.ylabel('Energy (eV)')
    plt.title('MoS2 Band Structure')
    plt.legend()
    plt.grid(True)
    
    # Set reasonable y-axis limits
    plt.ylim(-8, 1)
    
    plt.show()

if __name__ == "__main__":
    plot_band_demo()