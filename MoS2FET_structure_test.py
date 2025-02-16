import numpy as np
from scipy import constants as const
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class BandBendingSimulator:
    """
    Simulator for band bending in MoS2 FET with different doping concentrations
    """
    def __init__(self, params=None):
        # Physical constants
        self.q = const.e
        self.kb = const.k
        self.eps0 = const.epsilon_0
        self.h = const.h
        self.T = 300  # Temperature in Kelvin
        
        # MoS2 material parameters
        self.eps_r = 7.6  # Relative permittivity of MoS2
        self.chi = 4.0 * self.q  # Electron affinity
        self.Eg = 2.3 * self.q   # Bandgap
        self.Nc = 2.3e19         # Effective DOS in conduction band (m^-3)
        self.thickness = 0.65e-9  # MoS2 thickness (m)
        
        # Update with custom parameters if provided
        if params:
            for key, value in params.items():
                setattr(self, key, value)
                
        # Derived quantities
        self.eps = self.eps0 * self.eps_r
        self.ni = np.sqrt(self.Nc * self.Nc) * np.exp(-self.Eg/(2*self.kb*self.T))
        
    def poisson_eq(self, x, V, Nd1, Nd2):
        """Poisson equation for the heterojunction"""
        # Position-dependent doping profile
        if x < 0:  # Under Bi region
            Nd = Nd1
        else:      # Channel region
            Nd = Nd2
            
        # Space charge density
        rho = self.q * (Nd - self.ni * np.exp((self.q*V - self.Eg/2)/(self.kb*self.T)))
        
        # Poisson equation: d²V/dx² = -ρ/ε
        d2V = -rho/self.eps
        return d2V
    
    def solve_band_bending(self, x_range, Nd_Bi, Nd_channel):
        """
        Solve for band bending using depletion approximation
        
        Parameters:
        x_range: spatial points for calculation
        Nd_Bi: doping concentration under Bi (m^-3)
        Nd_channel: doping concentration in channel (m^-3)
        """
        # Built-in potential from doping difference
        kT = self.kb * self.T
        V_bi = kT/self.q * np.log(Nd_Bi/Nd_channel)
        
        # Depletion width estimation
        W = np.sqrt(2*self.eps*V_bi/(self.q*Nd_channel))
        
        # Initialize potential and field arrays
        V = np.zeros_like(x_range)
        E = np.zeros_like(x_range)
        
        # Calculate potential profile
        for i, x in enumerate(x_range):
            if x < 0:  # Under Bi region
                V[i] = V_bi * (1 - np.exp(x/W))
            else:      # Channel region
                V[i] = V_bi * np.exp(-x/W)
        
        # Calculate band energies
        Ec = -self.chi + self.q*V
        Ef = -self.chi - self.Eg/2 + self.kb*self.T*np.log(Nd_channel/self.ni)
        Ev = -self.chi - self.Eg + self.q*V
        
        return Ec/self.q, Ef/self.q, Ev/self.q, V_bi
        
    def plot_bands(self, x_range, Nd_Bi, Nd_channel):
        """Plot band diagram showing band bending"""
        Ec, Ef, Ev, V_bi = self.solve_band_bending(x_range, Nd_Bi, Nd_channel)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_range*1e9, Ec, 'b-', label='$E_c$')
        plt.plot(x_range*1e9, Ef, 'k--', label='$E_F$')
        plt.plot(x_range*1e9, Ev, 'r-', label='$E_v$')
        
        # Add annotations
        barrier_height = np.max(Ec) - np.min(Ec)
        plt.text(0, np.max(Ec)+0.1, f'Barrier Height = {barrier_height*1000:.0f} meV')
        
        plt.axvline(x=0, color='gray', linestyle=':')
        plt.text(-20, np.min(Ec)-0.2, '$\mathrm{MoS_2}$ $(n^{++})$')
        plt.text(20, np.min(Ec)-0.2, '$\mathrm{MoS_2}$ channel')
        
        plt.xlabel('Position (nm)')
        plt.ylabel('Energy (eV)')
        plt.title('Band Bending at $\mathrm{MoS_2}$ Junction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        return plt.gcf()

def main():
    # Initialize simulator
    simulator = BandBendingSimulator()
    
    # Set up spatial grid (-50nm to 50nm)
    x_range = np.linspace(-50e-9, 50e-9, 1000)
    
    # Define doping concentrations (converting from cm^-3 to m^-3)
    Nd_Bi = 1.42e18 * 1e6      # Under Bi region (n++)
    Nd_channel = 1.42e17 * 1e6  # Channel region
    
    # Plot band diagram
    simulator.plot_bands(x_range, Nd_Bi, Nd_channel)
    plt.show()

if __name__ == "__main__":
    main()