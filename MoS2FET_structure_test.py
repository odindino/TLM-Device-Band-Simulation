from BandStructure2D import BandStructure2D
from Heterostructure import Heterostructure
import matplotlib.pyplot as plt
import numpy as np

def test_mos2_fet_basic():
    """
    Basic test function for MoS2 FET band structure simulation.
    Creates a three-region device (source/channel/drain) with different doping levels.
    """
    
    # Create source region (heavily doped)
    source_mos2 = BandStructure2D(
        name="Source",
        Eg=2.3,
        chi=3.7,
        me_eff=0.35,
        mh_eff=0.54,
        thickness=0.7,
        carrier_density=1e11,
        work_function=4.5  # Metal-modified work function
    )
    
    # Channel region (intrinsic)
    channel_mos2 = BandStructure2D(
        name="Channel",
        Eg=2.3,
        chi=3.7,
        me_eff=0.35,
        mh_eff=0.54,
        thickness=0.7,
        carrier_density=1e10,
        work_function=4.0  # Intrinsic work function
    )
    
    # Drain region (with metal contact)
    drain_mos2 = BandStructure2D(
        name="Drain",
        Eg=2.3,
        chi=3.7,
        me_eff=0.35,
        mh_eff=0.54,
        thickness=0.7,
        carrier_density=1e11,
        work_function=4.5  # Metal-modified work function
    )
    
    # Define region lengths (nm)
    region_lengths = [2000, 2000, 2000]
    
    # Create and plot heterostructure
    fet = Heterostructure([source_mos2, channel_mos2, drain_mos2], region_lengths)
    fet.plot_bands()



def main():
    """
    Main function to run different tests.
    """
    print("Running basic MoS2 FET test...")
    test_mos2_fet_basic()


if __name__ == "__main__":
    main()