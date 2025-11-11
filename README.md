# TLM Device Band Simulation

This software package implements a Transfer Length Method (TLM) device model to simulate voltage and current distributions in source-channel-drain semiconductor devices. The model particularly focuses on the phenomena occurring at the metal-semiconductor interfaces in two-dimensional field-effect transistors.

## Overview

The TLM Device Band Simulation tool enables researchers to:

1. Model voltage distributions across source, channel, and drain regions
2. Calculate current flow patterns (horizontal and vertical)
3. Visualize electric potential profiles with high-quality publication-ready plots
4. Extract key device parameters such as contact resistivity, sheet resistance, and transfer length

This software was designed to support research published in Nature journals on contact resistivity in two-dimensional WSe₂ field-effect transistors.

## System Requirements

### Software Dependencies
- Python 3.8 or higher
- NumPy 1.20.0 or higher
- Matplotlib 3.5.0 or higher
- Vue.js 3.0 or higher (for web interface)

### Operating Systems
- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (10.15+)
- Windows 10/11

### Hardware Requirements
- Minimum 4GB RAM
- 2GHz dual-core processor or higher
- 500MB free disk space

No specialized hardware is required. All simulations run on standard CPU.

## Installation Guide

### Setting up the Python Environment

1. Clone the repository:
```bash
git clone https://github.com/odindino/TLM-Device-Band-Simulation
cd TLM-Device-Band-Simulation
```

2. Create a virtual environment (recommended):
```bash
python -m venv tlm_env
```

3. Activate the environment:
   - On Windows:
   ```bash
   tlm_env\Scripts\activate
   ```
   - On macOS/Linux:
   ```bash
   source tlm_env/bin/activate
   ```

4. Install required packages:
```bash
pip install numpy matplotlib
```

The typical installation time is less than 5 minutes on a standard desktop computer.

## Demo

### Running the Demo Simulation

To run a demonstration of the TLM device model with predefined parameters:

```bash
python TLM-model-simulation_for_manuscript_v2.py
```

This will:
1. Create a TLM device model with default parameters
2. Solve for voltage distribution across the device
3. Compute current flow patterns
4. Generate visualization plots

### Expected Output

The demo will generate three plots:
1. Node voltage distribution across source, channel, and drain regions
2. Horizontal current flow
3. Vertical current flow

Example voltage distribution output is included as `voltage_distribution_final.svg` in the repository.

Expected runtime for the demo is less than 10 seconds on a typical desktop computer.

## Instructions for Use

### Running Simulations with Custom Parameters

To run the simulation with your own device parameters, modify the device initialization in `TLM-model-simulation.py`:

```python
device = TLMDevice(
    N_s=10000, N_ch=5000, N_d=10000,    # Number of segments
    L_s=2.0, L_ch=2.0, L_d=2.0,         # Region lengths (μm)
    W=95.0,                             # Channel width (μm)
    R_sk_source=581000.0, rho_ck_source=2.466,  # Source contact parameters
    R_sh_channel=81600,                 # Channel sheet resistance
    R_sk_drain=581000.0, rho_ck_drain=2.466,    # Drain contact parameters
    V_source=0.0, V_drain=-0.5          # Applied voltages
)
```

Adjust these parameters according to your specific device characteristics.

### Using Your Own Data

To simulate devices based on experimental data:
1. Prepare a data file with your measured device parameters
2. Create a script that reads this data and initializes the TLMDevice with appropriate values
3. Call the `solve_voltage()` and `compute_currents()` methods
4. Visualize the results using the provided plotting functions

### Generating Publication-Quality Figures

The package includes the `plot_voltage_distribution()` function in `TLM-model-simulation_for_manuscript_v2.py` that creates publication-ready plots with:
- Professionally styled legends and labels
- Region highlighting
- Inset zoom plots for interface regions
- Consistent decimal formatting
- High-resolution SVG output

```python
from TLM_Device_Model import TLMDevice
from TLM-model-simulation_for_manuscript_v2 import plot_voltage_distribution

# Initialize and solve device model
device = TLMDevice(...)
device.solve_voltage()
device.compute_currents()

# Create publication-ready figure
plot_voltage_distribution(device, title="TLM Model Voltage Distribution", 
                         filename="figure1.svg")
```

## Physics Background

This simulation is based on the resistive network model for metal-semiconductor contacts in two-dimensional devices. The model accounts for:

1. Different sheet resistances in contact and channel regions
2. Contact resistivity at metal-semiconductor interfaces
3. Transfer length effects
4. Current crowding phenomena

The model solves a system of linear equations derived from Kirchhoff's laws to determine the voltage distribution across the device. The physics is described in detail in:

Moon, I., Choi, M.S., Lee, S. et al. Analytical measurements of contact resistivity in two-dimensional WSe₂ field-effect transistors. 2D Mater. 8, 045019 (2021).

## License

This software is released under the GNU General Public License v3.0.

## Contact

For questions or support, please open an issue on the GitHub repository:
https://github.com/odindino/TLM-Device-Band-Simulation

## Supplementary Information

### Theoretical Background

The TLM Device model implements a modified resistive network that accounts for the unique properties of 2D semiconductor devices. Unlike bulk semiconductor models, this approach considers:

1. Different sheet resistances under metal contacts (R_sk) and in channel regions (R_sh)
2. Contact resistivity (ρ_ck) at metal-semiconductor interfaces
3. Transfer length (L_Tk) calculation using both TLM and contact-end resistance methods

The voltage distribution V(x) and current distribution I(x) are derived from second-order differential equations:

```
d²I/dx² = I(x)·R_sk/ρ_ck = I(x)/L_Tk²
```

Where L_Tk = √(ρ_ck/R_sk) is the transfer length.

The contact resistance distribution R(x) is given by:

```
R(x)·W = (ρ_ck/L_Tk)·[coth(L/L_Tk)·cosh(x/L_Tk) - sinh(x/L_Tk)]
```

These equations form the theoretical foundation of the simulation tool.
