import matplotlib.pyplot as plt
import matplotlib
from TLM_Device_Model import TLMDevice
import numpy as np
import matplotlib.ticker as ticker

def setup_plot_style():
    """Set plotting style to meet academic publication standards"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 21
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
def plot_voltage_distribution(device, x_range=None, title=None, filename=None):
    """Create publication-ready TLM device voltage distribution plot
    
    Parameters:
    device - TLMDevice instance
    x_range - Optional x-axis range limit
    title - Plot title (set to None to omit)
    filename - If provided, save plot to this file
    """
    # Create square figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot voltage distribution with solid line
    ax.plot(device.x_nodes, device.V_nodes, '-', label='Voltage', color='#1f77b4', linewidth=2)
    
    # Calculate total length
    total_length = device.L_s + device.L_ch + device.L_d
    
    # Add background shading for different regions
    ax.axvspan(0, device.L_s, alpha=0.15, color='orange')
    ax.axvspan(device.L_s, device.L_s + device.L_ch, alpha=0.15, color='lightgreen')
    ax.axvspan(device.L_s + device.L_ch, total_length, alpha=0.15, color='orange')
    
    # Add region labels above the plot (outside the frame)
    source_center = device.L_s/2
    channel_center = device.L_s + device.L_ch/2
    drain_center = device.L_s + device.L_ch + device.L_d/2
    
    # Position labels above the top border of the plot
    y_position = ax.get_ylim()[1] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    ax.text(source_center, y_position, 'Source', color='darkorange', ha='center', 
            va='bottom', fontsize=12, weight='bold', transform=ax.transData)
    ax.text(channel_center, y_position, 'Channel', color='green', ha='center', 
            va='bottom', fontsize=12, weight='bold', transform=ax.transData)
    ax.text(drain_center, y_position, 'Drain', color='darkorange', ha='center', 
            va='bottom', fontsize=12, weight='bold', transform=ax.transData)
    
    # Set plot title if provided
    if title:
        fig.suptitle(title, y=0.95)
    
    # Set consistent decimal format for all axes
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    # Handle custom x-range if specified
    if x_range is not None:
        mask = (device.x_nodes >= x_range[0]) & (device.x_nodes <= x_range[1])
        y_data = device.V_nodes[mask]
        if len(y_data) > 0:
            y_min, y_max = np.min(y_data), np.max(y_data)
            y_margin = (y_max - y_min) * 0.05
            ax.set_xlim(x_range)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
    # Add inset zoom plots for transition regions
    if x_range is None:  # Only add insets to main figure
        # Source-channel interface (top right)
        inset_source = ax.inset_axes([0.65, 0.65, 0.25, 0.25])
        inset_source.plot(device.x_nodes, device.V_nodes, '-', color='#1f77b4', linewidth=1.5)
        source_channel_x = (1.98, 2.02)
        inset_source.set_xlim(source_channel_x)
        
        # Use consistent formatting
        inset_source.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        inset_source.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        # Dynamically set y-axis range
        mask = (device.x_nodes >= source_channel_x[0]) & (device.x_nodes <= source_channel_x[1])
        y_data = device.V_nodes[mask]
        if len(y_data) > 0:
            y_min, y_max = np.min(y_data), np.max(y_data)
            y_margin = (y_max - y_min) * 0.1
            inset_source.set_ylim(y_min - y_margin, y_max + y_margin)
        inset_source.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
        ax.indicate_inset_zoom(inset_source, edgecolor="green")
        
        # Channel-drain interface (bottom left)
        inset_drain = ax.inset_axes([0.15, 0.25, 0.25, 0.25])
        inset_drain.plot(device.x_nodes, device.V_nodes, '-', color='#1f77b4', linewidth=1.5)
        drain_channel_x = (3.98, 4.02)
        inset_drain.set_xlim(drain_channel_x)
        
        # Use consistent formatting
        inset_drain.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        inset_drain.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        # Dynamically set y-axis range
        mask = (device.x_nodes >= drain_channel_x[0]) & (device.x_nodes <= drain_channel_x[1])
        y_data = device.V_nodes[mask]
        if len(y_data) > 0:
            y_min, y_max = np.min(y_data), np.max(y_data)
            y_margin = (y_max - y_min) * 0.1
            inset_drain.set_ylim(y_min - y_margin, y_max + y_margin)
        inset_drain.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
        ax.indicate_inset_zoom(inset_drain, edgecolor="green")
    
    # Add axis labels
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Voltage (V)')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    
    # Position legend in the lower right (away from data and labels)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Adjust margins
    plt.tight_layout()
    
    # Add additional padding at the top for region labels
    plt.subplots_adjust(top=0.9)
    
    if filename:
        # Save as high-resolution SVG
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig

def main():
    # Create TLM device model
    device = TLMDevice(
        N_s=8000, N_ch=4000, N_d=8000,
        L_s=2.0, L_ch=2.0, L_d=2.0,
        W=95.0,
        R_sk_source=334680.0, rho_ck_source=1.420,
        R_sh_channel=74784,
        R_sk_drain=334680.0, rho_ck_drain=1.420,
        V_source=0.0, V_drain=-0.5
    )

    # Solve for voltage distribution
    device.solve_voltage()
    device.compute_currents()
    
    # Set plotting style
    setup_plot_style()

    # Generate figure with no title (or optionally use "TLM Model Voltage Distribution" as title)
    plot_voltage_distribution(device, title=None, filename='voltage_distribution_final.svg')

if __name__ == "__main__":
    main()