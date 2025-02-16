import matplotlib.pyplot as plt
import matplotlib
from TLM_Device_Model import TLMDevice
import numpy as np

def setup_plot_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 21
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 15
    matplotlib.rcParams['svg.fonttype'] = 'none'

def plot_voltage_distribution(device, x_range=None, title="TLM Model Voltage Distribution", filename=None):
    # 調整長寬比為更適合並排的尺寸
    fig, ax = plt.subplots(figsize=(6, 8))  # 修改圖片尺寸比例
    
    # 繪製數據並減小標記大小
    ax.plot(device.x_nodes, device.V_nodes, 'o-', label='Voltage', color='#1f77b4', markersize=2)
    
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(title)
    
    if x_range is not None:
        mask = (device.x_nodes >= x_range[0]) & (device.x_nodes <= x_range[1])
        y_data = device.V_nodes[mask]
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_margin = (y_max - y_min) * 0.05
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 優化網格線
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    
    # 優化圖例位置
    ax.legend(loc='best', frameon=True, framealpha=1)
    
    # 調整邊距
    plt.tight_layout(pad=1.2)
    
    if filename:
        # 提高解析度到 600 DPI
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.show()
    else:
        plt.show()
    
    return fig

def main():
    device = TLMDevice(
        N_s=8000, N_ch=4000, N_d=8000,
        L_s=2.0, L_ch=2.0, L_d=2.0,
        W=95.0,
        R_sk_source=581000.0, rho_ck_source=2.466,
        R_sh_channel=81600,
        R_sk_drain=581000.0, rho_ck_drain=2.466,
        V_source=0.0, V_drain=-0.5
    )

    device.solve_voltage()
    device.compute_currents()
    setup_plot_style()

    plot_voltage_distribution(device, filename='voltage_full_range.svg')
    plot_voltage_distribution(device, x_range=(1.98, 2.02), filename='voltage_source_channel.svg')
    plot_voltage_distribution(device, x_range=(3.98, 4.02), filename='voltage_drain_channel.svg')

if __name__ == "__main__":
    main()