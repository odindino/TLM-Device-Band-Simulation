# main_sim_class.py

import matplotlib.pyplot as plt
from TLM_Device_Model import TLMDevice

def main():
    # Initialize device
    device = TLMDevice(
        N_s=2000, N_ch=2000, N_d=2000,
        L_s=2.0, L_ch=2.0, L_d=2.0,
        W=95.0,
        R_sk_source=500.0, rho_ck_source=2.0,
        R_sh_channel=81.6,
        R_sk_drain=500.0, rho_ck_drain=2.0,
        V_source=0.0, V_drain=-0.5
    )

    # Solve for voltages
    device.solve_voltage()
    # Compute currents
    device.compute_currents()

    # Plot
    x_nodes = device.x_nodes
    V_nodes = device.V_nodes
    seg_x = device.segment_centers
    I_h = device.I_h
    I_perp = device.I_perp

    plt.figure()
    plt.plot(x_nodes, V_nodes, 'o-', label='Voltage')
    plt.xlabel('x (um)')
    plt.ylabel('Voltage (V)')
    plt.title('Node Voltage')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(seg_x, I_h, 'o-', label='I_h')
    plt.xlabel('x (um)')
    plt.ylabel('Current (A or kA)')
    plt.title('Horizontal Current')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(seg_x, I_perp, 'o-', label='I_perp')
    plt.xlabel('x (um)')
    plt.ylabel('Current (A or kA)')
    plt.title('Vertical Current')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()
