# tlm_device_model_class.py

import numpy as np

class TLMDevice:
    """
    A class to encapsulate the TLM resistor-network model for a Source-Channel-Drain device.
    
    R_sk: Sheet resistance under metal contacts (Ohm/sq)
    R_sh: Sheet resistance of channel (Ohm/sq)
    rho_ck: contact resistivity (ohm-um^2)
    L_s: Length of source region (um)
    L_ch: Length of channel region (um)
    L_d: Length of drain region (um)
    W: Width of channel (um)
    N_s: Number of segments in source region
    N_ch: Number of segments in channel region
    N_d: Number of segments in drain region
    V_source: Voltage at source contact (V)
    V_drain: Voltage at drain contact (V)
    L_TK: calibrated transfer length sqrt(rho_ck/R_sk) (um)
    L_T: Transfer length sqrt(rho_ck/R_sh) (um)
    """
    def __init__(self,
                 N_s, N_ch, N_d,
                 L_s, L_ch, L_d,
                 W,
                 R_sk_source, rho_ck_source,
                 R_sh_channel,
                 R_sk_drain, rho_ck_drain,
                 V_source=0.0, V_drain=1.0):
        """
        Initialize device parameters.
        """
        self.N_s = N_s
        self.N_ch = N_ch
        self.N_d = N_d
        self.L_s = L_s
        self.L_ch = L_ch
        self.L_d = L_d
        self.W = W
        self.R_sk_source = R_sk_source
        self.rho_ck_source = rho_ck_source
        self.R_sh_channel = R_sh_channel
        self.R_sk_drain = R_sk_drain
        self.rho_ck_drain = rho_ck_drain
        self.V_source = V_source
        self.V_drain = V_drain

        # Internal storage
        self.x_nodes = None
        self.V_nodes = None
        self.I_h = None
        self.I_perp = None
        self.segment_centers = None

    def solve_voltage(self):
        """
        Solve the 1D resistor network to obtain node voltages self.V_nodes.
        Also store self.x_nodes.
        """
        N_s, N_ch, N_d = self.N_s, self.N_ch, self.N_d
        L_s, L_ch, L_d = self.L_s, self.L_ch, self.L_d
        W = self.W
        R_sk_source, rho_ck_source = self.R_sk_source, self.rho_ck_source
        R_sh_channel = self.R_sh_channel
        R_sk_drain, rho_ck_drain = self.R_sk_drain, self.rho_ck_drain
        V_source, V_drain = self.V_source, self.V_drain

        # total segments
        N_total_segments = N_s + N_ch + N_d
        N_nodes = N_total_segments + 1

        dx_s = L_s / N_s if N_s>0 else 0
        dx_ch = L_ch / N_ch if N_ch>0 else 0
        dx_d = L_d / N_d if N_d>0 else 0

        x_nodes = [0.0]
        # fill source
        for i in range(N_s):
            x_nodes.append(x_nodes[-1] + dx_s)
        # fill channel
        for i in range(N_ch):
            x_nodes.append(x_nodes[-1] + dx_ch)
        # fill drain
        for i in range(N_d):
            x_nodes.append(x_nodes[-1] + dx_d)
        x_nodes = np.array(x_nodes)
        self.x_nodes = x_nodes

        G = np.zeros((N_nodes, N_nodes))
        Ivec = np.zeros(N_nodes)

        def add_conductance(n1, n2, g):
            G[n1, n1] += g
            G[n2, n2] += g
            G[n1, n2] -= g
            G[n2, n1] -= g

        for i in range(N_total_segments):
            n_left = i
            n_right = i+1

            if i < N_s:
                R_sheet = R_sk_source
                dx = dx_s
                rho_c = rho_ck_source
                metal_V = V_source
            elif i < (N_s + N_ch):
                R_sheet = R_sh_channel
                dx = dx_ch
                rho_c = None
                metal_V = None
            else:
                R_sheet = R_sk_drain
                dx = dx_d
                rho_c = rho_ck_drain
                metal_V = V_drain

            # horizontal
            if dx>0 and R_sheet>0:
                R_h = (R_sheet * dx)/W
                G_h = 1.0/R_h
                add_conductance(n_left, n_right, G_h)

            # vertical
            if rho_c is not None and dx>0:
                R_perp = rho_c/(W*dx)
                G_v = 1.0/R_perp
                # left node
                G[n_left, n_left] += G_v
                Ivec[n_left] += G_v*metal_V
                # right node
                G[n_right, n_right] += G_v
                Ivec[n_right] += G_v*metal_V

        V_nodes = np.linalg.solve(G, Ivec)
        self.V_nodes = V_nodes

    def compute_currents(self):
        """
        Compute horizontal and vertical currents for each segment,
        store them in self.I_h, self.I_perp, and segment centers in self.segment_centers.
        """
        N_s, N_ch, N_d = self.N_s, self.N_ch, self.N_d
        L_s, L_ch, L_d = self.L_s, self.L_ch, self.L_d
        W = self.W
        R_sk_source, rho_ck_source = self.R_sk_source, self.rho_ck_source
        R_sh_channel = self.R_sh_channel
        R_sk_drain, rho_ck_drain = self.R_sk_drain, self.rho_ck_drain
        V_source, V_drain = self.V_source, self.V_drain

        x_nodes, V_nodes = self.x_nodes, self.V_nodes
        N_total_segments = N_s + N_ch + N_d

        I_h = np.zeros(N_total_segments)
        I_perp = np.zeros(N_total_segments)

        segment_centers = 0.5*(x_nodes[:-1] + x_nodes[1:])

        for i in range(N_total_segments):
            n_left = i
            n_right = i+1

            if i < N_s:
                R_sheet = R_sk_source
                dx = L_s / N_s if N_s>0 else 0
                rho_c = rho_ck_source
                metal_V = V_source
            elif i < (N_s + N_ch):
                R_sheet = R_sh_channel
                dx = L_ch / N_ch if N_ch>0 else 0
                rho_c = None
                metal_V = None
            else:
                R_sheet = R_sk_drain
                dx = L_d / N_d if N_d>0 else 0
                rho_c = rho_ck_drain
                metal_V = V_drain

            # horizontal
            if dx>0 and R_sheet>0:
                R_h = (R_sheet*dx)/W
                I_h[i] = (V_nodes[n_left] - V_nodes[n_right])/R_h

            # vertical
            if rho_c is not None and dx>0:
                R_perp = rho_c/(W*dx)
                I_left = (V_nodes[n_left] - metal_V)/R_perp
                I_right = (V_nodes[n_right] - metal_V)/R_perp
                I_perp[i] = I_left + I_right

        self.I_h = I_h
        self.I_perp = I_perp
        self.segment_centers = segment_centers

