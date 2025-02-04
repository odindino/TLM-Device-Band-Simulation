import numpy as np
import matplotlib.pyplot as plt

def build_device_resistor_network(N_s, N_ch, N_d,
                                  L_s, L_ch, L_d,
                                  W,
                                  R_sk_source, rho_ck_source,
                                  R_sh_channel,
                                  R_sk_drain, rho_ck_drain,
                                  V_source=0.0, V_drain=1.0):
    """
    Build and solve a 1D resistor network for a source-contact region + channel + drain-contact region.
    
    Parameters:
    -----------
    N_s, N_ch, N_d : int
        Number of segments for source contact, channel, and drain contact regions, respectively.
    L_s, L_ch, L_d : float
        Length (um) of source contact, channel, and drain contact, respectively.
    W : float
        Width (um) of the device (assume uniform).
    R_sk_source : float
        Sheet resistance under metal in source contact region [kΩ/sq].
    rho_ck_source : float
        Contact resistivity for source side [kΩ·um^2].
    R_sh_channel : float
        Sheet resistance for the channel region [kΩ/sq].
    R_sk_drain : float
        Sheet resistance under metal in drain contact region [kΩ/sq].
    rho_ck_drain : float
        Contact resistivity for drain side [kΩ·um^2].
    V_source, V_drain : float
        Fixed metal potentials at source and drain in Volts.

    Returns:
    --------
    x_nodes : 1D array of length (N_s+N_ch+N_d+1)
        The positions of each semiconductor node (0-based).
    V_nodes : 1D array of the same length
        The solved semiconductor voltage at each node.
        
    Explanation:
    ------------
    1) The total number of segments is N_s + N_ch + N_d.
       Therefore, the number of nodes is N_s + N_ch + N_d + 1.
    2) We build a conductance matrix G (size NxN) and current vector I.
       For each segment, we add a horizontal resistor (sheet) and a possible vertical resistor (contact).
    3) Boundary conditions (fixed metal potentials) are imposed by injecting current from metal to node 
       through the vertical resistors. The metal is assumed to be at V_source or V_drain.
    4) Solve G*V = I to get the node voltages in the semiconductor.
    """
    
    # Total segments in the device
    N_total_segments = N_s + N_ch + N_d
    # Total semiconductor nodes
    N_nodes = N_total_segments + 1

    # Compute segment lengths for each region
    dx_s = L_s / N_s if N_s > 0 else 0
    dx_ch = L_ch / N_ch if N_ch > 0 else 0
    dx_d = L_d / N_d if N_d > 0 else 0

    # Build x_nodes array (the position of each node along the device)
    x_nodes = [0.0]
    # Fill source region
    for i in range(N_s):
        x_nodes.append(x_nodes[-1] + dx_s)
    # Fill channel region
    for i in range(N_ch):
        x_nodes.append(x_nodes[-1] + dx_ch)
    # Fill drain region
    for i in range(N_d):
        x_nodes.append(x_nodes[-1] + dx_d)
    x_nodes = np.array(x_nodes)

    # Initialize the conductance matrix G and current vector I
    G = np.zeros((N_nodes, N_nodes))
    Ivec = np.zeros(N_nodes)

    def add_conductance(n1, n2, g):
        """
        Add a conductance g between node n1 and n2 in the matrix G.
        This corresponds to G[n1,n1]+=g, G[n2,n2]+=g, G[n1,n2]-=g, G[n2,n1]-=g
        """
        G[n1, n1] += g
        G[n2, n2] += g
        G[n1, n2] -= g
        G[n2, n1] -= g

    # Build the resistor network by iterating over each segment i
    for i in range(N_total_segments):
        # Nodes at the ends of this segment
        n_left = i
        n_right = i + 1

        # Determine if this segment is in source, channel, or drain region
        if i < N_s:
            # Source-contact region
            R_sheet = R_sk_source
            dx = dx_s
            rho_c = rho_ck_source
            metal_V = V_source
        elif i < (N_s + N_ch):
            # Channel region
            R_sheet = R_sh_channel
            dx = dx_ch
            rho_c = None     # no contact here
            metal_V = None
        else:
            # Drain-contact region
            R_sheet = R_sk_drain
            dx = dx_d
            rho_c = rho_ck_drain
            metal_V = V_drain

        # Horizontal resistor: R_h = (R_sheet * dx)/W  [kΩ]
        if dx > 0 and R_sheet > 0:
            R_h = (R_sheet * dx) / W
            G_h = 1.0 / R_h
            add_conductance(n_left, n_right, G_h)

        # Vertical contact resistor (if in contact region)
        if rho_c is not None:
            # R_perp = rho_c / (W * dx)  [kΩ]
            R_perp = rho_c / (W * dx) if dx>0 else 1e12  # avoid /0
            G_v = 1.0 / R_perp

            # We'll connect this segment to the metal potential (metal_V) 
            # in a "shunt" manner: current = G_v * (metal_V - V_node)
            # In matrix form, that means: 
            #   I[node] += G_v*metal_V
            #   G[node,node] += G_v
            # We'll apply it to node i (left node).
            # You could also do it at node i+1, or split half to each, etc.

            G[n_left, n_left] += G_v
            Ivec[n_left] += G_v * metal_V

            # Optionally, also connect node i+1 to the same contact.
            # This is a matter of how we approximate the "strip" spanning from x_i to x_{i+1}.
            # For simplicity, let's do it as well:
            G[n_right, n_right] += G_v
            Ivec[n_right] += G_v * metal_V

    # Solve G*V = Ivec => V
    V_nodes = np.linalg.solve(G, Ivec)

    return x_nodes, V_nodes


if __name__ == "__main__":
    # -----------------------------
    # Example device parameters
    # -----------------------------
    # Region lengths [um]
    L_s = 2.0      # source contact length
    L_ch = 2.0    # channel length
    L_d = 2.0      # drain contact length

    # Number of segments in each region
    N_s = 2000
    N_ch = 2000
    N_d = 2000

    # Device width [um]
    W = 95.0

    # Resistances and contact properties
    R_sk_source = 500.0      # [kΩ/sq]
    rho_ck_source = 2.0   # [kΩ·um^2]
    R_sh_channel = 81.6     # [kΩ/sq] channel
    R_sk_drain = R_sk_source       # [kΩ/sq]
    rho_ck_drain = rho_ck_source    # [kΩ·um^2]

    # Metal potentials
    V_source = 0.0
    V_drain = -0.5

    # -----------------------------
    # 1) Solve for node voltages
    # -----------------------------
    x_nodes, V_nodes = build_device_resistor_network(
        N_s, N_ch, N_d,
        L_s, L_ch, L_d,
        W,
        R_sk_source, rho_ck_source,
        R_sh_channel,
        R_sk_drain, rho_ck_drain,
        V_source, V_drain
    )

    # -----------------------------
    # 2) Compute current distribution
    # -----------------------------
    # We'll define arrays to hold horizontal/vertical current in each segment.
    N_total_segments = N_s + N_ch + N_d
    I_h = np.zeros(N_total_segments)     # horizontal current in each segment
    I_perp = np.zeros(N_total_segments)  # vertical current (if contact)

    # For convenience, define the segment center positions
    segment_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    for i in range(N_total_segments):
        n_left = i
        n_right = i + 1

        # Identify region
        if i < N_s:
            # source contact
            R_sheet = R_sk_source
            dx = L_s / N_s if N_s>0 else 0
            rho_c = rho_ck_source
            metal_V = V_source
        elif i < (N_s + N_ch):
            # channel
            R_sheet = R_sh_channel
            dx = L_ch / N_ch if N_ch>0 else 0
            rho_c = None
            metal_V = None
        else:
            # drain contact
            R_sheet = R_sk_drain
            dx = L_d / N_d if N_d>0 else 0
            rho_c = rho_ck_drain
            metal_V = V_drain

        # Horizontal current: I_h = (V_left - V_right) / R_h
        if dx > 0 and R_sheet > 0:
            R_h = (R_sheet * dx)/W  # [kΩ]
            I_h[i] = (V_nodes[n_left] - V_nodes[n_right]) / R_h  # [kA], if consistent with R in kΩ

        # Vertical current: I_perp (if contact region)
        if rho_c is not None and dx>0:
            R_perp = rho_c / (W*dx)
            # Here we demonstrate connecting metal to node i
            # In the build function, we also connected node i+1 
            # for a "double" vertical shunt. So total might be split.
            # We'll just compute one side here for demonstration:
            
            nodeV_left = V_nodes[n_left]
            I_perp_left = (nodeV_left - metal_V) / R_perp  # from node to metal

            nodeV_right = V_nodes[n_right]
            I_perp_right = (nodeV_right - metal_V) / R_perp

            # Summation if both ends are connected
            I_perp[i] = I_perp_left + I_perp_right

    # -----------------------------
    # 3) Plot results
    # -----------------------------
    # Plot the node voltage
    plt.figure(figsize=(7,5))
    plt.plot(x_nodes, V_nodes, '-o', label='Voltage at nodes')
    plt.xlabel("x (um)")
    plt.ylabel("Voltage (V)")
    plt.title("Node Voltage Distribution (Source–Channel–Drain)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the horizontal current
    plt.figure(figsize=(7,5))
    plt.plot(segment_centers, I_h, '-o', label='Horizontal current')
    plt.xlabel("x (um)")
    plt.ylabel("Current (A or kA)")
    plt.title("Horizontal Current in Each Segment")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the vertical current
    plt.figure(figsize=(7,5))
    plt.plot(segment_centers, I_perp, '-o', color='r', label='Vertical current')
    plt.xlabel("x (um)")
    plt.ylabel("Current (A or kA)")
    plt.title("Vertical Current (Injection/Extraction) in Contact Regions")
    plt.grid(True)
    plt.legend()
    plt.show()
