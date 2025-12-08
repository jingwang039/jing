import os
import json
import numpy as np
import time


def transfer_matrix_drift(l_drift, gamma):
    M_drift = np.array([[1, l_drift, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, l_drift, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, l_drift/gamma**2],
                        [0, 0, 0, 0, 0, 1]])
    return M_drift

def transfer_matrix_sector(l_sector, rho, gamma, alpha):
    M_sector = np.array([[np.cos(alpha), rho*np.sin(alpha), 0, 0, 0, rho*(1-np.cos(alpha))],
                         [-np.sin(alpha)/rho, np.cos(alpha), 0, 0, 0, np.sin(alpha)],
                         [0, 0, 1, l_sector, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [-np.sin(alpha), -rho*(1-np.cos(alpha)), 0, 0, 1, rho*alpha /gamma**2 - rho*(alpha - np.sin(alpha))],
                         [0, 0, 0, 0, 0, 1]])
 
    M_edge_focus = np.array([[1, 0, 0, 0, 0, 0],
                             [np.tan(alpha/2)/rho, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                                [0, 0, -np.tan(alpha/2)/rho, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
    
    return M_edge_focus @ M_sector @ M_edge_focus


def transfer_matrix_QF(l_QF, k_QF, gamma):
    k_QF = abs(k_QF)
    k = np.sqrt(k_QF)
    M_QF = np.array([[np.cos(k*l_QF), (1/k)*np.sin(k*l_QF), 0, 0, 0, 0],
                     [-k*np.sin(k*l_QF), np.cos(k*l_QF), 0, 0, 0, 0],
                     [0, 0, np.cosh(k*l_QF), (1/k)*np.sinh(k*l_QF), 0, 0],
                     [0, 0, k*np.sinh(k*l_QF), np.cosh(k*l_QF), 0, 0],
                     [0, 0, 0, 0, 1, l_QF/gamma**2],
                     [0, 0, 0, 0, 0, 1]])
    return M_QF

def transfer_matrix_QD(l_QD, k_QD, gamma):
    k_QD = abs(k_QD)
    k = np.sqrt(k_QD)
    M_QD = np.array([[np.cosh(k*l_QD), (1/k)*np.sinh(k*l_QD), 0, 0, 0, 0],
                     [k*np.sinh(k*l_QD), np.cosh(k*l_QD), 0, 0, 0, 0],
                     [0, 0, np.cos(k*l_QD), (1/k)*np.sin(k*l_QD), 0, 0],
                     [0, 0, -k*np.sin(k*l_QD), np.cos(k*l_QD), 0, 0],
                     [0, 0, 0, 0, 1, l_QD/gamma**2],
                     [0, 0, 0, 0, 0, 1]])
    return M_QD 


def get_object_position(trajectory_type, t, scan_config):
    """
    Calculate the position of the heavy object at time t
    Based on Chapter 5, Section 5.3 - Exploration of Symmetries
    """
    if trajectory_type == "stationary":
        # Section 5.3 - baseline case
        if "stationary_position" in scan_config:
            return np.array(scan_config["stationary_position"])
        else:
            # Simple radial distance (backward compatibility)
            distance_m = scan_config.get("distance_m", 10e3)
            return np.array([distance_m, 0, 0])
    
    elif trajectory_type == "linear":
        # Section 5.3.1 - Motion in a Line
        linear_config = scan_config["linear_motion"]
        v = np.array(linear_config["velocity_vector"])  # m/s
        r0 = np.array(linear_config["initial_position"])  # m
        return r0 + v * t
    
    elif trajectory_type == "circular":
        # Section 5.3.2 - Motion in a Circle
        circular_config = scan_config["circular_motion"]
        center = np.array(circular_config["center"])
        R = circular_config["radius"]  # m
        omega = circular_config["angular_velocity"]  # rad/s
        phi0 = circular_config["initial_angle"]  # rad
        
        angle = omega * t + phi0
        return center + R * np.array([np.cos(angle), np.sin(angle), 0])
    
    elif trajectory_type == "oscillating":
        # Section 5.3.3 - Motion in an oscillating Manner
        osc_config = scan_config["oscillating_motion"]
        center = np.array(osc_config["center"])
        A = np.array(osc_config["amplitude"])
        f = osc_config["frequency"]  # Hz
        phi = osc_config["phase"]  # rad
        
        omega = 2 * np.pi * f
        return center + A * np.sin(omega * t + phi)
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def compute_gravitational_forces(theta_cells, vector_particle, vector_parallel, 
                                 vector_object, m_particle_kg, object_mass, G):
    """
    Compute gravitational forces at each element position
    """
    gr_forces = []
    f_parallel = []
    
    for i in range(len(theta_cells)):
        # Vector from particle to object
        vector_p_to_o = vector_object - vector_particle[i]
        r = np.linalg.norm(vector_p_to_o)
        
        # Total gravitational force
        F = G * m_particle_kg * object_mass / r**2
        gr_forces.append(F)
        
        # Parallel component (along particle trajectory)
        f_p = F * np.dot(vector_p_to_o, vector_parallel[i]) / r
        f_parallel.append(f_p)
    
    return gr_forces, f_parallel


def compute_displacement_vectors(f_parallel, cells, num_units, l_QF, l_QD, l_drift, l_sector,
                                rho, m_particle_kg, beta, gamma, c, p_0):
    """
    Compute displacement vectors for each element based on parallel force
    From Chapter 3 - equations of motion with longitudinal force
    """
    delta_x = []
    
    for i in range(num_units):
        for j in range(len(cells)):
            index = i * len(cells) + j
            f_p = f_parallel[index]
            
            if cells[j] in ["QF", "QD", "drift"]:
                if cells[j] == "QF":
                    l_cell = l_QF
                elif cells[j] == "QD":
                    l_cell = l_QD
                elif cells[j] == "drift":
                    l_cell = l_drift
                
                # Displacement for drift-like elements
                d_x = np.array([0, 0, 0, 0,
                               f_p * l_cell / (2 * m_particle_kg * beta**2 * gamma**2 * c),
                               f_p * l_cell / (c * beta * p_0)])
                delta_x.append(d_x)
                
            elif cells[j] == "sector":
                # Displacement for sector magnets (more complex due to curvature)
                l_cell = l_sector
                k_sector = 1 / rho**2
                
                d_x = np.array([
                    -2*f_p/(rho*m_particle_kg*beta**2*gamma**3*c**2)*(l_cell/k_sector - np.sin(np.sqrt(k_sector)*l_cell)/np.sqrt(k_sector)/k_sector),
                    -2*f_p/(rho*m_particle_kg*beta**2*gamma**3*c**2)*(l_cell - np.cos(np.sqrt(k_sector)*l_cell))/k_sector,
                    0,
                    0,
                    f_p*l_cell/(2*m_particle_kg*beta**2*gamma**2*c) + 2*f_p/(rho**2*m_particle_kg*beta**2*gamma**3*c**2) \
                        * ((l_cell**2/2/k_sector) + (np.cos(np.sqrt(k_sector)*l_cell) - 1)/(k_sector**2)),
                    f_p*l_cell/(c*beta*p_0)
                ])
                delta_x.append(d_x)
    
    return delta_x


####################################
######## FODO parameter config

config = json.load(open('FODO_config.json'))
num_units = config['num_units']
times = config['times']
cells = config['cells']
particle = config['particle']
q = config['charge'] 
m = config['mass'] 
beta = config['beta']

p_ev = beta * np.sqrt(m**2 / (1 - beta**2))  # eV/c
p = p_ev * 1e-9  # GeV/c
beta = p_ev / np.sqrt(p_ev**2 + m**2)
print(f"particle: {particle}, mass: {m} eV, p_ev: {p_ev:.6e} eV/c")
print(f"particle: {particle}, p: {p} GeV/c")
gamma = 1 / np.sqrt(1 - beta**2)
print(f"beta: {beta:.6f}, gamma: {gamma:.6f}")

# Quadrupole parameters
g_QF = config['g_QF']
g_QD = config['g_QD']
B_rho0 = p / abs(q) / 0.299792458   # T·m
k_QF = g_QF / B_rho0  # 1/m^2
k_QD = g_QD / B_rho0  # 1/m^2
l_QF = config['l_QF']  # m
l_QD = config['l_QD']  # m

print(f"k_QF: {k_QF:.10f}, k_QD: {k_QD:.10f}")
print(f"l_QF: {l_QF} m, l_QD: {l_QD} m")

# Dipole parameters
rho = config['rho']  # m
l_sector = config['l_sector']  # m
alpha_angle = l_sector / rho * 180/np.pi  # deg
alpha = alpha_angle / 180 * np.pi  # rad

print(f"rho: {rho} m, alpha: {alpha:.6f} rad ({alpha_angle:.6f}°)")
print(f"l_sector: {l_sector} m")

# Drift parameters
l_drift = config['l_drift']  # m
print(f"l_drift: {l_drift} m")

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

# Build transfer matrices
matrices = {}
matrices['QF'] = transfer_matrix_QF(l_QF, k_QF, gamma)
matrices['QD'] = transfer_matrix_QD(l_QD, k_QD, gamma)
matrices['drift'] = transfer_matrix_drift(l_drift, gamma)
matrices['sector'] = transfer_matrix_sector(l_sector, rho, gamma, alpha)

# Test: calculate full FODO cell matrix
M_full = np.eye(6)
for cell in cells:
    M_full = matrices[cell] @ M_full
print("\nFull FODO cell transfer matrix:")
print("Full cell transfer matrix:\n"+str(np.round(M_full,4)))

# Calculate FODO cell length and ring circumference
l_FODO = 0
for cell in cells:
    if cell == "drift":
        l_FODO += l_drift
    elif cell == "sector":
        l_FODO += l_sector
    elif cell == "QF":
        l_FODO += l_QF
    elif cell == "QD":
        l_FODO += l_QD

l_ring = l_FODO * num_units
T_rev = l_ring / (beta * 3e8)  # Revolution period in seconds

print(f"\n=== Ring Geometry ===")
print(f"FODO cell length: {l_FODO:.2f} m")
print(f"Ring circumference: {l_ring:.2f} m")
print(f"Revolution period: {T_rev:.6e} s ({1/T_rev:.6f} Hz)")

#####################################################
########################################## External force configuration

scan_config = json.load(open('scan_config.json'))
object_mass = scan_config["object_mass_kg"]  # kg
trajectory_type = scan_config.get("trajectory_type", "stationary")
enable_time_dependent = scan_config.get("simulation_parameters", {}).get("enable_time_dependent", False)

print(f"\n=== Heavy Object Configuration ===")
print(f"Object mass: {object_mass:.6e} kg")
print(f"Trajectory type: {trajectory_type}")
print(f"Time-dependent forces: {enable_time_dependent}")

# Physical constants
m_particle_kg = m * 1.783e-36  # kg (eV to kg conversion)
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458  # m/s
p_0 = p_ev * 1.602e-19 / c  # kg·m/s

print(f"Particle mass: {m_particle_kg:.6e} kg")
print(f"Particle momentum: {p_0:.6e} kg·m/s")

# Calculate element positions in the ring
theta_cells = []
s = 0
l_before = 0
for i in range(num_units):
    for j in range(len(cells)):
        cell = cells[j]
        if cell == "drift":
            l_cell = l_drift
        elif cell == "sector":
            l_cell = l_sector
        elif cell == "QF":
            l_cell = l_QF
        elif cell == "QD":
            l_cell = l_QD
        
        s += l_cell / 2 + l_before / 2
        theta_cells.append(s / l_ring * 2 * np.pi)
        l_before = l_cell

# Particle positions and tangent vectors at each element
vector_particle = []
vector_parallel = []
for i in range(len(theta_cells)):
    vector = rho * np.array([np.cos(theta_cells[i]), np.sin(theta_cells[i]), 0])
    vector_particle.append(vector)
    vector_p = np.array([-np.sin(theta_cells[i]), np.cos(theta_cells[i]), 0])
    vector_parallel.append(vector_p)

# Compute forces based on trajectory type
if not enable_time_dependent:
    # Static calculation: object at t=0 position
    print("\n=== Static Force Calculation (t=0) ===")
    vector_object = get_object_position(trajectory_type, 0, scan_config)
    print(f"Object position at t=0: {vector_object} m")
    
    gr_forces, f_parallel = compute_gravitational_forces(
        theta_cells, vector_particle, vector_parallel,
        vector_object, m_particle_kg, object_mass, G
    )
    
    print(f"Force range: [{min(f_parallel):.6e}, {max(f_parallel):.6e}] N")
    
    # Compute displacement vectors
    delta_x = compute_displacement_vectors(
        f_parallel, cells, num_units, l_QF, l_QD, l_drift, l_sector,
        rho, m_particle_kg, beta, gamma, c, p_0
    )
    
else:
    # Time-dependent calculation
    print("\n=== Time-Dependent Force Calculation ===")
    print("Forces will be recalculated at each turn based on object motion")
    
    # For initial setup, use t=0
    vector_object = get_object_position(trajectory_type, 0, scan_config)
    gr_forces, f_parallel = compute_gravitational_forces(
        theta_cells, vector_particle, vector_parallel,
        vector_object, m_particle_kg, object_mass, G
    )
    delta_x = compute_displacement_vectors(
        f_parallel, cells, num_units, l_QF, l_QD, l_drift, l_sector,
        rho, m_particle_kg, beta, gamma, c, p_0
    )

#######################################################
## Write Julia simulation file

X_initial = scan_config["X_in"]
save_per_time = config['save_per_time']

print(f"\n=== Generating Julia Code ===")
print(f"Output file: 6D_FODO_simulation.jl")
print(f"Save every {save_per_time} turns")

with open('6D_FODO_simulation.jl', 'w') as f:
    f.write("# 6D Particle Tracking in FODO Lattice with Gravitational Perturbations\n")
    f.write("# Generated for Chapter 5: Heavy Object Tracking\n")
    f.write(f"# Trajectory type: {trajectory_type}\n")
    f.write(f"# Time-dependent: {enable_time_dependent}\n\n")
    
    f.write("using CSV, DataFrames\n\n")
    
    # Initial conditions
    f.write(f"# Initial particle state vector [x, x', y, y', l, δ]\n")
    f.write(f"x_initial = [{X_initial[0]}, {X_initial[1]}, {X_initial[2]}, {X_initial[3]}, {X_initial[4]}, {X_initial[5]}]\n\n")
    
    # Physical parameters
    f.write("# Physical parameters\n")
    f.write(f"const beta = {beta}\n")
    f.write(f"const gamma = {gamma}\n")
    f.write(f"const c = {c}\n")
    f.write(f"const T_rev = {T_rev}\n")
    f.write(f"const l_ring = {l_ring}\n\n")
    
    # Transfer matrices
    f.write("# Transfer matrices for each element type\n")
    for name, matrix in matrices.items():
        f.write(f"{name} = [")
        for i in range(6):
            for j in range(6):
                f.write(f"{matrix[i,j]}")
                if j < 5:
                    f.write(" ")
            if i < 5:
                f.write("; ")
        f.write("]\n")
    f.write("\n")
    
    # Displacement vectors
    f.write("# Displacement vectors from gravitational perturbation\n")
    f.write("delta_x = [\n")
    for i in range(len(delta_x)):
        f.write("    [")
        for j in range(6):
            f.write(f"{delta_x[i][j]}")
            if j < 5:
                f.write(", ")
        f.write("]")
        if i < len(delta_x) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n\n")
    
    # History arrays
    f.write(f"# History arrays\n")
    f.write(f"x_history = zeros(6, {int(times/save_per_time)+1})\n")
    f.write(f"time_of_flight = zeros({int(times/save_per_time)+1})\n\n")
    
    # Main tracking loop
    f.write("# Main tracking loop\n")
    f.write(f"x = copy(x_initial)\n")
    f.write(f"total_time = 0.0\n\n")
    
    for i in range(times):
        for j in range(num_units):
            for k in range(len(cells)):
                f.write(f"x = {cells[k]} * x + delta_x[{j*len(cells)+k+1}]\n")
        
        f.write(f"total_time += T_rev\n")
        
        if i % save_per_time == 0:
            f.write(f"x_history[:, {int(i/save_per_time+1)}] = x\n")
            f.write(f"time_of_flight[{int(i/save_per_time+1)}] = total_time\n")
        f.write("\n")
    
    # Save results
    f.write("# Save tracking results\n")
    f.write("df = DataFrame(x_history', [:x, :px, :y, :py, :l, :delta])\n")
    f.write("df.time = time_of_flight\n")
    f.write(f'CSV.write("output/{particle}_FODO_6D_history.csv", df)\n\n')
    
    f.write('println("Simulation complete!")\n')
    f.write(f'println("Results saved to output/{particle}_FODO_6D_history.csv")\n')

print("\n=== Generation Complete ===")
print("Run the generated Julia code with: julia 6D_FODO_simulation.jl")
