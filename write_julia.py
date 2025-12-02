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



####################################
######## fodo parameter config

config = json.load(open('FODO_config.json'))
num_units = config['num_units']
times = config['times']
cells = config['cells']
particle = config['particle']
# p = config['p'] # GeV/c
q = config['charge'] 
m = config['mass'] 
beta = config['beta']

p_ev = beta * np.sqrt(m**2 / (1 - beta**2))  # GeV/c
p = p_ev*1e-9  # GeV/c
beta = p_ev / np.sqrt(p_ev**2 + m**2)
print(f"particle: {particle}, mass: {m}, p_ev: {p_ev}")
gamma = 1 / np.sqrt(1 - beta**2)
print(f"beta: {beta}, gamma: {gamma}")


# quadrupole 
g_QF = config['g_QF']
g_QD = config['g_QD']
B_rho0 = p / abs(q) / 0.299792458   # T·m
k_QF = g_QF / B_rho0  # 1/m^2
k_QD = g_QD / B_rho0  # 1/m^2
l_QF = config['l_QF']  # m
l_QD = config['l_QD']  # m

print(f"k_QF: {k_QF}, k_QD: {k_QD}, l_QF: {l_QF}, l_QD: {l_QD}")


# dipole
rho = config['rho']  # m
l_sector = config['l_sector']  # m
alpha_angle = l_sector / rho * 180/np.pi  # deg
# alpha_angle = config['alpha_angle'] 
alpha = alpha_angle/180*np.pi # rad

print(f"rho: {rho}, alpha: {alpha}, l_sector: {l_sector}")

# drift
l_drift = config['l_drift']  # m

print(f"l_drift: {l_drift}")

if not os.path.exists('output'):
    os.makedirs('output')
#matrices

M_drift = transfer_matrix_drift(l_drift, gamma)
M_sector = transfer_matrix_sector(l_sector, rho, gamma, alpha)
M_QF = transfer_matrix_QF(l_QF, k_QF, gamma)
M_QD = transfer_matrix_QD(l_QD, k_QD, gamma)

matrices = {"drift": M_drift, "sector": M_sector, "QF": M_QF, "QD": M_QD}

print("drift:\n"+str(M_drift))
print("edge focus* sector:\n"+str(M_sector))
print("QF:\n"+str(M_QF))
print("QD:\n"+str(M_QD))


n_turns = times * len(cells) * num_units
print(f"times: {times}")
print(f"num_units: {num_units}")
print(f"n_cells: {len(cells)}")
print(f"n_turns: {n_turns}")
print("FODO cells: "+str(cells))

M_full = np.eye(6)
for cell in cells:
    M_full = matrices[cell] @ M_full
print("Full cell transfer matrix:\n"+str(M_full))

#####################################################
########################################## external force
scan_config = json.load(open('scan_config.json'))
distance_m = scan_config["distance_m"]  # m
object_mass = scan_config["object_mass_kg"]  # kg
m_particle_kg = m * 1.783e-36  # kg
print (m_particle_kg)
G = 6.67430e-11  # m^3 kg^-1 s^-2

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
        s += l_cell / 2 + l_before /2
        theta_cells.append(s / l_ring * 2 * np.pi)
        l_before = l_cell


vector_particle = []
vector_parallel = []
for i in range(len(theta_cells)):
    vector= rho * np.array([np.cos(theta_cells[i]), np.sin(theta_cells[i]), 0])
    vector_particle.append(vector)
    vector_p = np.array([-np.sin(theta_cells[i]), np.cos(theta_cells[i]), 0])
    vector_parallel.append(vector_p)

vector_object = np.array([distance_m, 0, 0])

vector_p_to_o = []
for i in range(len(theta_cells)):
    vector_p_to_o.append(vector_object - vector_particle[i])

gr_forces = []
for i in range(len(theta_cells)):
    r = np.linalg.norm(vector_p_to_o[i])
    F = G * m_particle_kg * object_mass / r**2
    gr_forces.append(F)

f_parallel = []
for i in range(len(theta_cells)):
    f_p = gr_forces[i] * np.dot(vector_p_to_o[i], vector_parallel[i]) / np.linalg.norm(vector_p_to_o[i])
    f_parallel.append(f_p)



delta_x = []
c = 299792458  # m/s
p_0 = p_ev * 1.602e-19 / c  # kg·m/s
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
            d_x = np.array([0,0,0,0,
                            f_parallel[index]*l_cell/(2*m_particle_kg*beta*beta*gamma*gamma*c),
                            f_parallel[index]*l_cell/(c*beta*p_0)]) 
            delta_x.append(d_x)
        if cells[j] == "sector":
            l_cell = l_sector
            k_sector = 1/rho**2
            d_x = np.array([-2*f_parallel[index]/(rho*m_particle_kg*beta*beta*gamma*gamma*gamma*c*c)*(l_cell/k_sector - np.sin(np.sqrt(k_sector)*l_cell)/np.sqrt(k_sector)/k_sector),
                            -2*f_parallel[index]/(rho*m_particle_kg*beta*beta*gamma*gamma*gamma*c*c)*(l_cell - np.cos(np.sqrt(k_sector)*l_cell))/k_sector,
                            0,
                            0,
                            f_parallel[index]*l_cell/(2*m_particle_kg*beta*beta*gamma*gamma*c)+2*f_parallel[index]/(rho*rho*m_particle_kg*beta*beta*gamma*gamma*gamma*c*c) \
                            *((l_cell*l_cell/2/k_sector) +(np.cos(np.sqrt(k_sector)*l_cell) - 1)/(k_sector*k_sector) ),
                            f_parallel[index]*l_cell/(c*beta*p_0)])
            delta_x.append(d_x)



X_initial = scan_config["X_in"]


#######################################################
## write to julia file
with open('6D_FODO_simulation.jl', 'w') as f:
    f.write("using CSV, DataFrames\n")
    f.write(f"x_initial = [{X_initial[0]}, {X_initial[1]}, {X_initial[2]}, {X_initial[3]}, {X_initial[4]}, {X_initial[5]}]\n")
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
    f.write("]\n")
    f.write(f"x_history = zeros(6, {times})\n")

    f.write(f"x = x_initial\n")
    for i in range(times):
        for j in range(num_units):
            for k in range(len(cells)):
                f.write(f"x = {cells[k]} * x + delta_x[{j*len(cells)+k+1}]\n")
        f.write(f"x_history[: , {i+1}] = x\n")
        
    f.write("df = DataFrame(x_history', [:x, :px, :y, :py, :delta, :z])\n")
    f.write(f'CSV.write("output/{particle}_FODO_6D_history.csv", df)\n')

         


        
    
    


