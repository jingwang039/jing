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




# fodo parameter config
config = json.load(open('FODO_config.json'))

times = config['times']
cells = config['cells']
particle = config['particle']
p = config['p'] # GeV/c
q = config['charge'] 
m = config['mass'] 

p_ev = p * 1e9  # eV/c
beta = p_ev / np.sqrt(p_ev**2 + m**2)
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
alpha_angle = config['alpha_angle'] 
alpha = alpha_angle/180*np.pi # rad
l_sector = config['l_sector']  # m

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


n_turns = times * len(cells)
print(f"times: {times}")
print(f"n_cells: {len(cells)}")
print(f"n_turns: {n_turns}")
print("FODO cells: "+str(cells))

X_initial = np.array([1e-3, 0, 1e-3, 0, 0, 0]) 

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
    f.write(f"x_history = zeros(6, {n_turns})\n")

    f.write(f"x = x_initial\n")
    for i in range(times):
        for j in range(len(cells)):
            f.write(f"x = {cells[j]} * x\n")
            f.write(f"x_history[: , {i*len(cells)+j+1}] = x\n")
    
    f.write("df = DataFrame(x_history', [:x, :px, :y, :py, :delta, :z])\n")
    f.write(f'CSV.write("output/{particle}_FODO_6D_history.csv", df)\n')

         


        
    
    


