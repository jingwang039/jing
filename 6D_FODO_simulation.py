import numpy as np
import matplotlib.pyplot as plt
import os
import time


def transfer_matrix_drift(X, l_drift, gamma, beta, Force):
    M_drift = np.array([[1, l_drift, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, l_drift, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, l_drift/gamma**2],
                        [0, 0, 0, 0, 0, 1]])
    return M_drift @ X

def transfer_matrix_sector(X, l_sector, rho, gamma, beta, Force):
    alpha = l_sector / rho
    
    M_sector = np.array([[np.cos(alpha), rho*np.sin(alpha), 0, 0, 0, rho*(1-np.cos(alpha))],
                         [-np.sin(alpha)/rho, np.cos(alpha), 0, 0, 0, np.sin(alpha)],
                         [0, 0, 1, l_sector, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [-np.sin(alpha), -rho*(1-np.cos(alpha)), 0, 0, 1, rho*(alpha - np.sin(alpha))/gamma**2],
                         [0, 0, 0, 0, 0, 1]])
    
    # beta_eff = 
    M_edge_focus = np.array([[1, 0, 0, 0, 0, 0],
                             [np.tan(alpha/2)/rho, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                                [0, 0, -np.tan(alpha/2)/rho, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
    
    # c = 3e8 
    # m = 123
    # k = 1 / rho**2
    # delta_x_sec = np.array([-2*Force/(rho*beta*c*m*(gamma**3)*beta*c)*((l_sector/k)-(rho*np.sin(alpha)/k)),
    #                         -2*Force/(rho*beta*c*m*(gamma**3)*beta*c)*((l_sector/k)-(np.cos(alpha)/k)),
    #                                      0, 
    #                                      0, 
    #                                      0, 
    #                                      0])
    return M_edge_focus @ M_sector @ M_edge_focus @ X


def transfer_matrix_QF(X, l_QF, k_QF, gamma, beta, Force):
    k = np.sqrt(k_QF)
    M_QF = np.array([[np.cos(k*l_QF), (1/k)*np.sin(k*l_QF), 0, 0, 0, 0],
                     [-k*np.sin(k*l_QF), np.cos(k*l_QF), 0, 0, 0, 0],
                     [0, 0, np.cosh(k*l_QF), (1/k)*np.sinh(k*l_QF), 0, 0],
                     [0, 0, k*np.sinh(k*l_QF), np.cosh(k*l_QF), 0, 0],
                     [0, 0, 0, 0, 1, l_QF/gamma**2],
                     [0, 0, 0, 0, 0, 1]])
    return M_QF @ X

def transfer_matrix_QD(X, l_QD, k_QD, gamma, beta, Force):
    k = np.sqrt(k_QD)
    M_QD = np.array([[np.cosh(k*l_QD), (1/k)*np.sinh(k*l_QD), 0, 0, 0, 0],
                     [k*np.sinh(k*l_QD), np.cosh(k*l_QD), 0, 0, 0, 0],
                     [0, 0, np.cos(k*l_QD), (1/k)*np.sin(k*l_QD), 0, 0],
                     [0, 0, -k*np.sin(k*l_QD), np.cos(k*l_QD), 0, 0],
                     [0, 0, 0, 0, 1, l_QD/gamma**2],
                     [0, 0, 0, 0, 0, 1]])
    return M_QD @ X

class FODO_loop:
    def __init__(self, rho, k_QF, k_QD, l_QF, l_QD, l_sector, l_drift, beta, n_turns, X_in, Force=0):
        self.rho = rho
        self.k_QF = k_QF
        self.k_QD = k_QD
        self.l_QF = l_QF
        self.l_QD = l_QD
        self.l_sector = l_sector
        self.l_drift = l_drift
        self.beta = beta
        self.gamma = 1 / np.sqrt(1 - beta**2)
        self.n_turns = n_turns
        self.X_history = []
        self.s_positions = []
        self.X = X_in
        self.Force = Force

        for i in range(self.n_turns):
            self.X = transfer_matrix_QF(self.X, self.l_QF, self.k_QF, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            if i == 0:
                self.s_positions.append(self.l_QF)
            else:
                self.s_positions.append(self.s_positions[-1]+self.l_QF)

            self.X = transfer_matrix_drift(self.X, self.l_drift, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_drift)

            self.X = transfer_matrix_sector(self.X, self.l_sector, self.rho, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_sector)

            self.X = transfer_matrix_drift(self.X, self.l_drift, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_drift)

            self.X = transfer_matrix_QD(self.X, self.l_QD, self.k_QD, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_QD)

            self.X = transfer_matrix_drift(self.X, self.l_drift, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_drift)

            self.X = transfer_matrix_sector(self.X, self.l_sector, self.rho, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_sector)

            self.X = transfer_matrix_drift(self.X, self.l_drift, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_drift)

            self.X = transfer_matrix_QF(self.X, self.l_QF, self.k_QF, self.gamma, self.beta, self.Force)
            self.X_history.append(self.X.copy())
            self.s_positions.append(self.s_positions[-1]+self.l_QF)
        
        # print(self.X_history)
        # time.sleep(10)

        self.x_history = np.array([x[0] for x in self.X_history])
        self.x_prime_history = np.array([x[1] for x in self.X_history])
        self.y_history = np.array([x[2] for x in self.X_history])
        self.y_prime_history = np.array([x[3] for x in self.X_history])
        self.l_history = np.array([x[4] for x in self.X_history])
        self.delta_history = np.array([x[5] for x in self.X_history])


def plot_trajectories(path, X, s, labels):
    plt.figure(figsize=(10, 6))
    for i in range(len(X)):
        plt.plot(s,X[i].x_history*1e6, label=labels[i])
    plt.xlabel('s (m)')
    plt.ylabel('x (µm)')
    plt.title('X Direction Particle Trajectories through FODO Lattice')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(path + 'FODO_6D_x_trajectories.png', dpi=400)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(len(X)):
        plt.plot(s,X[i].x_prime_history*1e6, label=labels[i])
    plt.xlabel('s (m)')
    plt.ylabel("x prime")
    plt.title('X prime Trajectories through FODO Lattice')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(path + 'FODO_6D_x_prime_trajectories.png', dpi=400)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(len(X)):
        plt.plot(s,X[i].y_history*1e6, label=labels[i])
    plt.xlabel('s (m)')
    plt.ylabel('y (µm)')
    plt.title('Y Direction Particle Trajectories through FODO Lattice')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(path + 'FODO_6D_y_trajectories.png', dpi=400)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(len(X)):
        plt.plot(s,X[i].y_prime_history*1e6, label=labels[i])
    plt.xlabel('s (m)')
    plt.ylabel("y prime")
    plt.title('Y prime Trajectories through FODO Lattice')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(path + 'FODO_6D_y_prime_trajectories.png', dpi=400)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(len(X)):
        plt.plot(s,X[i].l_history*1e6, label=labels[i])
    plt.xlabel('s (m)')
    plt.ylabel('l (µm)')
    plt.title('Longitudinal Position Deviations through FODO Lattice')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(path + 'FODO_6D_l_trajectories.png', dpi=400)
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(len(X)):
        plt.plot(s,X[i].delta_history*1e6, label=labels[i])
    plt.xlabel('s (m)')
    plt.ylabel("delta")
    plt.title('Relative Momentum Deviations through FODO Lattice')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(path + 'FODO_6D_delta_trajectories.png', dpi=400)
    plt.close()


### config

l_ring = 26700 # m
times = 1
n_cells = 200
beta = 0.32
gamma = 1 / np.sqrt(1 - beta**2)

m_U = 221.696e9  # eV
p0 = m_U * (gamma - 1) *1e-9 # GeV/c
q = 1
q_p0 = q / p0 
T = 0.0299792458 # GeV / (q * c)
k_QF = 0.891*q_p0*T #1/m
k_QD = 1.656*q_p0*T
# k_QF = 0.891
# k_QD = 1.656
print(f'k_QF: {k_QF}, k_QD: {k_QD}')

l_QD = 3.25 # m
l_QF = 3.6 # m
l_sector = 47.1 # m
l_drift = 7.3 # m

# rho = l_sector /(2 * np.pi / n_cells /2)  # m
rho = l_ring / (2 * np.pi) # m
print(f'rho: {rho}')

Force = 0



###start scans

path = "./results/FODO_6D_without_force/x_scan/"
if not os.path.exists(path):
    os.makedirs(path)

X_in = []
labels = []
for i in range(11):
    x0 = (i - 5) * 2e-6  
    X_in.append(np.array([x0, 0, 0, 0, 0, 0]))
    labels.append(f'Initial x0={x0*1e6:.1f} µm')

X = []
for i in range(len(X_in)):
    X.append(FODO_loop(rho, k_QF, k_QD, l_QF, l_QD, l_sector, l_drift, beta, n_turns=times*n_cells, X_in=X_in[i]))

s = X[0].s_positions
plot_trajectories(path, X, s, labels)


path = "./results/FODO_6D_without_force/x_prime_scan/"

if not os.path.exists(path):
    os.makedirs(path)

X_in = []
labels = []
for i in range(11):
    x1 = (i - 5) * 2e-2  
    X_in.append(np.array([0, x1, 0, 0, 0, 0]))
    labels.append(f'Initial x_prime={x1}')

X = []
for i in range(len(X_in)):
    X.append(FODO_loop(rho, k_QF, k_QD, l_QF, l_QD, l_sector, l_drift, beta, n_turns=times*n_cells, X_in=X_in[i]))
    

s = X[0].s_positions
plot_trajectories(path, X, s, labels)


path = "./results/FODO_6D_without_force/y_scan/"
if not os.path.exists(path):
    os.makedirs(path)

X_in = []
labels = []
for i in range(11):
    y0 = (i - 5) * 2e-6  
    X_in.append(np.array([0, 0, y0, 0, 0, 0]))
    labels.append(f'Initial y0={y0*1e6:.1f} µm')

X = []
for i in range(len(X_in)):
    X.append(FODO_loop(rho, k_QF, k_QD, l_QF, l_QD, l_sector, l_drift, beta, n_turns=times*n_cells, X_in=X_in[i]))

s = X[0].s_positions
plot_trajectories(path, X, s, labels)

path = "./results/FODO_6D_without_force/y_prime_scan/"
if not os.path.exists(path):
    os.makedirs(path)

X_in = []
labels = []
for i in range(11):
    y1 = (i - 5) * 2e-2  
    X_in.append(np.array([0, 0, 0, y1, 0, 0]))
    labels.append(f'Initial y_prime={y1}')

X = []
for i in range(len(X_in)):
    X.append(FODO_loop(rho, k_QF, k_QD, l_QF, l_QD, l_sector, l_drift, beta, n_turns=times*n_cells, X_in=X_in[i]))

s = X[0].s_positions
plot_trajectories(path, X, s, labels)



        
       
