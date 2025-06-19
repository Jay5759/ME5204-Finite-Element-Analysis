import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def func(x):
    return np.exp(np.sin(0.25 * np.pi * x * x))

def Get_error(num_nodes):
    nodes = np.linspace(0, 3, num_nodes)
    print(nodes)
    elements = np.column_stack((np.arange(num_nodes - 1), np.arange(1, num_nodes)))
    
    A = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))
    gp = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
    wt = np.array([5/9, 8/9, 5/9])
    error = 0.0
    
    # Loop over elements
    for e in elements:
        # get element connectivity
        n1, n2 = int(e[0]), int(e[1])
    
        # get nodal connectivity
        coord = np.array([nodes[n1], nodes[n2]])
        
        # Loop over Gauss points
        for j in range(3):
            # get Gauss points
            pt = gp[j]
    
            # get Shape functions
            N = np.array([(1 - pt)/2, (1 + pt)/2])
    
            # get Derivates
            dN = np.array([-0.5, 0.5])
    
            # get Jacobian
            Jac = np.dot(dN, coord)
    
            # get Local a
            a = np.outer(N, N) * Jac * wt[j]
    
            # get Local f
            x = np.dot(N, coord.T)
            f = func(x) * N * wt[j] * Jac
    
            # Assembly of A
            A[n1][n1] += a[0][0]
            A[n1][n2] += a[0][1]
            A[n2][n1] += a[1][0]
            A[n2][n2] += a[1][1]
    
            # Assembly of F
            F[n1] += f[0]
            F[n2] += f[1]
    
    # Compute the inverse of A
    A_inverse = np.linalg.inv(A)
    
    # Compute C = A_inverse x F
    C = np.dot(A_inverse, F)
    
    # Loop over elements
    for e in elements:
        n1, n2 = int(e[0]), int(e[1])
    
        local_c = np.array([C[n1][0], C[n2][0]])
    
        # get nodal connectivity
        coord = np.array([nodes[n1], nodes[n2]])
        
        # Loop over Gauss points
        for j in range(3):
            # get Gauss points
            pt = gp[j]
    
            # get Shape functions
            N = np.array([(1 - pt)/2, (1 + pt)/2])
    
            # get Jacobian
            Jac = np.dot(dN, coord)
    
            # locate gauss points in physical space
            x = np.dot(N, coord.T)
            f_x = func(x)
            I_h_x = np.dot(N, local_c)
    
            error += (f_x - I_h_x) * (f_x - I_h_x) * Jac * wt[j]
    
    error = np.sqrt(error)
    print(error)
    return error

Get_error(3)

num_nodes_list = np.array([50, 100, 200, 300, 400, 500, 600, 700])
mesh = 3 / (num_nodes_list - 1)
error = []

# for num_node in num_nodes_list:
#     error.append(Get_error(num_node))

# error = np.array([error])

# # Convert to log scale
# log_mesh = np.log(mesh.T).flatten()
# log_error = np.log(error.T).flatten()

# # Perform linear regression to get the slope and intercept
# slope, intercept, r_value, p_value, std_err = stats.linregress(log_mesh, log_error)

# # Plotting the log-log plot
# plt.figure(figsize=(8, 6))
# plt.plot(log_mesh, log_error, marker='o', linestyle='-', label='Data')
# plt.plot(log_mesh, slope * log_mesh + intercept, linestyle='--', color='red', label=f'Fit: slope={slope:.15f}')
# plt.xlabel('log(mesh size)')
# plt.ylabel('log(error)')
# plt.title('Log-Log Plot of Error vs. Mesh size')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Print slope value
# print(f"Rate of Convergence is : {slope}")