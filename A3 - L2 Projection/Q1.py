import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

gp = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
wt = np.array([5/9, 8/9, 5/9])
num_nodes = 700

def func(x):
    return -4*x

def Assembly(a, f, A, F, n):
    for i in range(2):
        for j in range(2):
            A[n[i]][n[j]] += a[i][j]

    for i in range(2):
        F[n[i]] += f[i]

def Compute_Interpolated_Function(num_nodes):
    # nodes = np.linspace(0, 3, num_nodes)
    nodes = np.array([0, 1.5, 3])
    # elements = np.column_stack((np.arange(num_nodes - 1), np.arange(1, num_nodes)))
    elements = np.array([[0, 1], [1, 2]])
    num_nodes = 3

    A = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))

    for e in elements:
        n = np.array(e[:2], dtype=int)
        coord = np.array([nodes[n[0]], nodes[n[1]]])

        for j in range(3):
            pt = gp[j]
            N = np.array([(1 - pt)/2, (1 + pt)/2])
            dN = np.array([-0.5, 0.5])
            Jac = dN.T @ coord
            X = N @ coord
            a = np.outer(N, N) * Jac * wt[j]
            f = func(X) * wt[j] * Jac * N
            Assembly(a, f, A, F, n)

    # A_inv = np.linalg.inv(A)
    print(A)
    print(F)
    # C = A_inv @ F
    # return C

C = Compute_Interpolated_Function(num_nodes)

def Compute_Error(C, num_nodes):
    nodes = np.linspace(0, 3, num_nodes)
    elements = np.column_stack((np.arange(num_nodes - 1), np.arange(1, num_nodes)))
    error = 0.0

    for e in elements:
        n = np.array(e[:2], dtype=int)
        local_c = C[n, 0]
        coord = np.array([nodes[n[0]], nodes[n[1]]])

        for j in range(3):
            pt = gp[j]
            N = np.array([(1 - pt)/2, (1 + pt)/2])
            dN = np.array([-0.5, 0.5])
            Jac = dN.T @ coord
            X = N @ coord
            f_x = func(X)
            I_h_x = N @ local_c.T
            error += (f_x - I_h_x)**2 * Jac * wt[j]

    error = np.sqrt(error)
    return error

# error = Compute_Error(C, num_nodes)
# print(error)

def Plot_Error_Mesh():
    num_nodes_list = np.array([10, 25, 50, 100, 200, 300, 400, 500, 600, 700])
    mesh = 3 / (num_nodes_list - 1)
    error = []

    for num_node in num_nodes_list:
        C = Compute_Interpolated_Function(num_node)
        error.append(Compute_Error(C, num_node))
    
    error = np.array([error])
    log_mesh = np.log(mesh.T).flatten()
    log_error = np.log(error.T).flatten()
    
    # Perform linear regression to get the slope and intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_mesh, log_error)
    
    # Plotting the log-log plot
    plt.figure(figsize=(8, 6))
    plt.plot(log_mesh, log_error, marker='o', linestyle='-', label='Data')
    plt.plot(log_mesh, slope * log_mesh + intercept, linestyle='--', color='red', label=f'Fit: slope={slope:.15f}')
    plt.xlabel('log(mesh size)')
    plt.ylabel('log(error)')
    plt.title('Log-Log Plot of Error vs. Mesh size')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Print slope value
    print(f"Rate of Convergence is : {slope}")

# Plot_Error_Mesh()

def Plot_Comparison():
    num_nodes = 1000
    nodes = np.linspace(0, 3, num_nodes)
    C = Compute_Interpolated_Function(num_nodes)
    fine_points = np.linspace(0, 3, 10000)
    actual_func_values = func(fine_points)
    
    # Plot the actual function and the interpolated function
    plt.figure(figsize=(10, 6))
    
    # Plot actual function with a solid black line and no markers
    plt.plot(fine_points, actual_func_values, label='Actual Function', color='black', linestyle='-', linewidth=2)
    
    # Plot interpolated function with dashed line
    plt.plot(nodes, C, label='Interpolated Function', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.title('Actual Function vs Interpolated Function')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot_Comparison()