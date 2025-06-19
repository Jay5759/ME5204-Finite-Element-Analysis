import numpy as np
import matplotlib.pyplot as plt

# Function to approximate
def func(x):
    return np.exp(np.sin(0.25 * np.pi * x * x))

# Function to compute the interpolation solution
def Get_solution(num_nodes):
    nodes = np.linspace(0, 3, num_nodes)
    elements = np.column_stack((np.arange(num_nodes - 1), np.arange(1, num_nodes)))
    
    A = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))
    gp = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
    wt = np.array([5/9, 8/9, 5/9])
    
    # Loop over elements
    for e in elements:
        n1, n2 = int(e[0]), int(e[1])
        coord = np.array([nodes[n1], nodes[n2]])
        
        # Loop over Gauss points
        for j in range(3):
            pt = gp[j]
            N = np.array([(1 - pt)/2, (1 + pt)/2])
            dN = np.array([-0.5, 0.5])
            Jac = np.dot(dN, coord)
            a = np.outer(N, N) * Jac * wt[j]
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
    
    # Compute C = A_inverse * F
    A_inverse = np.linalg.inv(A)
    C = np.dot(A_inverse, F)

    return nodes, C.flatten()

# Number of nodes for interpolation
num_nodes = 1000

# Get the interpolation solution
nodes, C = Get_solution(num_nodes)

fine_points = np.linspace(0, 3, 1000)
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