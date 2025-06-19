import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import meshio

r = 1.5 * 2
centre = np.array([636.7, 355.1])
sig = r / 3
gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])

file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A3 - L2 Projection\GMSH files\IITM_Map.msh'

# Define the line ranges
file = meshio.read(file_path)
elements = file.cells_dict['triangle']
nodes = file.points
nodes = nodes[:, : 2]
print(elements.shape)
num_nodes, d = nodes.shape

mean = np.array([286.9, 260.6])
cov = np.array([[sig, 0], [0, sig]])

def func(x):
    return np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)) / np.sqrt(4 * np.pi ** 2 * np.linalg.det(cov))

def Get_error(num_nodes):
    A = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))
    error = 0.0
    
    # Loop over elements
    for e in elements:
        # get element connectivity
        n1, n2, n3 = int(e[0]) - 1, int(e[1]) - 1, int(e[2]) - 1
    
        # get nodal connectivity
        coord = np.array([nodes[n1], nodes[n2], nodes[n3]])
        
        # Loop over Gauss points
        for j in range(3):
            # get Gauss points
            pt = gp[j]
    
            # get Shape functions
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
    
            # get Derivates
            dN = np.array([[-1, -1], [1, 0], [0, 1]])
    
            # get Jacobian
            Jac = np.dot(dN.T, coord)
    
            # get Local a
            a = np.outer(N, N) * np.linalg.det(Jac) * wt[j]
            # print(a.shape)
    
            # get Local f
            x = np.dot(N, coord[:, 0].T)
            y = np.dot(N, coord[:, 1].T)
            X = np.array([x, y])
            f = func(X) * N * wt[j] * np.linalg.det(Jac)
    
            # Assembly of A
            A[n1][n1] += a[0][0]
            A[n1][n2] += a[0][1]
            A[n1][n3] += a[0][2]
            
            A[n2][n1] += a[1][0]
            A[n2][n2] += a[1][1]
            A[n2][n3] += a[1][2]
            
            A[n3][n1] += a[2][0]
            A[n3][n2] += a[2][1]
            A[n3][n3] += a[2][2]
    
            # Assembly of F
            F[n1] += f[0]
            F[n2] += f[1]
            F[n3] += f[2]

    print("Zero rows in A:", np.where(~A.any(axis=1))[0])
    print("Zero columns in A:", np.where(~A.any(axis=0))[0])
    A += np.eye(A.shape[0]) * 1e-10

    # Compute the inverse of A
    A_inverse = np.linalg.inv(A)
    
    # Compute C = A_inverse x F
    C = np.dot(A_inverse, F)
    
    # Loop over elements
    for e in elements:
        n1, n2, n3 = int(e[0]) - 1, int(e[1]) - 1, int(e[2]) - 1
    
        local_c = np.array([C[n1][0], C[n2][0], C[n3][0]])
    
        # get nodal connectivity
        coord = np.array([nodes[n1], nodes[n2], nodes[n3]])
        
        # Loop over Gauss points
        for j in range(3):
            # get Gauss points
            pt = gp[j]
    
            # get Shape functions
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
    
            # get Jacobian
            Jac = np.dot(dN.T, coord)
    
            # locate gauss points in physical space
            x = np.dot(N, coord[:, 0].T)
            f_x = func(x)
            I_h_x = np.dot(N, local_c)
    
            error += (f_x - I_h_x) * (f_x - I_h_x) * np.linalg.det(Jac) * wt[j]
    
    error = np.sqrt(abs(error))
    print(error)
    return error

# Get_error(num_nodes)

error = np.array([3.11910339518403, 3.5199767347495827, 2.49727548016349, 3.83224429631611, 4.91118997801645, 7.700701719629413, 7.894079842065574])
Elements = np.array([13805, 12534, 11364, 8629, 8383, 8074, 4881])
h = np.sqrt(1 / Elements)

# Convert to log scale
log_mesh = np.log(h.T).flatten()
log_error = np.log(error.T).flatten()

# Perform linear regression to get the slope and intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(log_mesh, log_error)

# Plotting the log-log plot
plt.figure(figsize=(8, 6))
plt.scatter(log_mesh, log_error, marker='o', linestyle='-', label='Data')
plt.plot(log_mesh, slope * log_mesh + intercept, linestyle='--', color='red', label=f'Fit: slope={slope:.15f}')
plt.xlabel('log(mesh size)')
plt.ylabel('log(error)')
plt.title('Log-Log Plot of Error vs. Mesh size')
plt.grid(True)
plt.legend()
plt.show()

# Print slope value
print(f"Rate of Convergence is : {slope}")