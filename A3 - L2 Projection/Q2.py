import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.tri import Triangulation

file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A3 - L2 Projection\GMSH files\IITM_Map.msh'

def Read_Data():
    file = meshio.read(file_path)
    nodes = file.points
    elements = file.cells_dict['triangle']
    nodes = nodes[:, : 2]
    return nodes, elements

nodes, elements = Read_Data()
num_nodes, d = nodes.shape

r = 3
sig = r / 3
mean = np.array([286.9, 260.6])
gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])
cov = np.array([[sig, 0], [0, sig]])
cov_det = np.linalg.det(cov)
cov_inv = np.linalg.inv(cov)

def func(x):
    numerator = np.exp(-0.5 * (x - mean).T @ cov_inv @ (x - mean))
    denominator = 2 * np.pi * np.sqrt(cov_det)
    return numerator / denominator

def Assembly(a, f, A, F, n):
    for i in range(3):
        for j in range(3):
            A[n[i]][n[j]] += a[i][j]

    for i in range(3):
        F[n[i]] += f[i]

def Compute_Interpolated_Function(num_nodes):
    A = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))

    for e in elements:
        n = np.array(e[:3], dtype=int)
        coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])
        for j in range(3):
            pt = gp[j]
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
            dN = np.array([[-1, -1], [1, 0], [0, 1]])
            Jac = dN.T @ coord
            X = N @ coord
            a = np.outer(N, N) * np.linalg.det(Jac) * wt[j]
            f = func(X) * wt[j] * np.linalg.det(Jac) * N
            Assembly(a, f, A, F, n)

    A += np.eye(A.shape[0]) * 1e-10
    A_inv = np.linalg.inv(A)
    C = A_inv @ F
    return C

C = Compute_Interpolated_Function(num_nodes)

def Compute_Error(C):
    error = 0.0

    for e in elements:
        n = np.array(e[:3], dtype=int)
        local_c = C[n, 0]
        coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])

        for j in range(3):
            pt = gp[j]
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
            dN = np.array([[-1, -1], [1, 0], [0, 1]])
            Jac = dN.T @ coord
            X = N @ coord
            f_x = func(X)
            I_h_x = N @ local_c.T
            error += (f_x - I_h_x)**2 * np.linalg.det(Jac) * wt[j]

    error = np.sqrt(error)
    return error

error = Compute_Error(C)
print("Error is",error)

def Plot_Error_Mesh_Size():
    error = np.array([0.132410239599194, 0.169716994230606, 0.202336699820492, 0.149912547070967, 0.178616928667426, 0.358493104489654, 0.418517213027262, 0.378775974974415, 0.386533158534293])
    Elements = np.array([12534, 8629, 8383, 8074, 7555, 5254, 4088, 3734, 3659])
    h = np.sqrt(130155.73500000025 / Elements)
    
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

# Plot_Error_Mesh_Size()

def plot_3d_comparison():
    # Compute actual Gaussian values for plotting
    f = np.array([func([x, y]) for x, y in nodes])

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    
    # Plot Interpolated Function
    ax1 = fig.add_subplot(121, projection='3d')
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax1.plot_trisurf(tri, C.flatten(), cmap='viridis', edgecolor='none')
    ax1.set_title('Interpolated Function')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Interpolated Value')
    
    # Plot Actual Gaussian Function
    ax2 = fig.add_subplot(122, projection='3d')
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax2.plot_trisurf(tri, f, cmap='viridis', edgecolor='none')
    ax2.set_title('Actual Gaussian Function')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Gaussian Value')
    
    plt.show()

# plot_3d_comparison()

def Plot_Error():
    f = np.array([func([x, y]) for x, y in nodes])

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    
    # Plot Interpolated Function
    ax1 = fig.add_subplot(111, projection='3d')
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax1.plot_trisurf(tri, f - C.flatten(), cmap='viridis', edgecolor='none')
    ax1.set_title('Error Plot')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Difference Value')

    plt.show()

# Plot_Error()