import numpy as np
import matplotlib.pyplot as plt
import meshio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

r = 1.5 * 2
sig = r / 3
gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])

file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A3 - L2 Projection\GMSH files\IITM_Map_8.msh'

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
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)
    return np.exp(exponent) / np.sqrt(4 * np.pi ** 2 * np.linalg.det(cov))

def Contour_Plots(num_nodes):
    A = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))
    
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

    A += np.eye(A.shape[0]) * 1e-10

    # Compute the inverse of A
    A_inverse = np.linalg.inv(A)
    
    # Compute C = A_inverse x F
    C = np.dot(A_inverse, F)

    # 3D Plotting
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot for nodes
    # ax.scatter(nodes[:, 0], nodes[:, 1], C.flatten(), c='r', marker='o', s=5)

    # # Triangular grid for surface plot
    # from matplotlib.tri import Triangulation
    # tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    # ax.plot_trisurf(tri, C.flatten(), cmap='viridis', edgecolor='none')

    # ax.set_title('3D Surface Plot')
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Function Value')
    # plt.show()

    return C

def plot_3d_comparison(num_nodes):
    """Plot FEM results and actual Gaussian function"""
    # Compute FEM results
    C_fem = Contour_Plots(num_nodes)

    # Add offset to ensure the peak is above the base
    C_fem_min = np.min(C_fem)
    C_fem_positive = C_fem - C_fem_min + 1e-2  # Ensure values are positive
    
    # Compute actual Gaussian values for plotting
    X, Y = np.meshgrid(np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 100),
                       np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), 100))
    Z_actual = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_actual[i, j] = func([X[i, j], Y[i, j]])

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    
    # Plot FEM Results
    ax1 = fig.add_subplot(121, projection='3d')
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax1.plot_trisurf(tri, C_fem_positive.flatten(), cmap='viridis', edgecolor='none')
    ax1.set_title('FEM Results')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('FEM Value')
    
    # Plot Actual Gaussian Function
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_actual, cmap='viridis', edgecolor='none')
    ax2.set_title('Actual Gaussian Function')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Gaussian Value')
    
    plt.show()

plot_3d_comparison(num_nodes)