import numpy as np
import matplotlib.pyplot as plt
import meshio
from matplotlib.tri import Triangulation

output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\Optimal_Mesh_OAT.msh'

def Read_Data():
    file = meshio.read(output_msh_path)
    nodes = file.points[:, :2]
    elements = file.cells_dict['triangle']
    return nodes, elements

gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])
fluctuataions = 0.001

def Assembly(k, K, m, M, n):
    for i in range(3):
        for j in range(3):
            M[n[i]][n[j]] += m[i][j]
            K[n[i]][n[j]] += k[i][j]

def Compute(K, M, nodes, elements):
    for e in elements:
        n = np.array(e[:3], dtype=int)
        coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])
        for j in range(3):
            pt = gp[j]
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
            dN = np.array([[-1, -1], [1, 0], [0, 1]])
            Jac = dN.T @ coord
            # X = N @ coord
            dphi = np.linalg.inv(Jac) @ dN.T
            m = np.outer(N, N) * np.linalg.det(Jac) * wt[j]
            k = np.outer(dphi, dphi) * np.linalg.det(Jac) * wt[j]
            Assembly(k, K, m, M, n)

nodes, elements = Read_Data()
num_nodes = nodes.shape[0]

K = np.zeros((num_nodes, num_nodes))
M = np.zeros((num_nodes, num_nodes))
Compute(K, M, nodes, elements)
L = np.linalg.inv(M) @ K
eigenvalues, eigenvectors = np.linalg.eig(L)

sorted_indices = np.argsort(eigenvalues)  
lamda = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
lamda = lamda + fluctuataions
BC = np.zeros((num_nodes, 1))
BC[3783] = 0.1
no_of_values = 15

P_without_bc = eigenvectors
P_with_bc = np.zeros((num_nodes, no_of_values))

for i in range(no_of_values):
    P_with_bc[:, i] = (np.linalg.inv(K - lamda[i] * M) @ BC).flatten()

def Plot_Pressure_Distribution(P_all):
    P_rms = np.sqrt(np.mean(P_all**2, axis=1))
    print(P_rms)
    P = 20 * np.log10(P_rms / (2 * 1e-5)) # in decibels
    P = P.flatten()
    triangles = elements[:, :3]
    triang = Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(triang, P, cmap='viridis', edgecolor='none')
    ax.set_title("Pressure Distribution")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Pressure (P) in decibels")
    fig.colorbar(ax.plot_trisurf(triang, P, cmap='viridis'), ax=ax, shrink=0.5, aspect=5)
    plt.show()

# without boundary conditions
# Plot_Pressure_Distribution(P_without_bc)

# with boundary conditions
# Plot_Pressure_Distribution(P_with_bc)

main_gate = 4744
student_gate = 3637
fountain = 1328
badminton_court = 4395

P_with_bc = np.sqrt(np.mean(P_with_bc**2, axis=1))
P_with_bc = 20 * np.log10(P_with_bc / (2 * 1e-5))

print("The sound level at main gate is", P_with_bc[main_gate], "dB")
print("The sound level at student gate is", P_with_bc[student_gate], "dB")
print("The sound level at fountain is", P_with_bc[fountain], "dB")
print("The sound level at badmintion court is", P_with_bc[badminton_court], "dB")