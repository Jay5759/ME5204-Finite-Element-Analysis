import numpy as np
import matplotlib.pyplot as plt
import meshio
import gmsh
from matplotlib.tri import Triangulation

geo_file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\OAT.geo'
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\OAT.msh'

def generate_msh(global_mesh_size_factor):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", global_mesh_size_factor)
    gmsh.open(geo_file_path)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_msh_path)
    gmsh.finalize()


def Read_Data():
    file = meshio.read(output_msh_path)
    nodes = file.points[:, :2]
    elements = file.cells_dict['triangle']
    no_of_elements = len(elements)
    return nodes, elements, no_of_elements

gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])

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


frequency_prev = np.zeros((500, 1))
mesh_size = np.array([2, 1, 0.95, 0.93])
# mesh_size = np.array([20, 16])
error = []
h = []
ele = []
nod = []

for ms in mesh_size:
    generate_msh(ms)
    nodes, elements, num_elements = Read_Data()
    num_nodes = nodes.shape[0]
    h.append(np.sqrt(130155.73500000025 / num_elements))
    ele.append(num_elements)
    nod.append(num_nodes)

    K = np.zeros((num_nodes, num_nodes))
    M = np.zeros((num_nodes, num_nodes))
    Compute(K, M, nodes, elements)
    L = np.linalg.inv(M) @ K
    eigenvalues, eigenvectors = np.linalg.eig(L)
    sorted_indices = np.argsort(eigenvalues)  

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    frequency = sorted_eigenvalues[:15]

    if ms == mesh_size[0]:
        # error.append(e)
        frequency_prev = frequency
        continue

    e = np.sum(((frequency[:15] - frequency_prev[:15]) / frequency_prev[:15])**2)
    error.append(e)
    frequency_prev = frequency

error = np.array(error)
np.set_printoptions(precision=7)
h = np.array(h)
ele = np.array(ele)
nod = np.array(nod)
print(error)
print(h)
# print(ele)
# print(nod)