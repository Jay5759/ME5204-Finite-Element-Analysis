import numpy as np
import meshio
import gmsh

geo_file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\IITM_Map_A4.geo'
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\IITM_Map_A4.msh'

def generate_msh(global_mesh_size_factor):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", global_mesh_size_factor)
    gmsh.open(geo_file_path)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_msh_path)
    gmsh.finalize()

global_mesh_size_factor = 100

generate_msh(global_mesh_size_factor)

def Read_Data():
    file = meshio.read(output_msh_path)
    nodes = file.points[:, :2]
    elements = file.cells_dict['triangle']
    surface_tags = file.cell_data_dict['gmsh:geometrical']['triangle']
    surface_elements = {}
    for tag in set(surface_tags):
        surface_elements[tag] = elements[surface_tags == tag]
    return nodes, surface_elements

nodes, surface_elements = Read_Data()

for surface_tag, elements in surface_elements.items():
    print(f"Surface Tag: {surface_tag}")
    print(f"Elements: {elements}\n")

num_nodes, d = nodes.shape

gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])
num_nodes = 3

def initial_condition(x):
    return np.sin(np.pi * x)

def Assembly(m, k, M, K, n):
    for i in range(2):
        for j in range(2):
            M[n[i]][n[j]] += m[i][j]

    for i in range(2):
        for j in range(2):
            K[n[i]][n[j]] += k[i][j]

def Compute_Interpolated_Function(num_nodes):
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    elements = np.array([[0, 1, 2]])

    M = np.zeros((num_nodes, num_nodes))
    K = np.zeros((num_nodes, num_nodes))

    for e in elements:
        n = np.array(e[:3], dtype=int)
        coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])
        # print(coord.shape)

        for j in range(3):
            pt = gp[j]
            N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
            dN = np.array([[-1, -1], [1, 0], [0, 1]])
            Jac = dN.T @ coord
            # X = N @ coord
            m = np.outer(N, N) * np.linalg.det(Jac) * wt[j]
            dphi = np.linalg.inv(Jac) @ dN.T
            k = np.outer(dphi, dphi) * np.linalg.det(Jac) * wt[j]
            Assembly(m, k, M, K, n)

    return M, K

M, K = Compute_Interpolated_Function(3)
print(M)
print(K)

nodes = np.linspace(0, 1, num_nodes)
C = []

for i in nodes:
    C.append(initial_condition(i))

C = np.array(C)
print('C at time t = 0', C)
delta_t = 0.01

timestep = (1.0 - 0.0) / delta_t

for i in np.arange(0.0, 1.0, delta_t):
    C_next = np.linalg.inv(M/delta_t) @ ((M / delta_t - K) @ C)
    C_next[0] = 0
    C_next[-1] = 0
    C = C_next

print(C_next)