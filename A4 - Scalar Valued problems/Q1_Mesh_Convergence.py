import numpy as np
import matplotlib.pyplot as plt
import meshio
import gmsh
from matplotlib.tri import Triangulation

geo_file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\Map.geo'
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\Map.msh'

def generate_msh(global_mesh_size_factor):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", global_mesh_size_factor)
    gmsh.open(geo_file_path)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_msh_path)
    gmsh.finalize()

no_of_elements = 0

def Read_Data():
    file = meshio.read(output_msh_path)
    nodes = file.points[:, :2]
    elements = file.cells_dict['triangle']
    no_of_elements = len(elements)
    surface_tags = file.cell_data_dict['gmsh:geometrical']['triangle']
    surface_elements = {}
    for tag in set(surface_tags):
        surface_elements[tag] = elements[surface_tags == tag]
    return nodes, surface_elements, no_of_elements

r = 5 * 1.5
sig = r / 3
sig *= (100 / 23)
P = 500 * 1e6
x0 = np.array([287.2, 260.5])
x0 *= (100 / 23)
gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])
k_therm = np.array([0.128 * 1020635.7277882794, # trees
                    0.1152 * 958774.6691871487, # Academic and residential zone
                    0.1152 * 334369.47069943274, # hostel zone
                    0.6 # water bodies
                  ])

def func(x):
    x *= (100 / 23)
    numerator = P * np.exp(-1.0 * (np.linalg.norm(x - x0))**2 / (2 * sig**2))
    denomenator = np.sqrt(2 * np.pi * sig**2)
    return numerator / denomenator

def Assembly(k, f, K, F, n):
    for i in range(3):
        for j in range(3):
            K[n[i]][n[j]] += k[i][j]
        F[n[i]] += f[i]

def Compute(K, F, k_therm_values, nodes, surface_elements):
    for surface_tag, elements in surface_elements.items():
        k_therm = k_therm_values[surface_tag - 1]
        for e in elements:
            n = np.array(e[:3], dtype=int)
            coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])
            for j in range(3):
                pt = gp[j]
                N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
                dN = np.array([[-1, -1], [1, 0], [0, 1]])
                Jac = dN.T @ coord
                X = N @ coord
                dphi = np.linalg.inv(Jac) @ dN.T
                k = np.outer(dphi, dphi) * np.linalg.det(Jac) * wt[j] * k_therm
                f = func(X) * wt[j] * np.linalg.det(Jac) * N
                Assembly(k, f, K, F, n)

    K_inv = np.linalg.inv(K)
    C = K_inv @ F
    return C 

C_prev = np.zeros((500, 1))
cmp = np.array([294, 429, 418, 200, 51, 258, 270, 432, 306])
mesh_size = np.array([16, 14, 12, 10, 8])
h = []
error = []
c = []
ele = []
nod = []

for ms in mesh_size:
    generate_msh(ms)
    nodes, surface_elements, num_elements = Read_Data()
    num_nodes = nodes.shape[0]
    h.append(np.sqrt(130155.73500000025 / num_elements))
    ele.append(num_elements)
    nod.append(num_nodes)

    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros((num_nodes, 1))
    C = Compute(K, F, k_therm, nodes, surface_elements)
    e = 0.0
    temp = 0
    for i in cmp:
        temp += C[i]
    c.append(temp)

    if ms == mesh_size[0]: 
        error.append(e)
        C_prev = C
        continue
    for i in cmp:
        e += ((C[i] - C_prev[i]) / C_prev[i])**2

    error.append(np.sum(e))
    C_prev = C

error = np.array(error)
h = np.array(h)
c = np.array(c)
ele = np.array(ele)
nod = np.array(nod)
print(error)
print(h)
# print(ele)
# print(nod)
# print(c)

plt.figure(figsize=(8, 6))
plt.plot(h[1:], error[1:], marker='o', linestyle='-', label='Data')
plt.xlabel('(mesh sizes)')
plt.ylabel('(Error values)')
plt.title('Mesh size vs. Error values')
plt.grid(True)
plt.gca().invert_xaxis()
plt.legend()
plt.show()