import numpy as np
import meshio
import gmsh

geo_file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_without_helicopter.geo'
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_without_helicopter.msh'

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
    surface_tags = file.cell_data_dict['gmsh:geometrical']['triangle']
    surface_elements = {}
    for tag in set(surface_tags):
        surface_elements[tag] = elements[surface_tags == tag]
    return nodes, surface_elements

ms = 0.5
generate_msh(ms)
nodes, surface_elements = Read_Data()
num_node = nodes.shape[0]

ndof = 2
num_node_per_element = 3
E = 210 * 1e9
nu = 0.3
bulk_modulus = 2.1 * 1e9
g = 9.81
density = np.array([2400, 1000])
gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])

def Compute(M, K, F, nodes, surface_elements):
    for surface_tag, elements in surface_elements.items():
        rho = density[surface_tag - 1]
        if surface_tag == 0:
            Elasticity_matrix = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]])
        else:
            Elasticity_matrix = np.array([[bulk_modulus, bulk_modulus, 0], [bulk_modulus, bulk_modulus, 0], [0, 0, 0]])
        body_load = np.array([0, rho * g])
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
                gindex = np.zeros(2 * num_node_per_element, dtype=int)
                for ind in range(num_node_per_element):
                    gindex[2 * ind - 1] = 2 * n[ind] - 1     # x DOF for node ind
                    gindex[2 * ind] = 2 * n[ind]  # y DOF for node ind
                # Assembly
                B = np.zeros((3, ndof * num_node_per_element))
                m = np.zeros((2, ndof * num_node_per_element))
                for i in range(num_node_per_element):
                    B[0][2 * i] = dphi[0][i]
                    B[1][2 * i + 1] = dphi[1][i]
                    B[2][2 * i] = dphi[1][i]
                    B[2][2 * i + 1] = dphi[0][i]
                    m[0][2 * i] = N[i]
                    m[1][2 * i + 1] = N[i]
    
                # Assembly with gindex
                Ke = (B.T @ Elasticity_matrix @ B) * np.linalg.det(Jac) * wt[j]
                Fe = m.T @ body_load * np.linalg.det(Jac) * wt[j]
                Me = m.T @ m * np.linalg.det(Jac) * wt[j]
                for i in range(2 * num_node_per_element):
                    for j in range(2 * num_node_per_element):
                        K[int(gindex[i]), int(gindex[j])] += Ke[i, j]
                        M[int(gindex[i]), int(gindex[j])] += Me[i, j] * rho
                    F[int(gindex[i])] += Fe[i]

K = np.zeros((ndof * num_node, ndof * num_node))
M = np.zeros((ndof * num_node, ndof * num_node))
F = np.zeros((ndof * num_node, 1))
Compute(M, K, F, nodes, surface_elements)
print(K)
print(M)
print(F)

displacement = np.linalg.inv(K) @ F
print(displacement)

# def Calculate_Stress(disp, Sigma, nodes, surface_elements):
#     for surface_tag, elements in surface_elements.items():
#         for e in elements:
#             n = np.array(e[:3], dtype=int)
#             coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])
#             for j in range(3):
#                 pt = gp[j]
#                 N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
#                 dN = np.array([[-1, -1], [1, 0], [0, 1]])
#                 Jac = dN.T @ coord
#                 X = N @ coord
#                 dphi = np.linalg.inv(Jac) @ dN.T
#                 u = np.zeros((6, 1))
#                 for ind in range(num_node_per_element):
#                     u[2 * ind - 1] = disp[2 * n[ind] - 1]
#                     u[2 * ind] = disp[2 * n[ind]]
#                 # Assembly
#                 B = np.zeros((3, ndof * num_node_per_element))
#                 for i in range(num_node_per_element):
#                     B[0][2 * i] = dphi[0][i]
#                     B[1][2 * i + 1] = dphi[1][i]
#                     B[2][2 * i] = dphi[1][i]
#                     B[2][2 * i + 1] = dphi[0][i]
    
#                 # Assembly with gindex
#                 Se = Elasticity_matrix @ B @ u * np.linalg.det(Jac) * wt[j] 
#                 for ind in range(num_node_per_element):
#                     Sigma[n[ind]] += Se[ind]

# Sigma = np.zeros((num_node, 1))
# Calculate_Stress(displacement, Sigma, nodes, surface_elements)
# print(Sigma)