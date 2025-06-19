import numpy as np
import matplotlib.pyplot as plt
import meshio
import gmsh

# Paths
geo_file_paths = {
    "MSB Tank without Helicopter": r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_without_helicopter.geo',
    "MSB Tank with Helicopter": r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_with_helicopter.geo',
    "Gym Tank without Helicopter": r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Gym_tank_without_helicopter.geo',
    "Gym Tank with Helicopter": r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Gym_tank_with_helicopter.geo'
}
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Optimal_Mesh.msh'
output_vtk_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\ModeShape'

# Mesh size configurations
mesh_sizes = {
    "MSB Tank without Helicopter": 0.45,
    "MSB Tank with Helicopter":0.45,
    "Gym Tank without Helicopter": 0.45,
    "Gym Tank with Helicopter": 0.5
}

# Global constants
ndof = 2
num_node_per_element = 3
bulk_modulus = 2.1 * 1e9
E = np.array([210 * 1e9, 3 * bulk_modulus * (1 - 2 * 0.49), 70 * 1e9])
nu = np.array([0.3, 0.49, 0.35])
g = 9.81
density = np.array([2400, 1000, 2800])
gp = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
wt = np.array([1/6, 1/6, 1/6])

def generate_msh(global_mesh_size_factor, geo_file_path, output_msh_path):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", global_mesh_size_factor)
    gmsh.open(geo_file_path)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_msh_path)
    gmsh.finalize()

def Read_Data(output_msh_path):
    file = meshio.read(output_msh_path)
    nodes = file.points[:, :2]
    elements = file.cells_dict['triangle']
    surface_tags = file.cell_data_dict['gmsh:geometrical']['triangle']
    surface_elements = {}
    for tag in set(surface_tags):
        surface_elements[tag] = elements[surface_tags == tag]
    no_of_elements = len(elements)
    return nodes, surface_elements, no_of_elements

def Compute(M, K, nodes, surface_elements):
    for surface_tag, elements in surface_elements.items():
        rho = density[surface_tag - 1]
        E_ele = E[surface_tag - 1]
        nu_ele = nu[surface_tag - 1]
        Elasticity_matrix = (E_ele / (1 - nu_ele**2)) * np.array([[1, nu_ele, 0], [nu_ele, 1, 0], [0, 0, (1 - nu_ele)/2]])
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
                for ind in range(0, num_node_per_element):
                    gindex[2 * ind] = 2 * n[ind]
                    gindex[2 * ind + 1] = 2 * n[ind] + 1
                B = np.zeros((3, ndof * num_node_per_element))
                m = np.zeros((2, ndof * num_node_per_element))
                for i in range(num_node_per_element):
                    B[0][2 * i] = dphi[0][i]
                    B[1][2 * i + 1] = dphi[1][i]
                    B[2][2 * i] = dphi[1][i]
                    B[2][2 * i + 1] = dphi[0][i]
                    m[0][2 * i] = N[i]
                    m[1][2 * i + 1] = N[i]
                Ke = (B.T @ Elasticity_matrix @ B) * np.linalg.det(Jac) * wt[j]
                Me = m.T @ m * np.linalg.det(Jac) * wt[j]
                for i in range(2 * num_node_per_element):
                    for j in range(2 * num_node_per_element):
                        K[int(gindex[i]), int(gindex[j])] += Ke[i, j]
                        M[int(gindex[i]), int(gindex[j])] += Me[i, j] * rho

def save_vtk(nodes, eigenvectors, eigenvalue_idx, filename, elements):
    # Get displacements from eigenvector for each node in the mode shape
    displacements = np.real(eigenvectors[:, eigenvalue_idx]).reshape(-1, 2)
    point_data = {"Displacement": displacements}
    
    # Write to VTK file using meshio
    mesh = meshio.Mesh(points=nodes, cells=[("triangle", np.array(elements))], point_data=point_data)
    mesh.write(filename)

def run_analysis(tank_type):
    geo_file_path = geo_file_paths[tank_type]
    mesh_size = mesh_sizes[tank_type]

    generate_msh(mesh_size, geo_file_path, output_msh_path)
    nodes, elements, num_elements = Read_Data(output_msh_path)
    all_elements = np.vstack(list(elements.values()))
    num_node = nodes.shape[0]
    K = np.zeros((ndof * num_node, ndof * num_node))
    M = np.zeros((ndof * num_node, ndof * num_node))
    Compute(M, K, nodes, elements)
    L = np.linalg.inv(M) @ K
    eigenvalues, eigenvectors = np.linalg.eig(L)
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    frequency = sorted_eigenvalues[:5]
    print(frequency)

    for mode_idx in range(2):
        save_vtk(
            nodes, eigenvectors, sorted_indices[mode_idx], 
            f"{output_vtk_path}_Mode_{mode_idx + 1}.vtk", all_elements
        )

# Run analysis for both configurations
run_analysis("MSB Tank with Helicopter")
# run_analysis("Gym Tank without Helicopter")
# run_analysis("MSB Tank without Helicopter")
# run_analysis("Gym Tank with Helicopter")