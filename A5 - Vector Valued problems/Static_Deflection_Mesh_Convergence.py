import numpy as np
import matplotlib.pyplot as plt
import meshio
import gmsh

# Define file paths
geo_files = {
    'MSB_with': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_with_helicopter.geo',
    'MSB_without': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_without_helicopter.geo',
    'Gym_with': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Gym_tank_with_helicopter.geo',
    'Gym_without': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Gym_tank_without_helicopter.geo'
}
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Output.msh'

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

def generate_msh(global_mesh_size_factor, geo_file_path):
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
    no_of_elements = len(elements)
    return nodes, surface_elements, no_of_elements

def Compute(K, F, nodes, surface_elements):
    for surface_tag, elements in surface_elements.items():
        rho = density[surface_tag - 1]
        E_ele = E[surface_tag - 1]
        nu_ele = nu[surface_tag - 1]
        Elasticity_matrix = (E_ele / (1 - nu_ele**2)) * np.array([[1, nu_ele, 0], [nu_ele, 1, 0], [0, 0, (1 - nu_ele)/2]])
        body_load = np.array([0, -rho * g])
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
                    gindex[2 * ind] = 2 * n[ind]    # x DOF for node ind
                    gindex[2 * ind + 1] = 2 * n[ind] + 1 # y DOF for node ind
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
                for i in range(2 * num_node_per_element):
                    for j in range(2 * num_node_per_element):
                        K[int(gindex[i]), int(gindex[j])] += Ke[i, j]
                    F[int(gindex[i])] += Fe[i]

def apply_dirichlet_boundary_conditions(K, F, border, num_dofs):
    all_indices = np.arange(num_dofs)
    free_indices = np.setdiff1d(all_indices, border)
    K_reduced = K[np.ix_(free_indices, free_indices)]
    F_reduced = F[free_indices]
    displacement_reduced = np.linalg.inv(K_reduced) @ F_reduced
    displacement_full = np.zeros(num_dofs)
    displacement_full[free_indices] = displacement_reduced.flatten()
    return displacement_full

def run_simulation(structure, helicopter_status, mesh_sizes, check_points):
    geo_file_key = f"{structure}_{helicopter_status}"
    geo_file_path = geo_files[geo_file_key]
    
    displacement_prev = np.zeros((500, 1))
    error = []
    h = []
    ele = []
    nod = []

    for ms in mesh_sizes:
        generate_msh(ms, geo_file_path)
        nodes, elements, num_elements = Read_Data()
        num_node = nodes.shape[0]
        
        # Boundary condition setup
        borders = []
        for i, node in enumerate(nodes):
            if node[1] == 0:  # Check if the y-coordinate is zero
                borders.append(2 * i)      # x DOF of the node
                borders.append(2 * i + 1)  # y DOF of the node
        
        ele.append(num_elements)
        nod.append(num_node)
        
        K = np.zeros((ndof * num_node, ndof * num_node))
        F = np.zeros((ndof * num_node, 1))
        
        Compute(K, F, nodes, elements)
        displacement = apply_dirichlet_boundary_conditions(K, F, borders, ndof * num_node)

        # Calculate error
        if ms == mesh_sizes[0]:
            error.append(0)
            displacement_prev = displacement
            continue

        e = 0
        for i in range(0, len(check_points), 2):
            u = displacement[check_points[i]]
            v = displacement[check_points[i + 1]]
            u_prev = displacement_prev[check_points[i]]
            v_prev = displacement_prev[check_points[i + 1]]
            net_disp = np.sqrt(u**2 + v**2)
            net_disp_prev = np.sqrt(u_prev**2 + v_prev**2)
            e += ((net_disp - net_disp_prev) / net_disp_prev) ** 2

        e = np.sqrt(e)
        error.append(e)
        displacement_prev = displacement

    return np.array(error), np.array(ele), np.array(nod)

def Plot_Error_vs_Mesh(ele, error):
    plt.figure(figsize=(8, 6))
    plt.plot(ele[1:], error[1:], marker='o', linestyle='-', label='Error vs Mesh Size')
    plt.xlabel('No. of elements')
    plt.ylabel('Error values')
    plt.title('Error values vs. No. of elements')
    plt.grid(True)
    # plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

# MSB tank without Helicopter
msb_without_mesh_size = np.array([0.8, 0.7, 0.61, 0.5, 0.45, 0.41])
msb_without_check_points = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
error, ele, nod = run_simulation('MSB', 'without', msb_without_mesh_size, msb_without_check_points)

# MSB tank with Helicopter
msb_without_mesh_size = np.array([0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.21])
msb_without_check_points = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
# error, ele, nod = run_simulation('MSB', 'with', msb_without_mesh_size, msb_without_check_points)

# Gym tank without Helicopter
gym_without_mesh_size = np.array([1, 0.74, 0.6, 0.5, 0.45, 0.4])
gym_without_check_points = np.array([4, 5, 6, 7, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
# error, ele, nod = run_simulation('Gym', 'without', gym_without_mesh_size, gym_without_check_points)

# Gym tank with Helicopter
gym_with_mesh_size = np.array([1, 0.8, 0.6, 0.5, 0.4, 0.36])
gym_with_check_points = np.array([4, 5, 6, 7, 12, 13, 14, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67])
# error, ele, nod = run_simulation('Gym', 'with', gym_with_mesh_size, gym_with_check_points)

print(nod)
print(ele)
print(error)
Plot_Error_vs_Mesh(ele, error)