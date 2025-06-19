import numpy as np
import meshio
import gmsh
import matplotlib.pyplot as plt

geo_files = {
    'MSB_with': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_with_helicopter.geo',
    'MSB_without': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\MSB_tank_without_helicopter.geo',
    'Gym_with': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Gym_tank_with_helicopter.geo',
    'Gym_without': r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Gym_tank_without_helicopter.geo'
}
output_msh_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Optimal_Mesh.msh'

# Optimal Mesh size factors
mesh_size_factors = {
    'MSB_with': 0.21,    
    'MSB_without': 0.41, 
    'Gym_with': 0.36,    
    'Gym_without': 0.4   
}

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

def Von_Mises_Stress(sig):
    sig_x, sig_y, sig_xy = sig[0], sig[1], sig[2]
    return np.sqrt(((sig_x - sig_y)**2 + sig_x**2 + sig_y**2 + 6 * (sig_xy**2)) / 2)

def Calculate_Stress(disp, nodes, surface_elements, filename):
    von_mises_stresses = np.zeros(nodes.shape[0])
    element_nodes = []
    max_stress = -np.inf
    max_stress_location = None

    for surface_tag, elements in surface_elements.items():
        E_ele = E[surface_tag - 1]
        nu_ele = nu[surface_tag - 1]
        Elasticity_matrix = (E_ele / (1 - nu_ele**2)) * np.array([[1, nu_ele, 0], [nu_ele, 1, 0], [0, 0, (1 - nu_ele)/2]])
        for e in elements:
            Se = np.zeros((3, 1))
            n = np.array(e[:3], dtype=int)
            coord = np.array([nodes[n[0]], nodes[n[1]], nodes[n[2]]])
            for j in range(3):
                pt = gp[j]
                N = np.array([1 - pt[0] - pt[1], pt[0], pt[1]])
                dN = np.array([[-1, -1], [1, 0], [0, 1]])
                Jac = dN.T @ coord
                X = N @ coord
                dphi = np.linalg.inv(Jac) @ dN.T
                u = np.zeros((6, 1))
                for ind in range(0, num_node_per_element):
                    u[2 * ind] = disp[2 * n[ind]]
                    u[2 * ind + 1] = disp[2 * n[ind] + 1]
                # Assembly
                B = np.zeros((3, ndof * num_node_per_element))
                for i in range(num_node_per_element):
                    B[0][2 * i] = dphi[0][i]
                    B[1][2 * i + 1] = dphi[1][i]
                    B[2][2 * i] = dphi[1][i]
                    B[2][2 * i + 1] = dphi[0][i]
                Se += Elasticity_matrix @ B @ u * np.linalg.det(Jac) * wt[j] 
            sig = Von_Mises_Stress(Se)
            von_mises_stresses[n[0]] = sig  # Assigning stress value to the first node of the element
            von_mises_stresses[n[1]] = sig  # Assigning stress value to the second node of the element
            von_mises_stresses[n[2]] = sig 
            element_nodes.append(n)

            # Check for maximum stress and update location
            if sig > max_stress:
                max_stress = sig
                max_stress_location = np.mean(nodes[n], axis=0)  # Average position of the triangle nodes

    # Print maximum stress location
    if max_stress_location is not None:
        print(f"Maximum Von Mises stress: {max_stress} at location: {max_stress_location}")

    ux = disp[::2]  # x-component of displacement at each node
    uy = disp[1::2]  # y-component of displacement at each node
    displacement_magnitude = np.sqrt(ux**2 + uy**2)  # Magnitude of displacement
    # Convert to VTK format
    cells = [("triangle", np.array(element_nodes))]
    point_data = {
        "Von_Mises_Stress": np.array(von_mises_stresses),
        "Displacement_X": ux,
        "Displacement_Y": uy,
        "Displacement_Magnitude": displacement_magnitude
    }
    
    # Create and save mesh as VTK file
    mesh = meshio.Mesh(
        points=np.array(nodes), 
        cells=cells, 
        point_data=point_data
    )
    mesh.write(filename)
    print(f"VTK file saved as {filename}")

def run_simulation(building_type, helicopter):
    # Set geo file and output file paths based on the building type and helicopter presence
    geo_file_key = f"{building_type}_{helicopter}"
    geo_file_path = geo_files[geo_file_key]
    global_mesh_size_factor = mesh_size_factors[geo_file_key]
    
    # Generate the .msh file from the .geo file
    generate_msh(global_mesh_size_factor, geo_file_path)
    
    # Read nodes, elements, and surface data
    nodes, surface_elements, num_elements = Read_Data()
    num_node = nodes.shape[0]
    
    # Initialize stiffness matrix and force vector
    K = np.zeros((ndof * num_node, ndof * num_node))
    F = np.zeros((ndof * num_node, 1))
    
    # Compute the stiffness matrix and force vector
    Compute(K, F, nodes, surface_elements)
    
    # Define boundary conditions for nodes with y-coordinate zero
    borders = []
    for i, node in enumerate(nodes):
        if node[1] == 0:
            borders.append(2 * i) 
            borders.append(2 * i + 1)
    
    # Solve for displacement
    displacement = apply_dirichlet_boundary_conditions(K, F, borders, ndof * num_node)
    
    # Generate VTK file paths for displacement and stress
    stress_vtk_file = f"{building_type}_{helicopter}_Von_Mises_Stress.vtk"
    
    # Calculate and save the stress field
    Calculate_Stress(displacement, nodes, surface_elements, stress_vtk_file)
    
    print(f"Simulation completed for {building_type} with helicopter: {helicopter}")
    print(f"Von Mises stress field saved as {stress_vtk_file}")

# Example usage:
run_simulation("MSB", "without")
# run_simulation("MSB", "with")
# run_simulation("Gym", "without")
# run_simulation("Gym", "with")