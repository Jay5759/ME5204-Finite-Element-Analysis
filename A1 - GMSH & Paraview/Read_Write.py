import meshio
import numpy as np
import os

# Change the file location accordingly for code to run
# for Triangular mesh
input_file = r"C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A1 - GMSH & Paraview\IITM_Map_Triangular.msh"
output_file = r"C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A1 - GMSH & Paraview\IITM_Map_Triangular.vtk"

# for Quadrilateral mesh
# input_file = r"C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A1 - GMSH & Paraview\IITM_Map_Quadrilateral.msh"
# output_file = r"C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A1 - GMSH & Paraview\IITM_Map_Quadrilateral.vtk"

# Check if the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"The input file {input_file} does not exist.")

# Read the .msh file
try:
    mesh = meshio.read(input_file)
except Exception as e:
    raise RuntimeError(f"Error reading the file {input_file}: {e}")

# Ensure 3D vectors for VTK compatibility
if mesh.points.shape[1] == 2:
    mesh.points = np.pad(mesh.points, ((0, 0), (0, 1)), 'constant')

# Convert data types to ensure VTK compatibility
def convert_data(data):
    if data.dtype == np.int32:
        return data.astype(np.int64)
    if data.dtype == np.float32:
        return data.astype(np.float64)
    return data

# Apply conversion to point and cell data
converted_point_data = {k: convert_data(v) for k, v in mesh.point_data.items()}
converted_cell_data = {k: [convert_data(d) for d in v] for k, v in mesh.cell_data.items()}

# Create a new mesh with converted data types
new_mesh = meshio.Mesh(
    points=mesh.points,
    cells=mesh.cells,
    point_data=converted_point_data,
    cell_data=converted_cell_data,
    field_data=mesh.field_data
)

# Write the new mesh to a .vtk file
try:
    meshio.write(output_file, new_mesh)
    print(f"Conversion complete: {output_file}")
except Exception as e:
    raise RuntimeError(f"Error writing the file {output_file}: {e}")