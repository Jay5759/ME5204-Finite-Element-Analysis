import numpy as np

file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A2 - Numerical Integration\IITM_Map_Triangular.msh'

# Define the line ranges
node_start_line1 = 377
node_end_line1 = 926
node_start_line2 = 2849
node_end_line2 = 4588
element_start_line = 5323
element_end_line = 8981

nodes = []
elements = []

with open(file_path, 'r') as file:
    lines = file.readlines()

    # Get Nodes for outer border points
    for i in range(node_start_line1 - 1, node_end_line1, 3):
        parts = lines[i].split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        nodes.append((x, y, z))
    
    # Get Nodes for inner mesh points
    for i in range(node_start_line2 - 1, node_end_line2):
        parts = lines[i].split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        nodes.append((x, y, z))

    # Get Elements
    for i in range(element_start_line - 1, element_end_line):
        parts = lines[i].split()
        t1, t2, t3 = int(parts[1]), int(parts[2]), int(parts[3])
        elements.append((t1, t2, t3))

nodes = np.array(nodes)
elements = np.array(elements)

def jacobian(triangles):
    # 0-based indexing
    t1, t2, t3 = triangles[0] - 1, triangles[1] - 1, triangles[2] - 1

    x0, y0 = nodes[t1][0], nodes[t1][1]
    x1, y1 = nodes[t2][0], nodes[t2][1]
    x2, y2 = nodes[t3][0], nodes[t3][1]

    del_x_del_ep = x1 - x0
    del_y_del_ep = y1 - y0
    del_x_del_enta = x2 - x0
    del_y_del_enta = y2 - y0

    return del_x_del_ep * del_y_del_enta - del_x_del_enta * del_y_del_ep

def Area_IITM():
    Area_of_IITM = 0
    Actual_Area = 2.5 # km^2 equivalent to 617 acres
    weight = 0.5

    for triangles in elements:
        J = jacobian(triangles)
        Area_of_IITM += (J * weight)
    
    # Scaling Factor applied and converted to sq.km
    print(Area_of_IITM)
    Area_of_IITM = Area_of_IITM * 0.1 * 0.1 / (23 * 23)
    print("Area of IIT Madras is", Area_of_IITM,"km^2")
    
    Error = (Actual_Area - Area_of_IITM) / Actual_Area * 100
    print("Error =", Error, "%")

# Question 1 
Area_IITM()