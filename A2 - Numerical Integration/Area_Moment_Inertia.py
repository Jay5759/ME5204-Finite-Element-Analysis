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

points = np.array([[2/3, 1/6], [1/6, 2/3], [1/6, 1/6]])
GC_coords = np.array([286.9, 260.6])

def jacobian(triangles):
    # 0-based indexing
    t1, t2, t3 = triangles[0] - 1, triangles[1] - 1, triangles[2] - 1
    centroid = np.zeros((1, 2))

    x0, y0 = nodes[t1][0], nodes[t1][1]
    x1, y1 = nodes[t2][0], nodes[t2][1]
    x2, y2 = nodes[t3][0], nodes[t3][1]

    centroid[0][0] = (x0 + x1 + x2)/3
    centroid[0][1] = (y0 + y1 + y2)/3

    del_x_del_ep = x1 - x0
    del_y_del_ep = y1 - y0
    del_x_del_enta = x2 - x0
    del_y_del_enta = y2 - y0

    return del_x_del_ep * del_y_del_enta - del_x_del_enta * del_y_del_ep, centroid

def f(points, triangle, centroid):
    # 0-based indexing
    t1, t2, t3 = triangle[0] - 1, triangle[1] - 1, triangle[2] - 1
    c0, c1 = centroid[0][0], centroid[0][1]
    epsilon, enta = points[0], points[1]

    x0, y0 = nodes[t1][0], nodes[t1][1]
    x1, y1 = nodes[t2][0], nodes[t2][1]
    x2, y2 = nodes[t3][0], nodes[t3][1]

    X = (1.0 - epsilon - enta) * x0 + epsilon * x1 + enta * x2
    Y = (1.0 - epsilon - enta) * y0 + epsilon * y1 + enta * y2

    return (X - c0)**2 + (Y - c1)**2

def Area_Moment():
    Area_Moment = 0
    Total_area = 0
    Centroid = np.zeros((1, 2))

    for triangle in elements:
        area_of_triangle, centroid_of_triangle = jacobian(triangle)
        area_of_triangle /= 2
        Total_area += area_of_triangle
        Centroid += area_of_triangle * centroid_of_triangle

    Centroid /= Total_area

    for triangle in elements:
        J, _ = jacobian(triangle)
        Area_Moment += ((f(points[0], triangle, Centroid) + f(points[1], triangle, Centroid) + f(points[2], triangle, Centroid)) * J) / 6

    print("Area Moment of Inertia about Centroid is",Area_Moment * 0.1 ** 4/ (23 ** 4),"km^4")

    # Question 3
    Area_Moment_GC = Area_Moment + Total_area * ((Centroid[0][0] - GC_coords[0])**2 + (Centroid[0][1] - GC_coords[1])**2)
    print("Area Moment of Inertia about Gajendra Circle is",Area_Moment_GC * 0.1 ** 4/ (23 ** 4),"km*4")

#Question 2
Area_Moment()