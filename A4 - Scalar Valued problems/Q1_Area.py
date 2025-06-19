import numpy as np
import meshio

file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A5 - Vector Valued problems\GMSH files\Output.msh'

def Read_Data():
    file = meshio.read(file_path)
    nodes = file.points[:, :2]
    elements = file.cells_dict['triangle']
    surface_tags = file.cell_data_dict['gmsh:geometrical']['triangle']
    surface_elements = {}
    for tag in set(surface_tags):
        surface_elements[tag] = elements[surface_tags == tag]
    return nodes, surface_elements

nodes, surface_elements = Read_Data()
num_nodes = nodes.shape[0]

def jacobian(triangles):
    t1, t2, t3 = triangles[0], triangles[1], triangles[2]

    x0, y0 = nodes[t1][0], nodes[t1][1]
    x1, y1 = nodes[t2][0], nodes[t2][1]
    x2, y2 = nodes[t3][0], nodes[t3][1]

    del_x_del_ep = x1 - x0
    del_y_del_ep = y1 - y0
    del_x_del_enta = x2 - x0
    del_y_del_enta = y2 - y0

    return del_x_del_ep * del_y_del_enta - del_x_del_enta * del_y_del_ep

def Area_IITM(elements):
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
    return Area_of_IITM

# Question 1 
area = 0
for surface_tags, elements in surface_elements.items():
    area += Area_IITM(elements)

print(area)