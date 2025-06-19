import numpy as np
import matplotlib.pyplot as plt
import meshio

file_path = r'C:\Users\jiten\OneDrive\Desktop\Sem 7\ME5204 - Finite Element Analysis\Assignments\A4 - Scalar Valued problems\GMSH files\Optimal_Mesh.msh'

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
rho = np.array([580, # trees
                522, # Academic and residential zone
                522, # hostel zone
                1000 # water bodies
               ])
cp = np.array([1760, # trees
               1584, # Academic and residential zone
               1584, # hostel zone
               4180 # water bodies
              ])

def func(x):
    x *= (100 / 23)
    numerator = P * np.exp(-1.0 * (np.linalg.norm(x - x0))**2 / (2 * sig**2))
    denomenator = np.sqrt(2 * np.pi * sig**2)
    return numerator / denomenator

def Assembly(k, f, K, F, m, M, n):
    for i in range(3):
        for j in range(3):
            M[n[i]][n[j]] += m[i][j]
            K[n[i]][n[j]] += k[i][j]
        F[n[i]] += f[i]

def Compute_Interpolated_Function(M, K, F, k_therm_values, rho_values, cp_values, nodes, surface_elements):
    for surface_tag, elements in surface_elements.items():
        k_therm_ele = k_therm_values[surface_tag - 1]
        rho_ele = rho_values[surface_tag - 1]
        cp_ele = cp_values[surface_tag - 1]
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
                m = np.outer(N, N) * np.linalg.det(Jac) * wt[j] * rho_ele * cp_ele
                k = np.outer(dphi, dphi) * np.linalg.det(Jac) * wt[j] * k_therm_ele
                f = func(X) * wt[j] * np.linalg.det(Jac) * N
                Assembly(k, f, K, F, m, M, n)

M = np.zeros((num_nodes, num_nodes))
K = np.zeros((num_nodes, num_nodes))
F = np.zeros((num_nodes, 1))
Compute_Interpolated_Function(M, K, F, k_therm, rho, cp, nodes, surface_elements)

cmp = np.array([4877, 3444, 4270, 4837])

def Neumann(del_t, time_t, scheme):
    Final_t = []
    Final_C = []
    for t in del_t:
        C = np.full((num_nodes, 1), 36.0)
        C_t = []
        time = 0.0
        while True:
            cnt = 0
            if scheme == 0: C_next = np.linalg.inv(M / t) @ ((M / t + F - K) @ C)
            else: C_next = np.linalg.inv(M / t + K) @ ((M / t + F) @ C)
            C = C_next
            for i in cmp:
                if abs(time - time_t) <= 1e-6:
                    C_t.append(C[i])
                print(C[i])
                if C[i] >= 46.0:
                    cnt += 1
            if cnt == cmp.shape[0]:
                break
            time += t
            print(f'Time: {time}')
        C_t = np.array(C_t)
        Final_t.append(time)
        Final_C.append(C_t)
    Final_t = np.array(Final_t)
    Final_C = np.array(Final_C, dtype=object)
    return Final_t, Final_C

def Dirichlet(del_t, time_t, scheme):
    Final_t = []
    Final_C = []
    num_fixed_nodes = 280
    num_total_nodes = num_nodes
    free_nodes = np.arange(num_fixed_nodes, num_total_nodes)
    for t in del_t:
        C = np.full((num_nodes, 1), 36.0)
        C_t = []
        time = 0.0
        while True:
            cnt = 0
            M_free = M[np.ix_(free_nodes, free_nodes)]
            K_free = K[np.ix_(free_nodes, free_nodes)]
            F_free = F[free_nodes]
            
            if scheme == 0: 
                right_side = (M_free / t + F_free - K_free) @ C[free_nodes]
                right_side -= (M_free[:, :num_fixed_nodes] @ (np.array([36.0 / t]) * np.ones(num_fixed_nodes).reshape(-1, 1)))
                C_next = np.linalg.inv(M_free / t) @ (right_side)
            else: 
                right_side = (M_free / t + F_free) @ C[free_nodes]
                right_side -= (M_free[:, :num_fixed_nodes] @ (np.array([36.0 / t]) * np.ones(num_fixed_nodes).reshape(-1, 1)) + K_free[:, :num_fixed_nodes] @ (np.array([36.0]) * np.ones(num_fixed_nodes).reshape(-1, 1))) 
                C_next = np.linalg.inv(M_free / t + K_free) @ (right_side)
            C[free_nodes] = C_next

            for i in cmp:
                if abs(time - time_t) <= 1e-6:
                    C_t.append(C[i])
                print(C[i])
                if C[i] >= 46.0:
                    cnt += 1
            if cnt == cmp.shape[0]:
                break
            time += t
            print(f'Time: {time}')
        C_t = np.array(C_t)
        Final_t.append(time)
        Final_C.append(C_t)
    Final_t = np.array(Final_t)
    Final_C = np.array(Final_C, dtype=object)
    return Final_t, Final_C
    
del_t = np.array([0.1, 0.5, 1.0])
Total_time_Explicit_Neumann, C_t_Neumann = Neumann(del_t, 2.0, 0)
print(Total_time_Explicit_Neumann)
# print(C_t_Neumann)
# Total_time_Implicit_Neumann, C_t_Neumann = Neumann(del_t, 2.0, 1)
# print(Total_time_Implicit_Neumann)
# print(C_t_Neumann)

Total_time_Explicit_Dirichlet, C_t_Dirichlet = Dirichlet(del_t, 2.0, 0)
print(Total_time_Explicit_Dirichlet)
print(C_t_Dirichlet)
# Total_time_Implicit_Dirichlet, C_t_Dirichlet = Dirichlet(del_t, 2.0, 1)
# print(Total_time_Implicit_Dirichlet)
# print(C_t_Dirichlet)

# plt.figure(figsize=(8, 6))
# for i, C in enumerate(C_t_Dirichlet):
#     x = np.arange(len(C))  # Indexes of C_t points on x-axis
#     plt.plot(x, C, label=f'del_t = {del_t[i]}')  # Plot C_t values for each del_t
# plt.xlabel('Index of C_t Points')
# plt.ylabel('Values of C_t')
# plt.title('C_t values for different del_t')
# plt.legend()
# plt.grid(True)
# plt.show()