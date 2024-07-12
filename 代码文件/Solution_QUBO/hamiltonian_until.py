import re
import math
import pennylane as qml
import numpy as np
# import matplotlib.pyplot as plt



# 生成线性方程组的 Hamiltonian 的系数，参考 Algorithm 1，但不全是他，仅针对正整数的 x  
def linear_equation_hamiltonian(A_matrix, b_vec, num_qubits=4):
    n = A_matrix.shape[0]  # 变量的数量
    m = num_qubits # 换个变量名，方便写
    
    # print("n = ", n)
    # total_num = n * m  # 每个变量使用 m 个 0 或 1表示 (二进制)
    
    coeffs = [] # 存储每一个 Pauli 项的系数
    obs = [] # 存储 Pauli 算子
    
    def binary_expansion(x):
        return 2**x
    
    # 一次项，参考推导中的公式
    for k in range(n):
        for i in range(n):
            for l in range(m):
                coeff_value = (A_matrix[k,i] ** 2) * binary_expansion(2 * l - 1)
                
                coeffs.append(coeff_value)
                obs.append(qml.Identity(m*i+l))
                
                coeffs.append(-coeff_value)
                obs.append(qml.PauliZ(m*i+l))    
                       
                
    # 二次项
    for k in range(n):
        for i in range(n):
            for l1 in range(m):
                for l2 in range(m):
                    if l1 != l2:
                        coeff_value_2 = (A_matrix[k,i] ** 2) * binary_expansion(l1 + l2 - 2)
                        
                        coeffs.append(coeff_value_2)
                        obs.append(qml.Identity(m*i+l1) @ qml.Identity(m*i+l2))             

                        coeffs.append(-coeff_value_2)
                        obs.append(qml.Identity(m*i+l1) @ qml.PauliZ(m*i+l2))   

                        coeffs.append(-coeff_value_2)
                        obs.append(qml.PauliZ(m*i+l1) @ qml.Identity(m*i+l2))   

                        coeffs.append(coeff_value_2)
                        obs.append(qml.PauliZ(m*i+l1) @ qml.PauliZ(m*i+l2))   


    # 二次项
    for k in range(n):
        for i in range(0,n-1):
            for j in range(i+1,n):
                for l1 in range(m):
                    for l2 in range(m):
                        coeff_value_3 = A_matrix[k,i] * A_matrix[k,j] * binary_expansion(l1 + l2 -1)
                        

                        coeffs.append(coeff_value_3)
                        obs.append(qml.Identity(m*i+l1))   

                        coeffs.append(-coeff_value_3)
                        obs.append(qml.Identity(m*i+l1) @ qml.PauliZ(m*j+l2))  

                        coeffs.append(-coeff_value_3)
                        obs.append(qml.PauliZ(m*i+l1) @ qml.Identity(m*j+l2))  

                        coeffs.append(coeff_value_3)
                        # print("m*i+l1 = ", m*i+l1)
                        # print('i = ', i)
                        # print("m*j+l2 = ", m*j+l2)
                        # print('j = ', j)
                        # print(qml.PauliZ(m*i+l1) @ qml.PauliZ(m*j+l2))
                        obs.append(qml.PauliZ(m*i+l1) @ qml.PauliZ(m*j+l2)) 


    # 一次项                    
    for k in range(n):
        for i in range(n):
            for l in range(m):
                coeff_value_4 = b_vec[k] * A_matrix[k,i] * binary_expansion(l)
                
                coeffs.append(-coeff_value_4)
                obs.append(qml.Identity(m*i+l))

                
                coeffs.append(coeff_value_4)
                obs.append(qml.PauliZ(m*i+l)) 
                
    # # 常数项
    # coeff_value_5 = 0
    # for i in range(len(b_vec)):
    #     coeff_value_5 += b_vec[i]**2
        
    # coeffs.append(coeff_value_5)
    # obs.append(qml.Identity(11))      
    
    
    
    return coeffs, obs            

# 查看特征向量，由哪些单位向量构成
def binary_conversion(state):
    r"""
    判断特征向量的每一项，大于 0 的数都要将其序号转换为二进制
    """
    binary_list = [] # 二进制
    element_list = [] # 系数
    
    # 把多维数组变成一维数组
    state = state.flatten()
    
    num_qubit = int(math.log(len(state),2))

    for i in range(len(state)):
        element = state[i]
        
        if abs(element) >= 1e-6:
            binary_i = bin(np.array(i))[2:]
            
            for i in range(num_qubit - len(binary_i)):
                binary_i = '0' + binary_i
                
            binary_list.append(binary_i)
            element_list.append(element)
            
    return binary_list, element_list



# 生成本题的哈密顿量

# 定义线性方程组的矩阵 A 和向量 b
A_matrix = np.array([[2, 5, -13],
              [1, -3, 1],
              [-5, 6, 8]]).astype(float)
b_vec = np.array([1000, 0, -600]).astype(float)
x_vec = np.array([1200, 500, 300]).astype(float)

# 对向量 b 和 x 进行缩放，以方便使用 4 位的二进制进行表示 x 中的每个元素
# 分别为 1100 0101 0011
b_vec = b_vec/100
x_vec = x_vec/100

# 每个变量用 num_qubits 进行编码，总共需要 tol_qubits
num_qubits = 4
tol_qubits = 12

# 生成对应的线性方程组哈密顿量
coeffs, obs  = linear_equation_hamiltonian(A_matrix, b_vec, num_qubits)
hamiltonian = qml.Hamiltonian(coeffs, obs)

# 哈密顿量转换成矩阵，并计算最小特征值及其特征向量
hamiltonian_sparse = hamiltonian.sparse_matrix()
hamiltonian_matrix = hamiltonian_sparse.toarray()

# 舍弃矩阵 hamiltonian_matrix 的虚部，仅保留实部，本身哈密顿量就是没有虚部
hamiltonian_matrix_real = hamiltonian_matrix.real

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(hamiltonian_matrix_real.astype(float))

# 找到最小特征值的索引
min_index = np.argmin(eigenvalues.astype(float))

# 最小特征值
min_eigenvalue = eigenvalues[min_index]

# 对应的特征向量
min_eigenvector = eigenvectors[:, min_index]

# 基态的二进制形式，应为 |001110101100>
bin_vec = binary_conversion(min_eigenvector)