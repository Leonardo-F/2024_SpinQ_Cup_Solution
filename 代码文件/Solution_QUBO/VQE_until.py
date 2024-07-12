r"""
导入线性方程组的哈密顿量，使用 VQE 算法进行求解，获得其基态及基态能量

"""

# 导入必要的库
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from pennylane import numpy as qml_np
# import jax

# 导入生成哈密顿量相关的库
import hamiltonian_until


# # 构造设备
tol_qubits = hamiltonian_until.tol_qubits
# dev = qml.device("lightning.qubit", wires=tol_qubits)
hamiltonian = hamiltonian_until.hamiltonian
hamiltonian_matrix = qml_np.array(hamiltonian_until.hamiltonian_matrix_real)

# 变分电路，用于学习基态
# 期望值计算
# @qml.qnode(dev, interface="autograd")
def variational_circuit(weights, deep_layer):

    for idx in range(tol_qubits):
        qml.Hadamard(wires=idx)

    for deep in range(deep_layer):
        # A very minimal variational circuit.
        for k in range(tol_qubits):
            qml.RY(weights[k+deep*tol_qubits], wires=k)
    
        for j in range(tol_qubits-1):
            qml.CNOT(wires=[j, j+1])
        qml.CNOT(wires=[tol_qubits-1, 0])


def variational_circuit_expval(weights, deep_layer):

    # 放置变分电路，返回哈密顿量期望值
    variational_circuit(weights, deep_layer)
        
    return qml.expval(hamiltonian)    

# 返回对应的量子态
# @qml.qnode(dev, interface="autograd")
def variational_circuit_state(weights, deep_layer):

    # 放置变分电路，返回电路量子态
    variational_circuit(weights, deep_layer)
    
    return qml.state()

def variational_circuit_prob(weights, deep_layer):
    # 放置变分电路，返回电路的理论上的概率
    variational_circuit(weights, deep_layer)
    
    return qml.probs()

# # 定义损失函数   
# def cost_fn(param, deep_layer, method='expval'):
    
#     if method == 'state':
#         # state 模式暂时还未在优化器中跑通
#         # 返回量子态，默认是列向量
#         circuit_state = variational_circuit_state(param, deep_layer)
        
#         # print("circuit_state = ", circuit_state)
#         # print(np.all(circuit_state.conj() == circuit_state))
#         # 由于采用的变分电路，生成的量子态不会产生虚部，所以内积计算我们可以不做共轭操作
        
#         # 量子态与哈密顿量的内积，作为损失值 
#         cost_value_1 = circuit_state.reshape(1,-1) @ hamiltonian_matrix @ circuit_state.reshape(-1,1)
#         cost_value = cost_value_1[0][0].real

#     elif method == 'expval':
#         # 使用 qml.expval 来计算量子态与哈密顿量的期望值
#         # 哈密顿量项数变多了，运算可能变得很慢
#         cost_value = variational_circuit_expval(param, deep_layer)
        
#     # print("cost_value = ", cost_value)
#     # print("type cost_value = ", type(cost_value))   
        
#     return cost_value


