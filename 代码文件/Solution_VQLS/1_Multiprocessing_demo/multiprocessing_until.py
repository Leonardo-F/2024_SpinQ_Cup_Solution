r"""
给出duo，运行 VQLS 需要的函数
"""

# 导入必要的库
import pennylane as qml
from pennylane import numpy as qml_np
import numpy as np

import multiprocessing as mp


# 定义矩阵 A 和向量 b
A_matrix = np.array([[2, 5, -13],
              [1, -3, 1],
              [-5, 6, 8]]).astype(float)

b_vec = np.array([1000, 0, -600]).astype(float)
x_vec = np.array([1200, 500, 300]).astype(float)

# 将矩阵 A 变成一个厄尔米特矩阵
H_matrix = np.block([[np.zeros((3, 3)), A_matrix], [A_matrix.T, np.zeros((3, 3))]])

# 将 H_matrix 变为 8×8，方便用 pauli 算子的线性组合来表示
H_matrix_2 = np.zeros((8, 8))

for i in range(H_matrix.shape[0]):
    for j in range(H_matrix.shape[0]):
        H_matrix_2[i,j] = H_matrix[i,j]

# 对厄密矩阵进行 pauli 分解
# 使用 qml.pauli_decompose 进行 Pauli 分解，使用 3 个 qubit
H_decomposition = qml.pauli_decompose(H_matrix_2, wire_order=[0, 1, 2])

coeffs = H_decomposition.coeffs
c = coeffs
ops = H_decomposition.ops

# 相对应的 b 和 x  也要进行维度拓展，以适应 H_matrixs
b_vec_2 = np.zeros((8))
b_vec_2[:3] = b_vec
b_vec_2 = b_vec_2.reshape((-1,1))

x_vec_2 = np.zeros((8))
x_vec_2[3:6] = x_vec
x_vec_2 = x_vec_2.reshape((-1,1))
x_norm = np.linalg.norm(x_vec_2)
x_normalized = (x_vec_2 / x_norm).reshape(-1)


# 计算范数
b_norm = np.linalg.norm(b_vec_2)
# 归一化向量，转回行向量，方便后续计算
b_normalized = (b_vec_2 / b_norm).reshape(-1)


# 构造设备
qubits = 3
ancilla_idx = qubits
deep_layer = 3
dev = qml.device("lightning.qubit", wires=qubits+1)

@qml.qnode(dev, interface="autograd")
# 生成 b_vec_2 的量子电路 ， |b_vec_2 > = U_b |0>
def U_b_circuit(state):
    qml.MottonenStatePreparation(state_vector=state, wires=[0,1,2])
    return qml.state()

# 生成解 |x> 的变分电路
def variational_x_circuit_2(weights, deep_layer):

    for idx in range(3):
        qml.Hadamard(wires=idx)

    for deep in range(deep_layer):
        for k in range(3):
            qml.RY(weights[k+deep*3], wires=k)
    
        for j in range(2):
            qml.CNOT(wires=[j, j+1])
        qml.CNOT(wires=[2, 0])    

        
# 损失函数计算

# Hadamard 测试，用于计算损失函数的一部分
@qml.qnode(dev)
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    # 辅助比特
    qml.Hadamard(wires=ancilla_idx)

    # 损失值 mu 的虚部计算
    # 相位门
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # |x>
    variational_x_circuit_2(weights, deep_layer)

    # 受控 A_l 
    H_l = qml.Hamiltonian(np.array([1.]), [ops[l]])
    H_l_matrix = H_l.sparse_matrix().toarray()
    qml.ControlledQubitUnitary(H_l_matrix, control_wires=[ancilla_idx], wires=range(qubits), control_values=[1])

    # U_b dagger
    qml.adjoint(qml.MottonenStatePreparation(state_vector=b_normalized, wires=range(qubits)))

    # 受控 Z
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # U_b
    qml.MottonenStatePreparation(state_vector=b_normalized, wires=range(qubits))

    # 受控 A_lp
    H_lp = qml.Hamiltonian(np.array([1.]), [ops[lp]])
    H_lp_matrix = H_lp.sparse_matrix().toarray()
    H_lp_matrix_dagger = H_lp_matrix.conj().T
    qml.ControlledQubitUnitary(H_lp_matrix_dagger, control_wires=[ancilla_idx], wires=range(qubits), control_values=[1])

    # 辅助位
    qml.Hadamard(wires=ancilla_idx)
    
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


# 使用并行化，多核心快速运算
def mu(weights, l=None, lp=None, j=None):
    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag
    
def compute_partial_norm(args):
    l, c, weights = args
    partial_sum = 0.0
    for lp in range(len(c)):
        partial_sum += c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)
    return partial_sum

def psi_norm(weights, c):
    with mp.Pool(processes=180) as pool:
        results = pool.map(compute_partial_norm, [(l, c, weights) for l in range(len(c))])
    norm = sum(results)
    return abs(norm)

def compute_partial_mu_sum(args):
    l, c, weights, qubits = args
    partial_sum = 0.0
    for lp in range(len(c)):
        for j in range(qubits):
            partial_sum += c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)
    return partial_sum

def cost_loc(weights):
    with mp.Pool(processes=180) as pool:
        results = pool.map(compute_partial_mu_sum, [(l, c, weights, qubits) for l in range(len(c))])
    mu_sum = sum(results)
    mu_sum = abs(mu_sum)

    # 计算 psi_norm
    norm = psi_norm(weights, c)

    return 0.5 - 0.5 * mu_sum / (qubits * norm)

# 创建一个函数来传递参数
def wrapped_cost(weights):
    return cost_loc(weights, c, qubits)


# 手动计算梯度, Parameter-shift Rule 
def numerical_gradient(cost_fn, weights, epsilon=np.pi/2):
    grad = np.zeros_like(weights)
    for i in range(len(weights)):
        weights_plus = np.copy(weights)
        weights_minus = np.copy(weights)
        weights_plus[i] += epsilon
        weights_minus[i] -= epsilon
        
        cost_plus = cost_fn(weights_plus)
        cost_minus = cost_fn(weights_minus)
        
        grad[i] = (cost_plus - cost_minus) / 2
    return grad

# 自定义优化器类
class CustomOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step_and_cost(self, cost_fn, weights):
        grad = numerical_gradient(cost_fn, weights)
        cost = cost_fn(weights)
        print(cost)
        weights -= self.learning_rate * grad
        cost = cost_fn(weights)
        return weights, cost
        

# 用于最终结果的采样
dev_2 = qml.device('lightning.qubit', wires=qubits, shots=2048)
@qml.qnode(dev_2)
def sample_and_state(weights, method='sample'):


    for idx in range(3):
        qml.Hadamard(wires=idx)

    for deep in range(deep_layer):
        # A very minimal variational circuit.
        for k in range(3):
            qml.RY(weights[k+deep*3], wires=k)
    
        for j in range(2):
            qml.CNOT(wires=[j, j+1])
        qml.CNOT(wires=[2, 0])    
    
    if method == 'sample':    
        return qml.sample()
    elif method == 'state':    
        return qml.state()