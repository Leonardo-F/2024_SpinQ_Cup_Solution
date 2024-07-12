r"""
给出含躁环境下，运行 VQLS 需要的函数

损失函数部分，使用的是 pennylane 教程中的形式 https://pennylane.ai/qml/demos/tutorial_vqls/

"""

# 导入必要的库
import pennylane as qml
from pennylane import numpy as qml_np
import numpy as np

# from qiskit_aer import noise
# from qiskit_aer import AerSimulator
# import pickle


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


# 构造含噪声的量子设备
qubits = 3
ancilla_idx = qubits
# shots = None
shots = 1024
# def configure_backend():
#     with open('NoiseModel/fakemontreal.pkl', 'rb') as file:
#         noise_model = noise.NoiseModel().from_dict(pickle.load(file))
#     backend = AerSimulator(
#         method='statevector',
#         noise_model=noise_model,
#     )

#     return backend

# dev_noise = qml.device('qiskit.aer', wires=qubits+1, shots=shots, backend=configure_backend())
dev_noise = qml.device('default.mixed', wires=qubits+1)


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

# 噪声系数
def sigmoid(x):
    return 1/(1+np.exp(-x))

        
# 损失函数计算

# Hadamard 测试，用于计算损失函数的一部分
@qml.qnode(dev_noise)
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    # 辅助比特
    qml.Hadamard(wires=ancilla_idx)

    # 损失值 mu 的虚部计算
    # 相位门
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # |x>
    variational_x_circuit_2(weights=weights, deep_layer=3)

    # 受控 A_l 
    H_l = qml.Hamiltonian(np.array([1.]), [ops[l]])
    H_l_matrix = H_l.sparse_matrix().toarray()
    qml.ControlledQubitUnitary(H_l_matrix, control_wires=[ancilla_idx], wires=range(qubits), control_values=[1])

    # U_b dagger
    qml.adjoint(qml.MottonenStatePreparation(state_vector=b_normalized, wires=range(qubits)))

    # # 受控 Z
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

    # 添加去极化噪声
    qml.DepolarizingChannel(0.01, wires=0)
    qml.DepolarizingChannel(0.01, wires=1)
    qml.DepolarizingChannel(0.01, wires=2)
    qml.DepolarizingChannel(0.01, wires=3)

    return qml.expval(qml.PauliZ(wires=ancilla_idx))

def mu(weights, l=None, lp=None, j=None):
    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag


def psi_norm(weights):
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)

def cost_loc(weights):
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    return 0.5 - 0.5 * mu_sum / (qubits * psi_norm(weights))


# 生成解 |x> 的变分电路
dev_noise_2 = qml.device('default.mixed', wires=qubits, shots=2048)
@qml.qnode(dev_noise_2)
def prepare_and_sample(weights, noise_x):

    deep_layer = 3
    for idx in range(3):
        qml.Hadamard(wires=idx)

    for deep in range(deep_layer):
        # A very minimal variational circuit.
        for k in range(3):
            qml.RY(weights[k+deep*3], wires=k)
    
        for j in range(2):
            qml.CNOT(wires=[j, j+1])
        qml.CNOT(wires=[2, 0])    
        
    qml.DepolarizingChannel(noise_x, wires=0)
    qml.DepolarizingChannel(noise_x, wires=1)
    qml.DepolarizingChannel(noise_x, wires=2)     
    
    return qml.probs(wires=[0, 1, 2])