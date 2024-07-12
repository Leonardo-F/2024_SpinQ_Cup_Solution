r"""
构造受控的 A_l 门
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


# 构造设备
qubits = 3
ancilla_idx = qubits
dev = qml.device("lightning.qubit", wires=qubits+1)

# 构造受控的 CA_l

# 识别各个 Pauli 算子并记录位置
def identify_pauli_operators(pauli_terms):
    r"""
        ops = [Identity(wires=[0]) @ Identity(wires=[1]) @ PauliX(wires=[2]),
                Identity(wires=[0]) @ PauliX(wires=[1]) @ Identity(wires=[2])]
    """
    identified_terms = []
    
    for term in pauli_terms:
        term_operators = []
        for op in term.obs:
            if isinstance(op, qml.Identity):
                operator_type = 'I'
            elif isinstance(op, qml.PauliX):
                operator_type = 'X'
            elif isinstance(op, qml.PauliY):
                operator_type = 'Y'
            elif isinstance(op, qml.PauliZ):
                operator_type = 'Z'
            else:
                operator_type = 'Unknown'
            term_operators.append((operator_type, op.wires.tolist()))
        identified_terms.append(term_operators)
    return identified_terms

# 定义受控量子门 A_l
def circuit_with_controlled_pauli(ops_list):
    # 获取识别结果
    identified_terms = identify_pauli_operators(ops_list)
    # 控制位为辅助比特位
    control_wire = ancilla_idx
    # 添加控制位
    for term in identified_terms:
        for op_type, wires in term:
            if op_type == 'X':
                for wire in wires:
                    qml.CNOT(wires=[control_wire, wire])
                    qml.PauliX(wires=wire)
                    qml.CNOT(wires=[control_wire, wire])                    
            elif op_type == 'Y':
                for wire in wires:
                    qml.CNOT(wires=[control_wire, wire])
                    qml.RY(np.pi / 2, wires=wire)
                    qml.PauliX(wires=wire)
                    qml.RY(-np.pi / 2, wires=wire)                    
                    qml.CNOT(wires=[control_wire, wire])                    
            elif op_type == 'Z':
                for wire in wires:
                    qml.CNOT(wires=[control_wire, wire])
                    qml.PauliZ(wires=wire)
                    qml.CNOT(wires=[control_wire, wire])                           



# 生成 b_vec_2 的量子电路 ， |b_vec_2 > = U_b |0>
def U_b_circuit():
    qml.RY(0.54, wires=1)
    qml.CNOT([0,1])
    qml.RY(0.54, wires=1)
    qml.CNOT([0,1])
    
    qml.CNOT([1,2])
    qml.CNOT([0,2])
    qml.CNOT([1,2])
    qml.CNOT([0,2])
    
    qml.RZ(-0.79, wires=0) 
    qml.RZ(0.79, wires=1) 
    qml.RZ(-0.79, wires=2) 
    
    qml.CNOT([0,1])
    qml.RZ(0.79, wires=1) 
    qml.CNOT([0,1])
    
    qml.CNOT([1,2])
    qml.RZ(0.79, wires=2)
    qml.CNOT([0,2]) 
    qml.RZ(0.79, wires=2)
    qml.CNOT([1,2]) 
    qml.RZ(-0.79, wires=1) 
    qml.CNOT([0,2]) 


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
    # 使用 RZ 门代替 PhaseShift 门，测量概率是相同的，量子态会不同
    if part == "Im" or part == "im":
        # qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
        # 使用 RZ 门代替 PhaseShift 门
        qml.RZ(-np.pi / 2, wires=ancilla_idx)

    # |x>
    variational_x_circuit_2(weights=weights, deep_layer=3)

    # 受控 A_l 
    H_l =[ops[l]]
    circuit_with_controlled_pauli(H_l)

    # U_b dagger
    qml.adjoint(U_b_circuit)()

    # 受控 Z
    if j != -1:
        qml.CNOT(wires=[ancilla_idx, j])
        qml.PauliZ(wires=j)
        qml.CNOT(wires=[ancilla_idx, j])

    # U_b
    U_b_circuit()

    # 受控 A_lp 的 dagger
    H_lp =[ops[lp]]
    qml.adjoint(circuit_with_controlled_pauli)(H_lp)

    # 辅助位
    qml.Hadamard(wires=ancilla_idx)

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


