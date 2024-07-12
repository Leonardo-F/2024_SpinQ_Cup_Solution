r"""
将 pennylane 的电路导出 OpenQASM 格式
"""



# 定义一个函数将量子脚本导出为 OpenQASM 格式
def export_to_openqasm(quantum_script):
    qasm_str = "OPENQASM 2.0;\n"
    qasm_str += "include \"qelib1.inc\";\n"
    qasm_str += f"qreg q[{quantum_script.num_wires}];\n"
    qasm_str += f"creg c[{quantum_script.num_wires}];\n"
    
    for operation in quantum_script.operations:
        name = operation.name
        wires = operation.wires.tolist()
        params = operation.parameters

        if name == "RX":
            qasm_str += f"rx({params[0]}) q[{wires[0]}];\n"
        elif name == "RY":
            qasm_str += f"ry({params[0]}) q[{wires[0]}];\n"
        elif name == "RZ":
            qasm_str += f"rz({params[0]}) q[{wires[0]}];\n"
        elif name == "PauliX":
            qasm_str += f"x q[{wires[0]}];\n" 
        elif name == "PauliZ":
            qasm_str += f"x q[{wires[0]}];\n" 
        
        elif name == "CNOT":
            qasm_str += f"cx q[{wires[0]}],q[{wires[1]}];\n"
        elif name == "Hadamard":
            qasm_str += f"h q[{wires[0]}];\n"
        elif name == "Measure":
            qasm_str += f"measure q[{wires[0]}] -> c[{wires[0]}];\n"
        else:
            raise ValueError(f"Unsupported gate: {name}")
    
    return qasm_str