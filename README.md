# 2024 量旋杯解题方案说明

本项目提供了两个量子算法来求解 2024 量旋杯的线性方程组问题，分别为 QUBO (Quadratic unconstrained binary optimization) + VQE (variational quantum eigensolver)、 VQLS (Variational Quantum Linear Solver)。实验下来，更倾向于使用 QUBO + VQE, VQE 也可以换成 QAOA。

本文档主要是为了说明本项目各个文件的作用，以及相对应的代码运行环境。推荐各个 `ipynb` 文件与 `2024量旋杯解题方案说明.pdf` 一块阅读。

## 环境准备

在项目中，构造了两套不同的 python 虚拟环境，其中有一个是为了引入 qiskit 模拟真机噪声的文件。

第一个运行环境，基于 `python = 3.11.9`, 其需要的第三方 python 包记录在文件 `requirements.txt` 中，主要使用 `pennylane` 来运行量子算法，项目中绝大多数代码文件均使用该环境运行。
```
pip install -r /path/to/requirements.txt
```
第二个运行环境，基于 `python = 3.9.18`, 使用文件 `requirements_noise.txt` 来搭建，主要使用 `pennylane` 特定版本的 `qiskit`，主要是为了从 `qiskit` 端加载模拟真机噪声的文件。使用过程中要注意 `pennylane` 和 `qiskit` 的版本适配问题。在项目中，会在 `ipynb` 文件的开头标注使用的是否是该噪声运行环境，如果未标注，则使用的是第一个运行环境。
```
pip install -r /path/to/requirements_noise.txt
```

## 文件说明

`说明文档` 中主要放置了解题方案说明 `2024量旋杯解题方案说明.pdf`，该文件包含解题过程中的理论推导、实验结果汇总、未完成的一些想法，以及参考的论文。

`代码文件` 分为两部分，分别 `Solution_QUBO` 和 `Solution_VQLS` 两套方案的代码，主要包含可运行的 `python` 文件。接下来，分别说明各个方案下的文件。


### Solution QUBO

`1_QUBO_introduction.ipynb` 和 `hamiltonian_until.py`：介绍如何构造线性方程组的哈密顿量，并使用特征值求解哈密顿量的矩阵形式，验证是否符合预期；

`2_QUBO_VQE_ideal_1.ipynb` 和 `VQE_until`：基于随机的初始参数，使用 VQE 在无噪的环境下，求解线性方程组哈密顿量的基态及其基态能量；

`3_circuit_params.ipynb`：使用变分电路训练，使得变分量子态与哈密顿量基态的内积为 1，即变分电路的最优参数，方便后续 VQE 的使用；

`4_QUBO_VQE_ideal_2.ipynb`：从 `3_circuit_params.ipynb` 获得变分电路的最优参数，使用随机扰动改变最优参数，并作为 VQE 的初始参数，进行测量和非测量模式下的迭代优化，并最终学到哈密顿量正确的基态及基态能量；

`5_QUBO_VQE_noise.ipynb`：使用 `requirements_noise` 构造的 `python` 环境，从 `qiskit` 引入模拟的真机噪声，使用随机扰动后的最优参数作为 VQE 的初始参数，进行非测量模式下的迭代优化，并与无噪环境下的训练进行对比，且最终学到了哈密顿量正确的基态及基态能量；

`6_circuit_qasm.ipynb` 和 `qasm_until.py`：用于生成变分电路的 OpenQASM 形式；

`NoiseModel`：`qiskit` 模拟的真机噪声文件，共有三个；


### Solution VQLS

`VQLS_introduction.ipynb`：介绍如何将线性方程组问题转换成 VQLS 能求解的形式，并使用矩阵运算验证最终生成的量子态 |x> 如何正确的恢复成解 x；

`1_Multiprocessing_demo` 和 `multiprocessing_demo`：采用多进程运算的写法，基于随机的初始参数，使用 VQLS 在无噪环境下，优化量子态 |x> 并恢复成经典解；

`2_Idea_demo` 和 `VQLS_train.ipynb`：从 `x_state_circuit.ipynb` 获得 VQLS 的最优参数，使用随机扰动更改最优参数，作用 VQLS 的初始参数，在无噪环境下进行优化，并达到终止条件。生成的量子态 |x> 恢复的解 x 与实际解的绝对误差极小，仅有 0.76；

`3_Noise_demo`： `VQLS_Noise_prob.ipynb` 展示不同程度的去极化噪声，在最优参数的情况下，对量子态测量概率的影响；`VQLS_Noise_train.ipynb` 展示从随机扰动后的最优参数出发，在具有 0.01 水平的去极化噪声下，进行 VQLS 的迭代优化，并恢复出最终的解，其中解的误差与实际的误差极大；

`4_circuit`：`x_state_circuit.ipynb` 学习最优的参数以构造真正的|x>；`circuit_qasm.ipynb` 和 `circuit_until` 则是将 `Hadamard test` 的电路转换成由基础门构成的电路，因为在前面的代码中，使用了特殊的命令来生成受控 $A_l$ 门和 $U_{b}$ 门，并经过数值验证，转换后的电路与原电路运算结果一致。但是 `pennylane` 不能直接转换成 `OpenQASM`，特殊指令如 `qml.adjoint` 无法使用 `qasm_until` 中的 `export_to_openqasm` 函数识别，略微麻烦，暂无时间进行更细致的转换。