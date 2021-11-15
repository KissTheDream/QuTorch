"""
目前共19个基本门，以及辅助函数
multi_kron,gate_expand_1toN,rx,ry,rz,rn,Hadamard,
rxx,ryy,rzz,two_qubit_control_gate,cnot,cz
multi_control_gate,multi_control_cnot
ptrace,measure,IsUnitary,dag
"""

import torch

sigma_x = torch.tensor([[0, 1], [1, 0]]) + 0j
sigma_y = torch.tensor([[0, -1j], [1j, 0]]) + 0j
sigma_z = torch.tensor([[1, 0], [0, -1]]) + 0j


# ==================================辅助函数===================================
def multi_kron(lst):
    rst = lst[0]
    for i in range(1, len(lst)):
        rst = torch.kron(rst, lst[i])
    return rst


def gate_expand_1toN(gate, N, index):
    '''
    不要直接用这个函数
    '''
    if N < 1:
        raise ValueError("number of qubits N must be >= 1")
    if index < 0 or index > N - 1:
        raise ValueError("index must between 0~N-1")
    lst1 = [torch.eye(2, 2)] * N
    lst1[index] = gate
    return multi_kron(lst1)


# ============================single qubit gate=================================
def rx(theta, N=-1, index=-1, expand=False):
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if not expand:
        Rx = (torch.cos(theta / 2) * torch.eye(2, 2) - 1j * torch.sin(theta / 2) * sigma_x)
        return Rx + 0j
    else:
        Rx = (torch.cos(theta / 2) * torch.eye(2, 2) - 1j * torch.sin(theta / 2) * sigma_x)
        return gate_expand_1toN(Rx, N, index) + 0j


def ry(theta, N=-1, index=-1, expand=False):
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if not expand:
        Ry = (torch.cos(theta / 2) * torch.eye(2, 2) - 1j * torch.sin(theta / 2) * sigma_y)
        return Ry + 0j
    else:
        Ry = (torch.cos(theta / 2) * torch.eye(2, 2) - 1j * torch.sin(theta / 2) * sigma_y)
        return gate_expand_1toN(Ry, N, index) + 0j


def rz(theta, N=-1, index=-1, expand=False):
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if not expand:
        Rz = (torch.cos(theta / 2) * torch.eye(2, 2) - 1j * torch.sin(theta / 2) * sigma_z)
        return Rz + 0j
    else:
        Rz = (torch.cos(theta / 2) * torch.eye(2, 2) - 1j * torch.sin(theta / 2) * sigma_z)
        return gate_expand_1toN(Rz, N, index) + 0j


def rn(alpha, theta, phi, N=-1, index=-1, expand=False):
    '''
    在bloch球中，绕着球坐标单位向量(theta,phi),逆时针旋转alpha度
    一般不建议使用
    '''
    if type(alpha) != type(torch.tensor(0.1)):
        alpha = torch.tensor(alpha)
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if type(phi) != type(torch.tensor(0.1)):
        phi = torch.tensor(phi)
    nx = torch.sin(theta) * torch.cos(phi) * sigma_x
    ny = torch.sin(theta) * torch.sin(phi) * sigma_y
    nz = torch.cos(theta) * sigma_z
    sigma_n = nx + ny + nz

    if not expand:
        Rn = (torch.cos(alpha / 2) * torch.eye(2) - 1j * torch.sin(alpha / 2) * sigma_n)
        return Rn + 0j
    else:
        Rn = (torch.cos(alpha / 2) * torch.eye(2) - 1j * torch.sin(alpha / 2) * sigma_n)
        return gate_expand_1toN(Rn, N, index) + 0j


def Hadamard(N=-1, index=-1, expand=False):
    if not expand:
        H = torch.sqrt(torch.tensor(0.5)) * torch.tensor([[1, 1], [1, -1]])
        return H + 0j
    else:
        H = torch.sqrt(torch.tensor(0.5)) * torch.tensor([[1, 1], [1, -1]])
        return gate_expand_1toN(H, N, index) + 0j


# =================================two qubit gates==============================


def rxx(theta, N=-1, qbit1=-1, qbit2=-1, expand=False):
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if not expand:
        Rxx = torch.cos(theta / 2) * torch.eye(4, 4) - 1j * torch.sin(theta / 2) * torch.kron(sigma_x, sigma_x)
        return Rxx + 0j
    else:
        if N < 1:
            raise ValueError("number of qubits N must be >= 1")
        if qbit1 < 0 or qbit1 > N - 1 or qbit2 < 0 or qbit2 > N - 1:
            raise ValueError("index must between 0~N-1")
        if qbit1 == qbit2:
            raise ValueError("qbit1 cannot be equal to qbit2")
        lst1 = [torch.eye(2, 2)] * N
        lst2 = [torch.eye(2, 2)] * N
        lst2[qbit1] = sigma_x
        lst2[qbit2] = sigma_x
        rst = torch.cos(theta / 2) * multi_kron(lst1) - 1j * torch.sin(theta / 2) * multi_kron(lst2)
        return rst + 0j


def ryy(theta, N=-1, qbit1=-1, qbit2=-1, expand=False):
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if not expand:
        Ryy = torch.cos(theta / 2) * torch.eye(4, 4) - 1j * torch.sin(theta / 2) * torch.kron(sigma_y, sigma_y)
        return Ryy + 0j
    else:
        if N < 1:
            raise ValueError("number of qubits N must be >= 1")
        if qbit1 < 0 or qbit1 > N - 1 or qbit2 < 0 or qbit2 > N - 1:
            raise ValueError("index must between 0~N-1")
        if qbit1 == qbit2:
            raise ValueError("qbit1 cannot be equal to qbit2")
        lst1 = [torch.eye(2, 2)] * N
        lst2 = [torch.eye(2, 2)] * N
        lst2[qbit1] = sigma_y
        lst2[qbit2] = sigma_y
        rst = torch.cos(theta / 2) * multi_kron(lst1) - 1j * torch.sin(theta / 2) * multi_kron(lst2)
        return rst + 0j


def rzz(theta, N=-1, qbit1=-1, qbit2=-1, expand=False):
    if type(theta) != type(torch.tensor(0.1)):
        theta = torch.tensor(theta)
    if not expand:
        Rzz = torch.cos(theta / 2) * torch.eye(4, 4) - 1j * torch.sin(theta / 2) * torch.kron(sigma_z, sigma_z)
        return Rzz + 0j
    else:
        if N < 1:
            raise ValueError("number of qubits N must be >= 1")
        if qbit1 < 0 or qbit1 > N - 1 or qbit2 < 0 or qbit2 > N - 1:
            raise ValueError("index must between 0~N-1")
        if qbit1 == qbit2:
            raise ValueError("qbit1 cannot be equal to qbit2")
        lst1 = [torch.eye(2, 2)] * N
        lst2 = [torch.eye(2, 2)] * N
        lst2[qbit1] = sigma_z
        lst2[qbit2] = sigma_z
        rst = torch.cos(theta / 2) * multi_kron(lst1) - 1j * torch.sin(theta / 2) * multi_kron(lst2)
        return rst + 0j


def two_qubit_control_gate(U, N, control, target):
    '''
    不建议直接使用该函数
    two_qubit_control_gate该函数可实现任意两比特受控门
    代码照抄田泽卉的，但建议用我这个函数名，注意这里的U是controlled-U里的U，而非controlled-U整体
    比如想实现cnot门，cnot表示controlled-not gate，那么U就是not门，即sigma_x(paulix)
    比如想实现cz门，cnot表示controlled-z gate，那么U就是z门，即sigma_z(pauliz)
    '''
    if N < 1:
        raise ValueError("number of qubits(interger N) must be >= 1")
    if max(control, target) > N - 1:
        raise ValueError("control&target must <= number of qubits - 1")
    if min(control, target) < 0:
        raise ValueError("control&target must >= 0")
    if control == target:
        raise ValueError("control cannot be equal to target")

    zero_zero = torch.tensor([[1, 0], [0, 0]]) + 0j
    one_one = torch.tensor([[0, 0], [0, 1]]) + 0j

    lst1 = [torch.eye(2, 2)] * N
    lst1[control] = zero_zero

    lst2 = [torch.eye(2, 2)] * N
    lst2[control] = one_one
    lst2[target] = U
    return multi_kron(lst1) + multi_kron(lst2)


def cnot(N, control, target):
    sigma_x = torch.tensor([[0, 1], [1, 0]]) + 0j
    return two_qubit_control_gate(sigma_x, N, control, target)


def cz(N, control, target):
    sigma_z = torch.tensor([[1, 0], [0, -1]]) + 0j
    return two_qubit_control_gate(sigma_z, N, control, target)


# ==========================多比特门（3/4/5... qubit）===========================

def multi_control_gate(U, N, control_lst, target):
    '''
    多控制比特受控门，比如典型的toffoli gate就是2个控制1个受控
    control_lst:一个列表，内部是控制比特的索引号
    '''
    if N < 1:
        raise ValueError("number of qubits(interger N) must be >= 1")

    if max(max(control_lst), target) > N - 1:
        raise ValueError("control&target must <= number of qubits - 1")

    if min(min(control_lst), target) < 0:
        raise ValueError("control&target must >= 0")

    for each in control_lst:
        if each == target:
            raise ValueError("control cannot be equal to target")

    U = U + 0j
    one_one = torch.tensor([[0, 0], [0, 1]]) + 0j

    lst1 = [torch.eye(2, 2)] * N
    for each in control_lst:
        lst1[each] = one_one
    lst1[target] = U

    lst2 = [torch.eye(2, 2)] * N

    lst3 = [torch.eye(2, 2)] * N
    for each in control_lst:
        lst3[each] = one_one
    # multi_kron(lst2) - multi_kron(lst3)对应不做操作的哪些情况
    return multi_kron(lst2) - multi_kron(lst3) + multi_kron(lst1)


def multi_control_cnot(N, control_lst, target):
    sigma_x = torch.tensor([[0, 1], [1, 0]]) + 0j
    return multi_control_gate(sigma_x, N, control_lst, target)


# ================================测量========================================

def ptrace(rho, N, trace_lst):
    '''
    trace_lst里面是想trace掉的qubit的索引号，须从小到大排列
    '''
    # 输入合法性检测
    if abs(torch.trace(rho) - 1) > 1e-6:
        raise ValueError("trace of density matrix must be 1")
    if rho.shape[0] != 2 ** N:
        raise ValueError('rho dim error')

    trace_lst.sort()  # 必须从小到大排列
    rho = rho + 0j
    if len(trace_lst) == 0:
        return rho + 0j

    id1 = torch.eye(2 ** (trace_lst[0])) + 0j
    id2 = torch.eye(2 ** (N - 1 - trace_lst[0])) + 0j
    id3 = torch.eye(2) + 0j
    rho_nxt = torch.tensor(0)
    for i in range(2):
        A = torch.kron(torch.kron(id1, id3[i]), id2) + 0j
        rho_nxt = rho_nxt + A @ rho @ dag(A)

    new_lst = [i - 1 for i in trace_lst[1:]]  # trace掉一个qubit，他后面的qubit索引号要减1

    return ptrace(rho_nxt, N - 1, new_lst) + 0j


def measure(rho, M, physic=False):
    if abs(torch.trace(rho) - 1) > 1e-6:
        raise ValueError("trace of density matrix must be 1")
    if dag(M) != M:
        raise ValueError("M must be hermitian")

    if not physic:  # physic=False，表示仅模拟量子线路，不考虑物理实现
        return torch.trace(torch.matmul(M, rho))
    else:
        # physic=True，此时要将对M的测量分解成：酉变换 + 计算基底测量
        # 此时需要对M进行本征分解
        pass


# ==================================辅助函数================================
def IsUnitary(in_matrix):
    '''
    判断一个矩阵是否是酉矩阵
    只需要判断每行是否归一，行与行是否正交（三重循环，十分耗时）
    '''
    if (in_matrix.shape)[0] != (in_matrix.shape)[1]:  # 验证是否为方阵
        raise ValueError("not square matrix!")
        return False

    n = in_matrix.shape[0]  # 行数

    if n < 1:
        raise ValueError("matrix has at least 1 row(column)")
        return False

    for i in range(n):  # 每行是否归一
        summ = 0.0
        for j in range(n):
            summ += (abs(in_matrix[i][j])) ** 2
        if abs(summ - 1) > 1e-6:
            print("not unitary")
            return False

    for i in range(n - 1):  # 行之间是否正交
        for k in range(i + 1, n):
            summ = 0.0 + 0.0 * 1j
            for j in range(n):
                summ += in_matrix[i][j] * (in_matrix[k][j]).conj()
            if abs(abs(summ) - 0) > 1e-6:
                print("not orthogonal")
                return False
    return True


def dag(x):
    """
    compute conjugate transpose of input matrix,对输入进行共轭转置
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag + 0j


# ==================================测试部分====================================
if __name__ == "__main__":
    a = torch.tensor(3.1416)
    b = 0
    # print(rz(a),'\n',rz(b))
    # print(multi_control_cnot(3,[0,1],2))

    N = 5
    rho = torch.rand(2 ** N, 2 ** N)
    tra = torch.trace(rho)
    rho1 = rho * (1 / tra)
    print("rho1", rho1)

    p_rho = ptrace(rho1, N, [0, 4, 2])
    print(p_rho)
