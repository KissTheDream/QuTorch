import torch
import torch.nn as nn
import torch.functional as F
from math import pi

# from qgate import gate_expand_1toN, rx, ry, rz, dag
# from qgate import measure
from Gate import dag, z_gate, I_gate, ptrace


def multi_kron(x_list):  # fixme QuTip tensor
    """
    kron the data in the list in order
    """
    x_k = torch.ones(1)
    for x in x_list:
        x_k = torch.kron(x_k, x)
    return x_k


class Circuit(object):
    def __init__(self, N):
        self.U = None
        self.n_qubits = N  # 总QuBit的个数
        self.gate_list = []  # 顺序保存门结构
        self.u = []  # 顺序保存酉矩阵

    #   内置函数 添加分拆资源
    def _add_gate(self, gate_name: str, target_qubit, gate_params):
        """add gate and its feature to the circuit by sequence.

        """
        # assert gate_name in list[]  #todo 创建一个可信池子
        self.gate_list.append({'gate': gate_name, 'theta': gate_params, 'which_qubit': target_qubit})

    def _add_u(self, u_matrix):
        """add u_matrix to the circuit by sequence.

        """
        # assert u_name in list[]  #todo 创建一个可信池子
        self.u.append(u_matrix)

    #   内置操作函数：张量积、  1toN、  两比特控制门toN、 两比特旋转门toN、  多比特控制门toN、 计算列表最终的酉矩阵
    def multi_kron(self, x_list):
        """
        kron the data in the list in order
        """
        x_k = torch.ones(1)
        for x in x_list:
            x_k = torch.kron(x_k, x)
        return x_k

    def gate_expand_1toN(self, U, N, target):
        """
        representing a one-qubit gate that act on a system with N qubits.

        """

        if N < 1:
            raise ValueError("integer N must be larger or equal to 1")

        if target >= N:
            raise ValueError("target must be integer < integer N")

        return self.multi_kron([torch.eye(2)] * target + [U] + [torch.eye(2)] * (N - target - 1))

    def gate_expand_2toN(self, U, N, targets):
        """
        representing a two-qubit gate that act on a system with N qubits.

        """

        if N < 2:
            raise ValueError("integer N must be larger or equal to 2")

        if targets[1] >= N:
            raise ValueError("target must be integer < integer N")

        return self.multi_kron([torch.eye(2)] * targets[0] + [U] + [torch.eye(2)] * (N - targets[1] - 1))

    def gate_sequence_product(self, left_to_right=True):
        """
        Calculate the overall unitary matrix for a given list of unitary operations.
        return: Unitary matrix corresponding to U_list.
        """

        U_overall = torch.eye(2 ** self.n_qubits, 2 ** self.n_qubits) + 0j
        for U in self.u:
            if left_to_right:
                U_overall = U @ U_overall
            else:
                U_overall = U_overall @ U
        self.U = U_overall
        return U_overall

    def two_qubit_control_gate(self, U, N, control, target):
        if N < 1:
            raise ValueError("integer N must be larger or equal to 1")
        if control >= N:
            raise ValueError("control must be integer < integer N")
        if target >= N:
            raise ValueError("target must be integer < integer N")
        if target == control:
            raise ValueError("control cannot be equal to target")

        zero_zero = torch.tensor([[1, 0], [0, 0]]) + 0j
        one_one = torch.tensor([[0, 0], [0, 1]]) + 0j
        list1 = [torch.eye(2)] * N
        list2 = [torch.eye(2)] * N
        list1[control] = zero_zero
        list2[control] = one_one
        list2[target] = U

        return self.multi_kron(list1) + self.multi_kron(list2)

    def two_qubit_rotation_gate(self, theta, N, qbit1, qbit2, way):
        # if type(theta) != type(torch.tensor(0.1)):
        #     theta = torch.tensor(theta)
        if N < 1:
            raise ValueError("number of qubits N must be >= 1")
        if qbit1 < 0 or qbit1 > N - 1 or qbit2 < 0 or qbit2 > N - 1:
            raise ValueError("index must between 0~N-1")
        if qbit1 == qbit2:
            raise ValueError("qbit1 cannot be equal to qbit2")
        lst1 = [torch.eye(2, 2)] * self.n_qubits
        lst2 = [torch.eye(2, 2)] * self.n_qubits
        if way == 'rxx':
            lst2[qbit1] = self._x_gate()
            lst2[qbit2] = self._x_gate()
        elif way == 'ryy':
            lst2[qbit1] = self._y_gate()
            lst2[qbit2] = self._y_gate()
        elif way == 'rzz':
            lst2[qbit1] = self._z_gate()
            lst2[qbit2] = self._z_gate()
        else:
            raise ValueError("Error gate")
        rst = torch.cos(theta / 2) * self.multi_kron(lst1) - 1j * torch.sin(theta / 2) * self.multi_kron(lst2)
        return rst + 0j

    def multi_control_gate(self, U, N, control_lst, target):
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
        return self.multi_kron(lst2) - self.multi_kron(lst3) + self.multi_kron(lst1)

    #   实例调用部分可使用的函数 添加门、run线路
    def rx(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('rx', target_qubit, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(self.gate_expand_1toN(self._rx(phi), self.n_qubits, target_qubit))
        else:
            self._add_u(self.gate_expand_1toN(self._rx(phi), self.n_qubits, target_qubit))

    def ry(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('ry', target_qubit, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(self.gate_expand_1toN(self._ry(phi), self.n_qubits, target_qubit))
        else:
            self._add_u(self.gate_expand_1toN(self._ry(phi), self.n_qubits, target_qubit))

    def rz(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('rz', target_qubit, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(self.gate_expand_1toN(self._rz(phi), self.n_qubits, target_qubit))
        else:
            self._add_u(self.gate_expand_1toN(self._rz(phi), self.n_qubits, target_qubit))

    def cnot(self, control_qubit: int, target_qubit: int):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('cnot', control_qubit, target_qubit)
        self._add_u(self.two_qubit_control_gate(self._x_gate(), self.n_qubits, control_qubit, target_qubit))

    def x_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('X', target_qubit, None)
        self._add_u(self.gate_expand_1toN(self._x_gate(), self.n_qubits, target_qubit))

    def z_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('Z', target_qubit, None)
        self._add_u(self.gate_expand_1toN(self._z_gate(), self.n_qubits, target_qubit))

    def y_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('Y', target_qubit, None)
        self._add_u(self.gate_expand_1toN(self._y_gate(), self.n_qubits, target_qubit))

    def Hcz(self, control_qubit, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"

        self._add_gate('cz', control_qubit, target_qubit)
        self._add_u(self.two_qubit_control_gate(self._z_gate(), self.n_qubits, control_qubit, target_qubit))

    def Hadamard(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('H', target_qubit, None)
        self._add_u(self.gate_expand_1toN(self._Hadamard(), self.n_qubits, target_qubit))

    def rxx(self, target_qubit01, target_qubit02, phi):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"
        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits

        self._add_gate('rxx', target_qubit01, phi)
        self._add_gate('rxx', target_qubit02, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(self.two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rxx'))
        else:
            self._add_u(self.two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rxx'))

    def ryy(self, target_qubit01, target_qubit02, phi):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"

        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits
        assert target_qubit01 != target_qubit02, \
            "target qubit should not be the same"

        self._add_gate('ryy', target_qubit01, phi)
        self._add_gate('ryy', target_qubit02, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(self.two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='ryy'))
        else:
            self._add_u(self.two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='ryy'))

    def rzz(self, target_qubit01, target_qubit02, phi):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"
        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits

        self._add_gate('rzz', target_qubit01, phi)
        self._add_gate('rzz', target_qubit02, phi)
        if type(phi) == float or type(phi) == int:
            phi = torch.tensor(phi)
            self._add_u(self.two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rzz'))
        else:
            self._add_u(self.two_qubit_rotation_gate(phi, self.n_qubits, target_qubit01, target_qubit02, way='rzz'))

    def multi_control_cnot(self, control_lst, target):
        self._add_u(self.multi_control_gate(self._x_gate(), self.n_qubits, control_lst, target))

    def run(self):
        return self.gate_sequence_product()

    #   内置函数
    def _I(self):
        """Single-qubit Identification gate
        -------
        result : torch.tensor for operator describing Identity matrix.
        """

        return torch.eye(2) + 0j

    def _rx(self, phi):
        """Single-qubit rotation for operator sigmax with angle phi.
        -------
        result : torch.tensor for operator describing the rotation.
        """

        return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1j * torch.sin(phi / 2).unsqueeze(dim=0),
                          -1j * torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)),
                         dim=0).reshape(2,
                                        -1)

    def _ry(self, phi):
        """Single-qubit rotation for operator sigmay with angle phi.
        -------
        result : torch.tensor for operator describing the rotation.
        """

        return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1 * torch.sin(phi / 2).unsqueeze(dim=0),
                          torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)), dim=0).reshape(2,
                                                                                                                    -1) + 0j

    def _rz(self, phi):
        """Single-qubit rotation for operator sigmaz with angle phi.
        -------
        result : torch.tensor for operator describing the rotation.
        """
        return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim=0), torch.zeros(1),
                          torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim=0)), dim=0).reshape(2, -1)

    def _z_gate(self):
        """
        Pauli z
        """
        return torch.tensor([[1, 0], [0, -1]]) + 0j

    def _x_gate(self):
        """
        Pauli x
        """
        return torch.tensor([[0, 1], [1, 0]]) + 0j

    def _y_gate(self):
        """
        Pauli x
        """
        return torch.tensor([[0, -1j], [1j, 0]]) + 0j

    def _Hadamard(self):
        H = torch.sqrt(torch.tensor(0.5)) * torch.tensor([[1, 1], [1, -1]]) + 0j
        return H

    def cswap(self, control_qubit, target_qubit01, target_qubit02):
        zero_zero = torch.tensor([[1, 0], [0, 0]]) + 0j
        one_one = torch.tensor([[0, 0], [0, 1]]) + 0j

        lst = [torch.eye(2, 2)] * self.n_qubits
        lst[control_qubit] = zero_zero

        swap = self.two_qubit_control_gate(self._x_gate(), self.n_qubits, target_qubit01, target_qubit02)
        swap = swap @ self.two_qubit_control_gate(self._x_gate(), self.n_qubits, target_qubit02, target_qubit01)
        swap = swap @ self.two_qubit_control_gate(self._x_gate(), self.n_qubits, target_qubit01, target_qubit02)

        self._add_gate('cswap', [control_qubit, target_qubit01, target_qubit02], None)
        self._add_u(self.multi_kron(lst) + swap @ self.gate_expand_1toN(one_one, self.n_qubits, control_qubit))

    # swap_test的过程对2维度相同的态进行内积，返回测量值在0到1之间。具体的操作是先对控制比特进行H门操作，之后对对应的比特用c_swap门进行交换，最后再
    # 对控制比特进行H门操作，通过偏迹求控制比特的密度矩阵，模拟对控制比特的测量。
    def swap_test(self, control_qubit, register1, register2):
        self.Hadamard(control_qubit)
        for qubit1, qubit2 in zip(register1, register2):
            self.cswap(control_qubit, qubit1, qubit2)
        self.Hadamard(control_qubit)


# def ptrace(rho, keep_qubit):
#     # 偏迹，输入密度矩阵，以及需要保留的比特位置即可。但请按比特顺序输入
#     total_number = torch.log2(torch.tensor(len(rho))).int()
#     keep_number = 0
#     residue_number = total_number
#
#     for i in range(total_number):
#         if i in keep_qubit:
#             keep_number += 1
#         # 如果i在需要保留的比特里面，则pass，不进行trace操作
#         else:
#             list1 = [torch.eye(2) + 0j] * residue_number
#             list2 = [torch.eye(2) + 0j] * residue_number
#             list1[keep_number] = torch.tensor([1, 0]) + 0j
#             list2[keep_number] = torch.tensor([0, 1]) + 0j
#             rho = multi_kron(list1) @ rho @ multi_kron(list1).T + multi_kron(list2) @ rho @ multi_kron(list2).T
#             residue_number -= 1
#     return rho
#

def measure(rho, M, physic=False):
    if torch.abs(torch.trace(rho) - 1) > 1e-4:
        raise ValueError("trace of density matrix must be 1")
    #     if dag(M) != M:
    #         raise ValueError("M must be hermitian")

    if not physic:  # physic=False，表示仅模拟量子线路，不考虑物理实现
        return torch.trace(torch.matmul(M, rho))
    else:
        # physic=True，此时要将对M的测量分解成：酉变换 + 计算基底测量
        # 此时需要对M进行本征分解
        pass


def encoding(x):
    """
    input: n*n matrix
    perform L2 regularization on x, x is complex
    """
    with torch.no_grad():
        if x.norm() != 1:
            xd = x.diag()
            xds = (xd.sqrt()).unsqueeze(1)
            xdsn = xds / (xds.norm() + 1e-12)
            xdsn2 = xdsn @ dag(xdsn)
            xdsn2 = xdsn2.type(dtype=torch.complex64)
        else:
            xdsn2 = x.type(dtype=torch.complex64)

    return xdsn2


def vector2rho(vector):
    """
    convert vector [torch.tensor] to qubit input
    为了防止没有归一化的问题
    """
    # todo 判断类型是不是torch.tensor
    n = vector.shape[0]  # dim of vector
    y = vector.reshape(1, n)  # convert to (1,n) shape
    yy = y.T @ y  # to matrix
    qinput = encoding(yy)

    return qinput


class Cir_Init_Feature(nn.Module):
    """
    用量子线路增强顶点特征 U(W)
    should be SWAP.test layer
    """

    def __init__(self, n_qubits: int = 2):
        super().__init__()
        self.total_qubits = n_qubits + 1  # 加一是因为辅助qubit
        self.n_qubits: int = int(n_qubits / 2)  # todo 先暂时这样

    def U(self):
        # 新增线路实例申明
        cir = Circuit(self.total_qubits)

        # 构建swap test
        regester1 = list(range(1, self.n_qubits + 1))  # First qubit register
        regester2 = list(range(self.n_qubits + 1, self.total_qubits))  # First qubit register
        cir.swap_test(0, regester1, regester2)

        U = cir.run()
        return U

    def forward(self, phi):
        U_out = self.U()
        weight_out = U_out @ phi @ dag(U_out)
        return weight_out


class Cir_Sum(nn.Module):
    """
    用量子线路求和两个量子态的输入
    """

    def __init__(self, theta_size=6, n_qubits=6, gain=2 ** 0.5, use_wscale=True,
                 lrmul=1):  # todo theta_size is changable
        super().__init__()

        # 标准初始化, why in this form unknow.
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        # weight learnable layer
        self.weight = nn.Parameter(
            nn.init.uniform_(torch.empty(theta_size), a=0.0, b=2 * pi) * init_std)  # theta_size=5

        # number of qubits
        self.total_qubits = n_qubits
        self.n_qubits: int = int(n_qubits / 2)  # todo 暂时先这样写

    def U(self):
        theta_W = self.weight * self.w_mul

        cir = Circuit(self.total_qubits)

        for which_q in range(0, self.n_qubits):
            cir.cnot(which_q, which_q + self.n_qubits)
        for which_q in range(0, self.total_qubits):
            cir.ry(which_q, theta_W[0])
            cir.rz(which_q, theta_W[1])
            cir.ry(which_q, theta_W[2])
            cir.rx(which_q, theta_W[3])
            cir.rx(which_q, theta_W[4])
            cir.rx(which_q, theta_W[5])

        U = cir.run()
        return U

    def forward(self, phi):
        U_out = self.U()
        weight_out = U_out @ phi @ dag(U_out)
        return weight_out


class Cir_XYX(nn.Module):
    """
    用量子线路运算融合注意力得分的输入值
    """

    def __init__(self, theta, n_qubits=2, gain=2 ** 0.5, use_wscale=True,
                 lrmul=1):  # todo theta_size is changable
        super().__init__()

        # 一条线路上3个单比特旋转门
        theta_size = 3 * n_qubits
        # 标准初始化, why in this form unknow.
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        # weight learnable layer
        self.weight = nn.Parameter(
            nn.init.uniform_(torch.empty(theta_size), a=0.0, b=2 * pi) * init_std)  # theta_size=5

        # number of qubits
        self.n_qubits: int = int(n_qubits / 2)  # todo 先暂时这样
        self.total_qubits = n_qubits
        # 唯一旋转角
        self.theta = theta

    def U(self):

        cir = Circuit(self.total_qubits)
        for which_q in range(0, self.total_qubits):
            cir.rx(which_q, self.theta)
            cir.ry(which_q, self.theta)
            cir.rx(which_q, self.theta)
        for idx in range(0, self.n_qubits):
            cir.cnot(idx, idx + self.n_qubits)

        U = cir.run()
        return U

    def forward(self, x):
        U_out = self.U()
        weight_out = U_out @ x @ dag(U_out)
        return weight_out
