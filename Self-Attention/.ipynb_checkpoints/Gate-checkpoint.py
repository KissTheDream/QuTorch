#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F

# single qubit gates


def I():
    """Single-qubit Identification gate
    -------
    result : torch.tensor for operator describing Identity matrix.
    """

    return torch.eye(2) + 0j


def rx(phi):
    """Single-qubit rotation for operator sigmax with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1j * torch.sin(phi / 2).unsqueeze(dim=0),
                      -1j * torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)), dim=0).reshape(2,
                                                                                                                      -1)


def ry(phi):
    """Single-qubit rotation for operator sigmay with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1 * torch.sin(phi / 2).unsqueeze(dim=0),
                      torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)), dim=0).reshape(2,
                                                                                                                -1) + 0j
    # return torch.tensor([[torch.cos(phi / 2), -torch.sin(phi / 2)],
    #              [torch.sin(phi / 2), torch.cos(phi / 2)]])


def rz(phi):
    """Single-qubit rotation for operator sigma_z with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim=0), torch.zeros(1),
                      torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim=0)), dim=0).reshape(2, -1)


def z_gate():
    """
    表明我们现在在 Pauli_Z表象下计算
    Pauli z
    """
    return torch.tensor([[1, 0], [0, -1]]) + 0j


def x_gate():
    """
    Pauli x
    """
    return torch.tensor([[0, 1], [1, 0]]) + 0j


def cnot():
    """
    torch.tensor representing the CNOT gate.
    control=0, target=1
    """
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]]) + 0j


def Hcz():
    """
    controlled z gate for measurement
    """
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, -1]]) + 0j


def rxx(phi):
    """
    torch.tensor representing the rxx gate with angle phi.
    """
    return torch.kron(rx(phi), rx(phi))


def ryy(phi):
    """
    torch.tensor representing the ryy gate with angle phi.
    """
    return torch.kron(ry(phi), ry(phi))


def rzz(phi):
    """
    torch.tensor representing the rzz gate with angle phi.
    """
    return torch.kron(rz(phi), rz(phi))


def dag(x):
    """
    compute conjugate transpose of input matrix
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag


def multi_kron(x_list):  # fixme QuTip tensor
    """
    kron the data in the list in order
    """
    x_k = torch.ones(1)
    for x in x_list:
        x_k = torch.kron(x_k, x)
    return x_k


def gate_expand_1toN(U, N, target):
    """
    representing a one-qubit gate that act on a system with N qubits.

    """

    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")

    return multi_kron([torch.eye(2)] * target + [U] + [torch.eye(2)] * (N - target - 1))


def gate_expand_2toN(U, N, targets):
    """
    representing a two-qubit gate that act on a system with N qubits.

    """

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if targets[1] >= N:
        raise ValueError("target must be integer < integer N")

    return multi_kron([torch.eye(2)] * targets[0] + [U] + [torch.eye(2)] * (N - targets[1] - 1))


def gate_sequence_product(U_list, n_qubits, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary operations.
    return: Unitary matrix corresponding to U_list.
    """

    U_overall = torch.eye(2 ** n_qubits, 2 ** n_qubits) + 0j
    for U in U_list:
        if left_to_right:
            U_overall = U @ U_overall
        else:
            U_overall = U_overall @ U

    return U_overall


def ptrace(rhoAB, dimA, dimB):
    """
    rhoAB : density matrix
    dimA: n_qubits A keep
    dimB: n_qubits B trash
    """
    mat_dim_A = 2 ** dimA
    mat_dim_B = 2 ** dimB

    id1 = torch.eye(mat_dim_A, requires_grad=True) + 0.j
    id2 = torch.eye(mat_dim_B, requires_grad=True) + 0.j

    pout = 0
    for i in range(mat_dim_B):
        p = torch.kron(id1, id2[i]) @ rhoAB @ torch.kron(id1, id2[i].reshape(mat_dim_B, 1))
        pout += p
    return pout


def expecval_ZI(state, nqubit, target):
    """
    state为 nqubit大小的密度矩阵，target为z门放置位置

    """
    zgate = z_gate()
    H = gate_expand_1toN(zgate, nqubit, target)
    expecval = (state @ H).trace()  # [-1,1]
    expecval_real = (expecval.real + 1) / 2  # [0,1]

    return expecval_real


def measure(state, nqubit):
    """
    测量nqubit次期望

    """
    measure = torch.zeros(nqubit, 1)
    for i in range(nqubit):
        measure[i] = expecval_ZI(state, nqubit, list(range(nqubit))[i])

    return measure


def encoding(x):
    """
    input: n*n matrix
    perform L2 regularization on x, x is complex
    """

    # if x.norm() != 1 :
    #     # print('l2norm:', x.norm())
    #     x = x / (x.norm() + 1e-10)
    # x = x.type(dtype=torch.complex64)
    # return x
    # from sklearn.preprocessing import normalize
    # xn = normalize(x, norm='l2', axis=0)
    with torch.no_grad():
        # x = x.squeeze()
        if x.norm() != 1:
            xd = x.diag()
            xds = (xd.sqrt()).unsqueeze(1)
            xdsn = xds / (xds.norm() + 1e-12)
            xdsn2 = xdsn @ dag(xdsn)
            xdsn2 = xdsn2.type(dtype=torch.complex64)
        else:
            xdsn2 = x.type(dtype=torch.complex64)
    # if x.norm() != 1:
    #     with torch.no_grad():
    #         xd = x.diag()
    #         xds = (xd.sqrt()).unsqueeze(1)
    #         xdsn = xds / (xds.norm() + 1e-12)
    #         xdsn2 = xdsn @ dag(xdsn)
    #         xdsn2 = xdsn2.type(dtype=torch.complex64)
    # else:
    #     xdsn2 = x.type(dtype=torch.complex64)
    return xdsn2


def vector2rho(vector):
    """
    convert vector [torch.tensor] to qubit input
    """
    # todo 判断类型是不是torch.tensor
    n = vector.shape[0]  # dim of vector
    y = vector.reshape(1, n)  # convert to (1,n) shape
    yy = y.T @ y  # to matrix
    qinput = encoding(yy)

    return qinput


class QEqualizedConv0(nn.Module):
    """
    Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
    放置5个量子门，也即有5个参数。

    todo qconv_q,qconv_k,qconv_V的线路构成没有随机
    """

    def __init__(self, n_qubits,  # n_qubits should be even
                 gain=2 ** 0.5,
                 use_wscale=True,
                 lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2 * np.pi) * init_std)  # todo 5-->size

        self.n_qubits = n_qubits

    def qconv0(self):

        w = self.weight * self.w_mul

        cir = []
        for which_q in range(0, self.n_qubits, 2):
            cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(rz(w[3]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)

        return U

    def qconv_q(self):

        w = self.weight * self.w_mul

        cir = []
        for which_q in range(0, self.n_qubits, 2):
            cir.append(gate_expand_1toN(ry(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(ry(w[3]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(ry(w[4]), self.n_qubits, which_q + 1))

        U = gate_sequence_product(cir, self.n_qubits)

        return U

    def qconv_k(self):

        w = self.weight * self.w_mul

        cir = []
        for which_q in range(0, self.n_qubits, 2):
            cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(rz(w[3]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)

        return U

    def qconv_v(self):

        w = self.weight * self.w_mul

        cir = []
        for which_q in range(0, self.n_qubits, 2):
            cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(rz(w[3]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)

        return U

    def forward(self, x):
        E_qconv0 = self.qconv0()
        qconv0_out = dag(E_qconv0) @ x @ E_qconv0

        return qconv0_out


# class Self_Attn(nn.Module):
#     """
#
#     """
#
#     def __init__(self, n_qubits,
#                  gain=2 ** 0.5,
#                  use_wscale=True,
#                  lrmul=1,
#                  size=100 ): # weight size
#         super().__init__()
#
#         he_std = gain * 5 ** (-0.5)  # He init
#         if use_wscale:
#             init_std = 1.0 / lrmul
#             self.w_mul = he_std * lrmul
#         else:
#             init_std = he_std / lrmul
#             self.w_mul = lrmul
#
#         self.weight = nn.Parameter(nn.init.uniform_(torch.empty(size), a=0.0, b=2 * np.pi) * init_std)
#         self.n_qubits = n_qubits
#
#     def qconv0(self):
#
#         w = self.weight * self.w_mul
#
#         cir = []
#         for which_q in range(0, self.n_qubits, 2):
#             cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
#             cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
#             cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
#             cir.append(gate_expand_1toN(rz(w[3]), self.n_qubits, which_q))
#             cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q + 1))
#         U = gate_sequence_product(cir, self.n_qubits)
#
#         return U
#
#     def forward(self, x):
#         E_qconv0 = self.qconv0()
#         qconv0_out = dag(E_qconv0) @ x @ E_qconv0
#
#         return qconv0_out
#
#
#
#     def cir_init_query(self):
#         input = self.input
#
#
#     def cir_init_key():
#
#     def cir_init_value():

class Circuit(object):
    def __init__(self, N):
        self.U = None
        self.n_qubits = N  # 总QuBit的个数
        self.gate_list = []  # 顺序保存门结构

    # def add_quibts(self, n):
    #     self.n_qubits += n

    def _add_gate(self, gate_name: str, target_qubit, gate_params):
        """add gate and its feature to the circuit by sequence.

        """
        # assert gate_name in list[]  #todo 创建一个可信池子
        self.gate_list.append({'gate': gate_name, 'theta': gate_params, 'which_qubit': target_qubit})

    def rx(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('rx', target_qubit, phi)

    def ry(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('ry', target_qubit, phi)

    def rz(self, target_qubit, phi):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('rz', target_qubit, phi)

    def x_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('X', target_qubit, None)

    def z_gate(self, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('Z', target_qubit, None)

    def cnot(self, control_qubit: int, target_qubit: int):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"
        self._add_gate('I', control_qubit, None)
        self._add_gate('X', target_qubit, None)

    def Hcz(self, control_qubit, target_qubit):
        assert isinstance(target_qubit, int), \
            "target qubit is not integer"
        assert isinstance(control_qubit, int), \
            "control qubit is not integer"
        assert control_qubit <= self.n_qubits
        assert 0 <= target_qubit < self.n_qubits, \
            "target qubit is not available"

        self._add_gate('I', control_qubit, None)
        self._add_gate('Z', target_qubit, None)

    def rxx(self, phi, target_qubit01, target_qubit02=None):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"
        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits

        self._add_gate('rx', target_qubit01, phi)
        self._add_gate('rx', target_qubit02, phi)
        self._add_gate('rx', target_qubit01, phi)
        self._add_gate('rx', target_qubit02, phi)

    def ryy(self, phi, target_qubit01, target_qubit02=None):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"

        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits
        "target qubit should not be the same"
        assert target_qubit01 != target_qubit02

        self._add_gate('ry', target_qubit01, phi)
        self._add_gate('ry', target_qubit02, phi)
        self._add_gate('ry', target_qubit01, phi)
        self._add_gate('ry', target_qubit02, phi)

    def rzz(self, phi, target_qubit01, target_qubit02=None):
        assert isinstance(target_qubit01, int), \
            "target qubit is not integer"
        assert isinstance(target_qubit02, int), \
            "target qubit is not integer"
        if not target_qubit02:
            target_qubit02 = target_qubit01 + 1
        assert target_qubit01 <= self.n_qubits
        assert target_qubit02 <= self.n_qubits

        self._add_gate('rz', target_qubit01, phi)
        self._add_gate('rz', target_qubit02, phi)
        self._add_gate('rz', target_qubit01, phi)
        self._add_gate('rz', target_qubit02, phi)

    def show_gates(self):
        print(f"\n gate in sequence is : {self.gate_list}")

    def read_gate(self):
        """
        get
        """
        # create U
        dim = 2 ** self.n_qubits
        U = torch.eye(dim, dtype=torch.complex64)

        for i, list_ele in enumerate(self.gate_list):
            # print(list_ele)
            # print(f"gate: {list_ele['gate']}")
            # print(f"param: {list_ele['theta']} \n")
            gate_matrix_temp = self._gate_to_matrix(list_ele['gate'], list_ele['theta'])
            cir_matrix_temp = gate_expand_1toN(gate_matrix_temp, self.n_qubits, list_ele['which_qubit'])

            U = cir_matrix_temp @ U  # Note: the sequence should be right

            print(f"\n index: {i}"
                  f"\n gate_list:{list_ele}"
                  f"\n gate_matrix: {gate_matrix_temp}"
                  f"\n circuit_matrix: {cir_matrix_temp}"
                  f"\n U: {U}")

        self.U = U

    def _gate_to_matrix(self, gate_name, params=None):
        # params to tensor
        if params is not None:
            params = torch.tensor(params)
        else:
            pass

        if gate_name is not str:
            gate_name = str(gate_name)

        # choose gate
        if gate_name == 'rx':
            gate_matrix = rx(params)
        elif gate_name == 'ry':
            gate_matrix = ry(params)
        elif gate_name == 'rz':
            gate_matrix = rz(params)
        elif gate_name == 'I':
            gate_matrix = I()
        # elif gate_name == 'x':
        #     gate_matrix = x_gate()
        # elif gate_name == 'Hcz':
        #     gate_matrix = Hcz()
        elif gate_name == 'Z':
            gate_matrix = z_gate()
        elif gate_name == 'cnot':
            gate_matrix = cnot()
        elif gate_name == 'Hcz':
            gate_matrix = Hcz()
        elif gate_name == 'rxx':
            gate_matrix = rxx(params)
        elif gate_name == 'ryy':
            gate_matrix = ryy(params)
        elif gate_name == 'rzz':
            gate_matrix = rzz(params)
        else:
            raise Exception("Gate name not accepted")

        return gate_matrix

    def run(self, A):
        if A == 'sim_cir':
            # todo 调用backend 代码模块

            pass

        if A == 'sim_light':
            pass
        if A == 'real':
            pass
