import torch
from torch import nn
import numpy as np
from numpy import diag
import json
import random
from scipy.linalg import sqrtm, logm
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def dag(x):
    """
    compute conjugate transpose of input matrix
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag


def encoding(x):
    """
    input: n*n matrix
    perform L2 regularization on x, x为complex
    """
    with torch.no_grad():
        # x = x.squeeze( )
        if x.norm() != 1:
            xd = x.diag()
            xds = (xd.sqrt()).unsqueeze(1)
            xdsn = xds / (xds.norm() + 1e-12)
            xdsn2 = xdsn @ dag(xdsn)  # dag() 自定义函数
            xdsn2 = xdsn2.type(dtype=torch.complex64)
        else:
            xdsn2 = x.type(dtype=torch.complex64)
    return xdsn2


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


def rz(phi):
    """Single-qubit rotation for operator sigmaz with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim=0), torch.zeros(1),
                      torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim=0)), dim=0).reshape(2, -1)


def z_gate():
    """
    Pauli z
    """
    return torch.tensor([[1, 0], [0, -1]]) + 0j


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


def multi_kron(x_list):
    """
    kron the data in the list in order
    """
    x_k = torch.ones(1)
    for x in x_list:
        x_k = torch.kron(x_k, x)
    return x_k


def ptrace(rhoAB, dimA, dimB):
    """
    rhoAB : density matrix（密度矩阵）
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


class QEqualizedConv0(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，也即有5个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def qconv0(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits, 2):
            # 旋转门
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


class QPool(nn.Module):
    """Quantum Pool layer.
       放置4个量子门，2个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def qpool(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits, 2):
            cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(ry(w[2]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[3]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[5]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(cnot(), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(rz(-w[5]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(ry(-w[3]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(rx(-w[1]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)
        return U

    def forward(self, x):
        E_qpool = self.qpool()
        qpool_out = E_qpool @ x @ dag(E_qpool)
        return qpool_out


class deQuConv(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，也即有5个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def de_qconv(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits, 2):
            # 旋转门
            cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(rz(w[3]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)
        U = dag(U)
        return U

    def forward(self, x):
        E_qconv = self.de_qconv()
        qconv0_out = dag(E_qconv) @ x @ E_qconv
        return qconv0_out


class deQuPool(nn.Module):
    """Quantum Pool layer.
       放置4个量子门，2个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6), a=0.0, b=2 * np.pi) * init_std)
        self.n_qubits = n_qubits

    def dequpool(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits, 2):
            cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(ry(w[2]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[3]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(rz(w[5]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_2toN(cnot(), self.n_qubits, [which_q, which_q + 1]))
            cir.append(gate_expand_1toN(rz(-w[5]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(ry(-w[3]), self.n_qubits, which_q + 1))
            cir.append(gate_expand_1toN(rx(-w[1]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)
        U = dag(U)
        return U

    def forward(self, x):
        E_qpool = self.dequpool()
        qpool_out = E_qpool @ x @ dag(E_qpool)
        return qpool_out


class Q_Encoder(nn.Module):
    def __init__(self):
        super(Q_Encoder, self).__init__()
        self.qconv1 = QEqualizedConv0(8)
        # 8比特量子进行一次池化
        self.pool = QPool(8)

    def forward(self, x):
        x = self.qconv1(x)
        x = self.pool(x)
        # dimA+dimB的值需要等于量子线路的数量
        # dimA的值取最大即量子线路数-1，dimB=1时，偏迹测量所得的值保真度最高；
        x = ptrace(x, 7, 1)
        return x


class Q_Decoder(nn.Module):
    def __init__(self):
        super(Q_Decoder, self).__init__()
        # 8量子比特进行池化
        self.depool = deQuPool(8)
        # 8比特量子进行卷积
        self.deqconv = deQuConv(8)

    def forward(self, x, y):
        # x: encode state
        # y1: trash state,healthy EEG; 1qubit kron
        # y2: 128
        deinput = torch.kron(x, y)
        out = self.depool(deinput)
        out = self.deqconv(out)
        return out


class Q_Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.depool = deQuPool(8)
        # 8比特量子进行卷积
        self.deqconv = deQuConv(8)

    def forward(self, x, y):
        # x: encode state
        # y1: trash state,healthy EEG; 1qubit kron
        # y2: 128
        deinput = block_diag(x.detach().numpy(), y.detach().numpy())  # 并集
        deinput = torch.tensor(deinput)
        out = self.depool(deinput)
        out = self.deqconv(out)
        return out


class Q_AEnet(nn.Module):
    def __init__(self):
        super(Q_AEnet, self).__init__()
        self.encoder = Q_Encoder()
        self.decoder = Q_Decoder()

    def forward(self, x, y):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output, y)
        return decoder_output


def encoder_import_data():
    path = "E:/data/preictal/chb01_01_time1.json"
    with open(path, "r") as f:
        _data = json.load(f)
        step = 128
        import_data = _data["FP1-F7"]
        d = [import_data[i:i + step] for i in range(0, len(import_data), step)]
        import_data = random.sample(d, 1)
    torch_import_data = torch.tensor(diag(import_data[0]))
    torch_import_data = torch_import_data.T @ torch_import_data
    out_torch_import_data = encoding(torch_import_data)
    return out_torch_import_data


rho_C = torch.tensor(np.diag([1, 0]))


# rho_C  = encoder_import_data()
def get_fid(true_sp, gen_sp, flag=1):
    """
    :param true_sp: 真实数据
    :param gen_sp: 生成数据
    :return: flag=1: Tr(AB) + sqrt((1-Tr(A^2)) * sqrt((1-Tr(B^2)))
             flag=2: Tr(sqrtm(sqrtm(in) @ out @ sqrtm(in)))
             flag=3: square(Tr(sqrtm(sqrtm(in) @ out @ sqrtm(in))))
    """
    # rho_in = encoding(true_sp)
    # rho_out = encoding(gen_sp)
    rho_in = true_sp
    rho_out = gen_sp
    if flag == 1:
        fid = (rho_in @ rho_out).trace() + torch.sqrt((1 - (rho_in @ rho_in).trace())) * \
              torch.sqrt((1 - (rho_out @ rho_out).trace()))
    elif flag == 2:
        f_inner = sqrtm(rho_in) @ rho_out @ sqrtm(rho_in)
        fid = sqrtm(f_inner).trace()
    elif flag == 3:
        f_inner = sqrtm(rho_in) @ rho_out @ sqrtm(rho_in)
        fid = np.square((sqrtm(f_inner)).trace())
    return fid.real


def Loss(true_sp, gen_sp):
    fid = get_fid(true_sp, gen_sp, flag=1)
    loss = 1 - fid
    return loss.requires_grad_(True)


def smiles2qstate_test():
    # dataset = pd.read_csv('E:/chb-mit-eeg-data/data_eeg.csv',header=None).values[:, 0]
    # data = diag(dataset[0:256])
    # data_torch = torch.tensor(data)
    # data_torch = data_torch.T@data_torch
    # out_data = encoding(data_torch)
    path = "E:/data/preictal/chb01_01_time1.json"
    with open(path, "r") as f:
        data = json.load(f)
        step = 256
        data_torch = data["FP1-F7"]
        d = [data_torch[i:i + step] for i in range(0, len(data_torch), step)]
        data_torch = d[0]
        data_torch = torch.tensor(diag(data_torch))
        data_torch = data_torch.T @ data_torch
        out_data = encoding(data_torch)
    return out_data


torch.manual_seed(90)
epochs = 2
model = Q_AEnet()
# print(model)
loss_func = Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for enpoch in range(epochs):
    fid_list = []
    loss_list = []
    path = "E:/data/preictal/chb01_01_time1.json"
    with open(path, "r") as f:
        data = json.load(f)
        step = 256
        data_torch = data["FP1-F7"]
        d = [data_torch[i:i + step] for i in range(0, len(data_torch), step)]
    for k in range(len(d)):
        data_torch = d[k]
        data_torch = torch.tensor(diag(data_torch))
        data_torch = data_torch.T @ data_torch
        drug = encoding(data_torch)
        # print(drug.shape)
        output = model(drug, rho_C)
        loss = loss_func(drug, output)
        loss_list.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        fid = get_fid(drug, output, flag=1)
        fid_list.append(fid)
    print('enpochs:', enpoch + 1, 'loss:', '%.4f' % (sum(loss_list) / len(loss_list)), 'fid:',
          '%.4f' % (sum(fid_list) / len(fid_list)))
