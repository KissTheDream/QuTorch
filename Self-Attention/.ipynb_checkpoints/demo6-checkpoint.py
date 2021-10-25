import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F

VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25}

VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def smiles2int(drug):
    return [VOCAB_LIGAND_ISO[s] for s in drug]


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]


def rx(phi):
    """Single-qubit rotation for operator sigmax with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim=0), -1j * torch.sin(phi / 2).unsqueeze(dim=0),
                      -1j * torch.sin(phi / 2).unsqueeze(dim=0), torch.cos(phi / 2).unsqueeze(dim=0)), dim=0).reshape(2,
                                                                                                                      -1)
    # return torch.tensor([[torch.cos(phi / 2), -1j * torch.sin(phi / 2)],
    #              [-1j * torch.sin(phi / 2), torch.cos(phi / 2)]])


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
    """Single-qubit rotation for operator sigmaz with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim=0), torch.zeros(1),
                      torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim=0)), dim=0).reshape(2, -1)
    # return torch.tensor([[torch.exp(-1j * phi / 2), 0],
    #              [0, torch.exp(1j * phi / 2)]])


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


def dag(x):
    """
    compute conjugate transpose of input matrix
    """
    x_conj = torch.conj(x)
    x_dag = x_conj.permute(1, 0)
    return x_dag


def multi_kron(x_list):
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
    state为nqubit大小的密度矩阵，target为z门放置位置
    
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
    perform L2 regularization on x, x为complex
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


class QEqualizedConv0(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，也即有5个参数。
    """

    def __init__(self, n_qubits,
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
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


class Q_seq_representation1(nn.Module):
    def __init__(self, embedding_num_drug, embedding_dim_drug, embedding_num_target,
                 embedding_num_pocket, embedding_dim_target=4, embedding_dim_pocket=4):
        super().__init__()
        # embedding_num_drug == dim of input, not size
        # embedding_dim_drug == dim of output
        # embedding_dim_target, _pocket = 4
        self.embed_drug = nn.Embedding(embedding_num_drug, embedding_dim_drug, padding_idx=0)  # dim=64
        self.embed_target = nn.Embedding(embedding_num_target, embedding_dim_target, padding_idx=0)
        self.embed_pocket = nn.Embedding(embedding_num_pocket, embedding_dim_pocket, padding_idx=0)

        self.qconv1 = QEqualizedConv0(6)
        self.qconv2 = QEqualizedConv0(8)
        self.qconv3 = QEqualizedConv0(10)

        self.pool = QPool(10)

    def forward(self, drug, target, pocket):
        x = self.embed_drug(drug)
        y = self.embed_target(target)
        z = self.embed_pocket(pocket)

        qinput_x = encoding(x.T @ x)
        qconv1_x = self.qconv1(qinput_x)
        qinput_y = encoding(y.T @ y)
        qinput_xy = encoding(torch.kron(qconv1_x, qinput_y))
        qconv2_y = self.qconv2(qinput_xy)
        qinput_z = encoding(z.T @ z)
        qinput_xyz = encoding(torch.kron(qconv2_y, qinput_z))
        qconv3_z = self.qconv3(qinput_xyz)

        qpool_out = self.pool(qconv3_z)

        classical_value = measure(qpool_out, 10)
        return classical_value


class Q_seq_representation2(nn.Module):
    def __init__(self, embedding_num_target, embedding_dim_target, embedding_num_drug,
                 embedding_num_pocket, embedding_dim_drug=4, embedding_dim_pocket=4):
        super().__init__()
        # embedding_num_drug == dim of input, not size
        # embedding_dim_drug == dim of output
        # embedding_dim_target, _pocket = 4
        self.embed_drug = nn.Embedding(embedding_num_drug, embedding_dim_drug, padding_idx=0)
        self.embed_target = nn.Embedding(embedding_num_target, embedding_dim_target, padding_idx=0)  # dim=64
        self.embed_pocket = nn.Embedding(embedding_num_pocket, embedding_dim_pocket, padding_idx=0)

        self.qconv1 = QEqualizedConv0(6)
        self.qconv2 = QEqualizedConv0(8)
        self.qconv3 = QEqualizedConv0(10)

        self.pool = QPool(10)

    def forward(self, drug, target, pocket):
        y = self.embed_drug(drug)
        x = self.embed_target(target)
        z = self.embed_pocket(pocket)

        qinput_x = encoding(x.T @ x)
        qconv1_x = self.qconv1(qinput_x)
        qinput_y = encoding(y.T @ y)
        qinput_xy = encoding(torch.kron(qconv1_x, qinput_y))
        qconv2_y = self.qconv2(qinput_xy)
        qinput_z = encoding(z.T @ z)
        qinput_xyz = encoding(torch.kron(qconv2_y, qinput_z))
        qconv3_z = self.qconv3(qinput_xyz)

        qpool_out = self.pool(qconv3_z)

        classical_value = measure(qpool_out, 10)
        return classical_value


class Q_seq_representation3(nn.Module):
    def __init__(self, embedding_num_pocket, embedding_dim_pocket, embedding_num_drug,
                 embedding_num_target, embedding_dim_drug=4, embedding_dim_target=4):
        super().__init__()
        # embedding_num_drug == dim of input, not size
        # embedding_dim_drug == dim of output
        # embedding_dim_target, _pocket = 4 xxxx
        self.embed_drug = nn.Embedding(embedding_num_drug, embedding_dim_drug, padding_idx=0)
        self.embed_target = nn.Embedding(embedding_num_target, embedding_dim_target, padding_idx=0)
        self.embed_pocket = nn.Embedding(embedding_num_pocket, embedding_dim_pocket, padding_idx=0)  # dim=64

        self.qconv1 = QEqualizedConv0(6)
        self.qconv2 = QEqualizedConv0(8)
        self.qconv3 = QEqualizedConv0(10)

        self.pool = QPool(10)

    def forward(self, drug, target, pocket):
        z = self.embed_drug(drug)
        y = self.embed_target(target)
        x = self.embed_pocket(pocket)

        qinput_x = encoding(x.T @ x)
        qconv1_x = self.qconv1(qinput_x)
        qinput_y = encoding(y.T @ y)
        qinput_xy = encoding(torch.kron(qconv1_x, qinput_y))
        qconv2_y = self.qconv2(qinput_xy)
        qinput_z = encoding(z.T @ z)
        qinput_xyz = encoding(torch.kron(qconv2_y, qinput_z))
        qconv3_z = self.qconv3(qinput_xyz)

        qpool_out = self.pool(qconv3_z)

        classical_value = measure(qpool_out, 10)
        return classical_value


class DTImodel(nn.Module):
    def __init__(self):  # , measure_num=10):
        super().__init__()
        self.linear1 = nn.Linear(30, 512)
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, 512)
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(512, 128)
        self.drop3 = nn.Dropout(0.1)
        self.out_layer = nn.Linear(128, 1)

    def forward(self, protein_x, pocket_x, ligand_x):
        x = torch.cat([protein_x, pocket_x, ligand_x], dim=0)
        x = x.T
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = F.relu(self.linear2(x))
        x = self.drop2(x)
        x = F.relu(self.linear3(x))
        x = self.drop3(x)
        x = self.out_layer(x)
        x = x.view(1)

        return x


model = DTImodel()
criterion = nn.MSELoss()
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20):
    running_loss = 0.0
    MSE = 0
    prelist = []
    explist = []
    avepre = 0
    aveexp = 0
    dataset = pd.read_csv("E:/QDDTA/data/training_dataset.csv")
    for i in range(dataset.shape[0] - 2, dataset.shape[0]):  # 0
        data = dataset.iloc[i,]
        drug, target, pocket, label = data['smiles'], data['sequence'], data['pocket'], data['label']
        drug = smiles2int(drug)
        if len(drug) < 150:
            drug = np.pad(drug, (0, 150 - len(drug)))
        else:
            drug = drug[:150]
        target = seqs2int(target)
        if len(target) < 1000:
            target = np.pad(target, (0, 1000 - len(target)))
        else:
            target = target[:1000]
        pocket = seqs2int(pocket)
        if len(pocket) < 63:
            pocket = np.pad(pocket, (0, 63 - len(pocket)))
        else:
            pocket = pocket[:63]
        drug, target, pocket, exp = torch.tensor(drug, dtype=torch.long), torch.tensor(target,
                                                                                       dtype=torch.long), torch.tensor(
            pocket, dtype=torch.long), torch.tensor(label, dtype=torch.float).unsqueeze(-1)
        embedding_num_drug = 64
        embedding_dim_drug = 64
        embedding_num_target = 25
        embedding_num_pocket = 25
        drugencoder = Q_seq_representation1(embedding_num_drug, embedding_dim_drug, embedding_num_target,
                                            embedding_num_pocket)
        ligand_x = drugencoder(drug, target, pocket)
        embedding_num_target = 25
        embedding_dim_target = 64
        embedding_num_drug = 64
        embedding_num_pocket = 25
        targetencoder = Q_seq_representation2(embedding_num_target, embedding_dim_target, embedding_num_drug,
                                              embedding_num_pocket)
        protein_x = targetencoder(drug, target, pocket)
        embedding_num_pocket = 25
        embedding_dim_pocket = 64
        embedding_num_drug = 64
        embedding_num_target = 25
        pocketencoder = Q_seq_representation3(embedding_num_pocket, embedding_dim_pocket, embedding_num_drug,
                                              embedding_num_target)
        pocket_x = pocketencoder(drug, target, pocket)
        pre = model(protein_x, pocket_x, ligand_x)
        optimizer.zero_grad()
        loss = criterion(pre, exp)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        error2 = (float(pre - exp)) ** 2
        MSE = MSE + error2
        prelist.append(pre)
        explist.append(exp)
        avepre = avepre + pre
        aveexp = aveexp + exp

    MSE = MSE / (int(dataset.shape[0]))
    RMSE = MSE ** 0.5
    avepre = avepre / (int(dataset.shape[0]))
    aveexp = aveexp / (int(dataset.shape[0]))
    c = 0
    d = 0
    e = 0
    for j in range(0, len(prelist)):
        a = prelist[j]
        b = explist[j]
        c = c + (a - avepre) * (b - aveexp)
        d = d + (a - avepre) ** 2
        e = e + (b - aveexp) ** 2
    Rp = c / (d * e) ** 0.5

print('Train:Rp:' + '%.3f' % Rp + '\n' + 'MSE:' + '%.2f' % MSE + '\n' + 'RMSE:' + '%.2f' % RMSE)
print('Finished training')

PATH = './demoQ.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    dataset = pd.read_csv("E:/QDDTA/data/validation_dataset.csv")
    MSE = 0
    prelist = []
    explist = []
    avepre = 0
    aveexp = 0
    for i in range(dataset.shape[0] - 1, dataset.shape[0]):  # 0
        data = dataset.iloc[i,]
        drug, target, pocket, label = data['smiles'], data['sequence'], data['pocket'], data['label']
        drug = smiles2int(drug)
        if len(drug) < 150:
            drug = np.pad(drug, (0, 150 - len(drug)))
        else:
            drug = drug[:150]
        target = seqs2int(target)
        if len(target) < 1000:
            target = np.pad(target, (0, 1000 - len(target)))
        else:
            target = target[:1000]
        pocket = seqs2int(pocket)
        if len(pocket) < 63:
            pocket = np.pad(pocket, (0, 63 - len(pocket)))
        else:
            pocket = pocket[:63]
        drug, target, pocket, exp = torch.tensor(drug, dtype=torch.long), torch.tensor(target,
                                                                                       dtype=torch.long), torch.tensor(
            pocket, dtype=torch.long), torch.tensor(label, dtype=torch.float).unsqueeze(-1)
        embedding_num_drug = 64
        embedding_dim_drug = 64
        embedding_num_target = 25
        embedding_num_pocket = 25
        drugencoder = Q_seq_representation1(embedding_num_drug, embedding_dim_drug, embedding_num_target,
                                            embedding_num_pocket)
        ligand_x = drugencoder(drug, target, pocket)
        embedding_num_target = 25
        embedding_dim_target = 64
        embedding_num_drug = 64
        embedding_num_pocket = 25
        targetencoder = Q_seq_representation2(embedding_num_target, embedding_dim_target, embedding_num_drug,
                                              embedding_num_pocket)
        protein_x = targetencoder(drug, target, pocket)
        embedding_num_pocket = 25
        embedding_dim_pocket = 64
        embedding_num_drug = 64
        embedding_num_target = 25
        pocketencoder = Q_seq_representation3(embedding_num_pocket, embedding_dim_pocket, embedding_num_drug,
                                              embedding_num_target)
        pocket_x = pocketencoder(drug, target, pocket)
        pre = model(protein_x, pocket_x, ligand_x)
        error2 = (float(pre - exp)) ** 2
        MSE = MSE + error2
        prelist.append(pre)
        explist.append(exp)
        avepre = avepre + pre
        aveexp = aveexp + exp

    MSE = MSE / (int(dataset.shape[0]))
    RMSE = MSE ** 0.5
    avepre = avepre / (int(dataset.shape[0]))
    aveexp = aveexp / (int(dataset.shape[0]))
    c = 0
    d = 0
    e = 0
    for j in range(0, len(prelist)):
        a = prelist[j]
        b = explist[j]
        c = c + (a - avepre) * (b - aveexp)
        d = d + (a - avepre) ** 2
        e = e + (b - aveexp) ** 2
    Rp = c / (d * e) ** 0.5

print('Validation:Rp:' + '%.3f' % Rp + '\n' + 'MSE:' + '%.2f' % MSE + '\n' + 'RMSE:' + '%.2f' % RMSE)

with torch.no_grad():
    dataset = pd.read_csv("E:/QDDTA/data/test_dataset.csv")
    MSE = 0
    prelist = []
    explist = []
    avepre = 0
    aveexp = 0
    for i in range(dataset.shape[0] - 1, dataset.shape[0]):  # 0
        data = dataset.iloc[i,]
        drug, target, pocket, label = data['smiles'], data['sequence'], data['pocket'], data['label']
        drug = smiles2int(drug)
        if len(drug) < 150:
            drug = np.pad(drug, (0, 150 - len(drug)))
        else:
            drug = drug[:150]
        target = seqs2int(target)
        if len(target) < 1000:
            target = np.pad(target, (0, 1000 - len(target)))
        else:
            target = target[:1000]
        pocket = seqs2int(pocket)
        if len(pocket) < 63:
            pocket = np.pad(pocket, (0, 63 - len(pocket)))
        else:
            pocket = pocket[:63]
        drug, target, pocket, exp = torch.tensor(drug, dtype=torch.long), torch.tensor(target,
                                                                                       dtype=torch.long), torch.tensor(
            pocket, dtype=torch.long), torch.tensor(label, dtype=torch.float).unsqueeze(-1)
        embedding_num_drug = 64
        embedding_dim_drug = 64
        embedding_num_target = 25
        embedding_num_pocket = 25
        drugencoder = Q_seq_representation1(embedding_num_drug, embedding_dim_drug, embedding_num_target,
                                            embedding_num_pocket)
        ligand_x = drugencoder(drug, target, pocket)
        embedding_num_target = 25
        embedding_dim_target = 64
        embedding_num_drug = 64
        embedding_num_pocket = 25
        targetencoder = Q_seq_representation2(embedding_num_target, embedding_dim_target, embedding_num_drug,
                                              embedding_num_pocket)
        protein_x = targetencoder(drug, target, pocket)
        embedding_num_pocket = 25
        embedding_dim_pocket = 64
        embedding_num_drug = 64
        embedding_num_target = 25
        pocketencoder = Q_seq_representation3(embedding_num_pocket, embedding_dim_pocket, embedding_num_drug,
                                              embedding_num_target)
        pocket_x = pocketencoder(drug, target, pocket)
        pre = model(protein_x, pocket_x, ligand_x)
        error2 = (float(pre - exp)) ** 2
        MSE = MSE + error2
        prelist.append(pre)
        explist.append(exp)
        avepre = avepre + pre
        aveexp = aveexp + exp

    MSE = MSE / (int(dataset.shape[0]))
    RMSE = MSE ** 0.5
    avepre = avepre / (int(dataset.shape[0]))
    aveexp = aveexp / (int(dataset.shape[0]))
    c = 0
    d = 0
    e = 0
    for j in range(0, len(prelist)):
        a = prelist[j]
        b = explist[j]
        c = c + (a - avepre) * (b - aveexp)
        d = d + (a - avepre) ** 2
        e = e + (b - aveexp) ** 2
    Rp = c / (d * e) ** 0.5

print('Test:Rp:' + '%.3f' % Rp + '\n' + 'MSE:' + '%.2f' % MSE + '\n' + 'RMSE:' + '%.2f' % RMSE)
