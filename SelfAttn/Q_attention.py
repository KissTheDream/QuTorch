#!/usr/bin/env python
# coding: utf-8

# # Attention - Qutorch

# In[72]:


import numpy as np
import math, copy, time
import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


"""
def smiles2int(drug):

    return [VOCAB_LIGAND_ISO[s] for s in drug]

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 
"""

def rx(phi):
    """Single-qubit rotation for operator sigmax with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim = 0), -1j * torch.sin(phi / 2).unsqueeze(dim = 0), 
                      -1j * torch.sin(phi / 2).unsqueeze(dim = 0), torch.cos(phi / 2).unsqueeze(dim = 0)),dim = 0).reshape(2,-1)
    # return torch.tensor([[torch.cos(phi / 2), -1j * torch.sin(phi / 2)],
    #              [-1j * torch.sin(phi / 2), torch.cos(phi / 2)]])

def ry(phi):
    """Single-qubit rotation for operator sigmay with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """

    return torch.cat((torch.cos(phi / 2).unsqueeze(dim = 0), -1 * torch.sin(phi / 2).unsqueeze(dim = 0), 
                      torch.sin(phi / 2).unsqueeze(dim = 0), torch.cos(phi / 2).unsqueeze(dim = 0)), dim = 0).reshape(2,-1) + 0j
    # return torch.tensor([[torch.cos(phi / 2), -torch.sin(phi / 2)],
    #              [torch.sin(phi / 2), torch.cos(phi / 2)]])
    
def rz(phi):
    """Single-qubit rotation for operator sigmaz with angle phi.
    -------
    result : torch.tensor for operator describing the rotation.
    """
    return torch.cat((torch.exp(-1j * phi / 2).unsqueeze(dim = 0), torch.zeros(1), 
                      torch.zeros(1), torch.exp(1j * phi / 2).unsqueeze(dim = 0)), dim = 0).reshape(2,-1)    
    # return torch.tensor([[torch.exp(-1j * phi / 2), 0],
    #              [0, torch.exp(1j * phi / 2)]])

def x_gate():
    """
    Pauli x
    """
    return torch.tensor([[0, 1], [1, 0]]) + 0j

def y_gate():
    """
    Pauli y
    """
    return torch.tensor([[0, 0-1j], [0+1j, 0]])

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


# In[3]:


def gate_control(U,N,control,target):
    if N<1:
        raise ValueError("integer N must be larger or equal to 1")
    if control >= N:
        raise ValueError("control must be integer < integer N")
    if target >= N:
        raise ValueError("target must be integer < integer N")
    if target==control:
        raise ValueError("control cannot be equal to target")
        
    zero_zero=torch.tensor([[1, 0],[0, 0]]) + 0j
    one_one=torch.tensor([[0, 0],[0, 1]]) + 0j
    list1=[torch.eye(2)]*N
    list2=[torch.eye(2)]*N
    list1[control]=zero_zero
    list2[control]=one_one
    list2[target]=U
    
    return multi_kron(list1)+multi_kron(list2)


# In[4]:


def gate_expand_1toN(U, N, target):
    """
    representing a one-qubit gate that act on a system with N qubits.

    """

    if N < 1:
        raise ValueError("integer N must be larger or equal to 1")

    if target >= N:
        raise ValueError("target must be integer < integer N")

    return multi_kron([torch.eye(2)]* target + [U] + [torch.eye(2)] * (N - target - 1))

def gate_expand_2toN(U, N, targets):
    """
    representing a two-qubit gate that act on a system with N qubits.
    
    """

    if N < 2:
        raise ValueError("integer N must be larger or equal to 2")

    if targets[1] >= N:
        raise ValueError("target must be integer < integer N")

    return multi_kron([torch.eye(2)]* targets[0] + [U] + [torch.eye(2)] * (N - targets[1] - 1))


# In[5]:


def gate_sequence_product(U_list, n_qubits, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary operations.
    return: Unitary matrix corresponding to U_list.
    """

    U_overall = torch.eye(2 ** n_qubits, 2 **  n_qubits) + 0j
    for U in U_list:
        if left_to_right:
            U_overall = U @ U_overall
        else:
            U_overall = U_overall @ U

    return U_overall

def gate_sequence_product(U_list, n_qubits, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary operations.
    return: Unitary matrix corresponding to U_list.
    """

    U_overall = torch.eye(2 ** n_qubits, 2 **  n_qubits) + 0j
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
    mat_dim_A = 2**dimA
    mat_dim_B = 2**dimB

    id1 = torch.eye(mat_dim_A, requires_grad=True) + 0.j
    id2 = torch.eye(mat_dim_B, requires_grad=True) + 0.j

    pout = 0
    for i in range(mat_dim_B):
        p = torch.kron(id1, id2[i]) @ rhoAB @ torch.kron(id1, id2[i].reshape(mat_dim_B, 1))
        pout += p
    return pout


# In[6]:


def expecval_ZI(state, nqubit, target):
    """
    state为nqubit大小的密度矩阵，target为z门放置位置
    
    """
    zgate=z_gate()
    H = gate_expand_1toN(zgate, nqubit, target)
    expecval = (state @ H).trace() #[-1,1]
    expecval_real = (expecval.real + 1) / 2 #[0,1]
    
    return expecval_real

def measure(state, nqubit):
    """
    测量nqubit次期望
    
    """
    measure = torch.zeros(nqubit, 1)
    for i in range(nqubit):
        measure[i] = expecval_ZI(state, nqubit, list(range(nqubit))[i])

    return measure


# In[7]:


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


# In[ ]:





# In[503]:


class init_cir_q(nn.Module):
    """初始化attn—q
    
    """

    def __init__(self, n_qubits=2, 
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)# theta_size=5
        
        self.n_qubits = n_qubits


    def queryQ(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits):
            cir.append(gate_expand_1toN(rx(w[which_q*3+0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[which_q*3+1]), self.n_qubits, which_q))        
            cir.append(gate_expand_1toN(rz(w[which_q*3+2]), self.n_qubits, which_q))
        #for which_q in range(0, self.n_qubits, 2):
            #cir.append(gate_expand_1toN(rx(w[0]), self.n_qubits, which_q))
            #cir.append(gate_expand_1toN(rx(w[1]), self.n_qubits, which_q + 1))
            #cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            #cir.append(gate_expand_1toN(ry(w[2]), self.n_qubits, which_q))        
            #cir.append(gate_expand_1toN(ry(w[3]), self.n_qubits, which_q + 1))
            #cir.append(gate_expand_1toN(rz(w[4]), self.n_qubits, which_q))        
            #cir.append(gate_expand_1toN(rz(w[5]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)
        return U


    def forward(self, x):
        E_out = self.queryQ()
        queryQ_out = E_out@ x @ dag(E_out)
        return queryQ_out


# In[504]:


class init_cir_k(nn.Module):
    """初始化attn—q
    
    """

    def __init__(self, n_qubits=2, 
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)# theta_size=5
        
        self.n_qubits = n_qubits


    def keyQ(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits):
            cir.append(gate_expand_1toN(rx(w[which_q*3+0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[which_q*3+1]), self.n_qubits, which_q))        
            cir.append(gate_expand_1toN(rz(w[which_q*3+2]), self.n_qubits, which_q))
        #for which_q in range(0, self.n_qubits, 2):
            #cir.append(gate_expand_1toN(ry(w[0]), self.n_qubits, which_q))
            #cir.append(gate_expand_1toN(ry(w[1]), self.n_qubits, which_q + 1))
            ##cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            #cir.append(gate_expand_1toN(rz(w[2]), self.n_qubits, which_q))        
            #cir.append(gate_expand_1toN(rz(w[3]), self.n_qubits, which_q + 1))
            #cir.append(gate_expand_1toN(rx(w[4]), self.n_qubits, which_q))        
            #cir.append(gate_expand_1toN(rx(w[5]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)
        return U


    def forward(self, x):
        E_out = self.keyQ()
        keyQ_out = E_out @ x @ dag(E_out)
        return keyQ_out


# In[505]:


class init_cir_v(nn.Module):
    """初始化attn—q
    
    """

    def __init__(self, n_qubits=2, 
                 gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(n_qubits*3), a=0.0, b=2*np.pi) * init_std)# theta_size=5
        
        self.n_qubits = n_qubits


    def valueQ(self):
        w = self.weight * self.w_mul
        cir = []
        for which_q in range(0, self.n_qubits):
            cir.append(gate_expand_1toN(rx(w[which_q*3+0]), self.n_qubits, which_q))
            cir.append(gate_expand_1toN(ry(w[which_q*3+1]), self.n_qubits, which_q))        
            cir.append(gate_expand_1toN(rz(w[which_q*3+2]), self.n_qubits, which_q))
        #for which_q in range(0, self.n_qubits, 2):
            #cir.append(gate_expand_1toN(rz(w[0]), self.n_qubits, which_q))
            #cir.append(gate_expand_1toN(rz(w[1]), self.n_qubits, which_q + 1))
            ##cir.append(gate_expand_2toN(ryy(w[2]), self.n_qubits, [which_q, which_q + 1]))
            #cir.append(gate_expand_1toN(rx(w[2]), self.n_qubits, which_q))        
            #cir.append(gate_expand_1toN(rx(w[3]), self.n_qubits, which_q + 1))
            #cir.append(gate_expand_1toN(ry(w[4]), self.n_qubits, which_q))        
            #cir.append(gate_expand_1toN(ry(w[5]), self.n_qubits, which_q + 1))
        U = gate_sequence_product(cir, self.n_qubits)
        return U


    def forward(self, x):
        E_out = self.valueQ()
        valueQ_out = E_out @ x @ dag(E_out)
        return valueQ_out


# In[506]:


def cal_query_key(queryQ_out, keyQ_out, dim_q, dim_k):
    """queryQ_out: type torch.Tensor
       keyQ_out: torch.Tensor
    """
    out = torch.kron(queryQ_out, keyQ_out)
    n_qubits = dim_q + dim_k
    
    U_list=[]
    for t in range(dim_k):
        U_list.append(gate_control(x_gate(),n_qubits,t,n_qubits-dim_k+t))
    U_overall=gate_sequence_product(U_list, n_qubits)
    
    out=U_overall @ out @ dag(U_overall)
    
    quantum_score = measure(out, n_qubits)
    
    return quantum_score


# In[507]:


def cal_src_value(quantum_src, valueQ_out, dim_s, dim_v):
    """input torch.Tensor
    """
    src=quantum_src.mean()
    phi=(src-0.5)*2*np.pi #phi=[-pi,pi]
    
    U_list=[]
    ux=rx(phi*0.5)
    uy=ry(phi*0.5)
    uz=rz(phi)
    for i in range(dim_v):
        U_list.append(gate_expand_1toN(ux, dim_v, i))
        U_list.append(gate_expand_1toN(uy, dim_v, i))
        U_list.append(gate_expand_1toN(uz, dim_v, i))
    
    U_overall=gate_sequence_product(U_list,dim_v)
    quantum_weighted_value = U_overall @ valueQ_out @ dag(U_overall)
    
    return quantum_weighted_value


# In[508]:


def cal_output(qwv_list, dim):
    
    out = multi_kron(qwv_list)
    n_qubits=len(qwv_list)*dim
    U_list=[]
    for i in range(len(qwv_list)-1):
        for t in range(dim):
            U_list.append(gate_control(x_gate(),n_qubits,i*dim+t,n_qubits-dim+t))
            
    U_overall=gate_sequence_product(U_list, n_qubits)
    
    out=U_overall @ out @ dag(U_overall)
        
    attnQ = ptrace(out, dim, n_qubits-dim)
    return attnQ


# In[562]:


def clones(module, N):
    #"Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[563]:


def q_attention(query, key, value, mask=None, dropout=None):
    #"Compute 'Scaled Dot Product Attention'"
    query_input=query.squeeze(0)
    key_input=key.squeeze(0)
    value_input=value.squeeze(0)
    #print(query_input.size(-1))
    n_qubits=math.ceil(math.log2(query_input.size(-1)))
    #print(n_qubits)
    
    qqs=[]
    qks=[]
    qvs=[]
    #print(query_input.size())
    for x in query_input.chunk(query_input.size(0),0):
        #print(x.size())
        #expand to 2**n_qubits length vector
        qx=nn.ZeroPad2d((0,2**n_qubits-query_input.size(-1),0,0))(x)
        #l2-regularization
        if qx.dim()>2:
            qx=qx.squeeze()
        #print(qx.size())
        qtmp=qx.T@qx
        qinput=encoding(qtmp)
        #print(qinput)
        init_q=init_cir_q(n_qubits=n_qubits)
        qqs.append(init_q(qinput))
        
    for x in key_input.chunk(key_input.size(0),0):
        #expand to 2**n_qubits length vector
        qx=nn.ZeroPad2d((0,2**n_qubits-key_input.size(-1),0,0))(x)
        #l2-regularization
        if qx.dim()>2:
            qx=qx.squeeze()
        qtmp=qx.T@qx
        qinput=encoding(qtmp)
        #print(qinput)
        init_k=init_cir_k(n_qubits=n_qubits)
        qks.append(init_k(qinput))
        
    for x in value_input.chunk(value_input.size(0),0):
        #expand to 2**n_qubits length vector
        qx=nn.ZeroPad2d((0,2**n_qubits-query_input.size(-1),0,0))(x)
        #l2-regularization
        if qx.dim()>2:
            qx=qx.squeeze()
        qtmp=qx.T@qx
        qinput=encoding(qtmp)
        #print(qinput)
        init_v=init_cir_v(n_qubits=n_qubits)
        qvs.append(init_v(qinput))
    
    outputs=[]
    for i in range(len(qqs)):
        qwvs_i=[]
        for j in range(len(qks)):
            score_ij=cal_query_key(qqs[i],qks[j],n_qubits,n_qubits)
            qwvs_i.append(cal_src_value(score_ij,qvs[j],n_qubits,n_qubits))
        out_i=measure(cal_output(qwvs_i,n_qubits),n_qubits).squeeze().unsqueeze(0)
        outputs.append(out_i)
        #print(out_i)
    
    return torch.cat(outputs)


# In[564]:


class Q_MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(Q_MultiHeadedAttention, self).__init__()
        #assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear = nn.Linear((d_model+1)//2,d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Apply attention on all the projected vectors in batch. 
        x = q_attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        #print(x)
        x=x.unsqueeze(0)
        #print(x)
        return self.linear(x)


# # ----------------------------------------