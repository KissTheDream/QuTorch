# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:07:29 2021

@author: shish
"""
import torch
import math

'''
用途在于衡量一个参数化量子线路的纠缠能力，即对所有参数取随机值，用本函数计算输出量子态的
纠缠度，重复很多次，计算平均纠缠度，就是对线路纠缠能力的衡量。
'''


def _D_wedge(u, v):
    '''
    对矢量u和v做wedge product得到新矢量，再对新矢量每个分量的模平方求和
    不要单独使用这个函数
    '''
    rst = 0.0
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            rst += abs(u[i] * v[j] - u[j] * v[i]) ** 2
    return rst


def _lj(j, b, phi, N):
    '''
    不要单独使用这个函数
    '''
    rst = []
    for i in range(2 ** N):
        s = format(i, "b")
        str_lst = ['0'] * (N - len(s)) + list(s)
        if str_lst[j - 1] == str(b):
            rst.append(phi[i])
    rst = torch.tensor(rst) + 0j
    return rst


def MW_entanglement_measure(phi, N):
    '''
    Meyer-Wallach纠缠度量，衡量一个纯态phi的纠缠度。
    纠缠度：衡量一个纯态的纠缠程度，输入一个态矢phi，和qubit数目N，输出一个0~1的实数
    要求满足：最大纠缠态，纠缠度=1；可分离态，纠缠度=0；单比特门操作，不改变纠缠度。
    N为qubit数目，phi为量子态态矢，一个维度为2^N的复数矢量
    '''
    if len(phi.shape) != 1:
        raise ValueError("phi should be a vector")
    if phi.shape[0] != 2 ** N:
        raise ValueError("dim of state should be 2^N")

    summ = 0.0
    for each in phi:
        summ += abs(each) ** 2
    if abs(summ - 1) > 1e-6:
        raise ValueError("state vector should be normalized to 1")

    phi = phi + 0j

    summ = 0.0
    for j in range(1, N + 1):
        u = _lj(j, 0, phi, N)
        v = _lj(j, 1, phi, N)
        summ += _D_wedge(u, v)

    return (4.0 / N) * summ


if __name__ == '__main__':
    # 随便做点测试，最大纠缠态纠缠度为1，可分离态纠缠度为0
    N = 2
    # bell纠缠态
    phi = torch.tensor([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
    print("bell:", MW_entanglement_measure(phi, N))
    phi = torch.tensor([1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)])
    print("bell:", MW_entanglement_measure(phi, N))
    phi = torch.tensor([0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0])
    print("bell:", MW_entanglement_measure(phi, N))
    phi = torch.tensor([0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0])
    print("bell:", MW_entanglement_measure(phi, N))

    # 可分离态
    phi = torch.tensor([0.5, 0.5, 0.5, 0.5])
    print(MW_entanglement_measure(phi, N))
    phi = torch.tensor([1, 0, 0, 0])
    print(MW_entanglement_measure(phi, N))

    # GHZ纠缠态
    N = 3
    phi = torch.tensor([1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)])
    print("GHZ:", MW_entanglement_measure(phi, N))

    phi = torch.tensor([1 / math.sqrt(3), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(3 / 2.0)])
    print("GHZ:", MW_entanglement_measure(phi, N))

    N = 4
    phi = torch.tensor([1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / math.sqrt(2)])
    print("GHZ:", MW_entanglement_measure(phi, N))

    N = 8
    phi = torch.rand(2 ** N)
    norm = 0.0
    for each in phi:
        norm += abs(each) ** 2

    phi = phi * (1.0 / math.sqrt(norm)) + 0j

    print("rand:", MW_entanglement_measure(phi, N))

    # ========test:一个态矢phi的纠缠度在单比特门操作下不变=====================
