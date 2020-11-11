import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing
#本部分分为两部分，边权数据的二次处理，graph类的构造

#边权数据二次处理
#加载原数据
B1 = np.load('ave_traveltime_晚高峰17to19.npz')['arr_0']
print(B1.shape)
B2 = np.load('ave_traveltime_平峰.npz')['arr_0']
print(B2.shape)
min_max_scaler = preprocessing.MinMaxScaler()
B1 = min_max_scaler.fit_transform(B1)
B2 = min_max_scaler.fit_transform(B2)


#考虑不连接的两点边权为0，自链接为1，所以边权取通行时间最大-最小归一化化之后的倒数，倒数值再归1化
#遍历每一个格点
def Bian(B):
    for j in range(0, 4):
        for i in range(0, 256):
            if B[i][j] > 0:
                B[i][j] = 1/B[i][j]
            else:
                B[i][j] = 0
    print('123124123')
    min_max_scaler = preprocessing.MinMaxScaler()
    B = min_max_scaler.fit_transform(B)
    BB = []
    for i in range(16):
        for j in range(16):
            #上
            if B[i*16+j][0] > 0:
                BB.append((i*16+j+1, i*16+j+16+1, B[i*16+j][0]))
                print((i*16+j+1, i*16+j+16+1, B[i*16+j][0]))
            else:
                print('1')
            #下
            if B[i*16+j][1] > 0:
                BB.append((i * 16 + j + 1, i * 16 + j+1-16, B[i * 16 + j][1]))
                print((i * 16 + j + 1, i * 16 + j + 1-16, B[i * 16 + j][1]))
            else:
                print('2')
            #左
            if B[i*16+j][2] > 0:
                BB.append((i * 16 + j + 1, i * 16 + j, B[i * 16 + j][2]))
                print((i * 16 + j + 1, i * 16 + j, B[i * 16 + j][2]))
            else:
                print('3')
            #右
            if B[i*16+j][3] > 0:
                BB.append((i * 16 + j + 1, i * 16 + j + 2, B[i * 16 + j][3]))
                print((i * 16 + j + 1, i * 16 + j + 2, B[i * 16 + j][3]))
            else:
                print('4')
    print(BB)
    print(len(BB))
    return BB
#高峰时间
BB1 = []
BB1 = Bian(B1)[:]
print(len(BB1))
#平峰时间
BB2 = []
BB2 = Bian(B2)[:]
print(BB2)
print(len(BB2))
#graph类构造
P = []
for i in range(256):
   P.append(i+1)
print(P)
G1 = nx.DiGraph()
G1.add_nodes_from(P)
G1.add_weighted_edges_from(BB1)
#nx.draw(G1, with_labels=True)
#plt.savefig("directed_graph1.png")
#plt.show()
#Graph2
G2 = nx.DiGraph()
G2.add_nodes_from(P)
G2.add_weighted_edges_from(BB2)
#nx.draw(G2, with_labels=True)
#plt.savefig("directed_graph2.png")
#plt.show()