import numpy as np
data = np.load('CNN_DIDI_withPOI.npz')['arr_0']
demand = data[:, :, :, :1]
supply = data[:, :, :, 1:2]


def classify(ori):
    classified = np.zeros(ori.shape)
    avg = 16
    for i in range(2880):
        for j in range(16):
            for k in range(16):
                if ori[i][j][k][0] < avg:
                    classified[i][j][k][0] = 0
                elif ori[i][j][k][0] < 2*avg:
                    classified[i][j][k][0] = 1
                elif ori[i][j][k][0] < 3*avg:
                    classified[i][j][k][0] = 2
                else:
                    classified[i][j][k][0] = 3
    return classified

demand_aux = classify(demand)
supply_aux = classify(supply)

auxiliary = np.concatenate((demand_aux, supply_aux), -1)
print(auxiliary.shape)
np.save('auxiliary', auxiliary)