import numpy as np
eventfile = np.load('LSTM_event.npz')['arr_0']
event = np.zeros((2880, 16, 16, 1))
for timeslice in eventfile:
    if timeslice[1] == -1:
        continue
    else:
        event[timeslice[0]][timeslice[1]][timeslice[2]][0] = 1