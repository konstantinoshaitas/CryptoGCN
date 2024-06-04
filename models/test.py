import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
#
# data = np.array((5,6,10,20,30,60,70), dtype=float)
# data = data.reshape(-1,1)
# sequence_length = 3
#
# xs, ys = [], []
# loop = len(data) - sequence_length
# for i in range(loop):
#     x = data[i:(i + sequence_length)]
#     y = data[i + sequence_length]
#     xs.append(x)
#     ys.append(y)
#
# print(f'xs: {xs}, ys: {ys}')
# print(len(ys))
#
# print(data[0:(0 + sequence_length)])
# print(data[0 + sequence_length])


data = np.array([[i] for i in range(1, 11)])
print("Dataset:")
print(data)

tscv = TimeSeriesSplit(n_splits=3)

# Print the train and test indices for each split
for i, (train_index, test_index) in enumerate(tscv.split(data)):
    print(f"Split {i+1}:")
    print("Train indices:", train_index)
    print("Test indices:", test_index)
    print("Train data:", data[train_index].flatten())
    print("Test data:", data[test_index].flatten())
    print("-" * 30)


