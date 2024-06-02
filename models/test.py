import numpy as np

data = np.array((5,6,10,20,30,60,70), dtype=float)
data = data.reshape(-1,1)
sequence_length = 3

xs, ys = [], []
loop = len(data) - sequence_length
for i in range(loop):
    x = data[i:(i + sequence_length)]
    y = data[i + sequence_length]
    xs.append(x)
    ys.append(y)

print(f'xs: {xs}, ys: {ys}')
print(len(ys))

print(data[0:(0 + sequence_length)])
print(data[0 + sequence_length])


