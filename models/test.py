import numpy as np

data = np.array((5,6,10,20,30), dtype=float)
sequence_length = 3

xs, ys = [], []
loop = len(data) - sequence_length
for i in range(loop):
    x = data[i:(i + sequence_length)]
    y = data[i + sequence_length]
    xs.append(x)
    ys.append(y)

print(f'xs: {xs}, ys: {ys}')


plt.plot()
