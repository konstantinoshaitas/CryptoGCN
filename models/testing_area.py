import numpy as np
import os

# Load using relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
sequential_data_dir = os.path.join(data_dir, 'processed_sequential_data')
results_dir = os.path.join(data_dir, 'tgcn_results')


a = np.load(os.path.join(results_dir, 'train_gcn_outputs.npy'))
print(a)
