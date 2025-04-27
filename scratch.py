import numpy as np


original = "D:/datasets/ESCount/exemplar_VideoMAEtokens_original_RepCount/train951.npz"
new_data = "D:/datasets/ESCount/exemplar_VideoMAEtokens_new_RepCount/train951.npz"

n1 = np.load(original)['arr_0']
n2 = np.load(new_data)['arr_0']

n = n1-n2

print(np.max(np.abs(n)))
