import pickle as pkl

with open('F1_scores.pkl', 'rb') as f :
    f1_scores = pkl.load(f)
with open('EM_scores.pkl', 'rb') as f :
    em_scores = pkl.load(f)

import numpy as np

idx = np.argmax(f1_scores)

print(f1_scores[idx])
print(em_scores[idx])