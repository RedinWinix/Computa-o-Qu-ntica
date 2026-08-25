import sys
import os

current_wd = os.getcwd()
sys.path.append(current_wd)

import time
import json
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#set the seed - identical to the PQK/QK scripts, so the comparison is fair
seed = 123
np.random.seed(seed)

# NOTE: same file + same split as run_PQK_checkpointed.py and (now aligned)
# run_QK_checkpointed.py, so all three rows of the comparison table are
# evaluated on identical train/test data.
data_file_csv = 'data/env.sel3.sk_sc.csv'
f_rate = 1
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)

Y = env['occupancy']
X = env[['illuminance', 'blinds', 'lamps', 'rh', 'co2', 'temp']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

#best hyperparameters reported in the paper's Table 1 for the classical SVM row
kernel_type = 'rbf'
C_value = 2
gamma_value = 4.0

print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train_np.shape}')
print(f'Test shape dataset {X_test_np.shape}')
print(f'Using kernel: {kernel_type}, C: {C_value}, gamma: {gamma_value}')

t_start = time.time()
svm = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value).fit(X_train_np, y_train_np)
t_training = time.time()

predictions = svm.predict(X_test_np)
t_prediction = time.time()

score = accuracy_score(predictions, y_test_np)
f1 = f1_score(y_test_np, predictions)
t_final = time.time()

run_id = f'run_SVM_seed{seed}'
checkpoint_dir = os.path.join('qfm', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

results_path = os.path.join(checkpoint_dir, f'{run_id}_results.json')
results = {
    'run_id': run_id,
    'method': 'Classical_SVM',
    'kernel': kernel_type,
    'C': C_value,
    'gamma': gamma_value,
    'accuracy': score,
    'f1': f1,
    'kernel_training_seconds': 0.0,   # no separate "kernel training" step for classical SVM
    'svm_training_seconds': t_training - t_start,
    'total_seconds': t_final - t_start,
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=3)

print(f'*******SCORE (accuracy): {score}')
print(f'*******F1 SCORE: {f1}')
print(f'Time training: {t_training - t_start} seconds.')
print(f'Time prediction: {t_prediction - t_training} seconds.')
print(f'Final time: {t_final - t_start} seconds')
print(f'*** Results saved to: {results_path}')
