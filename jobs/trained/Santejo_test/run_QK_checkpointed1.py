import sys
import os

#define working directory and package for QK
current_wd = os.getcwd()
sys.path.append(current_wd)

import time
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit

from qiskit_machine_learning.kernels.algorithms.quantum_kernel_trainer import QuantumKernelTrainer
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.kernels import TrainableFidelityStatevectorKernel
from qiskit_algorithms.utils import algorithm_globals

from pqk.QKCallback_checkpointed import QKCallback


#set the seed
seed = 1234
np.random.seed(seed)
algorithm_globals.random_seed = seed

#load dataset with panda
f_rate = 1  # rate of data sampling for testing purpose - reduce for a fast smoke test
# NOTE: aligned to the same scaled dataset as run_PQK_checkpointed.py
# (the original run_QK.py used env.sel3.minmax.csv instead - different
# preprocessing - which would make accuracy/F1 not directly comparable
# across methods. Using the same file + same split here so all three rows
# of the comparison table are apples-to-apples.)
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)

Y = env['occupancy']
X = env[['illuminance', 'blinds', 'lamps', 'rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

#---- TRAINING BUDGET ----
#kept intentionally under 20 iterations so a full-dataset comparison run
#finishes quickly - QK's kernel evaluation (state-fidelity overlaps) is much
#cheaper per-iteration than PQK's, so this is the main "make it fast" lever.
max_iter = 1

#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Max number of iteration used in kernel optimization: {max_iter}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Test shape dataset {X_test.shape}')

#build a feature map
NUM_QBIT = X_train.shape[1]
fm = QuantumCircuit(NUM_QBIT)
input_params = ParameterVector('x_par', NUM_QBIT)
training_params = ParameterVector('theta_par', NUM_QBIT)

# Create an initial rotation layer of trainable parameters
for i, param in enumerate(training_params):
    fm.ry(param, fm.qubits[i])

# Create a rotation layer of input parameters
for i, param in enumerate(input_params):
    fm.rz(param, fm.qubits[i])

print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())

# ---- CHECKPOINT / RESUME SETUP ----
# same auto-derived-run_id pattern as run_PQK_checkpointed.py, so this run
# can never collide with the PQK results (different prefix) or with itself
# under a different config.
run_id = f'run_QK_seed{seed}'
checkpoint_dir = os.path.join('qfm', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f'{run_id}_latest.json')
print(f'*** Run ID (drives all filenames for this config): {run_id}')

resumed = QKCallback.load_checkpoint(checkpoint_path)

if resumed is not None:
    init_point = np.array(resumed['parameters'])
    completed_iters = resumed['iteration']
    remaining_iters = max(0, max_iter - completed_iters)
    print(f'*** RESUMING from checkpoint: {checkpoint_path}')
    print(f'*** Checkpoint was at iteration {completed_iters} (of {max_iter}), '
          f'saved at {resumed["timestamp"]}')
    print(f'*** {remaining_iters} iterations remaining')
else:
    init_point = np.array([np.pi / 2 for _ in range(NUM_QBIT)])
    completed_iters = 0
    remaining_iters = max_iter
    print('*** No checkpoint found, starting fresh')

print(f'Initial point: {init_point}')

my_callback = QKCallback(checkpoint_path=checkpoint_path, checkpoint_every=1)
my_callback.num_iteration = completed_iters

#define the trainable kernel
q_kernel = TrainableFidelityStatevectorKernel(feature_map=fm, training_parameters=training_params)

if remaining_iters <= 0:
    print('*** Checkpoint already reached max_iter - skipping SPSA training, '
          'binding saved parameters directly into the kernel.')
    final_params = init_point
    training_kernel_start = training_kernel_end = time.time()
    q_kernel.assign_training_parameters(final_params)
    optimized_kernel = q_kernel
else:
    spsa_opt = SPSA(maxiter=remaining_iters, learning_rate=0.03, perturbation=0.01,
                     termination_checker=my_callback.callback)
    loss_func = SVCLoss(C=1.0)

    training_kernel_start = time.time()

    qk_trainer = QuantumKernelTrainer(quantum_kernel=q_kernel, loss=loss_func,
                                       initial_point=init_point, optimizer=spsa_opt)
    qkt_results = qk_trainer.fit(X_train, y_train)
    optimized_kernel = qkt_results.quantum_kernel
    final_params = np.array(qkt_results.optimal_point)

    training_kernel_end = time.time()

    my_callback.save(prefix='TR_QK_')

# ---- SAVE FINAL WEIGHTS ----
weights_path = os.path.join(checkpoint_dir, f'{run_id}_final_weights.json')
with open(weights_path, 'w') as f:
    json.dump({
        'parameters': final_params.tolist(),
        'training_parameter_names': [p.name for p in training_params],
        'max_iter': max_iter,
    }, f, indent=3)
print(f'*** Final trained weights saved to: {weights_path}')

#using optimized kernel in QSVC - always runs, regardless of which branch above produced it
qsvc = QSVC(quantum_kernel=optimized_kernel)

training_svm_start = time.time()
qsvc.fit(X_train, y_train)
training_svm_end = time.time()

predictions = qsvc.predict(X_test)
score = accuracy_score(predictions, y_test)
f1 = f1_score(y_test, predictions)

jobs_final_time = time.time()

# ---- SAVE FINAL MODEL ----
model_path = os.path.join(checkpoint_dir, f'{run_id}_qsvc_model.joblib')
try:
    joblib.dump(qsvc, model_path)
    print(f'*** Trained QSVC model saved to: {model_path}')
except Exception as e:
    print(f'*** WARNING: could not pickle the QSVC model ({e}). '
          f'The trained weights are still safe in {weights_path}.')

# ---- SAVE RESULTS (comparison-table row) ----
results_path = os.path.join(checkpoint_dir, f'{run_id}_results.json')
results = {
    'run_id': run_id,
    'method': 'QK_trainable',
    'max_iter': max_iter,
    'accuracy': score,
    'f1': f1,
    'kernel_training_seconds': training_kernel_end - training_kernel_start,
    'svm_training_seconds': training_svm_end - training_svm_start,
    'total_seconds': jobs_final_time - training_kernel_start,
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=3)

print(f'*******SCORE (accuracy): {score}')
print(f'*******F1 SCORE: {f1}')
print(f'Time kernel training: {training_kernel_end - training_kernel_start} seconds.')
print(f'Time training SVM: {training_svm_end - training_svm_start} seconds.')
print(f'Total jobs time: {jobs_final_time - training_kernel_start} seconds.')
print(f'*** Results saved to: {results_path}')

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
