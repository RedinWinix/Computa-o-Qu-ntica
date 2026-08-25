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
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels.algorithms.quantum_kernel_trainer import QuantumKernelTrainer
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss

from pqk.TrainablePQK_SVC import TrainablePQK_SVC
from pqk.QKCallback_checkpointed import QKCallback
from pqk.QMeasures import QMeasures
from pqk.CKernels import CKernels
from pqk.Circuits import Circuits
from qiskit_algorithms.utils import algorithm_globals


#set the seed
seed = 123
np.random.seed(seed)
algorithm_globals.random_seed = seed


my_obs = ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX',
          'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY',
          'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']

#load dataset with panda
f_rate = 1  # rate of data sampling for testing purpose
data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(frac=f_rate, random_state=seed)

Y = env['occupancy']
X = env[['illuminance', 'blinds', 'lamps', 'rh', 'co2', 'temp']]

#split design matrix (25% of the design matrix used for test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123)

#cast to numpy object
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

#define the maxiter paramenter
max_iter = 5

#check the shape of test and training dataset
print(f'Using dataset in datafile: {data_file_csv}')
print(f'Fraction rate used for this run: {f_rate * 100}%')
print(f'Max number of iteration used in kernel optimization: {max_iter}')
print(f'Shape of dataset: {env.shape}')
print(f'Training shape dataset {X_train.shape}')
print(f'Test shape dataset {X_test.shape}')

#build a feature map
NUM_QBIT = X_train.shape[1]

#define the circuits
encoding_circuit = Circuits.xyz_encoded(n_wire=NUM_QBIT, full_ent=False)
trainable_circuit = Circuits.zy_decomposition(param_prefix='tr', n_wire=NUM_QBIT, full_ent=False)
encoding_circuit.barrier()
fm = encoding_circuit.compose(trainable_circuit)
training_params = trainable_circuit.parameters
n_trainables = len(training_params)

print(f'*** TRAINABLE FEATURE MAP used in QSVC')
print(fm.draw())
print(f'Number of trainable paramenters: {n_trainables}')

# ---- CHECKPOINT / RESUME SETUP ----
run_id = 'run_PQK'  # change this if you want independent checkpoints for different configs
checkpoint_dir = os.path.join('qfm', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f'{run_id}_latest.json')

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
    init_point = np.random.uniform(size=n_trainables)
    completed_iters = 0
    remaining_iters = max_iter
    print('*** No checkpoint found, starting fresh')

print(f'Initial point: {init_point}')

#checkpoint every iteration - this workload is ~seconds-to-minutes per
#iteration, so writing a small JSON file each time is essentially free
my_callback = QKCallback(checkpoint_path=checkpoint_path, checkpoint_every=1)
# preserve iteration counter across a resume so logging/checkpoint numbering stays consistent
my_callback.num_iteration = completed_iters

q_kernel = TrainablePQK_SVC(feature_map=fm, training_parameters=training_params, obs=my_obs,
                             measure_fn=QMeasures.StateVectorEstimator, c_kernel=CKernels.rbf)

print(f'The QMeasure function used: {q_kernel.measure_fn.__name__}')
print(f'The classical kernel used: {q_kernel.c_kernel.__name__}')
print(f'The observables we use: {my_obs}')

if remaining_iters <= 0:
    print('*** Checkpoint already reached max_iter - skipping training, using saved parameters directly.')
    final_params = init_point
    training_kernel_start = training_kernel_end = time.time()
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

    # ---- SAVE FINAL WEIGHTS (clean, standalone file - not buried in history) ----
    weights_path = os.path.join(checkpoint_dir, f'{run_id}_final_weights.json')
    with open(weights_path, 'w') as f:
        json.dump({
            'parameters': final_params.tolist(),
            'training_parameter_names': [p.name for p in training_params],
            'max_iter': max_iter,
        }, f, indent=3)
    print(f'*** Final trained weights saved to: {weights_path}')

#using optimized kernel in QSVC
qsvc = QSVC(quantum_kernel=optimized_kernel)

training_svm_start = time.time()
qsvc.fit(X_train, y_train)
training_svm_end = time.time()

predictions = qsvc.predict(X_test)
score = accuracy_score(predictions, y_test)

jobs_final_time = time.time()

# ---- SAVE FINAL MODEL ----
model_path = os.path.join(checkpoint_dir, f'{run_id}_qsvc_model.joblib')
try:
    joblib.dump(qsvc, model_path)
    print(f'*** Trained QSVC model saved to: {model_path}')
except Exception as e:
    print(f'*** WARNING: could not pickle the QSVC model ({e}). '
            f'The trained weights are still safe in {weights_path} - '
            f'you can rebuild the model from those (see note below).')

#save the full optimization history too (original behaviour)
my_callback.save(prefix='TR_')

print(f'*******SCORE: {score}')
print(f'Time kernel training: {training_kernel_end - training_kernel_start} seconds.')
print(f'Time training SVM: {training_svm_end - training_svm_start} seconds.')
print(f'Total jobs time: {jobs_final_time - training_kernel_start} seconds.')

# once training finishes successfully, the live checkpoint has served its
# purpose - remove it so a future run doesn't accidentally "resume" into
# a finished, already-scored model.
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
