"""
Run trainable/fixed QK (fidelity-based quantum kernel) on a real IBM Quantum
backend.

IMPORTANTE (histórico): versões anteriores deste script usavam
qiskit_algorithms.state_fidelities.ComputeUncompute + SamplerV1, seguindo o
caminho "padrão" para QK no Qiskit. Isso PAROU DE FUNCIONAR em hardware real:
a infraestrutura da IBM hoje rejeita Primitivas V1 completamente no lado do
servidor:

    RuntimeJobFailureError: "...Error code 1513...'The VNone Primitives are
    not supported. Please use Primitives V2...'"

Como ComputeUncompute é estruturalmente amarrado ao protocolo V1
(sampler.run(circuits, parameter_values)), nenhum ajuste do lado do cliente
resolve isso - é uma rejeição do servidor. Por isso este script agora usa
pqk.QKRealHardwareKernel, uma reimplementação própria do teste
compute-uncompute usando SamplerV2 do início ao fim, com todos os pares
necessários agrupados numa única submissão (mesmo princípio do
prefetch_pqk_features criado para o script de PQK).

HOW TO SWITCH BETWEEN FAKE (local testing, no credentials) AND REAL HARDWARE:
Just change `USE_FAKE_BACKEND` below.
"""

import sys
import os

# Localiza a raiz do repositório a partir da posição do PRÓPRIO ARQUIVO
# (não do diretório de trabalho atual) - assim o script funciona não importa
# de onde você o execute (terminal na raiz, VS Code/debugpy de dentro de uma
# subpasta de teste, etc.). Procura subindo diretórios até achar uma pasta
# que contenha "pqk/" (a marca da raiz do repositório QK).
def _find_repo_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, 'pqk')):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # chegou na raiz do sistema de arquivos, não achou
            raise RuntimeError(
                f"Não foi possível localizar a raiz do repositório (uma pasta "
                f"contendo 'pqk/') subindo a partir de {start_path}. Verifique "
                f"se este script ainda está em algum lugar dentro do repositório QK."
            )
        current = parent

current_wd = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_wd)
# fix_qiskit_ibm_runtime_bug.py vive especificamente em jobs/trained/, não na
# raiz nem em pqk/ - precisa estar no path também, senão o import falha
# quando este script roda de uma subpasta (ex.: jobs/trained/algum_teste/).
sys.path.append(os.path.join(current_wd, 'jobs', 'trained'))
os.chdir(current_wd)  # garante que caminhos relativos (ex.: 'data/...') também funcionem
print(f'*** Raiz do repositório localizada em: {current_wd}')

# corrige três bugs conhecidos em qiskit_ibm_runtime==0.22.0 - ver
# fix_qiskit_ibm_runtime_bug.py para detalhes de cada um. Precisa vir antes
# de importar qiskit_ibm_runtime.
import fix_qiskit_ibm_runtime_bug  # noqa: F401

import logging
logging.getLogger('qiskit_ibm_runtime').setLevel(logging.ERROR)

import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from pqk.Circuits import Circuits
from pqk.QKRealHardwareKernel import QKRealHardwareKernel

seed = 123
np.random.seed(seed)

# ---- BACKEND SELECTION ----
USE_FAKE_BACKEND = False  # True: run locally against a realistic noise model,
                          # no IBM account needed. False: submit to a real
                          # queued IBM device (needs a saved account, costs
                          # real quota/time - start with a tiny dataset).

if USE_FAKE_BACKEND:
    # backend falso PEQUENO (7 qubits) para poder simular localmente de
    # verdade - ver fix_qiskit_ibm_runtime_bug.py (Patch 3) para o motivo de
    # precisarmos dessa correção extra rodando contra backends falsos.
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    backend = FakeLagosV2()
    print(f'*** Using FAKE backend (local, no credentials needed): {backend.name}')
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    # one-time setup, only needed once per machine:
    # QiskitRuntimeService.save_account(
    #     channel="ibm_cloud", token="YOUR_API_KEY",
    #     instance="YOUR_INSTANCE_CRN", set_as_default=True,
    # )
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=6)
    print(f'*** Using REAL backend: {backend.name} (queue may take time)')

# ---- ENCODING SELECTION ----
# same 6 encodings as the paper's Table 1 - see run_PQK_checkpointed.py /
# run_QK_checkpointed.py for the same pattern
full_ent = False
encoding_key = 'xyz'  # 'x'=RotX, 'xyz'=3D/3D-CNOT, 'zz'=ZZFeatureMap, 'IQP', 'Trotter'

encoding_builders = {
    'xyz':    lambda n: Circuits.xyz_encoded(n_wire=n, full_ent=full_ent),
    'x':      lambda n: Circuits.x_encoded(n_wire=n, full_ent=full_ent),
    'zz':     lambda n: Circuits.zzfeaturemap(n_wire=n),
    'IQP':    lambda n: Circuits.IQP_HuangE2(n_wire=n, full_ent=full_ent),
    'Trotter': lambda n: Circuits.Trotter_HuangE3(n_wire=n, full_ent=full_ent),
}

NUM_QBIT = 6
fm = encoding_builders[encoding_key](NUM_QBIT)
print(f'*** ENCODING: {encoding_key} (full_ent={full_ent})')
print(fm.draw())

# ---- DATA ----
# QK ainda é O(N^2) em número de pares (isso não muda) - mas agora TODOS os
# pares de uma chamada fit()/predict() vão numa única submissão de job
# (ver QKRealHardwareKernel), então o custo real é 1-2 filas de espera
# totais, não uma fila por par.
n_points = 21
shots = 1024

data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(n=n_points, random_state=seed)
X = env[['illuminance', 'blinds', 'lamps', 'rh', 'co2', 'temp']].to_numpy()
Y = env['occupancy'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=7)

print(f'Training points: {X_train.shape[0]}, test points: {X_test.shape[0]}')
print(f'Shots per circuit: {shots}')

# ---- BUILD + FIT + EVALUATE ----
qsvc = QKRealHardwareKernel(feature_map=fm, backend=backend, shots=shots, C=1.0)

t0 = time.time()
qsvc.fit(X_train, y_train)
t_train = time.time()

predictions = qsvc.predict(X_test)
score = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
t_final = time.time()

print(f'*******SCORE (accuracy): {score}')
print(f'*******F1 SCORE: {f1}')
print(f'Time training: {t_train - t0} seconds.')
print(f'Time total: {t_final - t0} seconds.')

# ---- SALVA OS RESULTADOS ----
# run_id inclui timestamp porque resultados de hardware real NÃO são
# reprodutíveis de uma execução para outra (ruído físico varia) - diferente
# dos scripts locais, aqui NUNCA queremos sobrescrever uma execução anterior
# silenciosamente, mesmo que a configuração seja idêntica.
import json
import datetime

backend_name = getattr(backend, 'name', str(backend))
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_id = (f'run_QK_realhw_{encoding_key}_ent{full_ent}_{backend_name}'
          f'_train{X_train.shape[0]}_test{X_test.shape[0]}_{timestamp}')

checkpoint_dir = os.path.join('qfm', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
results_path = os.path.join(checkpoint_dir, f'{run_id}_results.json')

results = {
    'run_id': run_id,
    'method': 'QK_real_hardware',
    'encoding': encoding_key,
    'full_ent': full_ent,
    'backend': backend_name,
    'use_fake_backend': USE_FAKE_BACKEND,
    'n_points': n_points,
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
    'shots': shots,
    'accuracy': score,
    'f1': f1,
    'hardware_seconds': t_train - t0,
    'total_seconds': t_final - t0,
    'timestamp': timestamp,
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=3)

print(f'*** Resultados salvos em: {results_path}')
