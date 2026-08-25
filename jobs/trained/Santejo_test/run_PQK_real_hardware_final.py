"""
Run PQK (Projected Quantum Kernel) on a real IBM Quantum backend, using
per-qubit / per-pair Pauli observable measurements via EstimatorV2.

Unlike QK, PQK needs only O(N) circuit evaluations (one per unique data
point, cached by PQK_SVC) instead of O(N^2) - see run_QK_real_hardware.py's
docstring for the contrast. This makes PQK considerably more practical to
actually run on a real, queued device.

HOW TO SWITCH BETWEEN FAKE (local testing, no credentials) AND REAL HARDWARE:
Just change `USE_FAKE_BACKEND` below.
"""

import sys
import os

# Localiza a raiz do repositório a partir da posição do PRÓPRIO ARQUIVO
# (não do diretório de trabalho atual) - assim o script funciona não importa
# de onde você o execute (terminal na raiz, VS Code/debugpy de dentro de uma
# subpasta de teste, etc.).
def _find_repo_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, 'pqk')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(
                f"Não foi possível localizar a raiz do repositório (uma pasta "
                f"contendo 'pqk/') subindo a partir de {start_path}."
            )
        current = parent

current_wd = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_wd)
sys.path.append(os.path.join(current_wd, 'jobs', 'trained'))
os.chdir(current_wd)
print(f'*** Raiz do repositório localizada em: {current_wd}')

# corrige um bug conhecido em qiskit_ibm_runtime==0.22.0 (UnboundLocalError
# em qubit_props_list_from_props) que só aparece ao falar com backends reais -
# ver fix_qiskit_ibm_runtime_bug.py para detalhes. Precisa vir antes de
# importar qiskit_ibm_runtime.
import fix_qiskit_ibm_runtime_bug  # noqa: F401

import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from pqk.Circuits import Circuits
from pqk.PQK_SVC import PQK_SVC
from pqk.aux_funcs import adjacent_qub_obs
# QMeasures_with_ibm.py = the QMeasures.py from this repo, with
# IBMQPUEstimator + make_ibm_measure_fn added (see the file itself for the
# exact diff if you're merging it into your own QMeasures.py)
from pqk.QMeasures_with_ibm import QMeasures

seed = 12345678
np.random.seed(seed)

# ---- BACKEND SELECTION ----
USE_FAKE_BACKEND = False  # True: local, no credentials. False: real queued device.

if USE_FAKE_BACKEND:
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    backend = FakeBrisbane()
    print(f'*** Using FAKE backend (local, no credentials needed): {backend.name}')
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    # one-time setup, only needed once per machine:
    # QiskitRuntimeService.save_account(
    #     channel="ibm_cloud", token="YOUR_API_KEY",
    #     instance="YOUR_INSTANCE_CRN", set_as_default=True,
    # )
    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
   #backend = service.least_busy(operational=True, simulator=False, min_num_qubits=6)
    print(f'*** Using REAL backend: {backend.name} (queue may take time)')

# ---- ENCODING SELECTION (same pattern as run_PQK_checkpointed.py) ----
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
encoding_circuit = encoding_builders[encoding_key](NUM_QBIT)
print(f'*** ENCODING: {encoding_key} (full_ent={full_ent})')

# ---- CARREGAR PESOS JÁ TREINADOS LOCALMENTE (opcional) ----
# Aponte para um arquivo *_final_weights.json produzido por
# run_PQK_checkpointed.py para usar, em hardware real, um kernel cujo
# circuito treinável JÁ foi otimizado localmente (barato, sem fila de
# espera) - em vez do circuito fixo/nunca-treinado usado por padrão.
#
# None = usa a codificação pura, sem camada treinável (comportamento
# original deste script).
WEIGHTS_PATH = 'qfm/checkpoints/run_PQK_xyz_entFalse_18obs_seed1234567_final_weights.json'
# exemplo: WEIGHTS_PATH = 'qfm/checkpoints/run_PQK_xyz_entFalse_18obs_seed123_final_weights.json'

if WEIGHTS_PATH is not None:
    import json as _json
    from pqk.Circuits import Circuits as _Circuits

    with open(WEIGHTS_PATH) as _f:
        _weights_data = _json.load(_f)

    # validação de compatibilidade: os pesos foram treinados para a MESMA
    # codificação que este script está configurado para usar? Sem essa
    # checagem, seria possível parear pesos treinados para uma codificação
    # com uma codificação diferente, produzindo um resultado sem sentido
    # científico, sem nenhum aviso.
    _saved_encoding = _weights_data.get('encoding')
    _saved_full_ent = _weights_data.get('full_ent')
    if _saved_encoding is None:
        print(f'*** AVISO: {WEIGHTS_PATH} não contém metadados de codificação '
              f'(arquivo salvo por uma versão anterior do script de treino). '
              f'Não é possível validar automaticamente se esses pesos foram '
              f'treinados para a codificação "{encoding_key}" - confira manualmente '
              f'pelo nome do arquivo (o run_id inclui a codificação usada).')
    elif _saved_encoding != encoding_key or _saved_full_ent != full_ent:
        raise ValueError(
            f'Os pesos em {WEIGHTS_PATH} foram treinados para a codificação '
            f'"{_saved_encoding}" (full_ent={_saved_full_ent}), mas este script está '
            f'configurado para "{encoding_key}" (full_ent={full_ent}). Usar pesos '
            f'treinados para uma codificação diferente não tem sentido - ajuste '
            f'encoding_key/full_ent para bater com o que foi treinado, ou aponte '
            f'WEIGHTS_PATH para os pesos corretos.'
        )
    else:
        print(f'*** Codificação dos pesos confirmada: "{_saved_encoding}" '
              f'(full_ent={_saved_full_ent}) - compatível com a configuração deste script.')

    # reconstrói a MESMA camada treinável usada em run_PQK_checkpointed.py
    # (decomposição de Euler Z-Y-Z, 18 parâmetros: 3 por qubit)
    _trainable_circuit = _Circuits.zy_decomposition(param_prefix='tr', n_wire=NUM_QBIT, full_ent=False)

    # vincula por NOME (não por posição!) - o Qiskit ordena .parameters
    # alfabeticamente, não pela ordem de inserção no circuito, então usar
    # uma lista posicional para vincular valores seria arriscado. Um dict
    # nome->valor elimina essa ambiguidade por completo.
    _name_to_value = dict(zip(_weights_data['training_parameter_names'], _weights_data['parameters']))
    _trainable_circuit_bound = _trainable_circuit.assign_parameters(_name_to_value)

    encoding_circuit.barrier()
    fm = encoding_circuit.compose(_trainable_circuit_bound)
    print(f'*** Pesos treinados carregados de: {WEIGHTS_PATH}')
    print(f'*** Parâmetros treináveis vinculados (agora fixos): {_name_to_value}')
    print(f'*** Parâmetros restantes no circuito (apenas os da codificação, por ponto): '
          f'{fm.num_parameters}')
else:
    fm = encoding_circuit
    print('*** Nenhum peso treinado carregado - usando codificação pura (sem camada treinável).')

print(fm.draw())

# ---- OBSERVABLE SET (M1 - cheapest: 18 single-qubit Paulis) ----
my_obs = ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX',
          'YIIIII', 'IYIIII', 'IIYIII', 'IIIYII', 'IIIIYI', 'IIIIIY',
          'ZIIIII', 'IZIIII', 'IIZIII', 'IIIZII', 'IIIIZI', 'IIIIIZ']
# for M2 instead, use: adjacent_qub_obs(['X','Y','Z'], n_qub=NUM_QBIT, n_measured_qub=2)

# ---- DATA ----
# PQK only needs O(N) circuit evaluations (cached per unique point), so this
# can scale up more comfortably than the QK script - but each unique point
# still means a real queued job on real hardware, so start modest anyway.
n_points = 30
shots = 1024

data_file_csv = 'data/env.sel3.sk_sc.csv'
env = pd.read_csv(data_file_csv).sample(n=n_points, random_state=seed)
X = env[['illuminance', 'blinds', 'lamps', 'rh', 'co2', 'temp']].to_numpy()
Y = env['occupancy'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=10)

print(f'Training points: {X_train.shape[0]}, test points: {X_test.shape[0]}')
print(f'Observables: {len(my_obs)}, Shots per circuit: {shots}')

# ---- MEDIÇÃO EM LOTE (uma única submissão para todos os pontos) ----
# IMPORTANTE: se você passar measure_fn=QMeasures.make_ibm_measure_fn(...)
# diretamente para o PQK_SVC (como versões anteriores deste script faziam),
# ele submete UM JOB SEPARADO POR PONTO ÚNICO - ou seja, uma fila de espera
# por ponto. Para n_points=15 isso significa até 15 filas de espera
# independentes, o que pode ser extremamente lento em hardware real (cada
# fila pode levar minutos a horas). prefetch_pqk_features() resolve isso:
# agrupa TODOS os pontos necessários (treino + teste) numa ÚNICA submissão,
# então fit()/predict() só usam o cache já preenchido - nenhuma chamada
# individual adicional acontece.
def _measure_fn_never_called(qc, observables):
    raise RuntimeError(
        'measure_fn foi chamado individualmente - isso significa que algum '
        'ponto não foi coberto pelo prefetch_pqk_features(). Verifique se '
        'todos os pontos de treino/teste foram incluídos na chamada.'
    )

pqk = PQK_SVC(circuit=fm, obs=my_obs, measure_fn=_measure_fn_never_called, c_kernel='rbf', C=8.0, gamma=4.0)

t0 = time.time()
n_prefetched = QMeasures.prefetch_pqk_features(
    pqk, np.vstack([X_train, X_test]), backend, shots=shots
)
t_prefetch = time.time()
print(f'*** {n_prefetched} ponto(s) único(s) medido(s) em uma única submissão de job '
      f'({t_prefetch - t0:.2f}s, incluindo fila + execução + transpilação).')

# ---- FIT + EVALUATE ----
t_fit_start = time.time()
pqk.fit(X_train, y_train)
t_train = time.time()

predictions = pqk.predict(X_test)
score = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
t_final = time.time()

print(f'*******SCORE (accuracy): {score}')
print(f'*******F1 SCORE: {f1}')
print(f'Time hardware (prefetch, 1 job): {t_prefetch - t0} seconds.')
print(f'Time training SVM (cache-only, sem hardware): {t_train - t_fit_start} seconds.')
print(f'Time total (fila + execução + treino): {t_final - t0} seconds.')

# ---- SALVA OS RESULTADOS ----
# run_id inclui timestamp porque resultados de hardware real NÃO são
# reprodutíveis de uma execução para outra (ruído físico varia) - diferente
# dos scripts locais, aqui NUNCA queremos sobrescrever uma execução anterior
# silenciosamente, mesmo que a configuração seja idêntica.
import json
import datetime

backend_name = getattr(backend, 'name', str(backend))
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_id = (f'run_PQK_realhw_{encoding_key}_ent{full_ent}_{backend_name}'
          f'_train{X_train.shape[0]}_test{X_test.shape[0]}_{timestamp}')

checkpoint_dir = os.path.join('qfm', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
results_path = os.path.join(checkpoint_dir, f'{run_id}_results.json')

results = {
    'run_id': run_id,
    'method': 'PQK_real_hardware',
    'encoding': encoding_key,
    'full_ent': full_ent,
    'backend': backend_name,
    'use_fake_backend': USE_FAKE_BACKEND,
    'weights_path': WEIGHTS_PATH,
    'used_pretrained_weights': WEIGHTS_PATH is not None,
    'n_points': n_points,
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
    'n_unique_points_measured': n_prefetched,
    'n_observables': len(my_obs),
    'shots': shots,
    'accuracy': score,
    'f1': f1,
    'hardware_seconds': t_prefetch - t0,
    'svm_training_seconds': t_train - t_fit_start,
    'total_seconds': t_final - t0,
    'timestamp': timestamp,
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=3)

print(f'*** Resultados salvos em: {results_path}')
