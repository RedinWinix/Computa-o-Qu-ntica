"""
Implementação própria de um kernel de fidelidade (QK) usando SamplerV2 puro.

POR QUE ISSO EXISTE: qiskit_algorithms.state_fidelities.ComputeUncompute (usado
por FidelityQuantumKernel, a abordagem "padrão" para QK) é estruturalmente
amarrado ao protocolo de Primitivas V1 (sampler.run(circuits, parameter_values)).
A infraestrutura REAL da IBM hoje rejeita primitivas V1 completamente no lado
do servidor:

    RuntimeJobFailureError: "...Error code 1513; Failed to execute program:
    'The VNone Primitives are not supported. Please use Primitives V2..."

Isso não é algo que transpilação ou qualquer ajuste do lado do cliente resolve -
é uma rejeição do SERVIDOR. A única solução real é não usar ComputeUncompute
nem SamplerV1 em nenhum ponto do caminho até o hardware real. Este módulo
reimplementa o teste "compute-uncompute" (fidelidade) manualmente, usando
SamplerV2 do início ao fim.

BÔNUS: como construímos isso do zero, já aproveitamos para resolver também o
problema de "um job por par" que existiria com uma reimplementação ingênua -
todos os pares de um mesmo fit()/predict() são agrupados numa única submissão
(mesmo princípio do prefetch_pqk_features criado para o PQK).
"""

import numpy as np
from qiskit import QuantumCircuit
from sklearn.svm import SVC


def _fidelity_circuit(feature_map, x, y, num_qubits, pass_manager):
    """
    Monta o circuito compute-uncompute: U(x) seguido do INVERSO de U(y),
    depois medição de todos os qubits. A probabilidade do resultado
    '00...0' é exatamente |<psi(x)|psi(y)>|^2 - a fidelidade entre os dois
    estados codificados.

    IMPORTANTE: a transpilação acontece por ÚLTIMO, sobre o circuito JÁ
    COMPLETO (com a inversão e a medição incluídas) - transpilar só o
    feature_map antes de invertê-lo não é suficiente, porque a inversão de
    portas nativas (ex.: sx) pode introduzir portas fora do conjunto nativo
    do backend (ex.: sxdg), que só existem DEPOIS da inversão.
    """
    qc = QuantumCircuit(feature_map.num_qubits, num_qubits)
    bound_x = feature_map.assign_parameters(x)
    bound_y_inv = feature_map.assign_parameters(y).inverse()
    qc.compose(bound_x, inplace=True)
    qc.compose(bound_y_inv, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return pass_manager.run(qc)


class QKRealHardwareKernel(SVC):
    """
    QSVC baseado em fidelidade (QK), pronto para hardware real via
    SamplerV2. Uso: igual ao PQK_SVC deste repositório - um SVC do
    scikit-learn com um kernel customizado.

        kernel = QKRealHardwareKernel(feature_map=fm, backend=backend,
                                       shots=1024, C=1.0)
        kernel.fit(X_train, y_train)
        pred = kernel.predict(X_test)

    Todos os pares necessários para uma chamada de fit()/predict() são
    agrupados e enviados numa ÚNICA submissão ao SamplerV2 (uma fila de
    espera, um backend), com resultados cacheados para reuso entre
    fit()/predict() (ex.: pares repetidos, ou o mesmo ponto aparecendo em
    treino e teste).
    """

    def __init__(self, feature_map: QuantumCircuit = None, backend=None,
                 shots: int = 1024, C: float = 1.0, _cache: dict = None):
        super().__init__(C=C, kernel=self._kernel_matrix)
        self.feature_map = feature_map
        self.backend = backend
        self.shots = shots
        self._cache = _cache if _cache is not None else {}
        self._pm = None  # gerenciador de passes, construído sob demanda uma única vez

    def _ensure_transpiled(self):
        if self._pm is None:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            self._pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        return self._pm

    def _kernel_matrix(self, A, B):
        num_qubits = self.feature_map.num_qubits
        pm = self._ensure_transpiled()

        A = np.asarray(A)
        B = np.asarray(B)
        is_symmetric = A.shape == B.shape and np.allclose(A, B)

        # identifica todos os pares (i,j) que realmente precisam de medição
        # (pula diagonal quando simétrico - fidelidade de um ponto consigo
        # mesmo é sempre 1, não precisa de hardware para saber isso)
        pending_pairs = []  # (i, j, key)
        for i in range(A.shape[0]):
            j_range = range(i, B.shape[0]) if is_symmetric else range(B.shape[0])
            for j in j_range:
                if is_symmetric and i == j:
                    continue
                key = (tuple(A[i]), tuple(B[j]))
                if key not in self._cache and (key[1], key[0]) not in self._cache:
                    pending_pairs.append((i, j, key))

        if pending_pairs:
            print(f'[QKRealHardwareKernel] Submetendo {len(pending_pairs)} par(es) '
                  f'em UMA única submissão ao backend {getattr(self.backend, "name", self.backend)}...')
            pubs = [_fidelity_circuit(self.feature_map, A[i], B[j], num_qubits, pm)
                    for i, j, _ in pending_pairs]
            from qiskit_ibm_runtime import SamplerV2
            sampler = SamplerV2(backend=self.backend)
            job = sampler.run(pubs, shots=self.shots)
            result = job.result()

            zero_bitstring = '0' * num_qubits
            for (i, j, key), res_i in zip(pending_pairs, result):
                counts = res_i.data.c.get_counts()
                fidelity = counts.get(zero_bitstring, 0) / self.shots
                self._cache[key] = fidelity
            print(f'[QKRealHardwareKernel] OK - {len(pending_pairs)} par(es) medido(s) e cacheado(s).')

        # monta a matriz de Gram a partir do cache (diagonal = 1 se simétrico)
        K = np.ones((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                if is_symmetric and i == j:
                    K[i, j] = 1.0
                    continue
                key = (tuple(A[i]), tuple(B[j]))
                if key in self._cache:
                    K[i, j] = self._cache[key]
                else:
                    K[i, j] = self._cache[(key[1], key[0])]
        return K
