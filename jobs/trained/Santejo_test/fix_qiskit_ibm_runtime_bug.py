"""
Corrige um bug conhecido em qiskit_ibm_runtime==0.22.0 (a versao fixada no
requirements.txt deste repositorio).

Bug original (qiskit_ibm_runtime/utils/backend_converter.py):

    try:
        frequency = properties.frequency(qubit)
    except Exception:
        t_2 = None   # <- deveria ser 'frequency = None', nao 't_2 = None'

Se properties.frequency(qubit) falhar para qualquer qubit - o que pode
acontecer dependendo dos dados de calibracao retornados pelo backend real da
IBM no momento da consulta - a variavel 'frequency' nunca chega a ser
definida, e o codigo quebra logo em seguida com:

    UnboundLocalError: cannot access local variable 'frequency'

Import este modulo ANTES de chamar QiskitRuntimeService()/service.backend()/
service.least_busy() em qualquer script que fale com hardware real da IBM:

    import fix_qiskit_ibm_runtime_bug  # aplica o patch so de ser importado
    from qiskit_ibm_runtime import QiskitRuntimeService
    ...

Nao precisa fazer mais nada - o patch e aplicado automaticamente ao importar.
"""

import qiskit_ibm_runtime.utils.backend_converter as _bc


def _qubit_props_list_from_props_fixed(properties):
    IBMQubitProperties = _bc.IBMQubitProperties
    qubit_props = []
    for qubit, _ in enumerate(properties.qubits):
        try:
            t_1 = properties.t1(qubit)
        except Exception:
            t_1 = None
        try:
            t_2 = properties.t2(qubit)
        except Exception:
            t_2 = None
        try:
            frequency = properties.frequency(qubit)
        except Exception:
            frequency = None  # <-- correcao do bug original
        try:
            anharmonicity = properties.qubit_property(qubit, 'anharmonicity')[0]
        except Exception:
            anharmonicity = None
        qubit_props.append(
            IBMQubitProperties(t1=t_1, t2=t_2, frequency=frequency, anharmonicity=anharmonicity)
        )
    return qubit_props


_bc.qubit_props_list_from_props = _qubit_props_list_from_props_fixed
print('[fix_qiskit_ibm_runtime_bug] Patch 1 aplicado: qubit_props_list_from_props corrigido.')


# ---------------------------------------------------------------------------
# Segundo bug conhecido: convert_to_target adiciona a instrucao "measure" ao
# Target DUAS VEZES quando o backend real reporta "measure" como uma entrada
# normal dentro de properties.gates (comportamento da API atual da IBM que o
# cliente 0.22.0, escrito antes dessa mudanca, nao esperava). O resultado e:
#
#   AttributeError: Instruction measure is already in the target
#
# Corrigido simplesmente ignorando qualquer gate chamado "measure" no loop
# generico - ela ja e adicionada corretamente logo em seguida, no bloco
# dedicado "Create measurement instructions".
# ---------------------------------------------------------------------------

from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    IGate, SXGate, XGate, CXGate, RZGate, ECRGate, CZGate,
)
from qiskit.circuit import Reset, Gate, Measure
from qiskit.circuit.controlflow import (
    IfElseOp, WhileLoopOp, ForLoopOp, SwitchCaseOp,
)
from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend_compat import CONTROL_FLOW_OP_NAMES
from qiskit.pulse import Delay


def _convert_to_target_fixed(configuration, properties=None, defaults=None):
    name_mapping = {
        "id": IGate(), "sx": SXGate(), "x": XGate(), "cx": CXGate(),
        "rz": RZGate(Parameter("λ")), "reset": Reset(), "ecr": ECRGate(),
        "cz": CZGate(), "if_else": IfElseOp, "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp, "switch_case": SwitchCaseOp,
    }
    custom_gates = {}
    target = None
    if properties is not None:
        qubit_properties = _bc.qubit_props_list_from_props(properties=properties)
        target = Target(num_qubits=configuration.n_qubits, qubit_properties=qubit_properties)
        gates = {}
        for gate in properties.gates:
            name = gate.gate
            if name == "measure":  # <-- correcao: nao processar "measure" aqui
                continue
            if name in name_mapping:
                if name not in gates:
                    gates[name] = {}
            elif name not in custom_gates:
                custom_gate = Gate(name, len(gate.qubits), [])
                custom_gates[name] = custom_gate
                gates[name] = {}
            qubits = tuple(gate.qubits)
            if any(not properties.is_qubit_operational(qubit) for qubit in qubits):
                continue
            if not properties.is_gate_operational(name, gate.qubits):
                continue
            gate_props = {}
            for param in gate.parameters:
                if param.name == "gate_error":
                    gate_props["error"] = param.value
                if param.name == "gate_length":
                    from qiskit.utils.units import apply_prefix
                    gate_props["duration"] = apply_prefix(param.value, param.unit)
            gates[name][qubits] = InstructionProperties(**gate_props)
        for gate, props in gates.items():
            inst = name_mapping.get(gate) if gate in name_mapping else custom_gates[gate]
            target.add_instruction(inst, props)
        measure_props = {}
        for qubit, _ in enumerate(properties.qubits):
            if not properties.is_qubit_operational(qubit):
                continue
            measure_props[(qubit,)] = InstructionProperties(
                duration=properties.readout_length(qubit),
                error=properties.readout_error(qubit),
            )
        target.add_instruction(Measure(), measure_props)
    else:
        target = Target(num_qubits=configuration.n_qubits)
        for gate in configuration.gates:
            name = gate.name
            gate_props = (
                {tuple(x): None for x in gate.coupling_map} if hasattr(gate, "coupling_map") else {None: None}
            )
            gate_len = len(gate.coupling_map[0]) if hasattr(gate, "coupling_map") else 0
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                custom_gate = Gate(name, gate_len, [])
                target.add_instruction(custom_gate, gate_props)
        target.add_instruction(Measure())
    if hasattr(configuration, "dt"):
        target.dt = configuration.dt
    if hasattr(configuration, "timing_constraints"):
        target.granularity = configuration.timing_constraints.get("granularity")
        target.min_length = configuration.timing_constraints.get("min_length")
        target.pulse_alignment = configuration.timing_constraints.get("pulse_alignment")
        target.acquire_alignment = configuration.timing_constraints.get("acquire_alignment")
    supported_instructions = set(getattr(configuration, "supported_instructions", []))
    control_flow_ops = CONTROL_FLOW_OP_NAMES.intersection(supported_instructions)
    for op in control_flow_ops:
        target.add_instruction(name_mapping[op], name=op)
    if defaults is not None:
        faulty_qubits = set()
        if properties is not None:
            faulty_qubits = set(properties.faulty_qubits())
        inst_map = defaults.instruction_schedule_map
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                sched = inst_map.get(inst, qarg)
                if inst in target:
                    try:
                        qarg = tuple(qarg)
                    except TypeError:
                        qarg = (qarg,)
                    if inst == "measure":
                        for qubit in qarg:
                            if qubit in faulty_qubits:
                                continue
                            target[inst][(qubit,)].calibration = sched
                    else:
                        if any(qubit in faulty_qubits for qubit in qarg):
                            continue
                        target[inst][qarg].calibration = sched
        if "delay" not in target:
            target.add_instruction(
                Delay(Parameter("t")),
                {(bit,): None for bit in range(target.num_qubits) if bit not in faulty_qubits},
            )
    return target


_bc.convert_to_target = _convert_to_target_fixed

# ibm_backend.py fez "from .utils.backend_converter import convert_to_target",
# criando uma referencia PROPRIA e independente ao carregar o pacote - so
# corrigir bc.convert_to_target (acima) nao afeta essa referencia ja
# vinculada. E preciso corrigir tambem o nome ja importado dentro de
# ibm_backend, que e o que de fato roda quando um IBMBackend real e
# construido (service.backend(...) / service.least_busy(...)).
import qiskit_ibm_runtime.ibm_backend as _ibm_backend_mod
_ibm_backend_mod.convert_to_target = _convert_to_target_fixed

print('[fix_qiskit_ibm_runtime_bug] Patch 2 aplicado: convert_to_target corrigido (measure duplicado).')


# ---------------------------------------------------------------------------
# BUG 3: BackendSamplerV2._prepare_memory (usado só na execução LOCAL contra
# FakeBackend, não no caminho real de nuvem da IBM) quebra sob NumPy 2.x:
#
#   ValueError: Unable to avoid copy while creating an array as requested.
#
# Causa: np.array(lst, copy=False) mudou de comportamento no NumPy 2.0 -
# antes, copy=False era "copie só se necessário" (comportamento do NumPy
# 1.x); agora significa "NUNCA copie, e falhe se precisar copiar". A
# correção é trocar por np.asarray(lst), que preserva o comportamento
# pretendido ("copie se necessário") nas duas versões do NumPy.
#
# Isso só afeta testes locais com FakeBackend via SamplerV2/EstimatorV2 -
# hardware real usa RuntimeJobV2 (caminho de nuvem), que não passa por essa
# função local.
# ---------------------------------------------------------------------------
try:
    import qiskit_ibm_runtime.qiskit.primitives.backend_sampler_v2 as _bsv2

    def _prepare_memory_fixed(results, num_bytes):
        lst = []
        for res in results:
            for exp in res.results:
                if hasattr(exp.data, 'memory') and exp.data.memory:
                    data = b''.join(int(i, 16).to_bytes(num_bytes, 'big') for i in exp.data.memory)
                    data = _bsv2.np.frombuffer(data, dtype=_bsv2.np.uint8).reshape(-1, num_bytes)
                else:
                    data = _bsv2.np.zeros((exp.shots, num_bytes), dtype=_bsv2.np.uint8)
                lst.append(data)
        ary = _bsv2.np.asarray(lst)  # <-- correção: asarray em vez de array(..., copy=False)
        return _bsv2.np.unpackbits(ary, axis=-1, bitorder='big')

    _bsv2._prepare_memory = _prepare_memory_fixed
    print('[fix_qiskit_ibm_runtime_bug] Patch 3 aplicado: _prepare_memory corrigido '
          '(incompatibilidade com NumPy 2.x, só afeta testes locais com FakeBackend).')
except ImportError:
    # modulo pode nao existir dependendo da versao/instalacao - sem problema,
    # so significa que esse caminho de codigo local nao sera usado mesmo
    pass


# ---------------------------------------------------------------------------
# BUG 4: resultados de hardware REAL (não local/FakeBackend) vêm com objetos
# do tipo "SamplerPubResult" ainda como dict cru, em vez de reconstruídos
# corretamente - causando:
#
#   AttributeError: 'dict' object has no attribute 'data'
#
# Causa: RuntimeDecoder.object_hook (o decodificador de JSON usado para
# reconstruir os resultados que o servidor da IBM devolve) só sabe
# reconstruir objetos marcados como "PubResult" ou "PrimitiveResult" - mas
# SamplerV2.run() (a partir do Qiskit 1.1) devolve resultados marcados como
# "SamplerPubResult", um tipo mais novo que esta versão do decodificador
# (0.22.0) nunca chegou a conhecer. Sem um caso correspondente no
# object_hook, o dicionário bruto passa direto, sem virar objeto.
#
# Isso só aparece em hardware REAL porque testes locais (FakeBackend) nunca
# fazem esse round-trip de JSON - os objetos já nascem prontos em memória.
#
# Correção: ensina o object_hook a também reconstruir "SamplerPubResult",
# usando exatamente o mesmo construtor de "PubResult" (os dois têm a mesma
# assinatura: data, metadata).
# ---------------------------------------------------------------------------
try:
    from qiskit_ibm_runtime.utils.json import RuntimeDecoder
    from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult

    _original_object_hook = RuntimeDecoder.object_hook

    def _object_hook_fixed(self, obj):
        if isinstance(obj, dict) and obj.get('__type__') == 'SamplerPubResult':
            return SamplerPubResult(**obj['__value__'])
        return _original_object_hook(self, obj)

    RuntimeDecoder.object_hook = _object_hook_fixed
    print('[fix_qiskit_ibm_runtime_bug] Patch 4 aplicado: RuntimeDecoder agora reconstrói '
          'SamplerPubResult corretamente (resultados de hardware real).')
except ImportError:
    print('[fix_qiskit_ibm_runtime_bug] AVISO: não foi possível aplicar o Patch 4 '
          '(SamplerPubResult não encontrado nesta instalação de qiskit).')
