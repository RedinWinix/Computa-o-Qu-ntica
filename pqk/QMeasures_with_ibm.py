import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.primitives import Estimator as PrimitiveEstimator 
from qiskit.primitives import StatevectorEstimator



class QMeasures:
    
    #measure using Aer
    @staticmethod
    def Aer(qc, observables,**kargs):    
        obs = [SparsePauliOp(label) for label in observables]
    
        estimator = AerEstimator() 
        estimator.options.default_precision = 0  

        obs = [
            observable.apply_layout(qc.layout) for observable in obs
        ]        
        
        # One pub, with one circuit to run against observables.
        job = estimator.run([(qc, obs)])
        
        # This is the result of the entire submission.  We submitted one Pub,
        # so this contains one inner result (and some metadata of its own).
        job_result = job.result()     

        return job_result[0].data.evs
    
    #measure using primitive estimator
    @staticmethod
    def PrimitiveEstimator(qc, observables, **kargs):

        #option for primitive
        my_options=None

        #get the number of shots
        nshots = kargs.get('nshots')
        if nshots is None or type(nshots) is not int:            
            nshots = 100
        
        #get the number of shots
        seed = kargs.get('seed')
        if seed is None or type(seed) is not int:
            my_options={'shots':nshots}
        else:             
            my_options={'shots':nshots, 'seed':seed}                    

        
        estimator = PrimitiveEstimator(options=my_options)         

        l = []         

        for itm in observables:
            job = estimator.run(qc, itm)
            job_result = job.result()
            l.append(job_result.values[0])   

        #return job_result[0].data.evs
        return np.array(l)

    #measure using state vector (evs is the expectation values of the measure)
    def StateVectorEstimator(qc, observables,**kargs):         
        
        estimator = StatevectorEstimator(default_precision=0)     

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def GPUAerStateVectorEstimator(qc, observables, **kargs):

        default_precision=0.0
        backend_options={
            "method":"statevector",
            "device":"GPU"
        }
        estimator=AerEstimator(
            options={
                "backend_options":backend_options,
                "default_precision":default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def CPUAerStateVectorEstimator(qc, observables, **kargs):

        default_precision=0.0
        backend_options={
            "method":"statevector",
            "device":"CPU"
        }
        estimator=AerEstimator(
            options={

                "backend_options":backend_options,
                "default_precision":default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def GPUAerVigoNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeVigoV2
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeVigoV2()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "GPU",
            "noise_model": noise_model
        }

        run_options={};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] =kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def CPUAerVigoNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeVigoV2
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeVigoV2()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "CPU",
            "noise_model": noise_model
        }
        run_options = {};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] = kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def CPUAerBrisbaneNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeBrisbane()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "CPU",
            "noise_model": noise_model
        }
        run_options = {};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] = kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    def GPUAerBrisbaneNoiseStateVectorEstimator(qc, observables, **kargs):
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        from qiskit_aer.noise import NoiseModel
        fake_backend = FakeBrisbane()
        noise_model = NoiseModel.from_backend(fake_backend)

        default_precision = 0.0
        backend_options = {
            "method": "statevector",
            "device": "GPU",
            "noise_model": noise_model
        }

        run_options={};
        if kargs.get("seed_simulator") is not None:
            run_options["seed_simulator"] =kargs["seed_simulator"]

        estimator = AerEstimator(
            options={
                "run_options": run_options,
                "backend_options": backend_options,
                "default_precision": default_precision
            }
        )

        obs = [SparsePauliOp(label) for label in observables]

        pub = (qc, obs)
        job = estimator.run([pub])
        result = job.result()[0]
        return result.data.evs

    @staticmethod
    def IBMQPUEstimator(qc, observables, **kargs):
        """
        Real IBM Quantum hardware estimator (or a FakeBackend for local
        testing - same code path, no credentials needed).
        Requires a saved account (see QiskitRuntimeService.save_account) when
        using a real backend name/least_busy selection.
        kargs:
          - backend: an already-constructed backend object (real IBMBackend
            or a FakeBackend). Takes priority over backend_name/service.
          - backend_name: str, optional. Used with `service` if `backend`
            is not given directly.
          - shots: int, optional (default 1024)
          - service: a pre-built QiskitRuntimeService, optional (avoids
            re-authenticating on every call - build once, pass in).
        NOTE: qiskit-ibm-runtime 0.22.0 (the version pinned in this repo's
        requirements.txt) uses Estimator(backend=...) - NOT Estimator(mode=...),
        which is a newer-version API. Using the wrong one raises a TypeError.
        """
        from qiskit_ibm_runtime import EstimatorV2 as Estimator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.quantum_info import SparsePauliOp

        backend = kargs.get('backend')
        if backend is None:
            service = kargs.get('service')
            backend_name = kargs.get('backend_name')
            if service is None:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
            backend = service.backend(backend_name) if backend_name else service.least_busy(
                operational=True, simulator=False, min_num_qubits=qc.num_qubits
            )

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_qc = pm.run(qc)

        obs = [SparsePauliOp(label) for label in observables]
        isa_obs = [o.apply_layout(isa_qc.layout) for o in obs]

        estimator = Estimator(backend=backend)
        estimator.options.default_shots = kargs.get('shots', 1024)

        job = estimator.run([(isa_qc, isa_obs)])
        result = job.result()
        return result[0].data.evs

    @staticmethod
    def make_ibm_measure_fn(backend, shots=1024):
        """
        Factory that returns a measure_fn closure matching PQK_SVC's fixed
        call signature: measure_fn(qc, observables=obs). PQK_SVC._qfKernel
        only ever calls the measure_fn with those two arguments, so this is
        how you pass a specific backend/shots through without modifying
        PQK_SVC itself.

        Usage:
            ibm_measure = QMeasures.make_ibm_measure_fn(backend, shots=2048)
            pqk = PQK_SVC(circuit=fm, obs=my_obs, measure_fn=ibm_measure, ...)

        NOTE: this submits ONE JOB PER CALL. Since PQK_SVC calls measure_fn
        once per unique data point, this means one job (and one queue wait)
        per unique point. For real hardware, prefer prefetch_pqk_features()
        below, which batches every needed point into a SINGLE job.
        """
        def _measure(qc, observables):
            return QMeasures.IBMQPUEstimator(qc, observables, backend=backend, shots=shots)
        _measure.__name__ = f'IBM({getattr(backend, "name", backend)}, shots={shots})'
        return _measure

    @staticmethod
    def prefetch_pqk_features(pqk_svc, points, backend, shots=1024):
        """
        Batches ALL the observable measurements a PQK_SVC instance will need
        for a given set of data points into a SINGLE real-hardware job
        submission (one queue wait total), instead of one job per unique
        point (which is what happens if PQK_SVC.fit()/.predict() are called
        directly with measure_fn=make_ibm_measure_fn(...) - see that
        docstring).

        How it works: builds the ISA-transpiled feature-map template ONCE,
        then for each unique point (deduplicated using the exact same
        str(row) keying PQK_SVC._qfKernel uses internally) binds that
        point's values into the already-transpiled template - avoiding
        re-transpiling per point - and submits every resulting circuit as
        one batch of PUBs in a single EstimatorV2.run() call. Results are
        written directly into pqk_svc._fm_dict, so subsequent fit()/predict()
        calls hit the cache for every point and never call measure_fn again.

        Args:
            pqk_svc: a PQK_SVC instance (not yet fit).
            points: iterable of 1D arrays/rows (e.g. concatenate X_train and
                X_test so predict() is also fully covered by the same batch).
            backend: a real IBMBackend or a FakeBackend.
            shots: shots per circuit (default 1024).

        Returns:
            int - number of unique points that required a real measurement
            (i.e. were not already cached).
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_ibm_runtime import EstimatorV2 as Estimator

        obs = pqk_svc.obs
        # dedupe using the exact same key format _qfKernel uses (str(row)),
        # so pre-populated cache entries are actually found later
        unique = {}
        for p in points:
            p = np.asarray(p)
            key = str(p)
            if key not in unique and key not in pqk_svc._fm_dict:
                unique[key] = p

        if not unique:
            print('[prefetch_pqk_features] Todos os pontos já estavam em cache - nada a submeter.')
            return 0

        if getattr(pqk_svc, 'fit_clear', False):
            print('[prefetch_pqk_features] AVISO: pqk_svc.fit_clear estava True - isso apagaria '
                  'este cache assim que fit() fosse chamado. Desativando automaticamente '
                  '(fit_clear=False) para preservar o pré-carregamento.')
            pqk_svc.fit_clear = False

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        fm_isa = pm.run(pqk_svc.circuit)

        isa_obs = [SparsePauliOp(label).apply_layout(fm_isa.layout) for label in obs]

        pubs = []
        keys_in_order = []
        for key, p in unique.items():
            bound_qc = fm_isa.assign_parameters(p, inplace=False)
            pubs.append((bound_qc, isa_obs))
            keys_in_order.append(key)

        print(f'[prefetch_pqk_features] Submetendo {len(pubs)} circuito(s) em UM único job '
              f'(uma fila de espera, não {len(pubs)})...')

        estimator = Estimator(backend=backend)
        estimator.options.default_shots = shots
        job = estimator.run(pubs)
        result = job.result()

        for key, res_i in zip(keys_in_order, result):
            pqk_svc._fm_dict[key] = res_i.data.evs

        print(f'[prefetch_pqk_features] OK - {len(pubs)} pontos únicos agora em cache.')
        return len(pubs)
