import qiskit
from qiskit.quantum_info import state_fidelity
import copy


class TranspileEvaluate:
    def __init__(self, logical_circuit, qhardware):
        self.backend = qiskit.Aer.get_backend('qasm_simulator')

        self.num_logical_qubits = logical_circuit.num_qubits
        self.roundtrip_target = '0' * self.num_logical_qubits

        self.logical_circuit = logical_circuit.copy()
        self.logical_circuit.snapshot_density_matrix(label='final')
        # self.logical_circuit.measure_all()

        self.qhardware = copy.deepcopy(qhardware)

        self.base_transpile_kwargs = {
            'circuits': self.logical_circuit,
            'basis_gates': qhardware.basis_gates,
            'coupling_map': qhardware.coupling_map,
        }
        self.base_execute_kwargs = {
            'backend': self.backend,
            'noise_model': qhardware.noisemodel,
            'optimization_level': 0,
        }

        self.default_extra_transpile_options = dict()
        self.default_extra_execute_options = dict()

        self.forward_fidelity_init_done = False

    def forward_fidelity_init(self):
        job = qiskit.execute(self.logical_circuit, self.backend, shots=1)
        self.noiseless_density_matrix = job.result().data()['snapshots'] \
                                        ['density_matrix']['final'][0]['value']
        self.forward_fidelity_init_done = True

    def set_default_extra_transpile_options(self, **kwarg):
        self.default_extra_transpile_options = copy.deepcopy(kwarg)

    def set_default_extra_execute_options(self, **kwarg):
        self.default_extra_execute_options = copy.deepcopy(kwarg)

    def transpile_to_physical(self, initial_layout, **kwargs):
        transpile_kwargs = {
            **self.base_transpile_kwargs,
            **self.default_extra_transpile_options, 'initial_layout':
            initial_layout,
            **kwargs
        }

        self.physical_circuit = qiskit.compiler.transpile(**transpile_kwargs)

        circuit_dag = qiskit.converters.circuit_to_dag(self.physical_circuit)

        ## Remove measurements and snapshots######
        for node in circuit_dag.named_nodes('measure'):
            circuit_dag.remove_op_node(node)

        for node in circuit_dag.named_nodes('snapshot'):
            circuit_dag.remove_op_node(node)
        ##########################################

        self.roundtrip_circuit = qiskit.converters.dag_to_circuit(circuit_dag)
        self.roundtrip_circuit += self.roundtrip_circuit.inverse()

        # Measuring qubits after the roundtrip ###
        cl_register = qiskit.ClassicalRegister(self.num_logical_qubits)
        self.roundtrip_circuit.add_register(cl_register)
        self.roundtrip_circuit.measure(initial_layout, cl_register)

    def get_forward_fidelity(self, shots=1024, seed_simulator=None, **kwargs):
        if not self.forward_fidelity_init_done:
            self.forward_fidelity_init()

        execute_kwargs = {
            **self.base_execute_kwargs,
            **self.default_extra_execute_options, 'experiments':
            self.physical_circuit,
            'shots': shots,
            'seed_simulator': seed_simulator,
            **kwargs
        }

        job = qiskit.execute(**execute_kwargs)
        density_matrix = job.result().data()['snapshots'] \
                                        ['density_matrix']['final'][0]['value']

        return state_fidelity(self.noiseless_density_matrix, density_matrix)

    def get_roundtrip_fidelity(self,
                               shots=1024,
                               seed_simulator=None,
                               **kwargs):
        execute_kwargs = {
            **self.base_execute_kwargs,
            **self.default_extra_execute_options, 'experiments':
            self.roundtrip_circuit,
            'shots': shots,
            'seed_simulator': seed_simulator,
            **kwargs
        }

        job = qiskit.execute(**execute_kwargs)
        all_zeros_count = job.result().get_counts()[self.roundtrip_target]

        return all_zeros_count / shots
