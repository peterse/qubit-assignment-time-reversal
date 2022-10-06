import qiskit.providers.aer.noise as qiskitnoise
import numbers
import copy


class QProcessor2DGrid:
    def __init__(
        self,
        nrows=5,
        ncols=5,
        coupling_map='nearest_neighbor',
        basis_1qubit_gates=["u1", "u2", "u3"],
        basis_2qubit_gates=["cx"],
    ):

        self.shape = (nrows, ncols)
        self.num_qubits = nrows * ncols

        self.basis_1qubit_gates = copy.copy(basis_1qubit_gates)
        self.basis_2qubit_gates = copy.copy(basis_2qubit_gates)
        self.basis_gates = basis_1qubit_gates + basis_2qubit_gates + ['id']

        # Coupling map #####################################
        if coupling_map == 'nearest_neighbor':
            self.coupling_map = []
            for x in range(nrows):
                for y in range(ncols):
                    i = self.idx_2d_to_1d(x, y)
                    if x < nrows - 1:
                        j = self.idx_2d_to_1d(x + 1, y)
                        self.coupling_map.append([i, j])
                        self.coupling_map.append([j, i])
                    if y < ncols - 1:
                        j = self.idx_2d_to_1d(x, y + 1)
                        self.coupling_map.append([i, j])
                        self.coupling_map.append([j, i])
        else:
            self.coupling_map = copy.deepcopy(coupling_map)

    def init_noise_from_constant_err(self,
                                     one_qubit_err=0.001,
                                     two_qubit_err=0.01):
        """Initialize a noise model from constant one- and two-qubit errors."""

        self.noisemodel = qiskitnoise.NoiseModel(basis_gates=self.basis_gates)

        if isinstance(one_qubit_err, numbers.Number):
            err = qiskitnoise.errors.depolarizing_error(one_qubit_err, 1)

            self.noisemodel.add_all_qubit_quantum_error(
                err, self.basis_1qubit_gates)
        else:
            for (x, y), err_param in one_qubit_err.items():
                i = self.idx_2d_to_1d(x, y)
                err = qiskitnoise.errors.depolarizing_error(err_param, 1)

                self.noisemodel.add_quantum_error(err, self.basis_1qubit_gates,
                                                  [i])

        if isinstance(two_qubit_err, numbers.Number):
            err = qiskitnoise.errors.depolarizing_error(two_qubit_err, 2)

            self.noisemodel.add_all_qubit_quantum_error(
                err, self.basis_2qubit_gates)
        else:
            for ((x, y), (u, v)), err_param in two_qubit_err.items():
                i = self.idx_2d_to_1d(x, y)
                j = self.idx_2d_to_1d(u, v)
                err = qiskitnoise.errors.depolarizing_error(err_param, 2)

                self.noisemodel.add_quantum_error(err, self.basis_2qubit_gates,
                                                  [i, j])

    def init_noise_from_graph(self, noise_graph):
        """Initialize a noise model from a pre-defined graph.

        Args:
            noise_graph (networkx.Graph): A Graph containing edges and nodes
                corresponding to connectivity and qubits respectively. Each
                node and edge should contain an dictionary of noise magnitudes.
                If the noise magnitude is not found, a message will print and
                default "0" noise will be used.
        """

        self.noisemodel = qiskitnoise.NoiseModel(basis_gates=self.basis_gates)

        # FIXME: I'm assuming that networkx flattens the graph node positions
        # in the same convention as used above...
        for k, node in enumerate(noise_graph.nodes()):
            # Get single qubit noise and add it to the specific qubit (node)
            one_qubit_err = noise_graph.nodes[node].get("weight")
            if one_qubit_err is None:
                print(
                    f"Found no single qubit depolarizing error at qubit {node}"
                )
                one_qubit_err = 0
            err = qiskitnoise.errors.depolarizing_error(one_qubit_err, 1)

            i = self.idx_2d_to_1d(*node)

            self.noisemodel.add_quantum_error(
                err,
                self.basis_1qubit_gates,
                qubits=[i],
            )
        for edge in noise_graph.edges():
            # Get two-qubit gate noise and apply it to the right pairs (edges)
            two_qubit_err = noise_graph.edges[edge].get("weight", None)
            if two_qubit_err is None:
                print(f"Found no two qubit depolarizing error at edge {edge}")
                two_qubit_err = 0

            err = qiskitnoise.errors.depolarizing_error(two_qubit_err, 2)
            i = self.idx_2d_to_1d(*edge[0])
            j = self.idx_2d_to_1d(*edge[1])
            self.noisemodel.add_quantum_error(err, self.basis_2qubit_gates,
                                              (i, j))

    def idx_1d_to_2d(self, idx):
        idx_x = idx // self.shape[1]
        idx_y = idx % self.shape[1]

        assert 0 <= idx_x < self.shape[0]
        assert 0 <= idx_y < self.shape[1]

        return (idx_x, idx_y)

    def idx_2d_to_1d(self, idx_x, idx_y):
        assert 0 <= idx_x < self.shape[0]
        assert 0 <= idx_y < self.shape[1]

        return idx_x * self.shape[1] + idx_y
