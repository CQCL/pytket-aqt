from pytket import Circuit, Qubit


def get_2q_gate_pairs_from_circuit(circuit: Circuit) -> list[tuple[int, int]]:
    pair_list: list[tuple[int, int]] = []
    for cmd in circuit.get_commands():
        n_args = len(cmd.args)
        if n_args == 1:
            continue
        elif (
            n_args == 2
            and isinstance(cmd.args[0], Qubit)
            and isinstance(cmd.args[1], Qubit)
        ):
            qubit0 = cmd.args[0].index[0]
            qubit1 = cmd.args[1].index[0]
            pair_list.append((qubit0, qubit1))
    return pair_list


def get_depth_list(
    n_qubits, gate_pairs: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    depth_list: list[list[tuple[int, int]]] = []
    current_depth_per_qubit: list[int] = [0] * n_qubits
    for pair in gate_pairs:
        qubit0 = pair[0]
        qubit1 = pair[1]
        depth = max(current_depth_per_qubit[qubit0], current_depth_per_qubit[qubit1])
        assert len(depth_list) >= depth
        if depth > 0:
            if (qubit0, qubit1) in depth_list[depth - 1] or (
                qubit1,
                qubit0,
            ) in depth_list[depth - 1]:
                depth_list[depth - 1].append((qubit0, qubit1))
                continue
        if len(depth_list) > depth:
            depth_list[depth].append((qubit0, qubit1))
        else:
            depth_list.append([(qubit0, qubit1)])
        current_depth_per_qubit[qubit0] = depth + 1
        current_depth_per_qubit[qubit1] = depth + 1
    return depth_list


def get_initial_depth_list(circuit: Circuit) -> list[list[tuple[int, int]]]:
    n_qubits = circuit.n_qubits
    gate_pairs = get_2q_gate_pairs_from_circuit(circuit)
    return get_depth_list(n_qubits, gate_pairs)
