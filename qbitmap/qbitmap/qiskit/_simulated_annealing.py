import random as _random
import math as _math


def anneal(num_phys_qubits,
           num_logical_qubits,
           score_func,
           temperatures,
           steps=100,
           track_history=False,
           anneal_seed=0):

    _random.seed(anneal_seed)

    # Initialize mapping ###################################
    qubit_map = _random.sample(population=range(num_phys_qubits),
                               k=num_logical_qubits)

    unused_phys = [i for i in range(num_phys_qubits) if i not in qubit_map]
    ########################################################

    # Swap used qubits vs replace used with unused qubit ###
    swap_prob = (num_logical_qubits - 1)
    replace_prob = 2 * (num_phys_qubits - num_logical_qubits)
    swap_prob = swap_prob / (swap_prob + replace_prob)
    replace_prob = 1 - swap_prob
    ########################################################

    # delta_T = (end_T - start_T)/steps
    successful_transitions = 0

    T = temperatures[0]
    cur_score = score_func(qubit_map)
    state_log = set([tuple(qubit_map)])

    if track_history:
        history_dict = {
            'assignment_history': [tuple(qubit_map)],
            'score_history': [cur_score],
            'temp_history': [T],
            'unique_states_counter': [len(state_log)]
        }

    for k in range(steps):
        choose_transition_type = _random.random()
        if choose_transition_type < swap_prob:
            i, j = _random.sample(population=range(num_logical_qubits), k=2)

            List_1 = qubit_map
            List_2 = qubit_map
        else:
            i = _random.randint(0, num_logical_qubits - 1)
            j = _random.randint(0, num_phys_qubits - num_logical_qubits - 1)

            List_1 = qubit_map
            List_2 = unused_phys

        List_1[i], List_2[j] = List_2[j], List_1[i]
        next_score = score_func(qubit_map)
        if _random.random() < _math.exp((next_score - cur_score) / T):
            cur_score = next_score
            successful_transitions += 1
        else:
            List_1[i], List_2[j] = List_2[j], List_1[i]

        T = temperatures[k]
        state_log.add(tuple(qubit_map))

        if track_history:
            history_dict['assignment_history'].append(tuple(qubit_map))
            history_dict['score_history'].append(cur_score)
            history_dict['temp_history'].append(T)
            history_dict['unique_states_counter'].append(len(state_log))

        print(f"{k}/{steps} steps completed", end='\r')

    print()
    to_return = {
        'qubit_map': tuple(qubit_map),
        'score': cur_score,
        'successful_transitions': successful_transitions
    }

    if track_history:
        to_return.update(history_dict)

    return to_return
