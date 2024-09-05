import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle

import os
import sys
import pytket
import pytket.qasm
import pytket.extensions.qiskit

from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend
from pytket.extensions.aqt.multi_zone_architecture.compilation_settings import CompilationSettings
from pytket.extensions.aqt.multi_zone_architecture.circuit_routing.settings import RoutingSettings, RoutingAlg

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    X_grid,
    small_grid,
    mid_grid,
    large_grid,
    racetrack,
    six_zones_in_a_line_102,
)

name_to_arch = {'X_grid': X_grid, 'small_grid': small_grid, 'mid_grid': mid_grid,
                'large_grid': large_grid, 'racetrack': racetrack, 'six_zones_in_a_line_102': six_zones_in_a_line_102
                }

plt.style.use('physrev.mplstyle')


import gen_circuit as gc
def named_circuit(circuit_name, num_qubits) -> Circuit:
    if circuit_name == "ghz":
        return gc.ghz_circuit(num_qubits)
    if circuit_name == "toric_code":
        return gc.toric_code_circuit(num_qubits)
    if circuit_name == "brickwork_1d":
        return gc.brickwork_1d_circuit(num_qubits)
    if circuit_name == "sequential_1d":
        return gc.sequential_1d_circuit(num_qubits)
    if circuit_name == "brickwork_2d":
        return gc.brickwork_2d_circuit(num_qubits)
    if circuit_name == "sequential_2d":
        return gc.sequential_2d_circuit(num_qubits)
    if circuit_name == "MoS2_JW":
        return pickle.load(open(f'circuits/JW_2trotter_{num_qubits}.pkl','rb'))
    if circuit_name == "MoS2_CE":
        return pickle.load(open(f'circuits/CE_2trotter_{num_qubits}.pkl','rb'))
    else:
        from mqt.bench import get_benchmark
        qcirc = get_benchmark(benchmark_name=circuit_name, level='alg', circuit_size=num_qubits)
        pytket_circ = pytket.extensions.qiskit.qiskit_to_tk(qcirc, preserve_param_uuid=False)
        C = pytket_circ
        C.flatten_registers()
        assert C.is_simple
        assert pytket.passes.RemoveBarriers().apply(C)
        return C

def test_circuit(circuit_name, num_qubits, graph_init, routing_alg,
                 device_name,
                 optimisation_level=2,
                 ) -> None:
    total_t = time.time()
    start = time.time()
    print("running %s circuit of size %d" %(circuit_name, num_qubits))
    C = named_circuit(circuit_name, num_qubits)
    # C = pytket.qasm.circuit_from_qasm(qasm_filename)
    print(C.get_commands())

    end = time.time()
    print("load time: ", end - start)

    print("depth uncompiled", C.depth_2q())
    start = time.time()
    # backend = AQTMultiZoneBackend(architecture=six_zones_in_a_line_102, #racetrack,
    # backend = AQTMultiZoneBackend(architecture=racetrack,
    arch = name_to_arch[device_name]
    backend = AQTMultiZoneBackend(architecture=arch,
                                  access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)

    if graph_init:
        initial_placement=None
    else:
        initial_placement = {}
        n_qubits = C.n_qubits
        n_zones = arch.n_zones
        n_per_trap = 2
        n_alloc = 0
        # # Placement of L6 for n-qubits
        # initial_placement = {}
        # n_qubits =  C.n_qubits
        # n_zones = 6
        # n_per_trap = 15
        # n_alloc = 0
        for i in range(n_zones):
            initial_placement[i] = list(range(n_alloc, min(n_alloc+n_per_trap, n_qubits)))
            n_alloc = min(n_alloc+n_per_trap, n_qubits)

        print(initial_placement)

    setting = CompilationSettings()
    assert strategy == routing_alg
    if routing_alg == "greedy":
        alg = RoutingAlg.greedy
    else:
        alg = RoutingAlg.graph_partition

    setting.routing = RoutingSettings(alg)
    # setting.pytket_optimisation_level = optimisation_level
    # setting.initial_placement =

    compile_start_time = time.time()
    mz_circuit = backend.compile_circuit_with_routing(C,
                                                      compilation_settings=setting
                                                      )
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    n_2qb_gates = mz_circuit.pytket_circuit.n_2qb_gates()
    n_1qb_gates = mz_circuit.pytket_circuit.n_1qb_gates()
    print("n_2qb_gates: ", n_2qb_gates)
    print("n_1qb_gates: ", n_1qb_gates)
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    total_time = end_total_t - total_t
    compile_time = end_total_t - compile_start_time
    print("total time: ", total_time)

    # result_dict = {"n_2qb_gates": n_2qb_gates,
    #                "n_1qb_gates": n_1qb_gates,
    #                "n_shuttles": n_shuttles,
    #                "n_pswaps": n_pswaps,
    #                "compile_time": compile_time,
    #                }
    list_of_variables_to_save = ["n_2qb_gates", "n_1qb_gates", "n_shuttles", "n_pswaps", "compile_time", "total_time"]
    result_dict = dict([(name, eval(name)) for name in list_of_variables_to_save])
    return result_dict

if __name__ == "__main__":
    # device_name = 'large_grid'
    device_name = 'mid_grid'
    # device_name = 'racetrack'

    # ValueError: Selected benchmark is not supported. Valid benchmarks are ['ae', 'dj', 'grover-noancilla', 'grover-v-chain', 'ghz', 'graphstate', 'portfolioqaoa', 'portfoliovqe', 'qaoa', 'qft', 'qftentangled', 'qnn', 'qpeexact', 'qpeinexact', 'qwalk-noancilla', 'qwalk-v-chain', 'random', 'realamprandom', 'su2random', 'twolocalrandom', 'vqe', 'wstate', 'shor', 'pricingcall', 'pricingput', 'groundstate', 'routing', 'tsp'].

    # circuit_name = 'grover-noancilla'
    circuit_name = 'ghz'  # 1d-obc
    circuit_name = 'qnn'  # 1d-obc
    circuit_name = 'graphstate'  # 1d-pbc
    circuit_name = 'qft'  # all-to-all

    # circuit_name_list = ['ghz', 'toric_code', 'qft', 'random'] #, 'qnn', 'graphstate', 'qft']
    list_of_CNL = [['ghz', 'toric_code', 'qft', 'random'],
                   ['ghz', 'brickwork_1d', 'sequential_1d'],  # all 1d circuit
                   ['toric_code', 'brickwork_2d', 'sequential_2d', 'MoS2_JW', 'MoS2_CE'],  # all 2d circuit
                   ]

    for circuit_name_list in list_of_CNL:

        num_example = len(circuit_name_list)
        fig, axes = plt.subplots(3, num_example,
                                 sharex=True,
                                 # sharey='row'
                                 figsize=(7, 4.6),
                                 )

        # The key should be
        # result / [device_name + init + strategy] / [circuit + n_q]

        for c_idx, circuit_name in enumerate(circuit_name_list):
            if c_idx > 0:
                pass
                # continue


            color_list = ['r', 'b', 'k']
            # strategy_list = ['greedy', 'score', 'partitioning']
            graph_init_list = [False, True]
            strategy_list = ['greedy', 'partitioning']
            for graph_init in graph_init_list:
                for idx, strategy in enumerate(strategy_list):
                    print("running graph init=", graph_init, "strategy=", strategy)
                    dir_name = f"result/{device_name}_{graph_init}_{strategy}/"
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)

                    n_shuttles_list = []
                    n_pswaps_list = []
                    compile_time_list = []
                    if device_name == 'racetrack':
                        n_q_list = [10, 15, 20, 30, 40, 50] # , 60, 80, 100] # , 150, 200]
                        if circuit_name in ["toric_code", "brickwork_2d", "sequential_2d", "MoS2_JW"]:
                            n_q_list = [9, 16, 25, 36, 49]
                        elif circuit_name in ["MoS2_CE"]:
                            n_q_list = [11, 20, 33, 48] #, 67, 88, 113]


                    elif device_name in ['mid_grid', 'large_grid']:
                        n_q_list = [10, 20, 30, 40, 50, 60, 80, 100]
                        if circuit_name == 'brickwork_1d':
                            n_q_list.remove(80)
                            n_q_list.remove(100)

                        if circuit_name in ["toric_code", "brickwork_2d", "sequential_2d", "MoS2_JW"]:
                            n_q_list = [9, 16, 25, 36, 49, 64, 81, 100]
                        elif circuit_name in ["MoS2_CE"]:
                            n_q_list = [11, 20, 33, 48, 67, 88, 113]

                    else:
                        raise NotImplementedError

                    # n_q_list = [5, 10, 15]
                    for num_qubits in n_q_list:
                        file_name = f"{circuit_name}_{num_qubits}.pkl"
                        file_path = dir_name + file_name
                        if os.path.isfile(file_path):
                            result_dict = pickle.load(open(file_path, 'rb'))
                        else:
                            result_dict = test_circuit(circuit_name, num_qubits,
                                                       graph_init, strategy,
                                                       device_name=device_name)
                            pickle.dump(result_dict, open(file_path, 'wb'))

                        # n_2qb_gates, n_1qb_gates, n_shuttles, n_pswaps, compile_time = result
                        n_shuttles_list.append(result_dict["n_shuttles"])
                        n_pswaps_list.append(result_dict["n_pswaps"])
                        compile_time_list.append(result_dict["compile_time"])



                    print(n_shuttles_list)
                    if graph_init:
                        # graph init
                        plot_setting = {'color': color_list[idx],
                                        'label': 'kahypar_init-' + strategy}
                        axes[0][c_idx].plot(n_q_list, n_shuttles_list, 'x',
                                            **plot_setting)
                        axes[1][c_idx].plot(n_q_list, n_pswaps_list, 'x',
                                            **plot_setting)
                        axes[2][c_idx].plot(n_q_list, compile_time_list, 'x',
                                            **plot_setting)
                        if strategy == 'partitioning':
                            def ax_to_power_b(x, a, b):
                                return a * (x ** b)

                            from scipy.optimize import curve_fit
                            start_fit_idx = 1
                            # fitting n_shuttles
                            parameters, covariance = curve_fit(ax_to_power_b, n_q_list[start_fit_idx:],
                                                               n_shuttles_list[start_fit_idx:])
                            a, b = parameters
                            axes[0][c_idx].plot(n_q_list, ax_to_power_b(n_q_list, a, b), '--', color=color_list[idx])
                            axes[0][c_idx].text(0.05, 0.85, "$\\beta_k=%.2f$" % b, transform=axes[0][c_idx].transAxes,
                                                color=color_list[idx])
                            # fitting n_pswaps
                            parameters, covariance = curve_fit(ax_to_power_b, n_q_list[start_fit_idx:],
                                                               n_pswaps_list[start_fit_idx:])
                            a, b = parameters
                            axes[1][c_idx].plot(n_q_list, ax_to_power_b(n_q_list, a, b), '--', color=color_list[idx])
                            axes[1][c_idx].text(0.05, 0.85, "$\\beta_k=%.2f$" % b, transform=axes[1][c_idx].transAxes,
                                                color=color_list[idx])
                    else:
                        plot_setting = {'fillstyle': 'none',
                                        'color': color_list[idx],
                                        'label': 'uniform_init-' + strategy}
                        axes[0][c_idx].plot(n_q_list, n_shuttles_list, 's',
                                            **plot_setting)
                        axes[1][c_idx].plot(n_q_list, n_pswaps_list, 's',
                                            **plot_setting)
                        axes[2][c_idx].plot(n_q_list, compile_time_list, 's',
                                            **plot_setting)
                        if strategy == 'greedy':
                            def ax_to_power_b(x, a, b):
                                return a * (x ** b)

                            from scipy.optimize import curve_fit
                            start_fit_idx = 1
                            # fitting n_shuttles
                            parameters, covariance = curve_fit(ax_to_power_b, n_q_list[start_fit_idx:],
                                                               n_shuttles_list[start_fit_idx:])
                            a, b = parameters
                            axes[0][c_idx].plot(n_q_list, ax_to_power_b(n_q_list, a, b), '--', color=color_list[idx])
                            axes[0][c_idx].text(0.05, 0.7, "$\\beta_g=%.2f$" % b, transform=axes[0][c_idx].transAxes,
                                                color=color_list[idx])
                            # fitting n_pswaps
                            parameters, covariance = curve_fit(ax_to_power_b, n_q_list[start_fit_idx:],
                                                               n_pswaps_list[start_fit_idx:])
                            a, b = parameters
                            axes[1][c_idx].plot(n_q_list, ax_to_power_b(n_q_list, a, b), '--', color=color_list[idx])
                            axes[1][c_idx].text(0.05, 0.7, "$\\beta_g=%.2f$" % b, transform=axes[1][c_idx].transAxes,
                                                color=color_list[idx])



            for a_idx in range(3):
                axes[a_idx][c_idx].set_xscale('log')
                axes[a_idx][c_idx].set_yscale('log')

            axes[2][c_idx].set_xlabel('num qubits')
            axes[2][c_idx].xaxis.set_major_formatter(ScalarFormatter())
            # plt.ticklabel_format(style='plain', axis='x')
            if c_idx == 0:
                axes[0][c_idx].set_ylabel('num shuttles')
                axes[1][c_idx].set_ylabel('num pswaps')
                axes[2][c_idx].set_ylabel('compile time')

            axes[0][c_idx].set_title(circuit_name)
            # if circuit_name in ['qft']:
            # plt.yscale('log')
            # plt.xscale('log')

        axes[-1][-1].legend()
        plt.savefig(device_name + "-" + '-'.join(circuit_name_list) + '.pdf')
        plt.savefig(device_name + "-" + '-'.join(circuit_name_list) + '.png', dpi=300)
        # plt.show()


