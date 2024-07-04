import time

from pytket import Circuit

from pytket.extensions.aqt.backends.aqt_multi_zone import AQTMultiZoneBackend

from pytket.extensions.aqt.multi_zone_architecture.named_architectures import (
    racetrack,
)


def test_large_circuit() -> None:
    total_t = time.time()
    start = time.time()
    Lx = 7
    Ly = 7

    N = Lx * Ly
    Ntrot = 30
    dt = 0.2
    hx = 2
    J = 1

    def c(x, y):
        if (x - y) % N == Lx or (y - x) % N == Lx:
            return 1
        if x // Lx == y // Lx and x == y - (y % Lx) + (((y % Lx) + 1) % Lx):
            return 1
        if x // Lx == y // Lx and y == x - (x % Lx) + (((x % Lx) + 1) % Lx):
            return 1
        return 0

    coupling_list = []
    for i in range(0, N):
        for j in range(i + 1, N):
            if c(i, j) == 1:
                coupling_list.append([i, j])

    U = Circuit(N)

    for t in range(Ntrot):
        n_trotter_steps = t

        if n_trotter_steps == 0:
            for j in range(N):
                U.Rx(-2 * dt * hx / 2, j)

        for coupling in coupling_list:
            U.ZZPhase(-2 * dt * J, coupling[0], coupling[1])
        for j in range(N):
            U.Rx(-2 * dt * hx, j)

        if n_trotter_steps == Ntrot - 1:
            for j in range(N):
                U.Rx(2 * dt * hx / 2, j)

    end = time.time()
    print("prep time: ", end - start)

    print("depth uncompiled", U.depth_2q())
    start = time.time()
    backend = AQTMultiZoneBackend(architecture=racetrack, access_token="invalid")
    end = time.time()
    print("backend construct time: ", end - start)
    mz_circuit = backend.compile_circuit_with_routing(U, optimisation_level=0)
    n_shuttles = mz_circuit.get_n_shuttles()
    n_pswaps = mz_circuit.get_n_pswaps()
    print("shuttles: ", n_shuttles)
    print("pswaps: ", n_pswaps)
    end_total_t = time.time()
    print("total time: ", end_total_t - total_t)
