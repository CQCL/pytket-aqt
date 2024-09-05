import pytket
import pytket.qasm
import pytket.extensions.qiskit

from pytket import Circuit


def ghz_circuit(N) -> Circuit:
    circuit = Circuit(N)
    circuit.H(0)
    for i in range(circuit.n_qubits - 1):
        circuit.CX(i, i + 1)
    circuit.measure_all()
    return circuit

def brickwork_1d_circuit(N, depth=3) -> Circuit:
    circuit = Circuit(N)
    for d_idx in range(depth):
        # random single qubit rotation
        for i in range(N):
            circuit.Rx(0.123, i)

        # Even 2-qubit gates
        for i in range(N // 2):
            circuit.CX(2*i, 2*i+1)

        # random single qubit rotation
        for i in range(N):
            circuit.Ry(0.456, i)

        # Odd 2-qubit gates
        for i in range((N-1) // 2):
            circuit.CX(2*i+1, 2*i+2)

    circuit.measure_all()
    return circuit

def sequential_1d_circuit(N, depth=3) -> Circuit:
    circuit = Circuit(N)
    for d_idx in range(depth):
        # random single qubit rotation
        for i in range(N):
            circuit.Rx(0.123, i)

        # Even 2-qubit gates
        for i in range(N-1):
            circuit.CX(i, i+1)

    circuit.measure_all()
    return circuit

lattice_map = {9: (3, 3),
               16: (4, 4),
               25: (5, 5),
               36: (6, 6),
               49: (7, 7),
               64: (8, 8),
               81: (9, 9),
               100: (10, 10),
               }

def toric_code_circuit(N=None, Lx=None, Ly=None) -> Circuit:
    """
    A sequential circuit generating the
    Toric code state with cirtcuit depth-Ly.
    We follow a top-down approach.
    """
    if Lx and Ly:
        N = Lx * Ly
    elif N:
        Lx, Ly = lattice_map[N]
    else:
        raise ValueError

    circuit = Circuit(N)
    list_of_plaquette = gen_plaquette(Lx, Ly)
    for plaquette in list_of_plaquette:
        max_idx = max(plaquette) # lower right
        plaquette.remove(max_idx)
        circuit.H(max_idx)
        # print("H on site", max_idx)

        # # This will generate long-range gate
        # for idx in plaquette:
        #     # print(f"CX between {max_idx}-{idx}")
        #     circuit.CX(max_idx, idx)

        if len(plaquette) == 3:
            min_idx = min(plaquette)
            plaquette.remove(min_idx)
            idx1, idx2 = plaquette
            circuit.CX(max_idx, idx1)
            circuit.CX(max_idx, idx2)
            circuit.CX(idx2, min_idx)
        elif len(plaquette) == 1:
            circuit.CX(max_idx, plaquette[0])
        else:
            raise ValueError

    circuit.measure_all()
    return circuit


def gen_plaquette(Lx, Ly):
    """
    Generate the plaquettes that need to be
    entangled
    """
    list_of_plaquette = []
    for y in range(1, Ly):
        for x in range(0, Lx, 2):
            index = y * Lx + x
            # print(f"x={x}, y={y}, idx ={index}")
            if y % 2 == 1: # plaquette to the up right
                if x != Lx - 1:
                    # x    x+1
                    # o -- o   y-1
                    # |    |
                    # x -- o   y
                    plaquette = [index,
                                 y*Lx + (x+1),  # right
                                 (y-1)*Lx + x,  # up
                                 (y-1)*Lx + (x+1),  #up+right
                                 ]
                    list_of_plaquette.append(plaquette)
                else:
                    plaquette = [index, (y-1)*Lx + x,]
                    list_of_plaquette.append(plaquette)
            else:  # plaquette to the up left
                if x != 0:
                    # x-1  x
                    # o -- o   y-1
                    # |    |
                    # o -- x   y
                    plaquette = [index,
                                 y*Lx + (x-1), # left
                                 (y-1)*Lx + x, # top
                                 (y-1)*Lx + (x-1), #top-left
                                 ]
                    list_of_plaquette.append(plaquette)
                else:
                    plaquette = [index, (y-1)*Lx + x,]
                    list_of_plaquette.append(plaquette)

        if y % 2 == 0 and Lx % 2 == 0:
            x = Lx-1
            index = y * Lx + x
            # print(f"adding x={x}, y={y}, idx ={index}")
            plaquette = [index, (y-1)*Lx + x,]
            list_of_plaquette.append(plaquette)

    # print("All plaquette = ", list_of_plaquette)
    return list_of_plaquette

def sequential_2d_circuit(N=None, Lx=None, Ly=None,
                          depth=3,) -> Circuit:
    """
    A 2d sequential circuit generating the
    Toric code state with cirtcuit depth-Ly.
    We follow a top-down approach.
    """
    if Lx and Ly:
        N = Lx * Ly
    elif N:
        Lx, Ly = lattice_map[N]
    else:
        raise ValueError

    circuit = Circuit(N)
    for d_idx in range(depth):
        for y in range(1, Ly):
            for x in range(1, Lx):
                # some random single qubit rotation
                for idx in [y*Lx + x,
                            y*Lx + (x-1),
                            (y-1)*Lx + x,
                            (y-1)*Lx + (x-1),]:
                    circuit.Rx(0.123 * d_idx, idx)

                circuit.CX(y*Lx + x, y*Lx + (x-1))
                circuit.CX(y*Lx + x, (y-1)*Lx + x)
                circuit.CX(y*Lx + (x-1), (y-1)*Lx + (x-1))

    circuit.measure_all()
    return circuit

def brickwork_2d_circuit(N=None, Lx=None, Ly=None,
                         depth=3) -> Circuit:
    """
    A 2d brickwork circuit of a given depth.
    """
    if Lx and Ly:
        N = Lx * Ly
    elif N:
        Lx, Ly = lattice_map[N]
    else:
        raise ValueError

    circuit = Circuit(N)
    for d_idx in range(depth):
        # random single qubit rotation
        for i in range(N):
            circuit.Rx(0.123, i)

        # horizontal Even 2-qubit gates
        for y in range(Ly):
            for i in range(Lx // 2):
                x = 2*i
                idx = y * Lx + x
                circuit.CX(idx, idx+1)

        for i in range(N):
            circuit.Ry(0.456, i)

        # horizontal odd 2-qubit gates
        for y in range(Ly):
            for i in range((Lx-1) // 2):
                x = 2*i + 1
                idx = y * Lx + x
                circuit.CX(idx, idx+1)

        # random single qubit rotation
        for i in range(N):
            circuit.Rx(0.123, i)

        # vertical Even 2-qubit gates
        for x in range(Lx):
            for i in range(Ly // 2):
                y = 2*i
                idx = y * Lx + x
                circuit.CX(idx, idx+Lx)

        for i in range(N):
            circuit.Ry(0.456, i)

        # horizontal odd 2-qubit gates
        for x in range(Lx):
            for i in range((Ly-1) // 2):
                y = 2*i + 1
                idx = y * Lx + x
                circuit.CX(idx, idx+Lx)

    circuit.measure_all()
    return circuit




if __name__ == "__main__":
    # gen_plaquette(4, 4)
    toric_code_circuit(4, 4)

