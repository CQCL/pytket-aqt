def gen_lattice(Lx, Ly):
    """
    Generate a Lx by Ly rectangular lattice.
    The nodes (zones) live on the edge of the lattice.
    The position of each vertex is given by a pair of integers.
    The position of the edge is defined as the average position
    between two vertices.
    For example:
        x - x
        |   |
        x - x
    The vertices have positions:
        [(0, 0), (1, 0), (0, 1), (1, 1)]
    The edges have positions:
        [(0.5, 0), (0, 0.5), (1, 0.5), (0.5, 1)]

    """
    pos_to_idx = {}
    idx = 0
    for y in range(Ly):
        for x in range(Lx-1):
            # print(x+0.5, y)
            pos_to_idx[(x+0.5, y)] = idx
            idx += 1

        for x in range(Lx):
            # print(x, y+0.5)
            pos_to_idx[(x, y+0.5)] = idx
            idx += 1

    return pos_to_idx

def gen_plaquette_from_lattice(Lx, Ly, pos_to_idx):
    '''
    The lattice is described by the dictionary mapping
    the position to indices.
    '''
    plaquette_list = []
    for y in range(Ly-1):
        for x in range(Lx-1):
            # print("gen plaquette:", x, y)
            plaquette = [pos_to_idx[x+0.5, y],  # bottom
                         pos_to_idx[x, y+0.5],  # left
                         pos_to_idx[x+1, y+0.5],  # right
                         pos_to_idx[x+0.5, y+1],  # top
                         ]
            plaquette_list.append(plaquette)

    return plaquette_list

def gen_connection_map(Lx, Ly, plaquette_list, pos_to_idx):
    connection_map = {}
    for plaquette in plaquette_list:
        bottom, left, right, up = plaquette

        bottom_map = connection_map.get(bottom, {})
        # print("bottom -- left:", bottom, left)
        bottom_map[left] = 'LD'
        # print("bottom -- right:", bottom, right)
        bottom_map[right] = 'RD'
        connection_map[bottom] = bottom_map

        left_map = connection_map.get(left, {})
        # print("left -- up:", left, up)
        left_map[up] = 'UL'
        # print("left -- bottom:", left, bottom)
        left_map[bottom] = 'DL'
        connection_map[left] = left_map

        right_map = connection_map.get(right, {})
        # print("right -- up", right, up)
        right_map[up] = 'UR'
        # print("right -- bottom", right, bottom)
        right_map[bottom] = 'DR'
        connection_map[right] = right_map

        up_map = connection_map.get(up, {})
        # print("up -- left:", up, left)
        up_map[left] = 'LU'
        # print("up -- right:", up, right)
        up_map[right] = 'RU'
        connection_map[up] = up_map

    # The straight connection
    for x in range(Lx-2):
        for y in range(Ly):
            left = pos_to_idx[x+0.5, y]
            right = pos_to_idx[x+1.5, y]

            left_map = connection_map.get(left, {})
            left_map[right] = 'RL'
            connection_map[left] = left_map

            right_map = connection_map.get(right, {})
            right_map[left] = 'LR'
            connection_map[right] = right_map

    for x in range(Lx):
        for y in range(Ly-2):
            bottom = pos_to_idx[x, y+0.5]
            up = pos_to_idx[x, y+1.5]

            bottom_map = connection_map.get(bottom, {})
            bottom_map[up] = 'UD'
            connection_map[bottom] = bottom_map

            up_map = connection_map.get(up, {})
            up_map[bottom] = 'DU'
            connection_map[up] = up_map

    return connection_map



Lx = 3
Ly = 3
pos_to_idx = gen_lattice(Lx, Ly)
print("lattice:", pos_to_idx)
plaquette_list = gen_plaquette_from_lattice(Lx, Ly, pos_to_idx)
print("all plaquette:", plaquette_list)

print("===" * 20)
connection_map = gen_connection_map(Lx, Ly, plaquette_list, pos_to_idx)
print("===" * 20)
print(connection_map)

