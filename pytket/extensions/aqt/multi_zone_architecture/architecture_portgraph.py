# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from networkx import (  # type: ignore
    Graph,
    single_source_dijkstra,
)

from .architecture import MultiZoneArchitectureSpec


def zone_port_to_port_id(zone: int, port: int) -> int:
    return 2 * zone + port


def port_id_to_zone_port(port_id: int) -> tuple[int, int]:
    return port_id // 2, port_id % 2


def port_path_to_zone_path(port_path: list[int]) -> list[int]:
    """Translate a path through zone ports to a path through zones

    A port path starts with the port of a zone (s_p), moves through the two ports
    of n >= 0 intermediary zones (i_p0, i_p1) and stops at a target port of a zone (t_p)

    s_p -> 0_p0 -> 0_p1 -> ... -> n_p0 -> n_p1 -> t_p
    """
    # add source zone
    result = [port_id_to_zone_port(port_path[0])[0]]
    # only use the first port of each intermediary zone
    result.extend([port_id_to_zone_port(port)[0] for port in port_path[1::2]])
    # add target zone
    result.append(port_id_to_zone_port(port_path[-1])[0])
    return result


class MultiZonePortGraph:
    def __init__(self, spec: MultiZoneArchitectureSpec):
        self.port_graph = Graph()

        for zone_id, zone in enumerate(spec.zones):
            zone_port0_id = zone_port_to_port_id(zone_id, 0)
            zone_port1_id = zone_port_to_port_id(zone_id, 1)
            self.port_graph.add_node(zone_port0_id)
            self.port_graph.add_node(zone_port1_id)
            # Add edges between ports of a zone. Weight 0 for now but later will be dynamically set to current occupancy
            self.port_graph.add_edge(
                zone_port0_id,
                zone_port1_id,
                gate_capacity=zone.max_ions_gate_op,
                transport_capacity=zone.max_ions_transport_op,
                weight=0,
            )

        for connection in spec.connections:
            zone0 = connection.zone_port_spec0.zone_id
            port0 = connection.zone_port_spec0.port_id
            portid0 = zone_port_to_port_id(zone0, port0.value)
            zone1 = connection.zone_port_spec1.zone_id
            port1 = connection.zone_port_spec1.port_id
            portid1 = zone_port_to_port_id(zone1, port1.value)
            # TODO: update arch spec to include connection shuttle cost and use that as weight
            self.port_graph.add_edge(portid0, portid1, weight=1)

    def update_zone_occupancy_weight(self, zone: int, new_weight: int):
        port_id0 = zone_port_to_port_id(zone, 0)
        port_id1 = zone_port_to_port_id(zone, 1)
        self.port_graph.edges[port_id0, port_id1]["weight"] = new_weight

    def shortest_port_path_lengths(
        self, start_zone: int, targ_zone: int
    ) -> tuple[tuple[list[int], int, int], tuple[list[int], int, int]]:
        """Return the shortest path lengths for going from start to target zone

        The return value is a tuple. The first value is a tuple of the shortest zone path
        from port 0 of the start port to the closest port of the target port,
        this paths (port path) length, and the value of the
        target port for this shortest path. The second is the same values starting from port 1 of
        the starting zone.
        """
        port_ids0 = zone_port_to_port_id(start_zone, 0)
        port_ids1 = zone_port_to_port_id(start_zone, 1)
        port_idt0 = zone_port_to_port_id(targ_zone, 0)
        port_idt1 = zone_port_to_port_id(targ_zone, 1)
        path_s0t0, length_s0t0 = single_source_dijkstra(
            self.port_graph, port_ids0, port_idt0, weight="weight"
        )
        path_s0t1, length_s0t1 = single_source_dijkstra(
            self.port_graph, port_ids0, port_idt1, weight="weight"
        )
        path_s1t0, length_s1t0 = single_source_dijkstra(
            self.port_graph, port_ids1, port_idt0, weight="weight"
        )
        path_s1t1, length_s1t1 = single_source_dijkstra(
            self.port_graph, port_ids1, port_idt1, weight="weight"
        )
        path_length_s0_targ_port = (
            (port_path_to_zone_path(path_s0t0), length_s0t0, 0)
            if length_s0t0 <= length_s0t1
            else (port_path_to_zone_path(path_s0t1), length_s0t1, 1)
        )
        path_length_s1_targ_port = (
            (port_path_to_zone_path(path_s1t0), length_s1t0, 0)
            if length_s1t0 <= length_s1t1
            else (port_path_to_zone_path(path_s1t1), length_s1t1, 1)
        )
        return path_length_s0_targ_port, path_length_s1_targ_port
