from dataclasses import dataclass


@dataclass
class GraphData:
    n_vertices: int
    vertex_weights: list[int]
    edges: list[tuple[int, int]]
    edge_weights: list[int]
