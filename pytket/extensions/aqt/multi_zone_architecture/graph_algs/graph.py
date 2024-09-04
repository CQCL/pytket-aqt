from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphData:
    """Graph data format used to submit qubit-zone graphs to be partitioned"""

    n_vertices: int
    """Number of vertices"""
    vertex_weights: list[int]
    """Vertex weights. One weight per vertex required"""
    edges: list[tuple[int, int]]
    """Edges between two vertices"""
    edge_weights: list[int]
    """Edge weights. One weight per edge required"""
    fixed_list: Optional[list[int]] = None
    """Optional list designating which partition a vertex
    should be fixed to. If given, one value per vertex is required.
    The i'th value determines the partition
    vertex i should be fixed to. A value of -1 means do not fix vertex"""

    def __post_init__(self) -> None:
        if len(self.vertex_weights) != self.n_vertices:
            raise ValueError("len(vertex_weights) must equal n_vertices")
        if len(self.edges) != len(self.edge_weights):
            raise ValueError("len(edge_weights) must equal len(edges)")
