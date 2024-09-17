# Copyright 2020-2024 Quantinuum
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
