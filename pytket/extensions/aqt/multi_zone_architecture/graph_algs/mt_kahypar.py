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
import mtkahypar  # type: ignore

from pytket.extensions.aqt.multi_zone_architecture.graph_algs.graph import GraphData


def graph_data_to_mtkahypar_graph(graph_data: GraphData) -> mtkahypar.Graph:
    return mtkahypar.Graph(
        graph_data.n_vertices,
        len(graph_data.edges),
        graph_data.edges,
        graph_data.vertex_weights,
        graph_data.edge_weights,
    )


class MtKahyparPartitioner:
    """Class that performs graph partitioning using mt-kahypar

    :param n_threads: The number of threads to use for partitioning algorithms
    :param log_level: How much partitioning information to log, 0 == silent

    """

    def __init__(self, n_threads: int, log_level: int = 0):
        mtkahypar.initializeThreadPool(n_threads)
        mtkahypar.setSeed(13)
        self.context = mtkahypar.Context()
        self.context.loadPreset(mtkahypar.PresetType.DEFAULT)
        self.context.logging = False
        self.log_level = log_level

    def partition_graph(
        self,
        graph_data: GraphData,
        num_parts: int,
    ) -> list[int]:
        """Partition vertices of graph into num_parts parts

        Returns a list whose i'th element is the part that vertex i is assigned to

        :param graph_data: Graph specification
        :param num_parts: Number of partitions
        """
        avg_part_weight = sum(graph_data.vertex_weights) / num_parts
        self.context.setPartitioningParameters(
            num_parts,
            0.5 / avg_part_weight,
            mtkahypar.Objective.CUT,
        )
        graph = graph_data_to_mtkahypar_graph(graph_data)
        if graph_data.fixed_list:
            graph.addFixedVertices(graph_data.fixed_list, num_parts)
        part_graph = graph.partition(self.context)
        if self.log_level > 0:
            print("cut_cost: ", part_graph.cut())  # noqa: T201
        vertex_part_id: list[int] = []
        for vertex in range(graph_data.n_vertices):
            vertex_part_id.append(part_graph.blockID(vertex))
        return vertex_part_id

    def map_graph_to_target_graph(
        self, graph_data: GraphData, target_graph_data: GraphData
    ) -> list[int]:
        """Partition vertices of graph onto the nodes of
         another graph minimizing Steiner tree metric

        Returns a list whose i'th element is the target
         graph node that vertex i is assigned to
        """
        avg_part_weight = sum(graph_data.vertex_weights) / target_graph_data.n_vertices
        self.context.setPartitioningParameters(
            target_graph_data.n_vertices,
            0.5 / avg_part_weight,
            mtkahypar.Objective.CUT,  # This doesn't matter, Steiner tree metric is used
        )
        graph = graph_data_to_mtkahypar_graph(graph_data)
        target_graph = graph_data_to_mtkahypar_graph(target_graph_data)
        part_graph = graph.mapOntoGraph(target_graph, self.context)
        vertex_part_id: list[int] = []
        for vertex in range(graph_data.n_vertices):
            vertex_part_id.append(part_graph.blockID(vertex))
        return vertex_part_id
