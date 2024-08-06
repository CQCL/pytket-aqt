from typing import Optional

import mtkahypar

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
    def __init__(self, n_threads):
        mtkahypar.initializeThreadPool(n_threads)
        mtkahypar.setSeed(13)
        self.context = mtkahypar.Context()
        self.context.loadPreset(mtkahypar.PresetType.DEFAULT)
        self.context.logging = False

    def partition_graph(
        self,
        graph_data: GraphData,
        num_parts: int,
        fixed_list: Optional[list[int]] = None,
    ) -> list[int]:
        """Partition vertices of graph into num_parts parts

        Returns a list whose i'th element is the part that vertex i is assigned to
        """
        avg_part_weight = sum(graph_data.vertex_weights) / num_parts
        self.context.setPartitioningParameters(
            num_parts,
            0.5 / avg_part_weight,
            mtkahypar.Objective.CUT,
        )
        graph = graph_data_to_mtkahypar_graph(graph_data)
        if fixed_list:
            graph.addFixedVertices(fixed_list, num_parts)
        part_graph = graph.partition(self.context)
        print("cut_cost: ", part_graph.cut())
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
