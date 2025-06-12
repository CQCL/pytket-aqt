API documentation
~~~~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.aqt
.. automodule:: pytket.extensions.aqt.extension_version
.. automodule:: pytket.extensions.aqt.backends
.. automodule:: pytket.extensions.aqt.backends.aqt

    .. autoclass:: AQTBackend
        :members:

    .. autoexception:: AqtAccessError

.. automodule:: pytket.extensions.aqt.backends.config

    .. autoclass:: AQTAccessToken

        .. automethod:: resolve
        .. automethod:: overwrite
        .. automethod:: reset
        .. automethod:: retrieve
    
    .. autoclass:: AQTConfig
        :members:
    
    .. autofunction:: set_aqt_config
    
    .. autofunction:: print_available_devices

.. automodule:: pytket.extensions.aqt.backends.aqt_api

    .. autoclass:: AqtApi

        .. automethod:: cancel_job
        .. automethod:: get_devices
        .. automethod:: get_job_result
        .. automethod:: post_aqt_job
        .. automethod:: print_device_table
    
    .. autoclass:: AqtDevice

        .. automethod:: from_aqt_resource
    
    .. autoclass:: AqtMockApi

        .. automethod:: cancel_job
        .. automethod:: get_devices
        .. automethod:: get_job_result
        .. automethod:: post_aqt_job
        .. automethod:: print_device_table
    
    .. autoclass:: AqtOfflineApi

        .. automethod:: cancel_job
        .. automethod:: get_devices
        .. automethod:: get_job_result
        .. automethod:: post_aqt_job
        .. automethod:: print_device_table
    
    .. autoclass:: AqtRemoteApi

        .. automethod:: cancel_job
        .. automethod:: get_devices
        .. automethod:: get_job_result
        .. automethod:: post_aqt_job
        .. automethod:: print_device_table
    
    .. autofunction:: unwrap

.. automodule:: pytket.extensions.aqt.backends.aqt_job_data

    .. autoclass:: PytketAqtJob
        :members:

    .. autoclass:: PytketAqtJobCircuitData
        :members:

.. automodule:: pytket.extensions.aqt.backends.aqt_multi_zone
    :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture
.. automodule:: pytket.extensions.aqt.multi_zone_architecture.architecture

    .. autoclass:: MultiZoneArchitectureSpec

        .. autoproperty:: model_config
        .. automethod:: get_zone_max_ions
    
    .. autoclass:: Operation
        :members:
    
    .. autoenum:: PortId
        :members:
    
    .. autoclass:: PortSpec
        :members:
    
    .. autoclass:: Zone
        :members:
    
    .. autoclass:: ZoneConnection
        :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit
.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit.helpers
    :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit.multizone_circuit

    .. autoexception:: AcrossZoneOperationError
    .. autoexception:: MoveError
    .. autoexception:: QubitPlacementError
    .. autoexception:: ValidationError
    
    .. autoclass:: Init
        :members:
    
    .. autoclass:: MultiZoneCircuit

        .. automethod:: add_gate
        .. automethod:: add_move_barrier
        .. automethod:: copy
        .. automethod:: CX
        .. automethod:: get_n_pswaps
        .. automethod:: get_n_shuttles
        .. automethod:: measure_all
        .. automethod:: move_qubit
        .. automethod:: validate
    
    .. autoclass:: Shuttle

        .. automethod:: append_to_circuit
    
    .. autoclass:: SwapWithinZone

        .. automethod:: append_to_circuit
    
    .. autoenum:: VirtualZonePosition
    
    .. autodata:: move_gate
    .. autodata:: shuttle_gate
    .. autodata:: swap_gate

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit_routing
.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit_routing.greedy_routing

    .. autoclass:: GreedyCircuitRouter
        :members:
    
    .. autoclass:: QubitTracker

        .. automethod:: current_zone
        .. automethod:: move_qubit
        .. automethod:: zone_occupants

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit_routing.partition_routing
    :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit_routing.route_circuit
    :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.circuit_routing.settings

    .. autoexception:: RoutingSettingsError
    
    .. autoclass:: RoutingAlg
        :members:
    
    .. autoclass:: RoutingSettings

        .. automethod:: default

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.compilation_settings

    .. autoexception:: CompilationSettingsError
    
    .. autoclass:: CompilationSettings

        .. automethod:: default

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.depth_list
.. automodule:: pytket.extensions.aqt.multi_zone_architecture.depth_list.depth_list

    .. autofunction:: get_2q_gate_pairs_from_circuit
    .. autofunction:: get_depth_list
    .. autofunction:: get_initial_depth_list
    .. autofunction:: get_updated_depth_list

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.graph_algs
.. automodule:: pytket.extensions.aqt.multi_zone_architecture.graph_algs.graph
    :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar

    .. autoclass:: MtKahyparConfig
        :members:
    
    .. autoclass:: MtKahyparPartitioner

        .. automethod:: graph_data_to_mtkahypar_graph
        .. automethod:: graph_data_to_mtkahypar_target_graph
        .. automethod:: map_graph_to_target_graph
        .. automethod:: partition_graph

    .. autofunction:: configure_mtkahypar

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.graph_algs.mt_kahypar_check
    :members:

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.initial_placement
.. automodule:: pytket.extensions.aqt.multi_zone_architecture.initial_placement.initial_placement_generators

    .. autoexception:: InitialPlacementError
    
    .. autoclass:: GraphMapInitialPlacement

        .. automethod:: get_arch_graph_data
        .. automethod:: get_circuit_graph_data
        .. automethod:: initial_placement
    
    .. autoclass:: InitialPlacementGenerator

        .. automethod:: initial_placement

    .. autoclass:: ManualInitialPlacement

        .. automethod:: initial_placement

    .. autoclass:: QubitOrderInitialPlacement

        .. automethod:: initial_placement
    
    .. autofunction:: get_initial_placement
    .. autofunction:: get_initial_placement_generator

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.initial_placement.settings

    .. autoexception:: InitialPlacementSettingsError
    
    .. autoenum:: InitialPlacementAlg
    
    .. autoclass:: InitialPlacementSettings

        .. automethod:: default

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.macro_architecture_graph

    .. autoclass:: MacroZoneConfig
        :members:
    
    .. autoclass:: MacroZoneData
        :members:

    .. autoclass:: MultiZoneArch

        .. automethod:: get_connected_ports
        .. automethod:: shortest_path
    
    .. autofunction:: empty_macro_arch_from_architecture

.. automodule:: pytket.extensions.aqt.multi_zone_architecture.named_architectures
    :members:
