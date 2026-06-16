"""Business services exposed through lazy compatibility imports."""

from importlib import import_module

__all__ = [
    'OntologyGenerator',
    'GraphBuilderService',
    'TextProcessor',
    'EntityReader',
    'EntityNode',
    'FilteredEntities',
    'OasisProfileGenerator',
    'OasisAgentProfile',
    'SimulationManager',
    'SimulationState',
    'SimulationStatus',
    'SimulationConfigGenerator',
    'SimulationParameters',
    'AgentActivityConfig',
    'TimeSimulationConfig',
    'EventConfig',
    'PlatformConfig',
    'SimulationRunner',
    'SimulationRunState',
    'RunnerStatus',
    'AgentAction',
    'RoundSummary',
    'GraphMemoryUpdater',
    'GraphMemoryManager',
    'AgentActivity',
    'SimulationIPCClient',
    'SimulationIPCServer',
    'IPCCommand',
    'IPCResponse',
    'CommandType',
    'CommandStatus',
]

_MODULES = {
    "OntologyGenerator": "ontology_generator",
    "GraphBuilderService": "graph_builder",
    "TextProcessor": "text_processor",
    "EntityReader": "entity_reader",
    "EntityNode": "entity_reader",
    "FilteredEntities": "entity_reader",
    "OasisProfileGenerator": "oasis_profile_generator",
    "OasisAgentProfile": "oasis_profile_generator",
    "SimulationManager": "simulation_manager",
    "SimulationState": "simulation_manager",
    "SimulationStatus": "simulation_manager",
    "SimulationConfigGenerator": "simulation_config_generator",
    "SimulationParameters": "simulation_config_generator",
    "AgentActivityConfig": "simulation_config_generator",
    "TimeSimulationConfig": "simulation_config_generator",
    "EventConfig": "simulation_config_generator",
    "PlatformConfig": "simulation_config_generator",
    "SimulationRunner": "simulation_runner",
    "SimulationRunState": "simulation_runner",
    "RunnerStatus": "simulation_runner",
    "AgentAction": "simulation_runner",
    "RoundSummary": "simulation_runner",
    "GraphMemoryUpdater": "graph_memory_updater",
    "GraphMemoryManager": "graph_memory_updater",
    "AgentActivity": "graph_memory_updater",
    "SimulationIPCClient": "simulation_ipc",
    "SimulationIPCServer": "simulation_ipc",
    "IPCCommand": "simulation_ipc",
    "IPCResponse": "simulation_ipc",
    "CommandType": "simulation_ipc",
    "CommandStatus": "simulation_ipc",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
