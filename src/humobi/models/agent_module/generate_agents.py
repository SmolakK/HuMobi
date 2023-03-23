from humobi.models.agent_module.agent_class import Agent
import pandas as pd


def _generate_agent(cluster_id, sig_locations, activity_area, temporal_mechanism=None, spatial_mechanism=None):
    return Agent(sig_locs=sig_locations, temporal_mechanism=temporal_mechanism, spatial_mechanism=spatial_mechanism,
                 cluster=cluster_id, activity_area=activity_area)


def generate_agents(n_agents, cluster_id, list_home, list_work, list_geometry):
    """
    Returns a list of generated Agent class objects.

    Args:
        n_agents: the number of agents to generate
        cluster_id: id of a cluster
        list_home: contains home positions
        list_work: contains work positions
        list_geometry: contains activity areas

    Returns:
        a list of generated agents
    """
    agents = []
    sig_locs = pd.concat((list_home, pd.concat(list_work, axis=1)), axis=1)
    for n in range(n_agents):
        agents.append(_generate_agent(cluster_id=cluster_id, sig_locations=sig_locs.iloc[n, :],
                                      activity_area=list_geometry.iloc[n], temporal_mechanism='simple',
                                      spatial_mechanism='simple'))
    return agents
