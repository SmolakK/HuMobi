from src.humobi.models.agent_module.agent_class import Agent


def _generate_agent(cluster_id, home, work, area):
	sig_locations = (home, work)
	return Agent(sig_locs=sig_locations, temporal_mechanism=None, spatial_mechanism=None,
	             cluster=cluster_id, activity_area=area)


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
	for n in range(n_agents):
		agents.append(_generate_agent(cluster_id, list_home[n], list_work[n], list_geometry[n]))
	return agents
