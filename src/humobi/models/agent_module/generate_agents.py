from src.humobi.models.agent_module.agent_class import Agent


def _generate_agent(cluster_id, home, work, area):
	sig_locations = (home, work)
	return Agent(sig_locs=sig_locations, temporal_mechanism=None, spatial_mechanism=None,
	             cluster=cluster_id, activity_area=area)


def generate_agents(n_agents, cluster_id, list_home, list_work, list_geometry):
	"""
	Returns a list of generated Agent class objects.
	:param n_agents: an amount of agents to generate
	:param cluster_id: id of a cluster
	:param list_home: contains home positions
	:param list_work: contains work positions
	:param list_geometry: contains activity areas
	:return: a list of generated agents
	"""
	agents = []
	for n in range(n_agents):
		agents.append(_generate_agent(cluster_id, list_home[n], list_work[n], list_geometry[n]))
	return agents
