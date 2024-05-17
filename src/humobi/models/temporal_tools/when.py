import numpy as np
from humobi.models.spatial_tools.misc import normalize_array
WEIGHT = False


def when(agent, slot):
	"""
	WHEN module of the model
	:param agent: Currently processed user
	:param slot: Currently processed time slot
	:param circadian_rhythm: Circadian rhythm of the group within which the user is
	"""
	if WEIGHT:
		probabilities = normalize_array(np.array([agent.circadian_rhythm[slot.hour],
												  agent.circadian_rhythm[slot.hour + 24],
												  agent.circadian_rhythm[slot.hour + 48]]))
		agent.current_loc = np.random.choice([0, 1, 2], size=1, p=probabilities)[0]
	if not WEIGHT:
		agent.current_loc = int(agent.circadian_rhythm[slot.hour])