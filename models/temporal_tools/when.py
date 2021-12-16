import numpy as np
from models.spatial_tools.misc import normalize_array
WEIGHT = False


def when(user,slot,circadian_rhythm):
	"""
	WHEN module of the model
	:param user: Currently processed user
	:param slot: Currently processed time slot
	:param circadian_rhythm: Circadian rhythm of the group within which the user is
	"""
	if WEIGHT:
		probabilities = normalize_array(np.array([circadian_rhythm[user.cluster][slot.hour],
												circadian_rhythm[user.cluster][slot.hour + 24],
												circadian_rhythm[user.cluster][slot.hour + 48]]))
		user.current_loc = np.random.choice([0, 1, 2], size=1, p=probabilities)[0]
	if not WEIGHT:
		user.current_loc = int(circadian_rhythm[user.cluster][slot.hour])