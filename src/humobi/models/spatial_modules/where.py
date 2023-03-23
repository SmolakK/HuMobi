import numpy as np


class Simple():
    """
    Simple version of the spatial movement of an agent. Moves to the desired significant place,
    or to the random location if other is selected.
    """

    def __init__(self):
        pass


def where(user):
    """
    WHERE module of the model
    :param user: currently processed user
    """
    try:
        user.history.append(user.sig_locs[user.current_loc])
    except KeyError:
        try:
            user.history.append(np.random.choice([user.activity_area[n][0] for n in range(len(user.activity_area))]).centroid)
        except KeyError:
            user.history.append(user.sig_locs[0])