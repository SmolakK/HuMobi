import numpy as np


class Simple():
    """
    Simple version of the spatial movement of an agent. Moves to the desired significant place,
    or to the random location if other is selected.
    """

    def __init__(self):
        pass


def where(agent):
    """
    WHERE module of the model
    :param agent: currently processed user
    """
    try:
        agent.history.append(agent.sig_locs.iloc[agent.current_loc])
    except KeyError:
        try:
            agent.history.append(np.random.choice([agent.activity_area[n][0] for n in range(len(agent.activity_area))]).centroid)
        except KeyError:
            agent.history.append(agent.sig_locs.iloc[0])