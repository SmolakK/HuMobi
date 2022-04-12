import numpy as np


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