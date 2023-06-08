import numpy as np
import pandas as pd
from src.humobi.structures.trajectory import TrajectoriesFrame


def _Mseq(places,length,prob):
	out_seq = []
	for n in range(length):
		seq_type = np.random.choice([True,False], 1, p=[prob,1-prob])
		if seq_type:
			out_seq.append(n%places)
		else:
			out_seq.append(np.random.randint(0, places, 1)[0])
	return out_seq


def markovian_sequences_generator(users,places,length,prob,return_params=False):
	"""
	Generates synthetic frame wtih Markovian sequences.

	Args:
		users: The number of users (sequences) to generate.
		places: The number of unique places (symbols) to choose from. Can be a range or list to randomly choose from.
		length: The length of sequences. Can be a range or list to randomly choose from.
		prob: The $p$ value determining the probability of generator following the deterministic sequence.
		Can be a range or list to randomly choose from.

	Returns:
		A DataFrame with synthetic Markovian sequences.
	"""
	if isinstance(places,list):
		places = np.random.choice(places,users)
	else:
		places = [places] * users
	if isinstance(length,list):
		length = np.random.choice(length,users)
	else:
		length = [length] * users
	if isinstance(prob,list):
		prob = np.random.choice(prob,users)
	else:
		prob = [prob] * users
	frames = []
	for uid in range(users):
		generated_track = _Mseq(places[uid],length[uid],prob[uid])
		tmstmps = pd.date_range(0,periods=length[uid],freq='h')
		generated_frame = pd.concat({uid: pd.DataFrame(generated_track,index=tmstmps)})
		frames.append(generated_frame)
	generated_frame = pd.concat(frames)
	generated_frame = generated_frame.reset_index()
	generated_frame.columns = ['user_id', 'datetime', 'labels']
	generated_frame = generated_frame.set_index(['user_id','datetime'])
	if return_params:
		return TrajectoriesFrame(generated_frame), pd.concat((pd.Series(places), pd.Series(length), pd.Series(prob)),axis=1)
	else:
		return TrajectoriesFrame(generated_frame)


def random_sequences_generator(users,places,length,return_params=False):
	"""
	Generates synthetic frame wtih random sequences.

	Args:
		users: The number of users (sequences) to generate.
		places: The number of unique places (symbols) to choose from. Can be a range or list to randomly choose from.
		length: The length of sequences. Can be a range or list to randomly choose from.

	Returns:
		A DataFrame with synthetic random sequences.
	"""
	if isinstance(places,list):
		places = np.random.choice(places,users)
	else:
		places = [places] * users
	if isinstance(length,list):
		length = np.random.choice(length,users)
	else:
		length = [length] * users
	frames = []
	for uid in range(users):
		generated_track = np.random.randint(0,places[uid],length[uid],dtype=np.int64)
		tmstmps = pd.date_range(0,periods=length[uid],freq='h')
		generated_frame = pd.concat({uid: pd.DataFrame(generated_track,index=tmstmps)})
		frames.append(generated_frame)
	generated_frame = pd.concat(frames)
	generated_frame = generated_frame.reset_index()
	generated_frame.columns = ['user_id', 'datetime', 'labels']
	generated_frame = generated_frame.set_index(['user_id','datetime'])
	if return_params:
		return TrajectoriesFrame(generated_frame), pd.concat((pd.Series(places), pd.Series(length)),
		                                                     axis=1)
	else:
		return TrajectoriesFrame(generated_frame)


def deterministic_sequences_generator(users,places,repeats):
	"""
	Generates synthetic frame wtih deterministic sequences.

	Args:
		users: The number of users (sequences) to generate.
		places: The number of unique places (symbols) to choose from. Can be a range or list to randomly choose from.
		repeats: The number of repeats determining the lenght of sequences. Can be a range or list to randomly choose
	from.

	Returns:
		A DataFrame with synthetic deterministic sequences.
	"""
	if isinstance(places,list):
		places = np.random.choice(places,users)
	else:
		places = [places] * users
	if isinstance(repeats,list):
		repeats = np.random.choice(repeats,users)
	else:
		repeats = [repeats] * users
	frames = []
	for uid in range(users):
		generated_track = [x for x in range(places[uid])] * repeats[uid]
		tmstmps = pd.date_range(0,periods=len(generated_track),freq='h')
		generated_frame = pd.concat({uid: pd.DataFrame(generated_track,index=tmstmps)})
		frames.append(generated_frame)
	generated_frame = pd.concat(frames)
	generated_frame = generated_frame.reset_index()
	generated_frame.columns = ['user_id', 'datetime', 'labels']
	generated_frame = generated_frame.set_index(['user_id','datetime'])
	return TrajectoriesFrame(generated_frame)


def exploratory_sequences_generator(users, places):
	"""
	Generates synthetic frame wtih exploratory sequences, where each next symbol is previously unseen.

	Args:
		users: The number of users (sequences) to generate.
		places: The number of unique places (symbols) to choose from. Also determine the legnth of the sequence.
		Can be a range or list to randomly choose from.

	Returns:
		A DataFrame with synthetic exploratory sequences.
	"""
	if isinstance(places,list):
		places = np.random.choice(places,users)
	else:
		places = [places] * users
	frames = []
	for uid in range(users):
		generated_track = np.random.permutation([x for x in range(places[uid])])
		tmstmps = pd.date_range(0,periods=len(generated_track),freq='h')
		generated_frame = pd.concat({uid: pd.DataFrame(generated_track,index=tmstmps)})
		frames.append(generated_frame)
	generated_frame = pd.concat(frames)
	generated_frame = generated_frame.reset_index()
	generated_frame.columns = ['user_id', 'datetime', 'labels']
	generated_frame = generated_frame.set_index(['user_id','datetime'])
	return TrajectoriesFrame(generated_frame)


def self_transitions_sequences_generator(users,places,length):
	"""
	Generates synthetic frame wtih self-transitions sequences. The number of self-transtions repeating after each other 
	is determined by the number of symbols and the legnth of the sequence.

	Args:
		users: The number of users (sequences) to generate.
		places: The number of unique places (symbols) to choose from. Can be a range or list to randomly choose from.
		length: The length of sequences. Can be a range or list to randomly choose from.

	Returns:
		A DataFrame with synthetic self-transitions sequences.
	"""
	if isinstance(places,list):
		places = np.random.choice(places,users)
	else:
		places = [places] * users
	if isinstance(length,list):
		length = np.random.choice(length,users)
	else:
		length = [length] * users
	frames = []
	for uid in range(users):
		per_place = length[uid]//places[uid]
		generated_track = []
		for n in range(places[uid]):
			generated_track += [n]*per_place
		tmstmps = pd.date_range(0,periods=len(generated_track),freq='h')
		generated_frame = pd.concat({uid: pd.DataFrame(generated_track,index=tmstmps)})
		frames.append(generated_frame)
	generated_frame = pd.concat(frames)
	generated_frame = generated_frame.reset_index()
	generated_frame.columns = ['user_id', 'datetime', 'labels']
	generated_frame = generated_frame.set_index(['user_id','datetime'])
	return TrajectoriesFrame(generated_frame)


def non_stationary_sequences_generator(users, places, states, length, return_params = False):
	"""
	Generates synthetic frame wtih non-stationary sequences. The algorithms chooses between states, each with different
	generation routine.

	Args:
		users: The number of users (sequences) to generate.
		places: The number of unique places (symbols) to choose from. Can be a range or list to randomly choose from.
		states: The number of states with different generation routine. Can be a range or list to randomly choose
		from.
		length: The length of sequences. Can be a range or list to randomly choose from.

	Returns:
		A DataFrame with synthetic non-stationary sequences.
	"""
	if isinstance(places,list):
		places = np.random.choice(places,users)
	else:
		places = [places] * users
	if isinstance(length,list):
		length = np.random.choice(length,users)
	else:
		length = [length] * users
	if isinstance(states,list):
		states = np.random.choice(states,users)
	else:
		states = [states] * users
	frames = []
	for uid in range(users):
		num_of_states = states[uid]
		states_probs = np.random.uniform(0, 1, num_of_states)
		states_probs /= sum(states_probs)
		states_steps = np.random.choice([x for x in range(num_of_states)],length[uid],p=states_probs)
		states_places = [np.random.uniform(0, 1, places[uid]) for x in range(num_of_states)]
		states_places = [x/sum(x) for x in states_places]
		generated_track = []
		for n in range(length[uid]):
			generated_track.append(np.random.choice([x for x in range(places[uid])], p=states_places[states_steps[n]]))
		tmstmps = pd.date_range(0, periods=len(generated_track), freq='h')
		generated_frame = pd.concat({uid: pd.DataFrame(generated_track, index=tmstmps)})
		frames.append(generated_frame)
	generated_frame = pd.concat(frames)
	generated_frame = generated_frame.reset_index()
	generated_frame.columns = ['user_id', 'datetime', 'labels']
	generated_frame = generated_frame.set_index(['user_id','datetime'])
	if return_params:
		return TrajectoriesFrame(generated_frame), pd.concat((pd.Series(places), pd.Series(length), pd.Series(states)),axis=1)
	else:
		return TrajectoriesFrame(generated_frame)
