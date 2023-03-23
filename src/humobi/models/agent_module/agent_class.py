
class Agent:
	"""
	An Agent class.

	Args:
		sig_locs: contains significant locations,
		temporal_mechanism: contains temporal mechanism,
		spatial_mechanism: contains spatial mechanism,
		cluster: contains the label of cluster,
		activity_area: contains the area of activity.
	"""

	def __init__(self, sig_locs, temporal_mechanism, spatial_mechanism, cluster, activity_area):
		"""
		Initialises and agent and assigns him temporal and spatial mechanisms of movement.
		"""
		self._sig_locs = sig_locs
		self._pref_return = {}
		self._history = []
		self._temporal_mechanism = temporal_mechanism
		self._spatial_mechanism = spatial_mechanism
		self._current_loc = None
		self._cluster = cluster
		self._activity_area = activity_area

	def __str__(self):
		return 'Current location: ' + str(self.current_loc) + ' Home position: ' + str(
			self.sig_locs[0]) + ' Cluster: ' + str(self.cluster)

	def __repr__(self):
		return 'Current location: ' + str(self.current_loc) + ' Home position: ' + str(
			self.sig_locs[0]) + ' Cluster: ' + str(self.cluster)

	@property
	def sig_locs(self):
		return self._sig_locs

	@sig_locs.setter
	def sig_locs(self, new_value):
		self._sig_locs = new_value

	@property
	def pref_return(self):
		return self._pref_return

	@pref_return.setter
	def pref_return(self, new_dict):
		self._pref_return = new_dict

	@property
	def history(self):
		return self._history

	@history.setter
	def history(self, new_list):
		self._history = new_list

	@property
	def temporal_mechanism(self):
		return self._temporal_mechanism

	@temporal_mechanism.setter
	def temporal_mechanism(self, new_value):
		self._temporal_mechanism = new_value

	@property
	def spatial_mechanism(self):
		return self._spatial_mechanism

	@spatial_mechanism.setter
	def spatial_mechanism(self, new_value):
		self._spatial_mechanism = new_value

	@property
	def current_loc(self):
		return self._current_loc

	@current_loc.setter
	def current_loc(self, new_value):
		self._current_loc = new_value

	@property
	def cluster(self):
		return self._cluster

	@cluster.setter
	def cluster(self, new_value):
		self._cluster = new_value

	@property
	def activity_area(self):
		return self._activity_area

	@activity_area.setter
	def activity_area(self, new_value):
		self._activity_area = new_value

	def move(self):
		pass