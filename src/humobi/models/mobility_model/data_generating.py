import pandas as pd, geopandas as gpd
import sys
sys.path.append("..")
from src.humobi.models.temporal_tools import when
from src.humobi.models.spatial_modules import where
# from src.humobi.misc import export_to_file
WEIGHT = False


def data_generator(start, agents, clusters, circadian_rhythm, end=None, duration=None, output_format = 'csv', output_style = 'user'):
	"""
	Produces a time-step-based simulation for every agent.
	:param start: contains a string value which determinates a start of simulation, format: 'day.month.year'
	:param end: contains a string value which determinates an end of simulation, format: 'day.month.year'
	:param agents: contains a list with instances of Agent Class
	:param clusters: contains a list with clusters numbers
	:param duration: (an optional parameter) contains a duration of simulation
	:param output_format: contains a string with the format of output file: csv,feather or parquet
	:param output_style: contains a string 'user' or 'slot' which determinates a layout of the output file
	:return: a list with agents for each slot of simulation
	"""
	start = pd.to_datetime(start)
	if duration is None:
		if end is None:
			raise ValueError("You have to declare the end value")
		end = pd.to_datetime(end)
		duration = end-start
	else:
		duration = pd.Timedelta('{} days'.format(duration))
		end = start + duration
	slots_amount = 24 * duration.days
	time_slots = pd.date_range(start, end, slots_amount + 1)
	for slot in time_slots:
		for cluster in clusters:
			users = [agent for agent in agents if agent.cluster == cluster]
			for user in users:
				when.when(user, slot, circadian_rhythm)
				where.where(user)
	# export_to_file.export_to_file(agents, time_slots, output_format, output_style)