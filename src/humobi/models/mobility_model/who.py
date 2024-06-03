import sys

sys.path.append("..")
from src.humobi.structures.trajectory import TrajectoriesFrame
from src.humobi.models.temporal_tools import cluster_traj
from src.humobi.models.spatial_tools import distributions, generating, filtering
from src.humobi.misc import create_grid
from src.humobi.models.spatial_tools.misc import rank_freq
from math import ceil
from src.humobi.models.agent_module.generate_agents import generate_agents
from src.humobi.models.spatial_modules.where import where
from src.humobi.models.temporal_tools.when import when
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class Config:
    weights: bool = False
    side: int = 1000
    crs: str = "EPSG:3857"
    path: str = r"D:/Humobi/data/v2/london_1H_111.7572900082951_1.csv"
    trajectory_settings: dict = field(default_factory=lambda: {
        'names': ['id', 'datetime', 'temp', 'lat', 'lon', 'labels', 'start', 'end', 'geometry'],
        'crs': "EPSG:3857",
        'delimiter': ',',
        'skiprows': 1,
        'nrows': 9386
    })
    quantity: int = 2
    to_generate: int = 29



class Who:
    def __init__(self, config, trajectories_frame, quantity=2, to_generate=29):
        self.config = config
        self._trajectories_frame = trajectories_frame
        self._quantity = quantity
        self._to_generate = to_generate
        self._sig_frame = self._calculate_sig_frame() # Initialize this once
        self.agents = None

    def __str__(self):
        return f"Who Simulation with {len(self.agents)} agents configured."

    def __repr__(self):
        return (f"Who(config={self._config}, trajectories_frame={self._trajectories_frame}, "
                f"quantity={self._quantity}, to_generate={self._to_generate})")

    def run_simulation(self):
        self._process_data()
        self._generate_agents()
        self._simulate_movement()
        return self.output

    def save_output(self, file_path, file_format='csv'):
        if not hasattr(self, '_output'):
            raise ValueError("No simulation output to save. Please run the simulation first.")

        if file_format == 'csv':
            self._output.to_csv(file_path, index=False)
        elif file_format == 'excel':
            self._output.to_excel(file_path, index=False)
        else:
            raise ValueError("Unsupported file format. Please choose either 'csv' or 'excel'.")


    @property
    def config(self):
        """Get the configuration object."""
        return self._config

    @property
    def output(self):
        if not hasattr(self, '_output'):
            return None
        return self._output

    @config.setter
    def config(self, value):
        """Set the configuration object with validation."""
        if not isinstance(value, Config):
            raise ValueError("Config must be an instance of Config")
        self._config = value

    @property
    def trajectories_frame(self):
        """Get the trajectories frame."""
        return self._trajectories_frame

    @trajectories_frame.setter
    def trajectories_frame(self, value):
        """Set the trajectories frame with validation."""
        if not isinstance(value, TrajectoriesFrame):
            raise ValueError("Trajectories frame must be an instance of TrajectoriesFrame")
        self._trajectories_frame = value

    @property
    def quantity(self):
        """Get the quantity used in calculations."""
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        """Set the quantity ensuring it is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Quantity must be a positive integer")
        self._quantity = value

    @property
    def to_generate(self):
        """Get the number of agents to generate."""
        return self._to_generate

    @to_generate.setter
    def to_generate(self, value):
        """Set the number of agents to generate ensuring it is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Number of agents to generate must be a positive integer")
        self._to_generate = value



    def _calculate_sig_frame(self):
        return rank_freq(self.trajectories_frame, quantity= self._quantity)

    def _process_data(self):
        self.circadian_collection, self.cluster_association, self.cluster_share = cluster_traj.cluster_trajectories(
            self._trajectories_frame, top_places=self._sig_frame, quantity= self._quantity,  weights=self.config.weights
        )
        self._generate_distributions()

    def _generate_distributions(self):
        self.commute_dist = distributions.commute_distances(self.trajectories_frame, sig_frame=self._sig_frame,  quantity= self._quantity)
        self.layer = create_grid.create_grid(self.trajectories_frame, resolution=self.config.side)
        self.layer = filtering.filter_layer(self.layer, self._trajectories_frame)
        print(self.layer)
        self.unique_labels = set(self.cluster_association.values()).difference(set([-1]))

        self.cluster_spatial_distributions = {}
        self.cluster_commute_distributions = {}
        for n in self.unique_labels:
            group_indicies = [k for k, v in self.cluster_association.items() if v == n]
            group_sig_frame = self._sig_frame.loc[group_indicies]
            group_commute_dist = {k: v.loc[group_indicies] for k, v in self.commute_dist.items()}
            dist_list = distributions.convert_to_2d_distribution(group_sig_frame, self.layer, crs=self.config.crs, return_centroids=True, quantity=self._quantity)
            commute_distributions = distributions.commute_distances_to_2d_distribution(group_commute_dist, self.layer, crs=self.config.crs, return_centroids=True)
            self.cluster_spatial_distributions[n] = dist_list
            self.cluster_commute_distributions[n] = commute_distributions

    def _generate_agents(self):
        for label, share in self.cluster_share.items():
            amount = ceil(share * self._to_generate)
            current_spatial_distributions = self.cluster_spatial_distributions[label]
            current_commute_distributions = self.cluster_commute_distributions[label]
            home_positions = generating.generate_points_from_distribution(current_spatial_distributions[0], amount)
            work_positions = generating.select_points_with_commuting(home_positions, current_spatial_distributions, current_commute_distributions)
            activity_areas = generating.generate_activity_areas('ellipse', home_positions, work_positions, self.layer, 1.0)
            circadian_rhythm = cluster_traj.circadian_rhythm_extraction(self.circadian_collection, [], 2, 24)
            circadian_rhythm = circadian_rhythm[0]  # Temporary
            self.agents = generate_agents(amount, label, home_positions, work_positions, activity_areas, circadian_rhythm)

    def _simulate_movement(self):
        sim_start = datetime.strptime('01-01-2020', '%d-%m-%Y')
        end_sim = datetime.strptime('10-01-2020', '%d-%m-%Y')
        timeslots = [sim_start + timedelta(hours=i) for i in range((end_sim - sim_start).days * 24 + 1)]

        for timeslot in timeslots:
            for agent in self.agents:
                when(agent, timeslot)
                where(agent)
        zipped_history = [pd.DataFrame(zip(agent.history, timeslots), columns=['geometry', 'datetime']) for agent in
                          self.agents]
        self._output = pd.concat([y.assign(user_id=x) for x, y in enumerate(zipped_history)])

if __name__ == "__main__":
    config = Config()
    trajectories_frame = TrajectoriesFrame(config.path, config.trajectory_settings)  # Assume loaded outside the class
    simulation = Who(config, trajectories_frame)
    simulation.run_simulation()