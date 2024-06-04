# test_who.py
import pytest
from unittest.mock import patch
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.models.temporal_tools import cluster_traj
from humobi.models.agent_module.generate_agents import generate_agents

# # Test the initialization of TrajectoriesFrame
# def test_trajectories_frame_initialization():
#     with patch('humobi.structures.trajectory.TrajectoriesFrame.__init__', return_value=None) as mock_init:
#         tf = TrajectoriesFrame('dummy_path', {'dummy': 'params'})
#         mock_init.assert_called_once_with('dummy_path', {'dummy': 'params'})
#
# # Test clustering functionality
# def test_cluster_trajectories():
#     with patch('humobi.models.temporal_tools.cluster_traj.cluster_trajectories') as mock_cluster:
#         mock_cluster.return_value = ([], {}, {})
#         trajectories_frame = TrajectoriesFrame('dummy_path', {'dummy': 'params'})
#         result = cluster_traj.cluster_trajectories(trajectories_frame, weights=False)
#         assert result == ([], {}, {}), "Expected empty clustering outputs"
#
# # Test agent generation from distributions
# def test_generate_agents():
#     with patch('humobi.models.agent_module.generate_agents.generate_agents') as mock_generate:
#         mock_generate.return_value = ['agent1', 'agent2']
#         agents = generate_agents(2, 'label1', ['pos1'], ['pos2'], ['area1'])
#         mock_generate.assert_called_once_with(2, 'label1', ['pos1'], ['pos2'], ['area1'])
#         assert agents == ['agent1', 'agent2'], "Expected two agents to be generated"
