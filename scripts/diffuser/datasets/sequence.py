from collections import namedtuple
import numpy as np
import torch
from diffuser.environments.gridworld import generate_ground_truth_paths
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment
from .normalization import DatasetNormalizer
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='gridworld-v2', horizon=64,
                 normalizer='LimitsNormalizer', preprocess_fns=[],
                 max_path_length=1000, termination_penalty=0,
                 use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.use_padding = use_padding

        # Set observation and action dimensions
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Collect sample data for normalization
        sample_data, path_lengths = self.collect_normalization_data(num_samples=1000)

        # Initialize normalizer with collected data and path_lengths
        self.normalizer = DatasetNormalizer(sample_data, normalizer, path_lengths=path_lengths)

    def collect_normalization_data(self, num_samples=1000):
        observations = []
        actions = []
        path_lengths = []

        for _ in range(num_samples):
            obs = self.env.reset()
            trajectory_observations = []
            trajectory_actions = []  # Initialize the list to store actions for each trajectory
            path_len = 0
            for _ in range(self.horizon):
                action = self.env.action_space.sample()
                next_obs, reward, done, info = self.env.step(action)
                trajectory_observations.append(obs)
                trajectory_actions.append(action)  # Append the sampled action here
                path_len += 1
                obs = next_obs
                if done:
                    break
            path_lengths.append(path_len)
            observations.append(np.array(trajectory_observations, dtype=np.float32))
            actions.append(np.array(trajectory_actions, dtype=np.float32))  # Append the actions here

        dataset = {
            'observations': observations,
            'actions': actions
        }
        return dataset, path_lengths


    def get_conditions(self, observations):
        '''
            Condition on current observation for planning.
        '''
        return {0: observations[0]}

    def __len__(self):
        return 1000000  # Arbitrary large number

    def __getitem__(self, idx):
        obs = self.env.reset()
        env_state = self.env.get_state()
        observations = []
        actions = []

        for _ in range(self.horizon):
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            observations.append(obs)
            actions.append(action)
            obs = next_obs
            if done:
                break

        if self.use_padding and len(observations) < self.horizon:
            pad_length = self.horizon - len(observations)
            observations.extend([np.zeros_like(observations[0])] * pad_length)
            actions.extend([np.zeros_like(actions[0])] * pad_length)

        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        # Normalize observations and actions
        normed_observations = self.normalizer.normalize(observations, 'observations')
        normed_actions = self.normalizer.normalize(actions, 'actions')

        # Prepare conditions
        conditions = self.get_conditions(normed_observations)

        # Concatenate actions and observations
        trajectories = np.concatenate([normed_actions, normed_observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, env, horizon, max_path_length, normalizer, preprocess_fns, use_padding, **kwargs):
        """
        Initializes the PathDataset.

        Args:
            env (str): The environment identifier (e.g., 'gridworld-medium-expert-v2').
            horizon (int): The number of waypoints in each trajectory.
            max_path_length (int): Maximum path length for trajectories.
            normalizer (object): Normalizer instance for data normalization.
            preprocess_fns (list): List of preprocessing functions to apply to data.
            use_padding (bool): Whether to pad trajectories to the horizon.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.env = env
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.normalizer = normalizer
        self.preprocess_fns = preprocess_fns
        self.use_padding = use_padding

        # Load the dataset based on the environment
        self.dataset = self.load_dataset(env)

        # Define observation_dim
        self.observation_dim = (self.horizon + 2) * 2  # (start + goal + waypoints) * 2

    def load_dataset(self, env):
        """
        Loads the dataset based on the environment.

        Args:
            env (str): The environment identifier.

        Returns:
            list: Loaded dataset.
        """
        if env == 'gridworld-medium-expert-v2':
            dataset = generate_ground_truth_paths(num_samples=1000, grid_size=10)
            return dataset
        else:
            raise ValueError(f"Unknown environment: {env}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        start = torch.tensor(data["start"], dtype=torch.float32)
        goal = torch.tensor(data["goal"], dtype=torch.float32)
        waypoints = torch.tensor(data["waypoints"], dtype=torch.float32)
        optimized_waypoints = torch.tensor(data["optimized_waypoints"], dtype=torch.float32)

        N = self.horizon

        # Sample or pad waypoints to have exactly N waypoints
        if len(waypoints) >= N:
            indices = np.linspace(0, len(waypoints) - 1, N).astype(int)
            waypoints = waypoints[indices]
            optimized_waypoints = optimized_waypoints[indices]
        else:
            pad_size = N - len(waypoints)
            padding = waypoints[-1].unsqueeze(0).repeat(pad_size, 1)
            waypoints = torch.cat([waypoints, padding], dim=0)
            padding_opt = optimized_waypoints[-1].unsqueeze(0).repeat(pad_size, 1)
            optimized_waypoints = torch.cat([optimized_waypoints, padding_opt], dim=0)

        # Combine start, goal, and waypoints as conditioning information
        condition = torch.cat([start.unsqueeze(0), goal.unsqueeze(0), waypoints], dim=0)  # Shape: (N+2, 2)

        # Define target
        target = optimized_waypoints  # Ground truth

        return Batch(conditions=condition, targets=target)
class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            Condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset_22(SequenceDataset):
    '''
        Adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.horizon)[:, None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/value ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        # Extract rewards from the trajectory
        rewards = np.zeros(self.horizon, dtype=np.float32)
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = Batch(batch.trajectories, batch.conditions, value)
        return value_batch
class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.horizon)[:, None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        # Extract rewards from the trajectory
        rewards = np.zeros(self.horizon, dtype=np.float32)
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(batch.trajectories, batch.conditions, value)
        return value_batch