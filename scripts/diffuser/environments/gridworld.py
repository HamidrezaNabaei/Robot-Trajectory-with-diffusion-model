import gym
from gym import spaces
import numpy as np
import random
from queue import PriorityQueue
from scipy.interpolate import interp1d

class RandomObstacleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, max_episode_steps=100):
        super(RandomObstacleEnv, self).__init__()

        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps  # Add this line
        self.current_step = 0

        # Continuous action space: agent can move in any direction
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space: agent's position (x, y), goal position (x, y), and flattened grid
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2 + 2 + self.grid_size * self.grid_size,), dtype=np.float32
        )

        # Initialize grid and positions
        self.grid = None
        self.position = None
        self.start_pos = None
        self.goal_pos = None
        self.path = None

        # Seed for reproducibility (optional)
        self.seed()

        # Reset environment state
        self.reset()

    def add_obstacle(self, x, y):
        """
        Adds an obstacle at the specified (x, y) position.
        """
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[x, y] = 1

    def set_state(self, state):
        self.grid = state['grid'].copy()
        self.position = state['position'].copy()
        self.goal_pos = state['goal_pos']  # Assign directly
        self.grid_size = state['grid_size']

    def get_state(self):
        return {
            'grid': self.grid.copy(),
            'position': self.position.copy(),
            'goal_pos': self.goal_pos,  # Remove .copy() since tuples are immutable
            'grid_size': self.grid_size,
            # Include any other necessary attributes
        }

    def get_normalized_score(self, total_reward):
        """
        Normalizes the total reward to a standard scale.

        Args:
            total_reward (float): The total reward obtained.

        Returns:
            normalized_score (float): The normalized score.
        """
        # Define the minimum and maximum possible returns
        min_return = -self._max_episode_steps * 0.1  # Assuming minimal movement penalty
        max_return = 100.0  # Reward for reaching the goal

        # Normalize the total_reward to a 0-100 scale
        normalized_score = 100 * (total_reward - min_return) / (max_return - min_return)
        # Ensure the score is within 0 and 100
        normalized_score = max(0.0, min(100.0, normalized_score))

        return normalized_score

    def state_vector(self):
        """Returns a representation of the environment's current state."""
        # Flatten the grid
        grid_flat = self.grid.flatten().astype(np.float32)
        # Concatenate agent's position, goal position, and grid
        state = np.concatenate([
            self.position.copy(),
            np.array(self.goal_pos, dtype=np.float32),
            grid_flat
        ])
        return state

    def reset(self):
        #print("resetted")
        # Reset step counter
        self.current_step = 0

        # Generate random grid with obstacles
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        num_obstacles = random.randint(0, self.grid_size * 2)
        for _ in range(num_obstacles):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[x, y] = 1  # Place obstacle

        # Randomize start and goal positions
        while True:
            self.start_pos = (
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
            )
            self.goal_pos = (
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
            )
            start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
            goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))

            # Ensure start and goal are not on obstacles
            if (
                self.grid[start_cell[0], start_cell[1]] == 0
                and self.grid[goal_cell[0], goal_cell[1]] == 0
            ):
                # Check for feasible path using A*
                path = self.astar(start_cell, goal_cell)
                if path is not None:
                    break  # Valid path found

        # Set agent's starting position
        self.position = np.array(self.start_pos, dtype=np.float32)

        # Store the interpolated A* path
        self.path = self.interpolate_path(path)
        self.path_index = 0

        return self._get_obs()

    def step(self, action=None):
        # Use precomputed A* path to determine action
        if self.path is not None and self.path_index < len(self.path) - 1:
            next_position = self.path[self.path_index + 1]
            action = np.array(next_position) - self.position
            self.path_index += 1
        else:
            action = np.zeros(2, dtype=np.float32)  # No movement if no path available

        # Ensure action and position are 1D arrays
        self.position = np.asarray(self.position, dtype=np.float32).reshape(-1)

        # Apply action to agent's position
        new_position = self.position + action
        new_position = np.clip(new_position, 0, self.grid_size - 1e-5)  # Keep within bounds

        # Ensure new_position is a 1D array
        new_position = np.asarray(new_position, dtype=np.float32).reshape(-1)

        # Extract scalar values using .item()
        x_new = int(new_position[0].item())
        y_new = int(new_position[1].item())

        collision = False
        if self.grid[x_new, y_new] == 1:
            collision = True
            # Optionally prevent movement into obstacle
            new_position = self.position  # Stay in place

        # Update position
        self.position = new_position

        # Compute reward
        reward = -0.1 * np.linalg.norm(action)  # Small penalty for movement

        # Check if reached goal
        distance_to_goal = np.linalg.norm(self.position - self.goal_pos)
        done = False
        if distance_to_goal < 0.5:
            done = True
            reward += 100.0  # Reward for reaching goal

        # Penalize collision
        if collision:
            reward -= 10.0

        # Increment step count
        self.current_step += 1

        # Check if maximum steps exceeded
        if self.current_step >= self.max_episode_steps:
            done = True

        # Set info dictionary (can include additional diagnostic info)
        info = {
            'distance_to_goal': distance_to_goal,
            'current_step': self.current_step,
            'collision': collision,
        }
        return self._get_obs(), reward, done, info

    def is_valid(self, x, y):
        """
        Checks if a cell is valid (i.e., not an obstacle and within bounds).
        """
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x, y] == 0
        return False

    def _get_obs(self):
        # Flatten the grid
        grid_flat = self.grid.flatten().astype(np.float32)
        # Concatenate agent's position, goal position, and grid
        return np.concatenate([
            self.position.copy(),
            np.array(self.goal_pos, dtype=np.float32),
            grid_flat
        ])

    def render(self, mode='human'):
        # Optional: Implement visualization of the grid and agent
        pass

    def close(self):
        pass

    def get_dataset(self, num_episodes=100):
        """Generates a dataset by collecting experiences from the environment."""
        data = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'rewards': [],
            'terminals': [],
            'timeouts': [],  # Include the 'timeouts' key
        }

        for episode in range(num_episodes):
            obs = self.reset()
            done = False
            steps = 0
            while not done and steps < self._max_episode_steps:
                action = None  # Action will be determined by the A* path
                next_obs, reward, done, info = self.step(action)

                steps += 1
                timeout = steps >= self._max_episode_steps
                terminal = done and not timeout

                data['observations'].append(obs)
                data['actions'].append(action)
                data['next_observations'].append(next_obs)
                data['rewards'].append(reward)
                data['terminals'].append(terminal)
                data['timeouts'].append(timeout)

                obs = next_obs

                if done:
                    break  # Exit loop if episode is done

        # Convert lists to NumPy arrays
        for key in data:
            data[key] = np.array(data[key])

        return data

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def astar(self, start, goal):
        """A* pathfinding algorithm to check path feasibility."""
        grid = self.grid
        grid_size = self.grid_size
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected grid

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
                # Path found
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if (
                    0 <= neighbor[0] < grid_size
                    and 0 <= neighbor[1] < grid_size
                    and grid[neighbor[0], neighbor[1]] == 0  # Not an obstacle
                ):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, goal)
                        open_set.put((f_score, neighbor))
                        came_from[neighbor] = current

        return None  # No path found

    def heuristic(self, a, b):
        # Use Manhattan distance as heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def remove_duplicate_points(self, path, epsilon=1e-5):
        """Removes duplicate points from a path to avoid interpolation issues."""
        cleaned_path = [path[0]]
        for i in range(1, len(path)):
            if np.linalg.norm(path[i] - cleaned_path[-1]) > epsilon:
                cleaned_path.append(path[i])
        return np.array(cleaned_path)

    def interpolate_path(self, path, num_points=50):
        """Interpolate a path to create a finer and smoother path."""
        path = np.array(path, dtype=np.float32)
        if len(path) < 2:
            return path

        # Remove duplicate points to avoid division by zero
        path = self.remove_duplicate_points(path)

        if len(path) < 2:
            return path

        x = path[:, 0]
        y = path[:, 1]
        distance = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))

        # Check for all-zero distances, which would cause issues in interpolation
        if np.all(distance == 0):
            return path

        # Ensure that distance is strictly increasing
        unique_distances, unique_indices = np.unique(distance, return_index=True)
        x = x[unique_indices]
        y = y[unique_indices]
        distance = unique_distances

        if len(distance) < 2:
            return path

        distance = distance / distance[-1]

        # Correct endpoints to ensure smooth interpolation
        distance = np.insert(distance, 0, 0.0)
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, y[0])

        try:
            interpolator_x = interp1d(distance, x, kind='linear', fill_value='extrapolate')
            interpolator_y = interp1d(distance, y, kind='linear', fill_value='extrapolate')
        except ValueError as e:
            print(f"Interpolation error: {e}")
            return path

        alpha = np.linspace(0, 1, num_points)
        x_smooth = interpolator_x(alpha)
        y_smooth = interpolator_y(alpha)

        return np.vstack((x_smooth, y_smooth)).T


def generate_ground_truth_paths(num_samples=1000, grid_size=10):
    """
    Generates ground truth paths using the A* algorithm in a grid environment.

    Args:
        num_samples (int): Number of paths to generate.
        grid_size (int): Size of the grid (grid_size x grid_size).

    Returns:
        list of dict: Each dict contains 'start', 'goal', 'waypoints', and 'optimized_waypoints'.
    """
    dataset = []
    for _ in range(num_samples):
        grid_env = RandomObstacleEnv(grid_size=grid_size)

        # Randomly place obstacles
        num_obstacles = random.randint(0, grid_size * 2)
        obstacles = set()
        available_positions = list(np.ndindex(grid_env.grid_size, grid_env.grid_size))
        random.shuffle(available_positions)
        for obstacle in available_positions:
            if len(obstacles) >= num_obstacles:
                break
            x, y = obstacle
            if grid_env.is_valid(x, y):
                grid_env.add_obstacle(x, y)
                obstacles.add(obstacle)

        # Randomly choose start and goal positions
        available_positions = [pos for pos in available_positions if pos not in obstacles]
        if len(available_positions) < 2:
            continue  # Skip if not enough free cells

        start, goal = random.sample(available_positions, 2)

        # Generate A* path using the existing astar method
        path = grid_env.astar(start, goal)
        if path is None:
            continue  # Skip if no path found

        # Prepare data entry
        dataset.append({
            "start": np.array(start, dtype=np.float32) + 0.5,  # Center the positions
            "goal": np.array(goal, dtype=np.float32) + 0.5,
            "waypoints": np.array(path, dtype=np.float32) + 0.5,  # Center the waypoints
            "optimized_waypoints": grid_env.interpolate_path(path) + 0.5,  # Interpolated path for smooth trajectory
            "grid": grid_env.grid.copy(),
        })

    return dataset

