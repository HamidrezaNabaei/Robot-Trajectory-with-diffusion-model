import gym
from gym import spaces
import numpy as np
import random
from queue import PriorityQueue
from scipy.interpolate import interp1d

class RandomObstacleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, max_episode_steps=100, padding=0.1):
        super(RandomObstacleEnv, self).__init__()

        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps  # Add this line
        self.current_step = 0
        self.padding = padding
        self.points_distance = 0.2

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
        }

    def get_normalized_score(self, total_reward):
        """
        Normalizes the total reward to a standard scale.

        Args:
            total_reward (float): The total reward obtained.

        Returns:
            normalized_score (float): The normalized score.
        """
        min_return = -self._max_episode_steps * 10  # Assuming minimal movement penalty
        max_return = 100.0  # Reward for reaching the goal

        normalized_score = 100 * (total_reward - min_return) / (max_return - min_return)
        normalized_score = max(0.0, min(100.0, normalized_score))

        return normalized_score

    def state_vector(self):
        """Returns a representation of the environment's current state."""
        grid_flat = self.grid.flatten().astype(np.float32)
        state = np.concatenate([
            self.position.copy(),
            np.array(self.goal_pos, dtype=np.float32),
            grid_flat
        ])
        return state

    def reset(self):
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        num_obstacles = random.randint(0, self.grid_size * 2)
        num_obstacles = random.randint(15,20)
        # for _ in range(num_obstacles):
        #     x = random.randint(0, self.grid_size - 1)
        #     y = random.randint(0, self.grid_size - 1)
        #     self.grid[x, y] = 1  # Place obstacle
        # Fixed obstacle positions
        obstacle_positions = [
            (1, 2), (2, 3), (4, 1), (4, 3)
        ]
        #print("reset")
        for pos in obstacle_positions:
            self.grid[pos] = 1  # Place obstacle
        while True:
            #print("randomrandom")
            self.start_pos = (
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
            )
            self.goal_pos = (
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
                np.random.randint(low=0, high=self.grid_size, dtype='int32'),
            )
            #print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")

            #self.start_pos = (np.int32(3),np.int32(4))
            #self.goal_pos = (np.int32(2),np.int32(0))
            start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
            goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))
            if (
                self.grid[start_cell[0], start_cell[1]] == 0
                and self.grid[goal_cell[0], goal_cell[1]] == 0
            ):
                path = self.astar(start_cell, goal_cell)
                if path is not None:
                    break
        #self.start_pos = (2.0,0.0)
        #self.goal_pos = (4.0,4.0)
        start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
        goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))
        self.position = np.array(self.start_pos, dtype=np.float32)
        #self.position = np.array(start_cell, dtype=np.float32)
        #self.path = self.interpolate_path(path)
        #self.path_index = 0

        return self._get_obs()
    def reset3(self):
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Fixed obstacle positions
        obstacle_positions = [
            (1, 2), (2, 3), (4, 1), (4, 3)
        ]
        for pos in obstacle_positions:
            self.grid[pos] = 1  # Place obstacle

        # Fixed start and goal positions
        self.start_pos = np.array((3, 4), dtype=np.int32)  # Green circle position
        self.goal_pos = np.array((2, 0), dtype=np.int32)  # Orange circle position
        self.start_pos = (self.start_pos[0], self.start_pos[1])
        self.goal_pos = (self.goal_pos[0], self.goal_pos[1])
        start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
        goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))

        # Ensure there is a valid path
        path = self.astar(start_cell, goal_cell)


        self.position = np.array(self.start_pos, dtype=np.float32)
        self.path = self.interpolate_path(path)
        self.path_index = 0

        return self._get_obs()
    def reset4(self):
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        num_obstacles = random.randint(0, self.grid_size * 2)
        #num_obstacles = random.randint(15,20)
        obstacle_positions = [
            (1, 2), (2, 3), (4, 1), (4, 3)
        ]
        for pos in obstacle_positions:
            self.grid[pos] = 1  # Place obstacle
        self.start_pos = (3, 4)  # Green circle position
        self.goal_pos = (2, 0)  # Orange circle position
        # while True:
        #     self.start_pos = (
        #         np.random.randint(low=0, high=self.grid_size, dtype='int32'),
        #         np.random.randint(low=0, high=self.grid_size, dtype='int32'),
        #     )
        #     self.goal_pos = (
        #         np.random.randint(low=0, high=self.grid_size, dtype='int32'),
        #         np.random.randint(low=0, high=self.grid_size, dtype='int32'),
        #     )
        #     start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
        #     goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))

        #     if (
        #         self.grid[start_cell[0], start_cell[1]] == 0
        #         and self.grid[goal_cell[0], goal_cell[1]] == 0
        #     ):
        #         path = self.astar(start_cell, goal_cell)
        #         if path is not None:
        #             break
        start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
        goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))

        # # Ensure there is a valid path (since the setup is fixed, a path must exist)
        # path = self.astar(start_cell, goal_cell)
        # if path is None:
        #     raise ValueError("No valid path found between start and goal in the fixed configuration")
        self.position = np.array(self.start_pos, dtype=np.float32)
        #self.path = self.interpolate_path(path)
        #self.path_index = 0

        return self._get_obs()
    def reset2(self):
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Set obstacle positions as per the image
        obstacle_positions = [
            (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)
        ]
        for pos in obstacle_positions:
            self.grid[pos] = 1  # Place obstacle

        # Set start and goal positions
        self.start_pos = (4, 1)  # Green circle position
        self.goal_pos = (2, 4)  # Orange circle position
        while True:
            self.start_pos = (
                np.random.randint(low=0, high=self.grid_size-1, dtype='int32'),
                np.random.randint(low=0, high=1, dtype='int32'),
            )
            self.goal_pos = (
                np.random.randint(low=0, high=1, dtype='int32'),
                np.random.randint(low=0, high=self.grid_size-1, dtype='int32'),
            )
            start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
            goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))

            if (
                self.grid[start_cell[0], start_cell[1]] == 0
                and self.grid[goal_cell[0], goal_cell[1]] == 0
            ):
                path = self.astar(start_cell, goal_cell)
                if path is not None:
                    break
        start_cell = (int(self.start_pos[0]), int(self.start_pos[1]))
        goal_cell = (int(self.goal_pos[0]), int(self.goal_pos[1]))

        # Ensure there is a valid path (since the setup is fixed, a path must exist)
        path = self.astar(start_cell, goal_cell)
        if path is None:
            raise ValueError("No valid path found between start and goal in the fixed configuration")

        self.position = np.array(self.start_pos, dtype=np.float32)
        self.path = self.interpolate_path(path)
        self.path_index = 0
        return self._get_obs()
    def step(self, action=None):
        reward = 0
        if action is not None:
            if self.path is not None and self.path_index < len(self.path) - 1:
                #next_position = self.path[self.path_index + 1]
                next_position = self.position + action
                #action = np.array(next_position) - self.position
                self.path_index += 1
            
        else:
            if self.path is not None and self.path_index < len(self.path) - 1:
                next_position = self.path[self.path_index + 1]
                #next_position = self.position + action
                action = np.array(next_position) - self.position
                self.path_index += 1
            else:
                action = np.zeros(2, dtype=np.float32)

        self.position = np.asarray(self.position, dtype=np.float32).reshape(-1)
        new_position = self.position + action
        new_position = np.clip(new_position, 0, self.grid_size - 1e-5)
        new_position = np.asarray(new_position, dtype=np.float32).reshape(-1)

        # x_new = int(new_position[0].item())
        # y_new = int(new_position[1].item())
        x_new = float(new_position[0].item())
        y_new = float(new_position[1].item())
        collision = False
        if self.is_in_obstacle_area(x_new, y_new):
            collision = True
            new_position = self.position

        self.position = new_position
        #reward = -0.1 * np.linalg.norm(action)

        distance_to_goal = np.linalg.norm(self.position - self.goal_pos)
        done = False
        if distance_to_goal < 0.05:
            done = True
            reward += 100.0

        if collision:
            reward -= 10.0

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
        #distance_penalty = 0.0
        # if self.path_index > 0:
        #     previous_position = self.path[self.path_index - 1]
        #     distance = np.linalg.norm(np.array(previous_position) - np.array(next_position))
        #     if distance > self.points_distance:
        #         reward -=5.0
        info = {
            'distance_to_goal': distance_to_goal,
            'current_step': self.current_step,
            'collision': collision,
        }
        return self._get_obs(), reward, done, info

    def is_in_obstacle_area(self, x, y):
        """
        Checks if the given point is in the obstacle area considering padding.
        """
        for obstacle_x in range(self.grid_size):
            for obstacle_y in range(self.grid_size):
                if self.grid[obstacle_x, obstacle_y] == 1:
                    if (
                        obstacle_x - (1 + self.padding) / 2 <= x <= obstacle_x + (1 + self.padding) / 2
                        and obstacle_y - (1 + self.padding) / 2 <= y <= obstacle_y + (1 + self.padding) / 2
                    ):
                        return True
        return False

    def is_valid(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x, y] == 0
        return False

    def _get_obs(self):
        grid_flat = self.grid.flatten().astype(np.float32)
        return np.concatenate([
            self.position.copy(),
            np.array(self.goal_pos, dtype=np.float32),
            grid_flat
        ])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_dataset(self, num_episodes=100):
        data = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'rewards': [],
            'terminals': [],
            'timeouts': [],
        }

        for episode in range(num_episodes):
            obs = self.reset()
            done = False
            steps = 0
            while not done and steps < self._max_episode_steps:
                action = None
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
                    break

        for key in data:
            data[key] = np.array(data[key])

        return data

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def astar(self, start, goal):
        grid = self.grid
        grid_size = self.grid_size
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while not open_set.empty():
            _, current = open_set.get()

            if current == goal:
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
                    and grid[neighbor[0], neighbor[1]] == 0
                ):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, goal)
                        open_set.put((f_score, neighbor))
                        came_from[neighbor] = current

        return None

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def remove_duplicate_points(self, path, epsilon=1e-5):
        cleaned_path = [path[0]]
        for i in range(1, len(path)):
            if np.linalg.norm(path[i] - cleaned_path[-1]) > epsilon:
                cleaned_path.append(path[i])
        return np.array(cleaned_path)

    def interpolate_path(self, path, num_points=50):
        path = np.array(path, dtype=np.float32)
        if len(path) < 2:
            return path

        path = self.remove_duplicate_points(path)

        if len(path) < 2:
            return path

        x = path[:, 0]
        y = path[:, 1]
        distance = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))

        if np.all(distance == 0):
            return path

        unique_distances, unique_indices = np.unique(distance, return_index=True)
        x = x[unique_indices]
        y = y[unique_indices]
        distance = unique_distances

        if len(distance) < 2:
            return path

        distance = distance / distance[-1]
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
    dataset = []
    for _ in range(num_samples):
        grid_env = RandomObstacleEnv(grid_size=grid_size)

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

        available_positions = [pos for pos in available_positions if pos not in obstacles]
        if len(available_positions) < 2:
            continue

        start, goal = random.sample(available_positions, 2)
        path = grid_env.astar(start, goal)
        if path is None:
            continue

        dataset.append({
            "start": np.array(start, dtype=np.float32) + 0.5,
            "goal": np.array(goal, dtype=np.float32) + 0.5,
            "waypoints": np.array(path, dtype=np.float32) + 0.5,
            "optimized_waypoints": grid_env.interpolate_path(path) + 0.5,
            "grid": grid_env.grid.copy(),
        })

    return dataset
