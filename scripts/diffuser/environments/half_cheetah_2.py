import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from queue import PriorityQueue
import random

class HalfCheetahFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, grid_size=5):
        asset_path = os.path.join(os.path.dirname(__file__), 'assets/grid_obstacle.xml')
        mujoco_env.MujocoEnv.__init__(self, asset_path, frame_skip=5)
        utils.EzPickle.__init__(self)
        self.grid_size = grid_size
        self.grid_env = GridEnvironment(grid_size)
        self.start = None
        self.goal = None
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def reset_model(self):
        while True:
            self.grid_env.reset()
            num_obstacles = random.randint(0, self.grid_size * 2)  # Random number of obstacles
            for _ in range(num_obstacles):
                x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                self.grid_env.add_obstacle(x, y)

            available_positions = list(np.ndindex(self.grid_size, self.grid_size))
            random.shuffle(available_positions)
            self.start, self.goal = available_positions.pop(), available_positions.pop()

            # Check if there is a feasible path between start and goal
            path = astar_path(self.grid_env, self.start, self.goal)
            if path is not None:
                break  # Valid path found, proceed with the environment reset

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set(self, state):
        qpos_dim = self.sim.data.qpos.size
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]
        self.set_state(qpos, qvel)
        return self._get_obs()

class GridEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # Initialize grid with free cells

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))

    def add_obstacle(self, x, y):
        if self.is_valid(x, y):
            self.grid[int(x), int(y)] = 1  # Mark obstacle cells with 1

    def is_valid(self, x, y):
        x = int(x)
        y = int(y)
        return (
            0 <= x < self.grid_size and
            0 <= y < self.grid_size and
            self.grid[x, y] == 0  # Check if the cell is free
        )

def heuristic(a, b):
    """Heuristic function for A* (Manhattan distance)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_path(grid_env, start, goal):
    """Finds a path from start to goal using the A* algorithm."""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    open_set = PriorityQueue()
    open_set.put((0, tuple(start), [tuple(start)]))
    g_score = {tuple(start): 0}
    visited = set()

    while not open_set.empty():
        _, current_position, path = open_set.get()
        if current_position == tuple(goal):
            return np.array(path)
        visited.add(current_position)
        for direction in directions:
            new_position = (
                current_position[0] + direction[0],
                current_position[1] + direction[1],
            )
            if (
                grid_env.is_valid(new_position[0], new_position[1]) and
                new_position not in visited
            ):
                tentative_g_score = g_score[current_position] + 1
                if (
                    new_position not in g_score or
                    tentative_g_score < g_score[new_position]
                ):
                    g_score[new_position] = tentative_g_score
                    f_score = tentative_g_score + heuristic(new_position, goal)
                    open_set.put((f_score, new_position, path + [new_position]))

    return None  # Return None if no valid path is found

