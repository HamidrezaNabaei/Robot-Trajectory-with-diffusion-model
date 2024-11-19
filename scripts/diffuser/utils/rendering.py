import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb
import matplotlib.patches as patches
from gym import Env
from .arrays import to_np
from .video import save_video, save_videos
from matplotlib.animation import FuncAnimation
import copy
from diffuser.datasets.d4rl import load_environment

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        Map D4RL dataset names to custom fully-observed
        variants for rendering.
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    elif 'grid' in env_name:
        return 'gridworld-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class GridWorldRenderer:
    def __init__(self, env):
        env = env_map(env)
        env = gym.make(env)
        if not isinstance(env, gym.Env):
            raise ValueError("env must be an instance of gym.Env")
        self.env = env

    def render(self, savepath=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        grid_size = self.env.grid_size
        grid = self.env.grid
        position = self.env.position
        goal_pos = self.env.goal_pos

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size + 1))
        ax.set_yticks(np.arange(0, grid_size + 1))
        ax.grid(True)

        # Draw obstacles
        obstacle_positions = np.argwhere(grid == 1)
        for pos in obstacle_positions:
            rect = patches.Rectangle(pos, 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        # Draw agent
        agent_circle = patches.Circle((position[0] + 0.5, position[1] + 0.5), 0.3, color='blue')
        ax.add_patch(agent_circle)

        # Draw goal
        goal_circle = patches.Circle((goal_pos[0] + 0.5, goal_pos[1] + 0.5), 0.3, color='green')
        ax.add_patch(goal_circle)

        ax.set_title('RandomObstacleEnv')
        plt.axis('equal')
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)
            plt.close(fig)
            print(f'Rendered environment saved to {savepath}')
        else:
            plt.show()

    def render_plan(self, savepath, actions, observations_pred, state, fps=5):
        """
        Renders the full planned trajectory and saves it into one PNG image.

        Args:
            savepath (str): Path to save the image.
            actions (np.ndarray): Array of planned actions.
            observations_pred (np.ndarray): Array of predicted observations.
            state (np.ndarray): Initial state vector.
            fps (int): Frames per second for the animation (not used here).
        """

        # Extract environment information from the observation
        grid_size = int(np.sqrt(len(state[4:])))
        grid = state[4:].reshape((grid_size, grid_size))
        position = state[:2]
        goal_pos = state[2:4]

        # Simulate the planned actions and collect positions
        positions = [position.copy()]
        for action in actions:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (2,):
                raise ValueError(f"Expected action of shape (2,), but got {action.shape}")
            new_position = positions[-1] + action
            new_position = np.clip(new_position, 0, grid_size - 1e-5)
            x_new, y_new = int(new_position[0]), int(new_position[1])
            if grid[x_new, y_new] == 1:  # If there's an obstacle, stay in place
                new_position = positions[-1]
            positions.append(new_position)

        # Render the full trajectory in one image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size + 1))
        ax.set_yticks(np.arange(0, grid_size + 1))
        ax.grid(True)

        # Draw obstacles
        obstacle_positions = np.argwhere(grid == 1)
        for pos in obstacle_positions:
            rect = plt.Rectangle(pos, 1, 1, color='black')
            ax.add_patch(rect)

        # Draw goal
        goal_circle = plt.Circle((goal_pos[0] + 0.5, goal_pos[1] + 0.5), 0.3, color='green')
        ax.add_patch(goal_circle)

        # Draw the full path
        positions = np.array(positions)
        x = positions[:, 0] + 0.5
        y = positions[:, 1] + 0.5
        ax.plot(x, y, marker='o', color='blue')

        # Draw starting position
        start_circle = plt.Circle((x[0], y[0]), 0.3, color='red')
        ax.add_patch(start_circle)

        ax.set_title('Planned Trajectory')
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close(fig)
        print(f'Plan image saved to {savepath}')

    def composite(self, savepath, paths):
        """
        Create a composite visualization of multiple paths.

        Args:
            savepath (str): Path to save the composite image.
            paths (list of arrays): A list of sequences of observations.
        """
        import matplotlib.pyplot as plt

        # Prepare the figure and axes for composite visualization
        num_paths = len(paths)
        cols = 5
        rows = (num_paths + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        axes = axes.flatten()

        for idx, (ax, path) in enumerate(zip(axes, paths)):
            if len(path) == 0:
                continue

            # Extract environment information from the first observation
            grid_size = int(np.sqrt(len(path[0][4:])))
            grid = path[0][4:].reshape((grid_size, grid_size))
            goal_pos = path[0][2:4]

            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_xticks(np.arange(0, grid_size + 1))
            ax.set_yticks(np.arange(0, grid_size + 1))
            ax.grid(True)

            # Draw obstacles
            obstacle_positions = np.argwhere(grid == 1)
            for pos in obstacle_positions:
                rect = plt.Rectangle(pos, 1, 1, color='black')
                ax.add_patch(rect)

            # Draw goal
            goal_circle = plt.Circle((goal_pos[0] + 0.5, goal_pos[1] + 0.5), 0.3, color='green')
            ax.add_patch(goal_circle)

            # Draw the full path
            positions = np.array([obs[:2] for obs in path])
            x = positions[:, 0] + 0.5
            y = positions[:, 1] + 0.5
            ax.plot(x, y, marker='o', color='blue')

            # Draw starting position
            start_circle = plt.Circle((x[0], y[0]), 0.3, color='red')
            ax.add_patch(start_circle)

            ax.set_title(f'Path {idx + 1}')

        # Hide any unused subplots
        for idx in range(num_paths, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)
            plt.close(fig)
            print(f'Composite image saved to {savepath}')
        else:
            plt.show()



class MuJoCoRenderer:
    '''
        Default MuJoCo renderer.
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        # -1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        # xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:, None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, grid_shape=(2, 5)):
        """
        Create a composite image showing multiple paths arranged in a grid.

        Args:
            savepath (str): Path to save the composite image.
            paths (list of arrays): A list of sequences of observations.
            grid_shape (tuple): Tuple indicating the grid layout (rows, columns).
        """
        import matplotlib.pyplot as plt

        num_paths = len(paths)
        rows, cols = grid_shape
        assert num_paths <= rows * cols, "Number of paths exceeds grid capacity"

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        for idx, (ax, path) in enumerate(zip(axes, paths)):
            grid_size = self.env.grid_size
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_xticks(np.arange(0, grid_size + 1))
            ax.set_yticks(np.arange(0, grid_size + 1))
            ax.grid(True)

            # Draw obstacles
            obstacle_positions = np.argwhere(self.env.grid == 1)
            for pos in obstacle_positions:
                rect = plt.Rectangle(pos, 1, 1, color='black')
                ax.add_patch(rect)

            # Draw goal
            goal_circle = plt.Circle((self.env.goal_pos[0] + 0.5, self.env.goal_pos[1] + 0.5), 0.3, color='green')
            ax.add_patch(goal_circle)

            # Draw the path
            positions = np.array([obs[0:2] for obs in path])
            x = positions[:, 0] + 0.5
            y = positions[:, 1] + 0.5
            ax.plot(x, y, marker='o', color='blue')

            # Draw starting position
            start_circle = plt.Circle((x[0], y[0]), 0.3, color='red')
            ax.add_patch(start_circle)

            ax.set_title(f'Example {idx + 1}')
        # Hide any unused subplots
        for idx in range(num_paths, rows * cols):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)
            plt.close(fig)
            print(f'Composite image saved to {savepath}')
        else:
            plt.show()

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            # [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        # If terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations)

