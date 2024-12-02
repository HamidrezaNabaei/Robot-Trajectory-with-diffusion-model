import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
import os
import numpy as np

class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        #print("grad_shape", grad.shape)
        std = torch.tensor([
            1.5686610e+00, 1.6020948e+00, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
            9.9999999e-09, 9.9999999e-09, 9.9999999e-09
        ], dtype=torch.float32)

        mean = torch.tensor([
            2.3058867, 1.8427615, 4.0, 4.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0
        ], dtype=torch.float32)
        std = std.to('cuda')
        mean = mean.to('cuda')

        extracted_grad = grad[0, :, 5:30]  # Shape: (256, 25)

        # Reshape to 5x5 grid for each sample
        grid_map_grad = extracted_grad.view(-1, 5, 5)  # Shape: (256, 5, 5)

        # Transpose the grid for each sample
        transposed_grid = grid_map_grad.permute(0, 2, 1)  # Transpose rows and columns (Shape: 256, 5, 5)

        # Flatten the transposed grid back into a 1D array for each sample
        flattened_grid = transposed_grid.reshape(grad.size(1), -1)  # Shape: (256, 25)

        # Add the flattened grid back to a gradient-like tensor
        updated_gradient = torch.cat([grad[0, :, :5], flattened_grid, grad[0, :, 30:]], dim=1)  # Shape: (256, 31)
        updated_gradient = updated_gradient.unsqueeze(0)  # Restore batch dimension (1, 256, 31)
        #grad = grad * std + mean
        #grad = grad * 100000000
        #save_path = self.generate_unique_filename('grad_plot/grad_0')
        #self.save_grid_map_heatmaps_with_unnormalized_markers(grad, save_path, mean, std)
        #self.save_grid_map_heatmaps(grad, save_path)
        x.detach()
        return y, grad
	

    # @staticmethod
    # def save_gradient_heatmap(gradient, save_path):
    #     """
    #     Save the gradient as a heatmap.
        
    #     :param gradient: Tensor of gradients
    #     :param save_path: File path to save the heatmap
    #     """
    #     # Move the gradient to the CPU and convert to numpy
    #     grad_np = gradient.squeeze().detach().cpu().numpy()
        
    #     # Plot the gradient heatmap
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(grad_np, cmap='viridis', aspect='auto')
    #     plt.colorbar(label='Gradient Magnitude')
    #     plt.title('Gradient Heatmap')
    #     plt.xlabel('Feature Dimension')
    #     plt.ylabel('Batch Index' if grad_np.ndim > 1 else 'Feature Index')
    #     plt.tight_layout()
        
    #     # Save the heatmap as an image
    #     plt.savefig(save_path)
    #     plt.close()

    # @staticmethod
    # def generate_unique_filename(file_path):
    #     """
    #     Generate a unique filename by appending a number if the file already exists.
        
    #     :param file_path: Original file path
    #     :return: Unique file path
    #     """
    #     base, ext = os.path.splitext(file_path)
    #     counter = 1

    #     # Check for existing file and modify the name if necessary
    #     while os.path.exists(file_path):
    #         file_path = f"{base}_{counter}{ext}"
    #         counter += 1
        
    #     return file_path

    def save_grid_map_heatmaps_with_unnormalized_markers(self, gradient, save_path, mean, std):
        """
        Save the grid map portion of the gradient as heatmaps and overlay
        unnormalized current position and goal markers extracted from the gradient.

        :param gradient: Tensor of gradients (e.g., shape [1, 256, 31])
        :param step: Current denoising step
        :param save_path: Base path to save the heatmaps
        :param mean: Array of mean values for unnormalization
        :param std: Array of standard deviation values for unnormalization
        """
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Move the gradient to CPU and convert to NumPy
        gradient = gradient.detach().cpu().numpy()
        
        # Extract the grid map portion (last 25 elements)
        grid_map_grad = gradient[0, :, -27:-2]  # Shape: [256, 5, 5]
        grid_map_grad = grid_map_grad * std[-27:-2] + mean[-27:-2]
        grid_map_grad = grid_map_grad.reshape(-1, 5, 5)  # Shape: [256, 5, 5]

        # Extract current positions and goals from the gradient
        current_positions = gradient[0, :, 2:4]  # Shape: [256, 2] (x, y)
        goals = gradient[0, :, 4:6]             # Shape: [256, 2] (x, y)

        # Unnormalize current positions and goals
        current_positions = current_positions * std[:2] + mean[:2]
        goals = goals * std[2:4] + mean[2:4]
    

        # Save a heatmap for each sample
        for i, grad_map in enumerate(grid_map_grad):
            plt.figure(figsize=(5, 5))
            plt.imshow(grad_map.T, cmap='viridis', aspect='auto')
            plt.colorbar(label='Gradient Magnitude')
            plt.title(f'Grid Map Heatmap - Sample {i + 1}')
            plt.gca().invert_yaxis()
            plt.xticks(ticks=np.arange(0, 5), labels=np.arange(0, 5))
            plt.yticks(ticks=np.arange(0, 5), labels=np.arange(0, 5))
            plt.xlabel('X-axis (Columns)')
            plt.ylabel('Y-axis (Rows)')

            # Overlay unnormalized current position
            current_x, current_y = current_positions[i]
            plt.plot(current_x + 0.5, current_y + 0.5, 'ro', label='Current Position')  # Red marker

            # Overlay unnormalized goal
            goal_x, goal_y = goals[i]
            plt.plot(goal_x - 0.5, goal_y - 0.5, 'go', label='Goal')  # Green marker

            plt.legend(loc='upper right')
            plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5)


            # Save the heatmap
            heatmap_path = f"{save_path}_sample_{i + 1:03d}.png"
            plt.savefig(heatmap_path)
            plt.close()
        print(f"Saved grid map heatmaps with unnormalized markers to {os.path.dirname(save_path)}")



    @staticmethod
    def save_grid_map_heatmaps(gradient,  save_path):
        """
        Save the grid map portion of the gradient as heatmaps and overlay
        current position and goal markers extracted from the gradient.
        
        :param gradient: Tensor of gradients (e.g., shape [1, 256, 31])
        :param step: Current denoising step
        :param save_path: Base path to save the heatmaps
        """
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Move the gradient to CPU and convert to NumPy
        gradient = gradient.detach().cpu().numpy()
        
        # Extract the grid map portion (last 25 elements)
        grid_map_grad = gradient[0, :, -25:].reshape(-1, 5, 5)  # Shape: [256, 5, 5]

        # Extract current positions and goals from the gradient
        current_positions = gradient[0, :, 2:4]  # Shape: [256, 2] (x, y)
        actions = gradient[0,:,:2]
        #print("actions", actions)
        #print("cur_pos",current_positions)
        goals = gradient[0, :, 4:6]             # Shape: [256, 2] (x, y)
        #print("goals", goals)
        #print("grid_maps", grid_map_grad)
        # Save a heatmap for each sample
        for i, grad_map in enumerate(grid_map_grad):
            plt.figure(figsize=(5, 5))
            plt.imshow(grad_map, cmap='viridis', aspect='auto')
            plt.colorbar(label='Gradient Magnitude')
            plt.title(f'Grid Map Heatmap - Sample {i + 1}')

            # Overlay current position
            current_x, current_y = current_positions[i]
            plt.plot(current_x + 0.5, current_y + 0.5, 'ro', label='Current Position')  # Red marker

            # Overlay goal
            goal_x, goal_y = goals[i]
            plt.plot(goal_x + 0.5, goal_y + 0.5, 'go', label='Goal')  # Green marker

            plt.legend(loc='upper right')
            plt.axis('off')

            # Save the heatmap
            heatmap_path = f"{save_path}_sample_{i + 1:03d}.png"
            plt.savefig(heatmap_path)
            plt.close()
        print(f"Saved grid map heatmaps with markers to {os.path.dirname(save_path)}")


    @staticmethod
    def generate_unique_filename(file_path):
        """
        Generate a unique filename by appending a number if the file already exists.
        
        :param file_path: Original file path
        :return: Unique file path
        """
        base, ext = os.path.splitext(file_path)
        counter = 1

        # Check for existing file and modify the name if necessary
        while os.path.exists(file_path):
            file_path = f"{base}_{counter}{ext}"
            counter += 1
        
        return file_path