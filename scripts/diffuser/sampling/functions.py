import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)

def save_gradient_heatmap(gradient, step, save_path):
    """
    Save the gradient as a heatmap.
    
    :param gradient: Tensor of gradients
    :param step: Current denoising step
    :param save_path: File path to save the heatmap
    """
    grad_np = gradient.squeeze().detach().cpu().numpy()

    # Plot the gradient heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(grad_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Gradient Magnitude')
    plt.title(f'Gradient Heatmap (Step {step + 1})')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Batch Index' if grad_np.ndim > 1 else 'Feature Index')
    plt.tight_layout()

    # Save the heatmap
    heatmap_path = os.path.join(save_path, f'grad_heatmap_step_{step + 1:03d}.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f'Saved gradient heatmap for step {step + 1} to {heatmap_path}')


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, save_path='logs_png/denoising_steps'
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)
    std = torch.tensor([
        1.6263416e+00, 1.6851393e+00, 1.3706232e+00, 1.4794788e+00, 
        9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
        9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
        9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
        9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
        9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
        9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 9.9999999e-09, 
        9.9999999e-09
    ], dtype=torch.float32)

    mean = torch.tensor([
        2.3332138, 2.4282227, 1.9322315, 2.001556, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 
        0.0
    ], dtype=torch.float32)
    std = std.to('cuda')
    mean = mean.to('cuda')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for step in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)
        #save_gradient_heatmap(grad, step, 'grad_plot_2/grad_plotting_0')
        if scale_grad_by_std:
            grad = model_var * grad
        
        grad[t < t_stopgrad] = 0
        x = x + scale * grad

        
        #print("gradient",grad[:, :, 2:4]*std[:2] + mean[:2])
        x = apply_conditioning(x, cond, model.action_dim)
        #print("shape of denoising",x.shape)
        #print("denoising_raw_trajectory",x[:, :, 2:4]*std[:2]+mean[:2])
        denoising_trajectory = (x[:, :, 2:4]).detach().cpu().numpy()
        df = pd.DataFrame(denoising_trajectory[0], columns=["x_position", "y_position"])  # For first batch
        df.to_csv("denoising_trajectory.csv", index=False)
        # Clone the tensor to avoid overwriting during denoising steps
        x_clone = x.clone()

        # Render and save the current state as an image
        #render_trajectory_step(x_clone, step, save_path)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

def render_trajectory_step(x, step, save_path):
    x_np = x.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    grid_size = int(np.sqrt(x_np.shape[-1]))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.grid(True)

    # Assuming x_np represents positions on a grid (for trajectory visualization)
    x_positions = x_np[0, :, 0]
    y_positions = x_np[0, :, 1]
    ax.plot(x_positions + 0.5, y_positions + 0.5, marker='o', color='blue', markersize=4)

    # Draw starting position
    start_circle = plt.Circle((x_positions[0] + 0.5, y_positions[0] + 0.5), 0.3, color='red')
    ax.add_patch(start_circle)

    # Draw goal position (assuming a fixed goal for this example)
    goal_circle = plt.Circle((grid_size - 1 + 0.5, 0.5), 0.3, color='green')
    ax.add_patch(goal_circle)

    ax.set_title(f'Trajectory Step {step + 1}')
    plt.axis('equal')
    plt.tight_layout()

    # Save the image with unique filename including the step number
    image_path = os.path.join(save_path, f'trajectory_step_{step + 1:03d}.png')
    plt.savefig(image_path)
    plt.close(fig)
    print(f'Saved trajectory step {step + 1} to {image_path}')






@torch.no_grad()
def n_step_guided_p_sample_backup(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad
        #print("scale", scale)
        grad[t < t_stopgrad] = 0
        #print("x_prev",x)
        x = x + scale * grad
        
        x = apply_conditioning(x, cond, model.action_dim)
        #print("x",x)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y
@torch.no_grad()
def n_step_guided_p_sample_check(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)
            
        # Log the guidance gradient and the state before applying it
        print(f"Step: t={t} | Gradient Norm: {torch.norm(grad)} | State (before guidance): {x}")
        
        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0
        x = x + scale * grad
        
        # Log the state after applying guidance
        print(f"Step: t={t} | State (after guidance): {x}")

        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # Log the model mean and std deviation
    print(f"Model Mean: {model_mean} | Model Std: {model_std}")

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y
