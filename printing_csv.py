import pandas as pd
import torch

# Load the CSV file
df = pd.read_csv("denoising_trajectory.csv")

# Convert CSV data to a PyTorch tensor
positions = torch.tensor(df.values, dtype=torch.float32)  # Shape: (256, 2)

# Example of new std and mean values
new_std = torch.tensor([1.62, 1.68], dtype=torch.float32)  # New standard deviation
new_mean = torch.tensor([2.81, 2.85], dtype=torch.float32)  # New mean

# Recalculate the trajectory with the new std and mean
new_trajectory = (positions*new_std) + new_mean

# Print the recalculated values
print("Recalculated Trajectory:")
print(new_trajectory)

# Optionally save the recalculated trajectory to a new CSV file
#new_df = pd.DataFrame(new_trajectory.numpy(), columns=["x_position", "y_position"])
#new_df.to_csv("recalculated_trajectory.csv", index=False)
print(f"Recalculated trajectory saved to recalculated_trajectory.csv")