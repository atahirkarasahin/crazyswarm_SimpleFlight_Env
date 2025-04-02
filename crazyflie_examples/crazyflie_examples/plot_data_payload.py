# Keys in first TensorDict: _StringKeys({'agents': TensorDict(
#     fields={
#         action: Tensor(shape=torch.Size([1, 1, 4]), device=cuda:0, dtype=torch.float32, is_shared=True),
#         drone_state: Tensor(shape=torch.Size([1, 28]), device=cuda:0, dtype=torch.float32, is_shared=True),
#         observation: Tensor(shape=torch.Size([1, 1, 42]), device=cuda:0, dtype=torch.float32, is_shared=True),
#         real_position: Tensor(shape=torch.Size([1, 3]), device=cuda:0, dtype=torch.float32, is_shared=True),
#         rewarstated: Tensor(shape=torch.Size([1, 1, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
#         state : Tensor(shape=torch.Size([1, 46]), device=cuda:0, dtype=torch.float32, is_shared=True),
#         target_position: Tensor(shape=torch.Size([1, 3]), device=cuda:0, dtype=torch.float32, is_shared=True)},
#     batch_size=torch.Size([1]),
#     device=cuda:0,
#     is_shared=True), 'done': tensor([[False]], device='cuda:0'), 'drone.action_logp': tensor([[[10.2182]]], 
# device='cuda:0'), 'state_value': tensor([[[-0.3227]]], device='cuda:0'), 'terminated': tensor([[[False]]], 
# device='cuda:0'), 'truncated': tensor([[[False]]], device='cuda:0')})

import torch
import numpy as np
import matplotlib.pyplot as plt

# Payload data
#Mean Euclidean distance (XY plane): 0.1209 m
data = torch.load("/home/taka/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/model/data/payload_3.pt")

# Extract position data from all TensorDicts
positions = torch.cat([d[("agents", "real_position")] for d in data], dim=0).cpu().numpy()
target_positions = torch.cat([d[("agents", "target_position")] for d in data], dim=0).cpu().numpy()
drone_state = torch.cat([d[("agents", "drone_state")] for d in data], dim=0).cpu().numpy()
action = torch.cat([d[("agents", "action")] for d in data], dim=0).cpu().numpy()

print("Action values: ", action[0,:])

print("Action values: ", action[-1,:])
# Extract x, y, z coordinates
x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]   
px, py, pz = drone_state[:, 28], drone_state[:, 29], drone_state[:, 30] 

vx, vy, vz = drone_state[..., 7], drone_state[..., 8], drone_state[..., 9]
p_vx, p_vy, p_vz = drone_state[..., 31], drone_state[..., 32], drone_state[..., 33]

target_x, target_y, target_z = target_positions[..., 0], target_positions[..., 1], target_positions[..., 2]

############# 3D Plot #############
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, label="Drone Trajectory", color="blue")
ax.scatter(x[0], y[0], z[0], color="green", label="Start")  # Start position
ax.scatter(x[-1], y[-1], z[-1], color="red", label="End")  # End position
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.legend()
plt.title("Drone Flight Trajectory")

plt.show()

############# 2D Plot #############

# Compute mean Euclidean distance in x and y
xy_distance = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
mean_xy_distance = np.mean(xy_distance)
print(f"Mean Euclidean distance (XY plane) Drone: {mean_xy_distance:.4f} m")

xy_distance_pay = np.sqrt((px - target_x) ** 2 + (py - target_y) ** 2)
mean_xy_distance_pay = np.mean(xy_distance_pay)
print(f"Mean Euclidean distance (XY plane) Payload: {mean_xy_distance_pay:.4f} m")

# Compute velocity magnitude
velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
payload_velocity_magnitude = np.sqrt(p_vx**2 + p_vy**2 + p_vz**2)
print(f"Mean velocity magnitude (Drone): {np.mean(velocity_magnitude):.4f} m/s")
print(f"Mean velocity magnitude (Payload): {np.mean(payload_velocity_magnitude):.4f} m/s")

# Normalize velocity for color mapping
norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())

# Create 3D figure
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

fig, ax = plt.subplots(figsize=(8, 6))
# Scatter plot with velocity-based color mapping
sc = ax.scatter(px, py, c=payload_velocity_magnitude, cmap='jet', norm=norm, marker='o', label="Actual Payload Trajectory")

#ax.scatter(x, y, c=velocity_magnitude, cmap='viridis', norm=norm, marker='o', label="Actual Drone Trajectory")

ax.plot(target_x, target_y, color='red', label="Reference Trajectory")

# Mark the initial position with a large red star
#ax.scatter(x[0], y[0], color='red', marker='*', s=200, label="Initial Position")

# Add color bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label(r"$ \text{v } \left [ \text{m} \text{s}^{-1} \right]$", fontsize=16, labelpad=10)


# Start and End Markers
ax.scatter(x[0], y[0], color="green", edgecolors="black", s=100, label="Start")
ax.scatter(x[-1], y[-1], color="red", edgecolors="black", s=100, label="End")

# Change colorbar tick size
cbar.ax.tick_params(labelsize=14)  # Set the font size of the tick labels

# Color bar for velocity
ax.set_xlabel(r"$ \text{ x [m]}$", fontsize=16, labelpad=15)
ax.set_ylabel(r"$ \text{ y [m]}$",fontsize=16, labelpad=10)
ax.tick_params(axis='both', labelsize=16)  # Adjusts font size for both x and y axis ticks
    
# Add legend
ax.legend(fontsize=16)

#ax.set_title("Drone XY Trajectory with Velocity")
plt.grid()
plt.show()



# # Compute Euclidean distances for Box Plot

# dx = positions[:, 0] - target_positions[:, 0]
# dy = positions[:, 1] - target_positions[:, 1]
# distances = np.sqrt(dx**2 + dy**2)
# mean_distance = np.mean(distances)


# # Create box plot for Euclidean distances
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.boxplot(distances, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"), whis=3.0)
# ax.scatter(1, mean_distance, color='red', marker='*' ,zorder=3)

# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

# ax.set_ylabel("Euclidean Distance (m)")
# ax.legend()
# plt.grid()
# plt.show()