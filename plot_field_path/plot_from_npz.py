import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
data = np.load('chance_constrained_plotting/field_epoch_50.npz')
from matplotlib.patches import Circle
# See all array names/keys
print(data.files)        # e.g. ['arr_0', 'arr_1', 'x', 'y']

# Access a specific array by name
print(data['X'].shape)         # the array itself
print(data['Y'].shape)   # its shape
# print(data['x'].dtype)   # its data type
import matplotlib.pyplot as plt


X = data['X']
Y = data['Y']
speed = data['speed']
# Gaussian (smoother, more natural)
speed = gaussian_filter(speed, sigma=1.6)  # increase sigma for more smoothing
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(X, Y, speed, cmap='viridis', shading='auto', vmin=0, vmax=1)
# plt.colorbar(label='Speed')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Speed over X/Y Grid')
# plt.xlim(-0.25, 0.15)
# plt.ylim(-0.1, 0.05)
# plt.tight_layout()
# plt.show()

traj = np.load('chance_constrained_plotting/epoch_0050_optimized.npy')

# plt.figure(figsize=(8, 6))
# x_range = 0.16 - (-0.23)  # 0.39
# y_range = 0.05 - (-0.15)  # 0.20
# aspect = x_range / y_range  # ~1.95

# plt.figure(figsize=(6 * aspect, 6))
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_aspect('equal')
plt.pcolormesh(X, Y, speed, cmap='viridis', shading='auto', vmin=0, vmax=1)
plt.colorbar(label='Speed')

# If traj is shape (N, 2) — x in col 0, y in col 1
plt.plot(traj[:, 0], traj[:, 1], color='pink', linewidth=1.5, alpha=0.9, label='Optimized Planned Trajectory')
# plt.scatter(traj[0, 0], traj[0, 1], color='cyan', zorder=5, s=36, label='Start')
plt.scatter(traj[-1, 0], traj[-1, 1], color='purple', marker='*', zorder=5, s=200, label='Goal')
# start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105, 
#                        color='black', fill=False, linewidth=2, zorder=5, label='Turtlebot')
# plt.gca().add_patch(start_circle)
plt.scatter(traj[0, 0], traj[0, 1], color='cyan', zorder=5, s=36, label='Start')
start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105, 
                        color='black', fill=False, linewidth=2, zorder=5)
ax.add_patch(start_circle)
ax.scatter([], [], facecolors='none', edgecolors='black', linewidths=2, s=100, label='Turtlebot')
travel_time = gaussian_filter(data['travel_time'], sigma=1.6)
plt.contour(X, Y, travel_time, levels=60, colors='black', linewidths=0.5, alpha=0.6)
# plt.legend()
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.16), ncol=1, borderaxespad=0)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Speed over X/Y Grid')
plt.xlim(-0.25, 0.15)
plt.ylim(-0.15, 0.05)
# plt.tight_layout()
plt.tight_layout(rect=[0, 0, 0.75, 1])  # was plt.tight_layout()
plt.show()