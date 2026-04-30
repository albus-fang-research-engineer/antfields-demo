import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

epoch = 100
save_plot = True
base_path = 'pair1/baseline_success'

# data = np.load(f'{base_path}/field_epoch_100.npz')
data = np.load(f'{base_path}/field_epoch_{epoch}.npz')
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
speed[speed < 0.76] -= 0.56
speed[speed < 0.90] -= 0.026
speed[speed > 0.92] += 0.02
# Gaussian (smoother, more natural)
speed = gaussian_filter(speed, sigma=1.8)  # increase sigma for more smoothing
def get_traversed_path(current_epoch, base_path=base_path):
    if current_epoch == 50:
        return None
    
    segments = []
    epoch = 50
    while epoch < current_epoch:
        traj = np.load(f'{base_path}/planned_path_{epoch}.npy')
        next_traj = np.load(f'{base_path}/planned_path_{epoch+50}.npy')
        next_start = next_traj[0]
        
        dists = np.linalg.norm(traj - next_start, axis=1)
        idx = np.argmin(dists)
        segments.append(traj[:idx+1])  # up to and including the next start point
        
        epoch += 50
    
    return np.concatenate(segments, axis=0)

# traj = np.load(f'{base_path}/epoch_0100_optimized.npy')
traj = np.load(f'{base_path}/planned_path_{epoch}.npy')
# plt.figure(figsize=(8, 6))
# x_range = 0.16 - (-0.23)  # 0.39
# y_range = 0.05 - (-0.15)  # 0.20
# aspect = x_range / y_range  # ~1.95

# plt.figure(figsize=(6 * aspect, 6))
plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.set_aspect('equal')
plt.pcolormesh(X, Y, speed, cmap='viridis', shading='auto', vmin=0, vmax=1)
plt.colorbar(label='Speed')

# If traj is shape (N, 2) — x in col 0, y in col 1
plt.plot(traj[:, 0], traj[:, 1], color='pink', linewidth=1.5, alpha=0.9, label='Optimized Planned Trajectory')
# plt.scatter(traj[0, 0], traj[0, 1], color='cyan', zorder=5, s=36, label='Start')
plt.scatter(traj[-1, 0], traj[-1, 1], color='cyan', marker='*', zorder=5, s=260, label='Goal')
# start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105, 
#                        color='black', fill=False, linewidth=2, zorder=5, label='Turtlebot')
# plt.gca().add_patch(start_circle)
# plt.scatter(traj[0, 0], traj[0, 1], color='cyan', zorder=5, s=36, label='Start')
start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105, 
                        color='black', fill=False, linewidth=2, zorder=5)
ax.add_patch(start_circle)
ax.scatter([], [], facecolors='none', edgecolors='black', linewidths=2, s=100, label='Turtlebot')
# obstacles = np.load(f'{base_path}/obstacle_points_50.npy')
obstacles = np.load(f'{base_path}/surface_points_{epoch}.npy')
plt.scatter(obstacles[:, 0], obstacles[:, 1], color='red', s=2, zorder=5, label='Detected Obstacle Points')

# traversed = get_traversed_path(100)  # change 50 to whatever current epoch you're plotting
traversed = get_traversed_path(epoch)
# plt.plot(traversed[:, 0], traversed[:, 1], color='blue', linewidth=2, 
#          alpha=0.8, label='Traversed Path', zorder=4)
if traversed is not None:
    plt.plot(traversed[:, 0], traversed[:, 1], color='blue', linewidth=2, 
             alpha=0.8, label='Traversed Path', zorder=4)
start = traversed[0] if traversed is not None else traj[0]
plt.scatter(start[0], start[1], color='cyan', zorder=5, s=36, label='Start')
travel_time = gaussian_filter(data['travel_time'], sigma=1.6)
# travel_time = data['travel_time']
plt.contour(X, Y, travel_time, levels=30, colors='black', linewidths=0.5, alpha=0.6)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.16), ncol=1, borderaxespad=0)
handles, labels = ax.get_legend_handles_labels()
# Define the desired order by label name
desired_order = [
    'Optimized Planned Trajectory',
    'Traversed Path',
    'Start',
    'Goal',
    'Turtlebot',
    'Detected Obstacle Points',
]

# Build ordered lists (skips any label not present, e.g. if traversed is None)
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[l] for l in desired_order if l in label_to_handle]
ordered_labels  = [l for l in desired_order if l in label_to_handle]

ax.legend(ordered_handles, ordered_labels,
          loc='lower center', bbox_to_anchor=(0.5, 1.2),
          ncol=1, borderaxespad=0)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Speed over X/Y Grid')
plt.xlim(-0.25, 0.15)
plt.ylim(-0.15, 0.05)
# plt.tight_layout()
plt.tight_layout(rect=[0, 0, 0.75, 1])  # was plt.tight_layout()
# plt.savefig(f'{base_path}/epoch_100_field_and_path')
if save_plot:
    plt.savefig(f'pair1/plots/baseline_epoch_{epoch}_field_and_path')
plt.show()