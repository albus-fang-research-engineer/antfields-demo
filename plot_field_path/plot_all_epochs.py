import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.pyplot as plt
import trimesh
from matplotlib.lines import Line2D
# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = [50, 100, 150, 200, 250]
SAVE_PLOT  = True
BASE_PATH  = 'chance_constrained_plotting'
MESH_PATH  = 'gibson/mesh.obj'
CEILING_Z  = 0.02
FLOOR_Z    = -0.12

# ── Build wall / floor geometry once ─────────────────────────────────────────
mesh_3d    = trimesh.load(MESH_PATH, force='mesh')
normals    = mesh_3d.face_normals
wall_mask  = np.abs(normals[:, 2]) < 0.3
wall_faces = mesh_3d.faces[wall_mask]

edges = np.vstack([
    wall_faces[:, [0, 1]],
    wall_faces[:, [1, 2]],
    wall_faces[:, [0, 2]],
])
edges     = np.unique(np.sort(edges, axis=1), axis=0)
z_verts   = mesh_3d.vertices[:, 2]
edge_max_z = z_verts[edges].max(axis=1)
edges     = edges[edge_max_z < 0.05]
verts     = mesh_3d.vertices[:, :2]
segments_2d = verts[edges]

face_z    = mesh_3d.vertices[mesh_3d.faces, 2]
floor_mask = (
    (np.abs(normals[:, 2]) > 0.7) &
    (face_z.mean(axis=1) < FLOOR_Z + 0.05)
)
vertices_2d = mesh_3d.vertices[:, :2]
floor_polys = vertices_2d[mesh_3d.faces[floor_mask]]

# ── Helper: build traversed path ─────────────────────────────────────────────
def get_traversed_path(current_epoch):
    if current_epoch == 50:
        return None
    segments = []
    epoch = 50
    while epoch < current_epoch:
        traj      = np.load(f'{BASE_PATH}/epoch_{epoch:04d}_optimized.npy')
        next_traj = np.load(f'{BASE_PATH}/epoch_{epoch+50:04d}_optimized.npy')
        next_start = next_traj[0]
        dists = np.linalg.norm(traj - next_start, axis=1)
        idx   = np.argmin(dists)
        segments.append(traj[:idx+1])
        epoch += 50
    return np.concatenate(segments, axis=0)

# ── Figure layout ─────────────────────────────────────────────────────────────
n_epochs = len(EPOCHS)
fig, axes = plt.subplots(1, n_epochs, figsize=(5 * n_epochs, 5),
                         constrained_layout=False)

legend_handles = None
legend_labels  = None

for ax, epoch in zip(axes, EPOCHS):
    # ── Load data ──────────────────────────────────────────────────────────
    data  = np.load(f'{BASE_PATH}/field_epoch_{epoch}.npz')
    X     = data['X']
    Y     = data['Y']
    speed = data['speed'].copy()

    speed[speed < 0.76] -= 0.56
    speed[speed < 0.90] -= 0.026
    speed[speed > 0.92] += 0.02
    speed = gaussian_filter(speed, sigma=1.8)

    traj      = np.load(f'{BASE_PATH}/epoch_{epoch:04d}_optimized.npy')
    obstacles = np.load(f'{BASE_PATH}/obstacle_points_{epoch}.npy')
    traversed = get_traversed_path(epoch)
    travel_time = gaussian_filter(data['travel_time'], sigma=1.6)

    # ── Draw ───────────────────────────────────────────────────────────────
    ax.set_aspect('equal')

    col = PolyCollection(floor_polys, facecolors='white', linewidths=0.0,
                         alpha=0.05, zorder=1)
    ax.add_collection(col)

    lc = LineCollection(segments_2d, colors='white', linewidths=0.6,
                        alpha=0.6, zorder=2)
    ax.add_collection(lc)

    pcm = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto',
                        vmin=0, vmax=1, alpha=0.9)

    ax.plot(traj[:, 0], traj[:, 1], color='pink', linewidth=1.5, alpha=0.9,
            label='Optimized Planned Trajectory')

    ax.scatter(traj[-1, 0], traj[-1, 1], color='cyan', marker='*',
               zorder=5, s=260, label='Goal')

    start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105,
                           color='black', fill=False, linewidth=2, zorder=5)
    ax.add_patch(start_circle)
    ax.scatter([], [], facecolors='none', edgecolors='black',
               linewidths=2, s=100, label='Turtlebot')

    ax.scatter(obstacles[:, 0], obstacles[:, 1], color='red', s=2,
               zorder=5, label='Detected Obstacle Points')

    if traversed is not None:
        ax.plot(traversed[:, 0], traversed[:, 1], color='blue', linewidth=2,
                alpha=0.8, label='Traversed Path', zorder=4)

    start = traversed[0] if traversed is not None else traj[0]
    ax.scatter(start[0], start[1], color='cyan', zorder=5, s=36, label='Start')

    ax.contour(X, Y, travel_time, levels=60, colors='black',
               linewidths=0.5, alpha=0.6)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val * 10:.2g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val * 10:.2g}'))

    ax.set_xlim(-0.25, 0.15)
    ax.set_ylim(-0.15, 0.05)
    ax.set_xlabel('X (m)')
    ax.set_title(f'Epoch {epoch}')

    if epoch == EPOCHS[0]:                 # only leftmost gets Y label
        ax.set_ylabel('Y (m)')

    # ── Grab legend from first non-50 epoch ───────────────────────────────
    if legend_handles is None and epoch != 50:
        desired_order = [
            'Optimized Planned Trajectory',
            'Traversed Path',
            'Start',
            'Goal',
            'Turtlebot',
            'Detected Obstacle Points',
        ]
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        legend_handles = [label_to_handle[l] for l in desired_order if l in label_to_handle]
        legend_labels  = [l for l in desired_order if l in label_to_handle]

# ── Shared colorbar ───────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.12, top=0.78, left=0.05, right=0.93, wspace=0.15)
cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.66])
fig.colorbar(pcm, cax=cbar_ax, label='Speed')

# ── Shared legend (centered above all subplots) ───────────────────────────────
fig.legend(legend_handles, legend_labels,
           loc='upper center', bbox_to_anchor=(0.47, 0.816),
           ncol=len(legend_labels), borderaxespad=0, fontsize=9)

if SAVE_PLOT:
    plt.savefig('epoch_sweep_50_250.png', dpi=150, bbox_inches='tight')

plt.show()