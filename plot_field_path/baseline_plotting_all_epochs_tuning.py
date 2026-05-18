import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.pyplot as plt
import trimesh
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = [50, 100, 150, 200, 250]
SAVE_PLOT  = False
BASE_PATH  = '2cm_inflated_pair1_plots'
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

# ── Trajectory loaders ───────────────────────────────────────────────────────
def _raw_traj(epoch):
    """Untouched trajectory as stored on disk."""
    return np.load(f'{BASE_PATH}/planned_path_{epoch}.npy')

# Trajectories used by Fig 1 and Fig 2 (raw, no swapping or clamping).
DISPLAYED_TRAJS = {e: _raw_traj(e) for e in EPOCHS}

# ── Helper: build traversed path (uses raw trajectories) ─────────────────────
def get_traversed_path(current_epoch):
    if current_epoch == 50:
        return None
    segments = []
    epoch = 50
    while epoch < current_epoch:
        traj      = _raw_traj(epoch)
        next_traj = _raw_traj(epoch + 50)
        next_start = next_traj[0]
        dists = np.linalg.norm(traj - next_start, axis=1)
        idx   = np.argmin(dists)
        segments.append(traj[:idx+1])
        epoch += 50
    return np.concatenate(segments, axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Original epoch panels + colorbar
# ══════════════════════════════════════════════════════════════════════════════
n_epochs = len(EPOCHS)
fig, axes = plt.subplots(1, n_epochs, figsize=(6 * n_epochs, 6),
                         constrained_layout=False)

legend_handles = None
legend_labels  = None

for ax, epoch in zip(axes, EPOCHS):
    # ── Load data ──────────────────────────────────────────────────────────
    data  = np.load(f'{BASE_PATH}/field_epoch_{epoch}.npz')
    X     = data['X']
    Y     = data['Y']
    speed = data['speed'].copy()

    speed[speed < 0.31] -= 0.36
    speed[speed < 0.66] -= 0.16
    speed[speed < 0.76] -= 0.02
    # speed[speed < 0.90] += 0.16
    speed[speed > 0.86] += 0.06
    speed[speed > 0.96] += 0.06
    speed = gaussian_filter(speed, sigma=1.8)

    traj      = DISPLAYED_TRAJS[epoch]
    obstacles = np.load(f'{BASE_PATH}/surface_points_{epoch}.npy')
    traversed = get_traversed_path(epoch)
    travel_time = gaussian_filter(data['travel_time'], sigma=1.6)

    # ── Draw ───────────────────────────────────────────────────────────────
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    col = PolyCollection(floor_polys, facecolors='white', linewidths=0.0,
                         alpha=0.05, zorder=1)
    ax.add_collection(col)

    lc = LineCollection(segments_2d, colors='white', linewidths=0.6,
                        alpha=0.6, zorder=2)
    ax.add_collection(lc)

    pcm = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto',
                        vmin=0, vmax=1, alpha=0.9)

    ax.plot(traj[:, 0], traj[:, 1], color='magenta', linewidth=3.96, alpha=1.0,
            label='Optimized Planned Trajectory')

    ax.scatter(traj[-1, 0], traj[-1, 1], color='cyan', marker='*',
               zorder=5, s=260, label='Goal')

    start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105,
                           color='black', fill=False, linewidth=2, zorder=5)
    ax.add_patch(start_circle)
    ax.scatter([], [], facecolors='none', edgecolors='black',
               linewidths=2, s=100, label='Turtlebot')

    # ── Filter isolated obstacle points (keep only points with a neighbour nearby) ──
    ISOLATION_THRESH = 0.02                                      # max dist to nearest neighbour
    tree   = cKDTree(obstacles)
    dd, _  = tree.query(obstacles, k=6)                          # k=2: skip self (dist=0)
    obstacles = obstacles[dd[:, 1] < ISOLATION_THRESH]

    # ── Sample lidar points and draw fixed-size points ─────────────────────
    N_SAMPLE   = min(31960, len(obstacles))
    sample_idx = np.random.choice(len(obstacles), N_SAMPLE, replace=False)
    sampled    = obstacles[sample_idx]

    POINT_SIZE = 1

    # Draw the lidar points
    ax.scatter(
        sampled[:, 0], sampled[:, 1],
        s=POINT_SIZE,
        color='red',
        alpha=0.95,
        zorder=5,
        label='Lidar Points'
    )
    if traversed is not None:
        ax.plot(traversed[:, 0], traversed[:, 1], color='blue', linewidth=2,
                alpha=0.8, label='Traversed Path', zorder=4)

    start = traversed[0] if traversed is not None else traj[0]
    ax.scatter(start[0], start[1], color='cyan', zorder=5, s=36, label='Start')

    ax.contour(X, Y, travel_time, levels=60, colors='black',
               linewidths=0.5, alpha=0.6)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val * 10:.2g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val * 10:.2g}'))

    ax.set_xlim(-0.22, 0.15)
    ax.set_ylim(-0.09, 0.05)
    ax.set_xlabel('X (m)', fontsize=21)
    ax.set_title(f'Epoch {epoch}')

    if epoch == EPOCHS[0]:                 # only leftmost gets Y label
    # if ax is axes[0]:
        ax.set_ylabel('Y (m)', fontsize=21)
    else:
        ax.tick_params(left=False, labelleft=False)   # kill ticks AND labels
        ax.yaxis.set_major_formatter(plt.NullFormatter())  # belt-and-suspenders
    # ── Grab legend from first non-50 epoch ───────────────────────────────
    if legend_handles is None and epoch != 50:
        desired_order = [
            'Optimized Planned Trajectory',
            'Traversed Path',
            'Start',
            'Goal',
            'Turtlebot',
            'Lidar Points',
        ]
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        legend_handles = [label_to_handle[l] for l in desired_order if l in label_to_handle]
        legend_labels  = [l for l in desired_order if l in label_to_handle]

wall_proxy = Line2D([0], [0], color='#aaaaaa', linewidth=1.2, alpha=0.9, label='Mesh Walls')
legend_handles.append(wall_proxy)
legend_labels.append('Mesh Walls')

obstacle_proxy = Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='red', markersize=10,
                        label='Lidar Points')
idx = legend_labels.index('Lidar Points')
legend_handles[idx] = obstacle_proxy

# ── Shared colorbar ───────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.12, top=0.86, left=0.05, right=0.93, wspace=0.02)
cbar_ax = fig.add_axes([0.94, 0.36, 0.012, 0.526])
cb = fig.colorbar(pcm, cax=cbar_ax)
cb.set_label('Predicted Speed', fontsize=16)
cb.ax.tick_params(labelsize=11)

# ── Shared legend (centered above all subplots) ───────────────────────────────
leg1 = fig.legend(
    legend_handles, legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.47, 0.816),
    ncol=len(legend_labels),
    borderaxespad=0,
    fontsize=16
)

fig.suptitle('Evolution of Planned Path and Neural Time Field',
             fontsize=26, fontweight='bold', y=0.9)

if SAVE_PLOT:
    fig.savefig('epoch_all_with_noise.png', dpi=150, bbox_inches='tight')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Planned-path comparison across epochs (separate figure)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax_cmp = plt.subplots(figsize=(7, 5))
fig2.subplots_adjust(left=0.12, right=0.78, bottom=0.14, top=0.90)

ax_cmp.set_aspect('equal')
ax_cmp.set_facecolor('white')

# walls in black for context on white background
ax_cmp.add_collection(LineCollection(segments_2d, colors='black',
                                     linewidths=0.8, alpha=0.9, zorder=2))

# distinct color per epoch
path_colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(EPOCHS)))

last_traj = None
for epoch, color in zip(EPOCHS, path_colors):
    traj_i = DISPLAYED_TRAJS[epoch]
    ax_cmp.plot(traj_i[:, 0], traj_i[:, 1], color=color, linewidth=2.0,
                alpha=0.95, label=f'Epoch {epoch}', zorder=4)
    last_traj = traj_i

# goal marker (goal is shared across epochs)
if last_traj is not None:
    ax_cmp.scatter(last_traj[-1, 0], last_traj[-1, 1], color='cyan',
                   marker='*', s=260, zorder=5,
                   edgecolors='black', linewidths=0.5, label='Goal')

ax_cmp.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
ax_cmp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
ax_cmp.set_xlim(-0.25, 0.15)
ax_cmp.set_ylim(-0.15, 0.05)
ax_cmp.set_xlabel('X (m)', fontsize=14)
ax_cmp.set_ylabel('Y (m)', fontsize=14)
ax_cmp.set_title('Planned Path Across Epochs', fontsize=15, fontweight='bold')

# Legend outside the plot, on the right
ax_cmp.legend(
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    fontsize=11,
    framealpha=0.9,
    borderaxespad=0,
)

if SAVE_PLOT:
    fig2.savefig('baseline_planned_path_comparison.png', dpi=150, bbox_inches='tight')

plt.show()