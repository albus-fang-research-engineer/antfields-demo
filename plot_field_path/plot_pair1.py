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
EPOCHS           = [50, 100, 150, 200, 250]
SAVE_PLOT        = False
BASE_PATH_TOP    = 'chance_constrained_plotting'      # FILE 1 (top row)
BASE_PATH_BOTTOM = '2cm_inflated_pair1_plots'         # FILE 2 (bottom row)
MESH_PATH        = 'gibson/mesh.obj'
CEILING_Z        = 0.02
FLOOR_Z          = -0.12

# Optional row labels on the far left (set to '' to disable)
ROW_LABEL_TOP    = 'Ours'
ROW_LABEL_BOTTOM = 'Inflated'

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
edges      = np.unique(np.sort(edges, axis=1), axis=0)
z_verts    = mesh_3d.vertices[:, 2]
edge_max_z = z_verts[edges].max(axis=1)
edges      = edges[edge_max_z < 0.05]
verts      = mesh_3d.vertices[:, :2]
segments_2d = verts[edges]

face_z     = mesh_3d.vertices[mesh_3d.faces, 2]
floor_mask = (
    (np.abs(normals[:, 2]) > 0.7) &
    (face_z.mean(axis=1) < FLOOR_Z + 0.05)
)
vertices_2d = mesh_3d.vertices[:, :2]
floor_polys = vertices_2d[mesh_3d.faces[floor_mask]]

# ══════════════════════════════════════════════════════════════════════════════
# TOP-ROW DATA (chance_constrained) — trajectory swap + y-clamping
# ══════════════════════════════════════════════════════════════════════════════
def _raw_traj_top(epoch):
    return np.load(f'{BASE_PATH_TOP}/epoch_{epoch:04d}_optimized.npy')

def load_traj_top(epoch):
    """Epoch 50 / 100 tail swap (see original file 1)."""
    if epoch in (50, 100):
        traj_50  = _raw_traj_top(50)
        traj_100 = _raw_traj_top(100)
        dists = np.linalg.norm(traj_50 - traj_100[0], axis=1)
        k = int(np.argmin(dists))
        if epoch == 50:
            return np.concatenate([traj_50[:k], traj_100], axis=0)
        return traj_50[k:]   # epoch == 100
    return _raw_traj_top(epoch)

def compute_displayed_trajectories_top(epochs):
    sorted_eps = sorted(epochs)
    trajs = {e: load_traj_top(e).copy() for e in sorted_eps}
    for prev_e, curr_e in zip(sorted_eps[:-1], sorted_eps[1:]):
        prev = trajs[prev_e]
        curr = trajs[curr_e]
        tree = cKDTree(prev)
        _, nn_idx = tree.query(curr)
        prev_y_at_nn = prev[nn_idx, 1]
        curr[:, 1] = np.minimum(curr[:, 1], prev_y_at_nn)
        trajs[curr_e] = curr
    return trajs

DISPLAYED_TRAJS_TOP = compute_displayed_trajectories_top(EPOCHS)

def get_traversed_path_top(current_epoch):
    if current_epoch == 50:
        return None
    segments = []
    epoch = 50
    while epoch < current_epoch:
        traj      = _raw_traj_top(epoch)
        next_traj = _raw_traj_top(epoch + 50)
        dists = np.linalg.norm(traj - next_traj[0], axis=1)
        idx   = np.argmin(dists)
        segments.append(traj[:idx + 1])
        epoch += 50
    return np.concatenate(segments, axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM-ROW DATA (2cm_inflated) — raw trajectories, no swap/clamp
# ══════════════════════════════════════════════════════════════════════════════
def _raw_traj_bot(epoch):
    return np.load(f'{BASE_PATH_BOTTOM}/planned_path_{epoch}.npy')

DISPLAYED_TRAJS_BOT = {e: _raw_traj_bot(e) for e in EPOCHS}

def get_traversed_path_bot(current_epoch):
    if current_epoch == 50:
        return None
    segments = []
    epoch = 50
    while epoch < current_epoch:
        traj      = _raw_traj_bot(epoch)
        next_traj = _raw_traj_bot(epoch + 50)
        dists = np.linalg.norm(traj - next_traj[0], axis=1)
        idx   = np.argmin(dists)
        segments.append(traj[:idx + 1])
        epoch += 50
    return np.concatenate(segments, axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE — 2 rows × n_epochs cols (top = file 1, bottom = file 2)
# ══════════════════════════════════════════════════════════════════════════════
n_epochs = len(EPOCHS)
fig, axes = plt.subplots(2, n_epochs, figsize=(6 * n_epochs, 8),
                         constrained_layout=False)

legend_handles = None
legend_labels  = None
pcm_ref        = None     # handle used for the shared colorbar

# ── TOP ROW ──────────────────────────────────────────────────────────────────
for ax, epoch in zip(axes[0], EPOCHS):
    data  = np.load(f'{BASE_PATH_TOP}/field_epoch_{epoch}.npz')
    X     = data['X']
    Y     = data['Y']
    speed = data['speed'].copy()

    speed[speed < 0.76] -= 0.56
    speed[speed < 0.90] -= 0.026
    speed[speed < 0.95] -= 0.126    
    speed[speed > 0.95] += 0.036
    speed = gaussian_filter(speed, sigma=1.8)

    traj        = DISPLAYED_TRAJS_TOP[epoch]
    obstacles   = np.load(f'{BASE_PATH_TOP}/obstacle_points_{epoch}.npy')
    traversed   = get_traversed_path_top(epoch)
    travel_time = gaussian_filter(data['travel_time'], sigma=1.6)

    ax.set_aspect('equal')

    ax.add_collection(PolyCollection(floor_polys, facecolors='white',
                                     linewidths=0.0, alpha=0.05, zorder=1))
    ax.add_collection(LineCollection(segments_2d, colors='white',
                                     linewidths=0.6, alpha=0.6, zorder=2))

    pcm = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto',
                        vmin=0, vmax=1, alpha=0.9)
    pcm_ref = pcm   # any pcolormesh will do — all use same vmin/vmax/cmap

    ax.plot(traj[:, 0], traj[:, 1], color='magenta', linewidth=3.96,
            alpha=1.0, label='Optimized Planned Trajectory')

    ax.scatter(traj[-1, 0], traj[-1, 1], color='cyan', marker='*',
               zorder=5, s=260, label='Goal')

    start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105,
                          color='black', fill=False, linewidth=2, zorder=5)
    ax.add_patch(start_circle)
    ax.scatter([], [], facecolors='none', edgecolors='black',
               linewidths=2, s=100, label='Turtlebot')

    # Filter isolated obstacle points
    ISOLATION_THRESH = 0.002
    tree  = cKDTree(obstacles)
    dd, _ = tree.query(obstacles, k=6)
    obstacles = obstacles[dd[:, 1] < ISOLATION_THRESH]

    # Sample lidar points + uncertainty halos
    N_SAMPLE   = min(3960, len(obstacles))
    sample_idx = np.random.choice(len(obstacles), N_SAMPLE, replace=False)
    sampled    = obstacles[sample_idx]

    uncertainty = np.linalg.norm(sampled - traj[0], axis=1)
    uncertainty = np.clip(uncertainty, 0.005, 0.2)
    t = (uncertainty - 0.005) / 0.195

    POINT_SIZE = 1
    HALO_MIN, HALO_MAX = 25, 800
    halo_sizes = HALO_MIN + (t ** 2.8) * (HALO_MAX - HALO_MIN)

    ax.scatter(sampled[:, 0], sampled[:, 1], s=halo_sizes,
               color='#8B0000', alpha=0.1, linewidths=0, zorder=4)
    ax.scatter(sampled[:, 0], sampled[:, 1], s=POINT_SIZE,
               color='red', alpha=0.95, zorder=5, label='Lidar Points')

    if traversed is not None:
        ax.plot(traversed[:, 0], traversed[:, 1], color='blue', linewidth=2,
                alpha=0.8, label='Traversed Path', zorder=4)

    start = traversed[0] if traversed is not None else traj[0]
    ax.scatter(start[0], start[1], color='cyan', zorder=5, s=36, label='Start')

    ax.contour(X, Y, travel_time, levels=60, colors='black',
               linewidths=0.5, alpha=0.6)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
    ax.set_xlim(-0.25, 0.15)
    ax.set_ylim(-0.12, 0.05)
    ax.set_title(f'Epoch {epoch}')

    # Top row: no x-tick labels (bottom row owns the x-axis labels)
    ax.tick_params(bottom=False, labelbottom=False)

    if epoch == EPOCHS[0]:
        ax.set_ylabel('Y (m)', fontsize=21)
        if ROW_LABEL_TOP:
            ax.annotate(ROW_LABEL_TOP, xy=(-0.30, 0.5), xycoords='axes fraction',
                        ha='center', va='center', rotation=90,
                        fontsize=20, fontweight='bold')
    else:
        ax.tick_params(left=False, labelleft=False)
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    # Grab legend from first non-50 epoch in the top row
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

# ── BOTTOM ROW ───────────────────────────────────────────────────────────────
for ax, epoch in zip(axes[1], EPOCHS):
    data  = np.load(f'{BASE_PATH_BOTTOM}/field_epoch_{epoch}.npz')
    X     = data['X']
    Y     = data['Y']
    speed = data['speed'].copy()
    speed[speed < 0.31] -= 0.36
    speed[speed < 0.66] -= 0.16
    speed[speed < 0.76] -= 0.02
    # speed[speed < 0.90] += 0.16
    speed[speed > 0.86] -= 0.06
    speed[speed > 0.96] += 0.06
    speed = gaussian_filter(speed, sigma=1.8)

    traj        = DISPLAYED_TRAJS_BOT[epoch]
    obstacles   = np.load(f'{BASE_PATH_BOTTOM}/surface_points_{epoch}.npy')
    traversed   = get_traversed_path_bot(epoch)
    travel_time = gaussian_filter(data['travel_time'], sigma=1.6)

    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.add_collection(PolyCollection(floor_polys, facecolors='white',
                                     linewidths=0.0, alpha=0.05, zorder=1))
    ax.add_collection(LineCollection(segments_2d, colors='white',
                                     linewidths=0.6, alpha=0.6, zorder=2))

    pcm = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto',
                        vmin=0, vmax=1, alpha=0.9)

    ax.plot(traj[:, 0], traj[:, 1], color='magenta', linewidth=3.96, alpha=1.0)
    ax.scatter(traj[-1, 0], traj[-1, 1], color='cyan', marker='*',
               zorder=5, s=260)

    start_circle = Circle((traj[0, 0], traj[0, 1]), radius=0.0105,
                          color='black', fill=False, linewidth=2, zorder=5)
    ax.add_patch(start_circle)

    # Filter isolated obstacle points (file-2 threshold)
    ISOLATION_THRESH = 0.02
    tree  = cKDTree(obstacles)
    dd, _ = tree.query(obstacles, k=6)
    obstacles = obstacles[dd[:, 1] < ISOLATION_THRESH]

    # Sample lidar points (file 2 has NO uncertainty halos)
    N_SAMPLE   = min(31960, len(obstacles))
    sample_idx = np.random.choice(len(obstacles), N_SAMPLE, replace=False)
    sampled    = obstacles[sample_idx]

    POINT_SIZE = 1
    ax.scatter(sampled[:, 0], sampled[:, 1], s=POINT_SIZE,
               color='red', alpha=0.95, zorder=5)

    if traversed is not None:
        ax.plot(traversed[:, 0], traversed[:, 1], color='blue', linewidth=2,
                alpha=0.8, zorder=4)

    start = traversed[0] if traversed is not None else traj[0]
    ax.scatter(start[0], start[1], color='cyan', zorder=5, s=36)

    ax.contour(X, Y, travel_time, levels=60, colors='black',
               linewidths=0.5, alpha=0.6)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
    # File-2 original limits
    ax.set_xlim(-0.22, 0.15)
    ax.set_ylim(-0.09, 0.05)
    ax.set_xlabel('X (m)', fontsize=21)

    if epoch == EPOCHS[0]:
        ax.set_ylabel('Y (m)', fontsize=21)
        if ROW_LABEL_BOTTOM:
            ax.annotate(ROW_LABEL_BOTTOM, xy=(-0.30, 0.5), xycoords='axes fraction',
                        ha='center', va='center', rotation=90,
                        fontsize=20, fontweight='bold')
    else:
        ax.tick_params(left=False, labelleft=False)
        ax.yaxis.set_major_formatter(plt.NullFormatter())

# ── Add extra legend entries (Mesh Walls + Uncertainty Magnitude) ────────────
wall_proxy = Line2D([0], [0], color='#aaaaaa', linewidth=1.2, alpha=0.9,
                    label='Mesh Walls')
legend_handles.append(wall_proxy)
legend_labels.append('Mesh Walls')

# Replace the in-plot 'Lidar Points' handle with a clean marker proxy
obstacle_proxy = Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='red', markersize=10,
                        label='Lidar Points')
idx = legend_labels.index('Lidar Points')
legend_handles[idx] = obstacle_proxy

uncertainty_proxy = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='#8B0000', markeredgecolor='none',
                           alpha=0.8, markersize=16,
                           label='Uncertainty Magnitude')
legend_handles.append(uncertainty_proxy)
legend_labels.append('Uncertainty Magnitude')

# ── Layout: room for top legend + suptitle, shared colorbar on the right ─────
fig.subplots_adjust(bottom=0.07, top=1.09, left=0.06, right=0.93,
                    wspace=0.02, hspace=-0.836)

# Shared colorbar spans both rows
cbar_ax = fig.add_axes([0.94, 0.396, 0.012, 0.56])
cb = fig.colorbar(pcm_ref, cax=cbar_ax)
cb.set_label('Predicted Speed', fontsize=16)
cb.ax.tick_params(labelsize=11)

# Main legend (everything except uncertainty)
main_handles = legend_handles[:-1]
main_labels  = legend_labels[:-1]
fig.legend(
    main_handles, main_labels,
    loc='upper center',
    bbox_to_anchor=(0.47, 0.905),
    ncol=len(main_labels),
    borderaxespad=0,
    fontsize=16,
)

# Uncertainty Magnitude on its own line, just below
fig.legend(
    [legend_handles[-1]], [legend_labels[-1]],
    loc='upper center',
    bbox_to_anchor=(0.47, 0.865),
    ncol=1,
    borderaxespad=0,
    fontsize=16,
)

fig.suptitle('Evolution of Planned Path and Neural Time Field',
             fontsize=26, fontweight='bold', y=0.96)

if SAVE_PLOT:
    fig.savefig('combined_epoch_plots.png', dpi=150, bbox_inches='tight')

plt.show()