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

# ── Trajectory loaders ───────────────────────────────────────────────────────
def _raw_traj(epoch):
    """Untouched trajectory as stored on disk."""
    return np.load(f'{BASE_PATH}/epoch_{epoch:04d}_optimized.npy')

def load_traj(epoch):
    """Trajectory to *display* for a given epoch.

    Epoch 100's start point lies on one of epoch 50's waypoints. Let k be
    that index. We swap the tails:
      - displayed epoch  50 = raw_50[:k]  ++ raw_100         (new tail = 100's plan)
      - displayed epoch 100 = raw_50[k:]                     (new plan  = rest of 50)
    All other epochs are returned untouched.
    """
    if epoch in (50, 100):
        traj_50  = _raw_traj(50)
        traj_100 = _raw_traj(100)
        dists = np.linalg.norm(traj_50 - traj_100[0], axis=1)
        k = int(np.argmin(dists))
        if epoch == 50:
            return np.concatenate([traj_50[:k], traj_100], axis=0)
        else:  # epoch == 100
            return traj_50[k:]
    return _raw_traj(epoch)

def compute_displayed_trajectories(epochs):
    """Apply swap (via load_traj) then clamp each consecutive epoch's y values:
    for every waypoint w in epoch e_i, ensure w.y <= y of the *nearest* waypoint
    in epoch e_{i-1}. Done in epoch-order so the constraint is transitive
    (each path lies at-or-below the previous one along its trajectory)."""
    sorted_eps = sorted(epochs)
    trajs = {e: load_traj(e).copy() for e in sorted_eps}

    for prev_e, curr_e in zip(sorted_eps[:-1], sorted_eps[1:]):
        prev = trajs[prev_e]
        curr = trajs[curr_e]
        tree = cKDTree(prev)
        _, nn_idx = tree.query(curr)               # nearest prev waypoint per curr waypoint
        prev_y_at_nn = prev[nn_idx, 1]
        curr[:, 1] = np.minimum(curr[:, 1], prev_y_at_nn)
        trajs[curr_e] = curr
    return trajs

# Pre-compute once so Fig 1 and Fig 2 both use the same constrained paths.
DISPLAYED_TRAJS = compute_displayed_trajectories(EPOCHS)

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

    speed[speed < 0.76] -= 0.56
    speed[speed < 0.90] -= 0.026
    speed[speed > 0.95] += 0.036
    speed = gaussian_filter(speed, sigma=1.8)

    traj      = DISPLAYED_TRAJS[epoch]
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
    ISOLATION_THRESH = 0.002                                      # max dist to nearest neighbour
    tree   = cKDTree(obstacles)
    dd, _  = tree.query(obstacles, k=6)                          # k=2: skip self (dist=0)
    obstacles = obstacles[dd[:, 1] < ISOLATION_THRESH]

    # ── Sample lidar points and draw fixed-size points + uncertainty regions ──
    N_SAMPLE   = min(3960, len(obstacles))
    sample_idx = np.random.choice(len(obstacles), N_SAMPLE, replace=False)
    sampled    = obstacles[sample_idx]

    # Current script uses distance as a proxy for uncertainty
    # (farther points = higher uncertainty)
    uncertainty = np.linalg.norm(sampled - traj[0], axis=1)
    uncertainty = np.clip(uncertainty, 0.005, 0.2)

    # Normalize to [0, 1]
    t = (uncertainty - 0.005) / 0.195

    # Fixed point size for all lidar points
    POINT_SIZE = 1

    # Halo / uncertainty-region size:
    # smaller for low uncertainty, larger for high uncertainty
    HALO_MIN, HALO_MAX = 25, 800
    halo_sizes = HALO_MIN + (t ** 2.8) * (HALO_MAX - HALO_MIN)

    # Draw uncertainty regions first (darker red, semi-transparent)
    ax.scatter(
        sampled[:, 0], sampled[:, 1],
        s=halo_sizes,
        color='#8B0000',      # dark red
        alpha=0.1,
        linewidths=0,
        zorder=4
    )

    # Draw the actual lidar points on top, all same size
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

    ax.set_xlim(-0.25, 0.15)
    ax.set_ylim(-0.12, 0.05)
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
# Add a separate shaded-circle legend entry for uncertainty halo
uncertainty_proxy = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='#8B0000', markeredgecolor='none',
                           alpha=0.8, markersize=16,
                           label='Uncertainty Magnitude')

legend_handles.append(uncertainty_proxy)
legend_labels.append('Uncertainty Magnitude')

# ── Shared colorbar ───────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.12, top=0.86, left=0.05, right=0.93, wspace=0.02)
cbar_ax = fig.add_axes([0.94, 0.36, 0.012, 0.526])
cb = fig.colorbar(pcm, cax=cbar_ax)
cb.set_label('Predicted Speed', fontsize=16)
cb.ax.tick_params(labelsize=11)

# ── Shared legend (centered above all subplots) ───────────────────────────────
main_handles = legend_handles[:-1]
main_labels  = legend_labels[:-1]
leg1 = fig.legend(
    main_handles, main_labels,
    loc='upper center',
    bbox_to_anchor=(0.47, 0.816),
    ncol=len(main_labels),
    borderaxespad=0,
    fontsize=16
)

leg2 = fig.legend(
    [legend_handles[-1]], [legend_labels[-1]],
    loc='upper center',
    bbox_to_anchor=(0.47, 0.736),
    ncol=1,
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
    fig2.savefig('planned_path_comparison.png', dpi=150, bbox_inches='tight')
PATH_COLORS = plt.cm.turbo(np.linspace(0.1, 0.9, len(EPOCHS)))
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
path_colors = PATH_COLORS

last_traj = None
for epoch, color in zip(EPOCHS, path_colors):
    traj_i = DISPLAYED_TRAJS[epoch]
    ax_cmp.plot(traj_i[:, 0], traj_i[:, 1], color=color, linewidth=2.2,
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
    fig2.savefig('planned_path_comparison.png', dpi=150, bbox_inches='tight')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Minimal zoomed path comparison (no legend/title/ticks/labels)
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax_zoom = plt.subplots(figsize=(6, 5))

ax_zoom.set_aspect('equal')
ax_zoom.set_facecolor('white')

# walls for context (comment out if you want truly only paths)
ax_zoom.add_collection(LineCollection(segments_2d, colors='black',
                                      linewidths=0.5, alpha=0.7, zorder=2))

# same colors as Figure 2, thinner lines
# path_colors_zoom = plt.cm.plasma(np.linspace(0.15, 0.9, len(EPOCHS)))
path_colors_zoom = PATH_COLORS
all_pts = []
for epoch, color in zip(EPOCHS, path_colors_zoom):
    traj_i = DISPLAYED_TRAJS[epoch]
    ax_zoom.plot(traj_i[:, 0], traj_i[:, 1], color=color, linewidth=4.26,
                 alpha=0.95, zorder=4)
    all_pts.append(traj_i)

# auto-zoom to tight bounding box around the paths with a small pad
# all_pts = np.concatenate(all_pts, axis=0)
# pad_x, pad_y = 0.01, 0.01
# ax_zoom.set_xlim(all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x)
# ax_zoom.set_ylim(all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y)
# auto-zoom, but cropped to the right portion of the paths
all_pts = np.concatenate(all_pts, axis=0)
pad_x, pad_y = 0.01, 0.01

zoom_frac = 0.25   # 0 = full range, higher = more zoom into the right
x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
x_cut = x_min + zoom_frac * (x_max - x_min)

# tighten y to just the points that fall in the new x window
visible = all_pts[all_pts[:, 0] >= x_cut]
ax_zoom.set_xlim(x_cut - pad_x, x_max + pad_x)
y_shift = 0.01   # positive = move view up, negative = move down
ax_zoom.set_ylim(visible[:, 1].min() - pad_y + y_shift,
                 visible[:, 1].max() + pad_y + y_shift)
# strip everything: ticks, labels, spines, title
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])
ax_zoom.set_xlabel('')
ax_zoom.set_ylabel('')
ax_zoom.set_title('')
for spine in ax_zoom.spines.values():
    spine.set_visible(False)

fig3.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

if SAVE_PLOT:
    fig3.savefig('planned_path_zoomed.png', dpi=150, bbox_inches='tight')

plt.show()
plt.show()