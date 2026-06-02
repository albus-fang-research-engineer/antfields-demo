import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.pyplot as plt
import trimesh
from matplotlib.lines import Line2D
from PIL import Image
import imageio.v2 as imageio

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = [50, 100, 150, 200, 250]
BASE_PATH  = '2cm_inflated_pair1_plots'        # <-- bottom-row data
MESH_PATH  = 'gibson/mesh.obj'
CEILING_Z  = 0.02
FLOOR_Z    = -0.12

# Movie config
FRAMES_DIR = 'movie_frames_inflated'
MOVIE_OUT  = 'inflated_epochs_movie.mp4'
FPS        = 10
TITLE_HOLD = 25   # ~2.5 s on title card
EPOCH_HOLD = 15   # ~1.5 s per epoch frame
FINAL_HOLD = 20   # extra ~2 s on the last frame

# Title-card text
TITLE_MAIN = '2cm Inflated Baseline'
TITLE_SUB  = 'Evolution of the Planned Path and Neural Time Field'

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

face_z    = mesh_3d.vertices[mesh_3d.faces, 2]
floor_mask = (
    (np.abs(normals[:, 2]) > 0.7) &
    (face_z.mean(axis=1) < FLOOR_Z + 0.05)
)
vertices_2d = mesh_3d.vertices[:, :2]
floor_polys = vertices_2d[mesh_3d.faces[floor_mask]]

# ── Trajectory loaders (bottom row: raw planned paths, no swap/clamp) ────────
def _raw_traj(epoch):
    return np.load(f'{BASE_PATH}/planned_path_{epoch}.npy')

# Displayed trajectories are simply the raw planned paths.
DISPLAYED_TRAJS = {e: _raw_traj(e) for e in EPOCHS}

# ── Helper: traversed path from raw trajectories ─────────────────────────────
def get_traversed_path(current_epoch):
    if current_epoch == 50:
        return None
    segments = []
    epoch = 50
    while epoch < current_epoch:
        traj       = _raw_traj(epoch)
        next_traj  = _raw_traj(epoch + 50)
        next_start = next_traj[0]
        dists = np.linalg.norm(traj - next_start, axis=1)
        idx   = np.argmin(dists)
        segments.append(traj[:idx + 1])
        epoch += 50
    return np.concatenate(segments, axis=0)

# ── Per-epoch panel drawer (content of one old File-1 bottom-row column) ─────
def draw_epoch_panel(ax, epoch):
    data  = np.load(f'{BASE_PATH}/field_epoch_{epoch}.npz')
    X, Y  = data['X'], data['Y']
    speed = data['speed'].copy()
    speed[speed < 0.31] -= 0.36
    speed[speed < 0.66] -= 0.16
    speed[speed < 0.76] -= 0.02
    # speed[speed < 0.90] += 0.16
    speed[speed > 0.86] -= 0.06
    speed[speed > 0.96] += 0.06
    speed = gaussian_filter(speed, sigma=1.8)

    traj        = DISPLAYED_TRAJS[epoch]
    obstacles   = np.load(f'{BASE_PATH}/surface_points_{epoch}.npy')
    traversed   = get_traversed_path(epoch)
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

    ax.add_patch(Circle((traj[0, 0], traj[0, 1]), radius=0.0105,
                        color='black', fill=False, linewidth=2, zorder=5))

    # filter isolated surface points (bottom-row threshold)
    tree  = cKDTree(obstacles)
    dd, _ = tree.query(obstacles, k=6)
    obstacles = obstacles[dd[:, 1] < 0.02]

    # bottom row samples densely and has NO uncertainty halos
    N_SAMPLE   = min(31960, len(obstacles))
    sample_idx = np.random.choice(len(obstacles), N_SAMPLE, replace=False)
    sampled    = obstacles[sample_idx]

    ax.scatter(sampled[:, 0], sampled[:, 1], s=1, color='red',
               alpha=0.95, zorder=5)

    if traversed is not None:
        traversed[-1] = traj[0]   # snap traversed tail to current start
        ax.plot(traversed[:, 0], traversed[:, 1], color='blue',
                linewidth=2, alpha=0.8, zorder=4)

    start = traversed[0] if traversed is not None else traj[0]
    ax.scatter(start[0], start[1], color='cyan', zorder=5, s=36)

    ax.contour(X, Y, travel_time, levels=60, colors='black',
               linewidths=0.5, alpha=0.6)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 10:.2g}'))
    ax.set_xlim(-0.22, 0.15)
    ax.set_ylim(-0.09, 0.05)
    ax.set_xlabel('X (m)', fontsize=18)
    ax.set_ylabel('Y (m)', fontsize=18)
    ax.set_title(f'Epoch {epoch}', fontsize=20, fontweight='bold')
    return pcm

# ── Fixed legend (identical on every frame; no uncertainty halo entry) ───────
def build_legend_handles():
    return [
        Line2D([0], [0], color='magenta', linewidth=3,
               label='Optimized Planned Trajectory'),
        Line2D([0], [0], color='blue', linewidth=2,
               label='Traversed Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
               markersize=8, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan',
               markersize=14, label='Goal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='black', markersize=10, markeredgewidth=2,
               label='Turtlebot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Lidar Points'),
        Line2D([0], [0], color='#aaaaaa', linewidth=1.2, label='Mesh Walls'),
    ]

# ══════════════════════════════════════════════════════════════════════════════
# Build movie: title card → one frame per epoch → MP4
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(FRAMES_DIR, exist_ok=True)

# 1. Title card
fig_t, ax_t = plt.subplots(figsize=(8, 8))
fig_t.patch.set_facecolor('white')
ax_t.set_facecolor('white')
ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
ax_t.axis('off')
ax_t.text(0.5, 0.60, TITLE_MAIN,
          ha='center', va='center',
          fontsize=44, fontweight='bold', color='black')
ax_t.text(0.5, 0.34, TITLE_SUB,
          ha='center', va='center',
          fontsize=15, color='#444', style='italic')
title_path = f'{FRAMES_DIR}/_title.png'
fig_t.savefig(title_path, dpi=130, bbox_inches='tight', facecolor='white')
plt.close(fig_t)

# 2. One PNG per epoch with the legend on top
frame_paths = []
for epoch in EPOCHS:
    fig, ax = plt.subplots(figsize=(8, 8))
    pcm = draw_epoch_panel(ax, epoch)

    fig.legend(
        handles=build_legend_handles(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=4, fontsize=10, frameon=True,
    )

    fig.subplots_adjust(left=0.12, right=0.86, top=0.78, bottom=0.10)
    cax = fig.add_axes([0.88, 0.10, 0.025, 0.68])
    cb  = fig.colorbar(pcm, cax=cax)
    cb.set_label('Predicted Speed', fontsize=12)
    cb.ax.tick_params(labelsize=10)

    out = f'{FRAMES_DIR}/epoch_{epoch:04d}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    frame_paths.append(out)

# 3. Stitch into MP4
all_paths = [title_path] + frame_paths
imgs = [Image.open(p).convert('RGB') for p in all_paths]

# unify dims and force even dimensions for H.264
w = min(im.width  for im in imgs)
h = min(im.height for im in imgs)
w -= w % 2
h -= h % 2
imgs = [im.resize((w, h), Image.LANCZOS) for im in imgs]

sequence = [imgs[0]] * TITLE_HOLD
for im in imgs[1:]:
    sequence += [im] * EPOCH_HOLD
sequence += [imgs[-1]] * FINAL_HOLD

with imageio.get_writer(
    MOVIE_OUT,
    fps=FPS,
    codec='libx264',
    quality=8,
    macro_block_size=1,
    pixelformat='yuv420p',
) as writer:
    for im in sequence:
        writer.append_data(np.array(im))

print(f'Wrote {MOVIE_OUT}')