import numpy as np
import torch

def rollout_nominal(start, path):
    traj = [start.copy()]
    for wp in path:
        traj.append(wp.copy())
    return traj

def _to_numpy(x):
    # torch Tensor → numpy (CPU)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    # already numpy / list / tuple → numpy
    return np.asarray(x)

def rollout_optimized(start, path, obstacle_points, solver, model, device, epoch, folder=""):
    p = _to_numpy(start).copy()
    traj = [p.copy()]
    nominal = [_to_numpy(start).copy()]
    active_flags = []   # per-waypoint: did the QP's chance constraints bind?
    for wp in path:
        wp_np = _to_numpy(wp)
        nominal.append(wp_np.copy())
        p, mu, sigma, active = solver(p, wp_np, obstacle_points, model, device, epoch, start, folder)
        p = _to_numpy(p).copy()
        traj.append(p.copy())
        active_flags.append(bool(active))
    if folder is not None:
        plot_epoch_paths(nominal,traj,obstacle_points,folder,epoch)
    nominal_np = np.asarray(nominal)
    traj_np = np.asarray(traj)
    if folder is not None:
        np.save(f"{folder}/epoch_{epoch:04d}_nominal.npy", nominal_np)
        np.save(f"{folder}/epoch_{epoch:04d}_optimized.npy", traj_np)

    return traj, active_flags

def rollout_optimized_plot(start, path, obstacle_points, solver, model, device):
    p = start.copy()

    traj = [p.copy()]
    mus = []
    sigmas = []

    for wp in path:
        p, mu, sigma, active = solver(p, wp, obstacle_points, model, device)

        traj.append(p.copy())
        mus.append(mu)
        sigmas.append(sigma)

    return traj, mus, sigmas


def control_effort(path):
    """Quadratic control effort: sum of squared xy step displacements."""
    p = np.asarray(path)
    if len(p) < 2:
        return 0.0
    d = np.diff(p[:, :2], axis=0)
    return float(np.sum(d ** 2))


def modified_segments(flags):
    """Contiguous runs of modified waypoints, expanded by one waypoint on
    each side (the waypoint prior and after), merged when overlapping.
    Returns a list of (start_idx, end_idx) inclusive index pairs."""
    flags = np.asarray(flags, dtype=bool)
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    runs = []
    s = e = idx[0]
    for i in idx[1:]:
        if i == e + 1:
            e = i
        else:
            runs.append((s, e))
            s = e = i
    runs.append((s, e))
    expanded = [(max(0, s - 1), min(len(flags) - 1, e + 1)) for s, e in runs]
    merged = [expanded[0]]
    for s, e in expanded[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged



def plot_epoch_paths(nominal_path, optimized_traj, obstacle_points, folder, epoch):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)

    nominal = np.asarray(nominal_path)
    opt = np.asarray(optimized_traj)

    plt.figure(figsize=(6,6))

    # obstacles
    if obstacle_points is not None:
        if hasattr(obstacle_points, "detach"):
            obstacle_points = obstacle_points.detach().cpu().numpy()

        plt.scatter(obstacle_points[:,0], obstacle_points[:,1], color='black',
                    s=2, alpha=0.2, label="obstacles")

    # nominal path
    plt.plot(nominal[:,0], nominal[:,1],
             '--', color='orange', linewidth=2, label="nominal path")

    plt.scatter(nominal[:,0], nominal[:,1], s=36, color='orange')

    # optimized path
    plt.plot(opt[:,0], opt[:,1],
             '-',  color = 'blue', linewidth=2, label="optimized path")

    plt.scatter(opt[:,0], opt[:,1], s=36, color='blue')

    # start and goal
    plt.scatter(opt[0,0], opt[0,1], c="green", s=100, label="start")
    plt.scatter(nominal[-1,0], nominal[-1,1], c="purple", s=100, label="goal")
    # ---- zoom to path region ----
    points = np.vstack([nominal[:,:2], opt[:,:2]])

    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    padding = 0.1

    plt.xlim(xmin - padding, xmax + padding)
    plt.ylim(ymin - padding, ymax + padding)

    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.legend()
    plt.title(f"Epoch {epoch} path comparison")

    plt.savefig(f"{folder}/epoch_{epoch:04d}_path.png")
    plt.close()