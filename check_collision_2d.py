import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery


def load_mesh(mesh_path, scale_factor=10.0, to_meters=True):
    """
    Load Gibson mesh and optionally convert to meters.
    Gibson often uses scale_factor=10 (0.1 units = 1m)
    """
    mesh = trimesh.load(mesh_path)

    if to_meters:
        mesh.apply_scale(1.0 / scale_factor)

    mesh = mesh.process()
    return mesh


def load_trajectory(traj_path, mesh):
    """
    Load trajectory (Nx2 or Nx3)
    """
    traj = np.load(traj_path)

    if traj.shape[1] == 2:
        # print("trajectory is 2D")
        floor_z = mesh.vertices[:, 2].min()
        z_offset = 0.002  # small lift above floor

        z = np.full((traj.shape[0], 1), floor_z + z_offset)
        traj = np.hstack([traj, z])
        # traj = np.hstack([traj, np.zeros((traj.shape[0], 1))])
    if traj.shape[1] == 3:
        floor_z = mesh.vertices[:, 2].min()
        traj[:, 2] = floor_z + 0.036
    return traj
def remove_floor_ceiling_by_z(mesh):
    """
    Removes floor and ceiling based purely on Z range.
    Keeps only faces strictly inside (z_min, z_max).
    """
    z_vals = mesh.vertices[:, 2]
    z_min = z_vals.min()
    z_max = z_vals.max()

    centers = mesh.triangles_center  # (F, 3)
    z = centers[:, 2]

    # Keep faces strictly inside bounds
    keep_faces = (z > z_min+0.05) & (z < z_max-0.05)

    filtered_mesh = mesh.submesh([keep_faces], append=True)
    return filtered_mesh

def compute_signed_distance(mesh, points):
    """
    Compute signed distance from points to mesh
    Positive = outside, Negative = inside
    """
    pq = ProximityQuery(mesh)
    signed_dist = pq.signed_distance(points)
    return signed_dist


def check_collision(dist, robot_radius):
    """
    Collision check using UNSIGNED distance
    """
    collision_mask = dist < robot_radius

    results = {
        "collision": bool(np.any(collision_mask)),
        "collision_rate": float(np.mean(collision_mask)),
        "num_collisions": int(np.sum(collision_mask)),
        "min_distance": float(np.min(dist)),
        "min_clearance": float(np.min(dist - robot_radius)),
        "collision_indices": np.where(collision_mask)[0],
    }

    return results

def visualize(mesh, traj, collision_mask):
    import open3d as o3d
    import numpy as np

    print("Visualizing...")

    # ===== Mesh =====
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # light gray

    # ===== Trajectory (all points) =====
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(traj)
    traj_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # blue

    # ===== Collision points =====
    if np.any(collision_mask):
        collision_pts = traj[collision_mask]
        coll_pcd = o3d.geometry.PointCloud()
        coll_pcd.points = o3d.utility.Vector3dVector(collision_pts)
        coll_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
    else:
        coll_pcd = None

    # ===== Start & Goal =====
    start = traj[0]
    goal = traj[-1]

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0105)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # green

    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0105)
    goal_sphere.translate(goal)
    goal_sphere.paint_uniform_color([1.0, 0.0, 0.0])  

    # ===== Draw =====
    geometries = [mesh_o3d, traj_pcd, start_sphere, goal_sphere]

    if coll_pcd is not None:
        geometries.append(coll_pcd)

    o3d.visualization.draw_geometries(geometries)
def main():
    import os

    # ===== CONFIG =====
    mesh_path = "/antfields/data/mesh.obj"
    root_dir = "/antfields/Experiments/TUNED_BASELINE_GLOBAL_RUN_2/"

    SCALE_FACTOR = 10.0
    ROBOT_RADIUS = 0.0105

    # ===== LOAD MESH =====
    print("Loading mesh...")
    mesh = load_mesh(mesh_path, scale_factor=SCALE_FACTOR, to_meters=False)

    run_dirs = [d for d in os.listdir(root_dir) if d.startswith("RUN_")]

    # extract indices
    run_indices = [int(d.split("_")[1]) for d in run_dirs]

    max_idx = max(run_indices)
    num_trajs = 0
    num_collided_trajs = 0

    # ===== LOOP OVER ALL TRAJECTORIES =====
    for i in sorted(run_indices):
        run = f"RUN_{i}"
        traj_path = os.path.join(root_dir, run, "full_trajectory.npy")

        if not os.path.exists(traj_path):
            continue

        print(f"\nProcessing {run}...")

        traj = load_trajectory(traj_path, mesh)

        # ===== DISTANCE =====
        signed_dist = compute_signed_distance(mesh, traj)
        dist = np.abs(signed_dist)   

        collision_mask = dist < ROBOT_RADIUS+0.01

        collided = np.any(collision_mask)

        num_trajs += 1
        if collided:
            num_collided_trajs += 1
            # visualize(mesh, traj, collision_mask)

        print(f"{run}: collided={collided}")

    # ===== FINAL RESULT =====
    traj_collision_rate = num_collided_trajs / num_trajs

    print("\n=== FINAL RESULT ===")
    print(f"Trajectory collision rate: {traj_collision_rate:.6f}")
    print(f"{num_collided_trajs} / {num_trajs} trajectories collided")

if __name__ == "__main__":
    main()