# SYSTEMS AND METHODS FOR PHYSICS-INFORMED NEURAL NETWORKS FOR ROBOT MAPPING
# Copyright © 2025 Purdue University.
# Developed by Yuchen Liu, Ruiqi Ni and CORAL Lab.
# Purdue Research Foundation Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

import numpy as np
import argparse
from models import model_igibson as md

# Define modes
EXPLORATION = 1 
READ_FROM_COOKED_DATA = 2


# Define the model path
scale_factor = 10
modelPath = './Experiments'

def main():
    parser = argparse.ArgumentParser(description='Train the model with or without exploration.')
    parser.add_argument('--no_explore', action='store_true', help='Disable exploration mode.')
    
    args = parser.parse_args()

    # Set the mode based on the --no_explore flag
    if args.no_explore:
        mode = READ_FROM_COOKED_DATA
    else:
        mode = EXPLORATION  # Default to exploration mode

    renderer = None
    if mode in [EXPLORATION]:
        from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
        meshpath = "data/mesh_superior_normalized.obj"
        meshpath = "data/mesh_allensville_normalized.obj"
        # meshpath = "data/mesh_denmark_normalized.obj"
        # meshpath = "data/mesh2.obj"
        # meshpath = "data/default_mesh.obj"
        renderer = MeshRenderer(width=1200, height=680)
        renderer.load_object(meshpath, scale=np.array([1, 1, 1]) * scale_factor)
        renderer.add_instance_group([0])
        camera_pose = np.array([0, 0, 1])    
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 0])
        renderer.set_fov(90)

    # Initialize and train the model
    # model = md.Model(modelPath, 3, scale_factor, mode, renderer, device='cuda:0')
    # model.train()
    global_base = modelPath
    import os

    # ===== CREATE GLOBAL RUN FOLDER =====
    global_base = modelPath

    global_run_id = 0
    while True:
        global_folder = os.path.join(global_base, f"CHANCE_CONSTRAINED_GLOBAL_RUN_{global_run_id}")
        if not os.path.exists(global_folder):
            os.makedirs(global_folder)
            break
        global_run_id += 1

    print(f"Global run folder: {global_folder}")
    lengths = []
    num_runs = 5
    collisions = 0
    all_policy_times = []          # <-- add
    all_optimizer_times = []       # <-- add
    all_total_times = []           # <-- add

    # ---- chance-constrained optimizer stats ----
    all_modified = []              # per planning call: optimizer changed nominal path?
    all_nominal_efforts = []       # planned path control effort (nominal)
    all_optimized_efforts = []     # planned path control effort (optimized)
    all_segment_efforts = []       # modified segments +/- 1 waypoint (optimized)
    all_segment_nominal_efforts = []
    all_dev_means = []             # nominal-vs-optimized deviation per call
    all_dev_maxes = []
    all_nominal_path_lengths = []  # nominal planned path xy length per call
    traversed_efforts = []         # executed trajectory control effort per run

    for i in range(num_runs):
        print(f"\n===== Run {i+1}/{num_runs} =====")

        model = md.Model(global_folder, 3, scale_factor, mode, renderer, device='cuda:0')

        result = model.train()
        all_policy_times.extend(result["policy_times"])              # <-- add
        all_optimizer_times.extend(result["optimizer_times"])        # <-- add
        all_total_times.extend(result["total_planning_times"])       # <-- add
        all_modified.extend(result["optimizer_modified"])
        all_nominal_efforts.extend(result["nominal_efforts"])
        all_optimized_efforts.extend(result["optimized_efforts"])
        all_segment_efforts.extend(result["segment_efforts"])
        all_segment_nominal_efforts.extend(result["segment_nominal_efforts"])
        all_dev_means.extend(result["deviation_means"])
        all_dev_maxes.extend(result["deviation_maxes"])
        all_nominal_path_lengths.extend(result["nominal_path_lengths"])
        if result["traversed_effort"] is not None:
            traversed_efforts.append(result["traversed_effort"])
        if result["collision"]:
            collisions += 1
            print("Run ended with collision")
        elif result["length"] is not None:
            lengths.append(result["length"])
        else:
            print("Warning: run did not reach goal")
        # if path_length is not None:
        #     lengths.append(path_length)
        # else:
        #     print("Warning: run did not reach goal")

    lengths = np.array(lengths)
    policy_times = np.array(all_policy_times)
    optimizer_times = np.array(all_optimizer_times)
    total_times = np.array(all_total_times)
    print("\n===== FINAL RESULTS =====")
    print(f"Runs completed: {len(lengths)}")
    print(f"Mean path length: {np.mean(lengths) * 10:.4f} m")
    print(f"Std: {np.std(lengths)*10:.4f} m")
    print(f"Collision rate: {collisions / num_runs:.2f}")
    if len(total_times) > 0:
        print(f"Planning calls:        {len(total_times)}")
        print(f"Mean policy time:      {np.mean(policy_times)*1000:.2f} ms  (std {np.std(policy_times)*1000:.2f})")
        print(f"Mean optimizer time:   {np.mean(optimizer_times)*1000:.2f} ms  (std {np.std(optimizer_times)*1000:.2f})")
        print(f"Mean total plan time:  {np.mean(total_times)*1000:.2f} ms  (std {np.std(total_times)*1000:.2f})")
    else:
        print("No planning calls recorded.")

    # ---- chance-constrained optimizer stats ----
    # Control effort = sum of squared xy step displacements; x scale_factor^2 -> m^2.
    effort_scale = scale_factor ** 2
    print("\n----- Chance-constrained optimizer stats -----")
    if len(all_modified) > 0:
        n_mod = int(np.sum(all_modified))
        print(f"Paths modified by optimizer:  {100.0 * n_mod / len(all_modified):.1f}% "
              f"({n_mod}/{len(all_modified)} planning calls)")
    if len(traversed_efforts) > 0:
        print(f"Traversed path control effort:   mean {np.mean(traversed_efforts) * effort_scale:.4f} m^2 per run "
              f"(std {np.std(traversed_efforts) * effort_scale:.4f})")
    if len(all_optimized_efforts) > 0:
        print(f"Planned path control effort:     nominal {np.mean(all_nominal_efforts) * effort_scale:.6f} m^2 "
              f"(std {np.std(all_nominal_efforts) * effort_scale:.6f}), "
              f"optimized {np.mean(all_optimized_efforts) * effort_scale:.6f} m^2 "
              f"(std {np.std(all_optimized_efforts) * effort_scale:.6f}) (mean per planning call)")
        nominal_arr = np.array(all_nominal_efforts)
        optimized_arr = np.array(all_optimized_efforts)
        valid = nominal_arr > 0
        added_pcts = 100.0 * (optimized_arr[valid] - nominal_arr[valid]) / nominal_arr[valid]
        print(f"Added control effort vs nominal: mean {np.mean(added_pcts):+.2f}% "
              f"(std {np.std(added_pcts):.2f}) (per planning call, all plans)")
        # Same per-plan percentage, restricted to plans the optimizer actually
        # modified. all_modified is the full-path flag (wp_flags.any()), indexed
        # per planning call alongside the efforts, so mask and metric share scope.
        cc_active = np.array(all_modified, dtype=bool)
        if len(cc_active) == len(nominal_arr):
            valid_cc = valid & cc_active
            if valid_cc.any():
                cc_pcts = 100.0 * (optimized_arr[valid_cc] - nominal_arr[valid_cc]) / nominal_arr[valid_cc]
                print(f"Added control effort vs nominal: mean {np.mean(cc_pcts):+.2f}% "
                      f"(std {np.std(cc_pcts):.2f}) (per planning call, "
                      f"CC-active plans only, n={int(valid_cc.sum())})")
        else:
            print("  (CC-active subset skipped: modified-flag count != effort count)")
        # Added effort normalized by nominal path length. Effort scales with
        # scale_factor^2, length with scale_factor, so the ratio scales with
        # scale_factor -> units m^2/m = m.
        length_arr = np.array(all_nominal_path_lengths)
        if len(length_arr) == len(nominal_arr):
            valid_len = length_arr > 0
            norm_added = (optimized_arr[valid_len] - nominal_arr[valid_len]) / length_arr[valid_len] * scale_factor
            if valid_len.any():
                print(f"Added effort / path length:      mean {np.mean(norm_added):+.6f} m^2/m "
                      f"(std {np.std(norm_added):.6f}) (per planning call, all plans, n={int(valid_len.sum())})")
            if len(cc_active) == len(nominal_arr):
                valid_len_cc = valid_len & cc_active
                if valid_len_cc.any():
                    norm_added_cc = (optimized_arr[valid_len_cc] - nominal_arr[valid_len_cc]) / length_arr[valid_len_cc] * scale_factor
                    print(f"Added effort / path length:      mean {np.mean(norm_added_cc):+.6f} m^2/m "
                          f"(std {np.std(norm_added_cc):.6f}) (per planning call, "
                          f"CC-active plans only, n={int(valid_len_cc.sum())})")
        else:
            print("  (Normalized added effort skipped: path-length count != effort count)")
    if len(all_segment_efforts) > 0:
        print(f"Optimized-segment control effort: optimized {np.mean(all_segment_efforts) * effort_scale:.6f} m^2 "
              f"(std {np.std(all_segment_efforts) * effort_scale:.6f}), "
              f"nominal {np.mean(all_segment_nominal_efforts) * effort_scale:.6f} m^2 "
              f"(std {np.std(all_segment_nominal_efforts) * effort_scale:.6f}) "
              f"(mean over {len(all_segment_efforts)} modified segments, +/- 1 waypoint)")
    if len(all_dev_means) > 0:
        print(f"Nominal-vs-optimized deviation:  mean {np.mean(all_dev_means) * scale_factor:.6f} m "
              f"(std {np.std(all_dev_means) * scale_factor:.6f}), "
              f"max {np.max(all_dev_maxes) * scale_factor:.6f} m")

if __name__ == '__main__':
    main()
