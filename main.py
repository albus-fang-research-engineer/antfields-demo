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
        meshpath = "data/mesh_denmark_normalized.obj"
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
    import os

    # ===== CREATE GLOBAL RUN FOLDER =====
    global_base = modelPath

    global_run_id = 0
    while True:
        global_folder = os.path.join(global_base, f"TUNED_BASELINE_GLOBAL_RUN_{global_run_id}")
        # global_folder = os.path.join(global_base, f"PAIR1_DEMO_{global_run_id}")
        if not os.path.exists(global_folder):
            os.makedirs(global_folder)
            break
        global_run_id += 1

    print(f"Global run folder: {global_folder}")
    lengths = []
    num_runs =5
    collisions = 0
    all_planning_times = []
    control_efforts = []
    traversed_control_efforts = []
    cc_segment_control_efforts = []
    all_planned_path_efforts = []
    all_nominal_path_efforts = []
    all_cc_active_flags = []
    all_cc_active_full_flags = []
    all_deviations = []
    total_paths_planned = 0
    total_paths_cc_triggered = 0
    cc_modified_dir = os.path.join(global_folder, "cc_modified_paths")
    total_cc_modified_saved = 0
    for i in range(num_runs):
        print(f"\n===== Run {i+1}/{num_runs} =====")

        model = md.Model(global_folder, 3, scale_factor, mode, renderer, device='cuda:0')

        result = model.train()
        all_planning_times.extend(result["planning_times"])
        control_efforts.append(result["control_effort"])
        traversed_control_efforts.append(result["traversed_control_effort"])
        cc_segment_control_efforts.append(result["cc_segment_control_effort"])
        all_planned_path_efforts.extend(result["planned_path_control_efforts"])
        all_nominal_path_efforts.extend(result["nominal_path_control_efforts"])
        all_cc_active_flags.extend(result["cc_active_per_plan"])
        all_cc_active_full_flags.extend(result["cc_active_full_path_per_plan"])
        all_deviations.extend(result["deviations"])
        total_paths_planned += result["paths_planned"]
        total_paths_cc_triggered += result["paths_cc_triggered"]
        # Save the nominal/optimized pair for every planning call where CC modified the path
        cc_modified = result["cc_modified_paths"]
        if len(cc_modified) > 0:
            os.makedirs(cc_modified_dir, exist_ok=True)
            for rec in cc_modified:
                np.savez(
                    os.path.join(cc_modified_dir, f"run{i:02d}_epoch{rec['epoch']:04d}.npz"),
                    nominal=rec["nominal"],
                    optimized=rec["optimized"],
                    traj_ind=rec["traj_ind"],
                    cc_active_flags=rec["cc_active_flags"],
                    max_deviation=rec["max_deviation"],
                    mean_deviation=rec["mean_deviation"],
                )
            total_cc_modified_saved += len(cc_modified)
            print(f"CC modified {len(cc_modified)} planned path(s) this run "
                  f"(max deviation {max(r['max_deviation'] for r in cc_modified) * scale_factor:.4f} m), "
                  f"saved to {cc_modified_dir}")
        print(f"Control effort (planned, executed segments): {result['control_effort'] * scale_factor**2:.4f} m^2")
        print(f"Control effort (whole traversed path):       {result['traversed_control_effort'] * scale_factor**2:.4f} m^2")
        print(f"Control effort (CC-optimized segments +/-1): {result['cc_segment_control_effort'] * scale_factor**2:.4f} m^2")
        if len(result["planned_path_control_efforts"]) > 0:
            print(f"Control effort (full planned path, mean per plan): {np.mean(result['planned_path_control_efforts']) * scale_factor**2:.4f} m^2")
            run_full_flags = np.array(result["cc_active_full_path_per_plan"], dtype=bool)
            if run_full_flags.any():
                run_cc_efforts = np.array(result["planned_path_control_efforts"])[run_full_flags]
                print(f"Control effort (CC-optimized plans only, mean per plan, n={int(run_full_flags.sum())}): {np.mean(run_cc_efforts) * scale_factor**2:.4f} m^2")
        run_nominal = np.array(result["nominal_path_control_efforts"])
        run_optimized = np.array(result["planned_path_control_efforts"])
        if len(run_nominal) > 0:
            run_added = run_optimized - run_nominal
            run_valid = run_nominal > 0
            print(f"CC added effort vs nominal (mean per plan, this run): {np.mean(run_added) * scale_factor**2:.4f} m^2"
                  + (f" ({np.mean(100.0 * run_added[run_valid] / run_nominal[run_valid]):+.2f}%)" if run_valid.any() else ""))
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
    planning_times = np.array(all_planning_times)
    print("\n===== FINAL RESULTS =====")
    print(f"Runs completed: {len(lengths)}")
    print(f"Mean path length: {np.mean(lengths) * 10:.4f} m")
    print(f"Std: {np.std(lengths)*10:.4f} m")
    print(f"Collision rate: {collisions / num_runs:.2f}")
    print(f"Planning calls:     {len(planning_times)}")
    print(f"Mean planning time: {np.mean(planning_times)*1000:.2f} ms")
    print(f"Std planning time:  {np.std(planning_times)*1000:.2f} ms")

    control_efforts = np.array(control_efforts)
    traversed_control_efforts = np.array(traversed_control_efforts)
    cc_segment_control_efforts = np.array(cc_segment_control_efforts)
    deviations = np.array(all_deviations)
    print(f"Mean control effort per run (planned, executed segments): {np.mean(control_efforts) * scale_factor**2:.4f} m^2")
    print(f"Std control effort per run  (planned, executed segments): {np.std(control_efforts) * scale_factor**2:.4f} m^2")
    print(f"Mean control effort per run (whole traversed path):       {np.mean(traversed_control_efforts) * scale_factor**2:.4f} m^2")
    print(f"Std control effort per run  (whole traversed path):       {np.std(traversed_control_efforts) * scale_factor**2:.4f} m^2")
    print(f"Mean control effort per run (CC-optimized segments +/-1): {np.mean(cc_segment_control_efforts) * scale_factor**2:.4f} m^2")
    print(f"Std control effort per run  (CC-optimized segments +/-1): {np.std(cc_segment_control_efforts) * scale_factor**2:.4f} m^2")
    planned_path_efforts = np.array(all_planned_path_efforts)
    if len(planned_path_efforts) > 0:
        print(f"Planning calls with full path logged: {len(planned_path_efforts)}")
        print(f"Mean control effort per plan (full planned path):         {np.mean(planned_path_efforts) * scale_factor**2:.4f} m^2")
        print(f"Std control effort per plan  (full planned path):         {np.std(planned_path_efforts) * scale_factor**2:.4f} m^2")
        cc_full_flags = np.array(all_cc_active_full_flags, dtype=bool)
        if len(cc_full_flags) == len(planned_path_efforts) and cc_full_flags.any():
            cc_plan_efforts = planned_path_efforts[cc_full_flags]
            print(f"Plans where CC optimized the path: {int(cc_full_flags.sum())} / {len(planned_path_efforts)}")
            print(f"Mean control effort per plan (CC-optimized plans only):   {np.mean(cc_plan_efforts) * scale_factor**2:.4f} m^2")
            print(f"Std control effort per plan  (CC-optimized plans only):   {np.std(cc_plan_efforts) * scale_factor**2:.4f} m^2")
    nominal_path_efforts = np.array(all_nominal_path_efforts)
    cc_active_flags = np.array(all_cc_active_flags, dtype=bool)
    if len(nominal_path_efforts) > 0 and len(nominal_path_efforts) == len(planned_path_efforts):
        added_efforts = planned_path_efforts - nominal_path_efforts
        valid = nominal_path_efforts > 0
        pct_added = 100.0 * added_efforts[valid] / nominal_path_efforts[valid]
        print("--- CC added control effort relative to nominal path (per planning call, full planned path) ---")
        print(f"Mean added effort (all plans):    {np.mean(added_efforts) * scale_factor**2:+.4f} m^2 ({np.mean(pct_added):+.2f}%)")
        print(f"Std added effort  (all plans):    {np.std(added_efforts) * scale_factor**2:.4f} m^2 ({np.std(pct_added):.2f}%)")
        if cc_active_flags.any():
            valid_cc = valid & cc_active_flags
            added_cc = added_efforts[cc_active_flags]
            pct_cc = 100.0 * added_efforts[valid_cc] / nominal_path_efforts[valid_cc]
            print(f"Mean added effort (CC-active plans only, n={int(cc_active_flags.sum())}): "
                  f"{np.mean(added_cc) * scale_factor**2:+.4f} m^2 ({np.mean(pct_cc):+.2f}%)")
            print(f"Std added effort  (CC-active plans only): "
                  f"{np.std(added_cc) * scale_factor**2:.4f} m^2 ({np.std(pct_cc):.2f}%)")
    if len(deviations) > 0:
        print(f"Waypoints compared:          {len(deviations)}")
        print(f"Mean deviation from nominal: {np.mean(deviations) * scale_factor:.4f} m")
        print(f"Std deviation from nominal:  {np.std(deviations) * scale_factor:.4f} m")
        print(f"Max deviation from nominal:  {np.max(deviations) * scale_factor:.4f} m")
    if total_paths_planned > 0:
        print(f"Paths planned:               {total_paths_planned}")
        print(f"Paths with CC active:        {total_paths_cc_triggered} "
              f"({100.0 * total_paths_cc_triggered / total_paths_planned:.1f}%)")
    if total_cc_modified_saved > 0:
        print(f"CC-modified paths saved:     {total_cc_modified_saved} -> {cc_modified_dir}")
if __name__ == '__main__':
    main()
