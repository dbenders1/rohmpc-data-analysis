from os import path
import argparse
import json
import logging
import math
import yaml

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

FLOAT_TOL = 1e-6

np.set_printoptions(threshold=np.inf)


def quaternion_to_zyx_euler(q):
    """
    Convert a quaternion into ZYX Euler angles (roll, pitch, yaw)
    q = [qw, qx, qy, qz]
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


if __name__ == "__main__":
    # Log settings
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="something")
    parser.add_argument("-v", "--verbose", action="count", default=0, dest="verbosity")
    args = parser.parse_args()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN - 10 * args.verbosity)

    # User settings
    package_dir = Path(__file__).parents[1]
    runtime_json_dir = f"{package_dir}/../rmpc/mpc_tools/recorded_data"
    if not path.exists(runtime_json_dir):
        log.warning(
            f"Directory {runtime_json_dir} does not exist! Please ensure that the rmpc submodule is cloned"
        )
        exit(1)
    ros_rec_json_dir = f"{package_dir}/../rosbag2json/data/converted_bags"
    if not path.exists(ros_rec_json_dir):
        log.warning(
            f"Directory {ros_rec_json_dir} does not exist! Please ensure that the rosbag2json submodule is cloned"
        )
        exit(1)
    config_dir = f"{package_dir}/config"
    config_path = f"{config_dir}/scripts/print_plot_tmpc_prediction.yaml"
    data_dir = f"{package_dir}/data"
    data_sel_dir = f"{data_dir}/selected_data"

    # Read configuration parameters
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    runtime_json_name = config["data"]["runtime_json_name"]
    ros_rec_json_name = config["data"]["ros_rec_json_name"]
    t_to_print = config["data"]["t_to_print"]
    t_to_plot = config["data"]["t_to_plot"]

    # Read ROS runtime json data
    runtime_json_path = f"{runtime_json_dir}/{runtime_json_name}.json"
    log.warning(f"Selected runtime json file: {runtime_json_path}")
    if not path.exists(runtime_json_path):
        raise FileNotFoundError(f"Runtime json path {runtime_json_path} not found")
    with open(runtime_json_path, "r") as file:
        runtime_data = json.load(file)

    runtime_static_data = runtime_data["static_data"]
    Q = np.array(runtime_static_data["Q"])
    R = np.array(runtime_static_data["R"])

    # Read ROS recording json data
    ros_rec_json_path = f"{ros_rec_json_dir}/{ros_rec_json_name}.json"
    log.warning(f"Selected ROS recording json file: {ros_rec_json_path}")
    if not path.exists(ros_rec_json_path):
        raise FileNotFoundError(
            f"ROS recording json path {ros_rec_json_path} not found"
        )
    with open(ros_rec_json_path, "r") as file:
        ros_rec_data = json.load(file)

    data_x_cur = ros_rec_data["/falcon/ground_truth/odometry"]
    t_x_cur = np.array(data_x_cur["t"])
    p_cur = np.array(data_x_cur["p"])
    q_cur = np.array(data_x_cur["q"])
    v_cur = np.array(data_x_cur["v"])
    wb_cur = np.array(data_x_cur["wb"])
    # Convert quaternion to Euler angles
    eul_cur = np.zeros((3, q_cur.shape[1]))
    for t in range(q_cur.shape[1]):
        eul_cur[:, t] = quaternion_to_zyx_euler(q_cur[:, t])
    x_cur = np.concatenate((p_cur, eul_cur, v_cur, wb_cur), axis=0)
    x_cur = x_cur.T
    print(f"x_cur.shape: {x_cur.shape})")

    data_x_cur_est = ros_rec_data["/mpc/rec/current_state"]
    t_x_cur_est = np.array(data_x_cur_est["t"])
    x_cur_est = np.array(data_x_cur_est["current_state"])
    print(f"x_cur_est.shape: {np.array(x_cur_est).shape}")

    data_ref_traj = ros_rec_data["/mpc/rec/reference_trajectory/0"]
    t_ref_traj = np.array(data_ref_traj["t"])
    u_ref_traj = np.array(data_ref_traj["u_ref"])
    x_ref_traj = np.array(data_ref_traj["x_ref"])
    print(f"u_ref_traj_.shape: {u_ref_traj.shape}")
    print(f"x_ref_traj_.shape: {x_ref_traj.shape}")

    data_pred_traj = ros_rec_data["/mpc/rec/predicted_trajectory/0"]
    t_pred_traj = np.array(data_pred_traj["t"])
    u_pred_traj = np.array(data_pred_traj["u_pred"])
    x_pred_traj = np.array(data_pred_traj["x_pred"])
    print(f"u_pred_traj_.shape: {u_pred_traj.shape}")
    print(f"x_pred_traj_.shape: {x_pred_traj.shape}")

    # Set times to a specific precision
    time_precision = 5
    t_x_cur = np.round(t_x_cur, time_precision)
    t_x_cur_est = np.round(t_x_cur_est, time_precision)
    t_ref_traj = np.round(t_ref_traj, time_precision)
    t_pred_traj = np.round(t_pred_traj, time_precision)

    # Select data to print
    if t_to_print < 0:
        idx_to_print = -1
    else:
        t_cur_idx = np.abs(t_x_cur - t_to_print).argmin()
        t_cur_est_idx = np.abs(t_x_cur_est - t_to_print).argmin()
        t_ref_traj_idx = np.abs(t_ref_traj - t_to_print).argmin()
        t_pred_traj_idx = np.abs(t_pred_traj - t_to_print).argmin()

    # Check if the estimated state overlaps with the ground truth state
    x_cur_mpc_start = x_cur[np.where(np.isin(t_x_cur, t_x_cur_est))[0], :]
    states_equal = np.all(np.abs(x_cur_mpc_start - x_cur_est) < FLOAT_TOL, axis=1)
    if np.all(states_equal):
        log.warning(
            "The estimated state matches the ground truth state at the MPC start times"
        )
    else:
        log.warning(
            "The estimated state does not match the ground truth state at the MPC start times"
        )
        first_diff_idx = np.where(~states_equal)[0][0]
        if first_diff_idx < x_cur_mpc_start.shape[0]:
            log.warning(
                f"First difference at index {first_diff_idx}: "
                f"First difference at time {t_x_cur_est[first_diff_idx]} s: "
                f"x_cur_mpc_start: {x_cur_mpc_start[first_diff_idx]}, "
                f"x_cur_est: {x_cur_est[first_diff_idx]}"
            )
