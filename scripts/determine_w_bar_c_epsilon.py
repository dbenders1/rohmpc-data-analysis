import argparse
import json
import logging
import math
import yaml

import matplotlib.pyplot as plt
import numpy as np

from os import path
from pathlib import Path
from rmpc_data_analysis import helpers

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
    config_path = f"{config_dir}/scripts/determine_w_bar_c_epsilon.yaml"
    data_dir = f"{package_dir}/data"
    data_sel_dir = f"{data_dir}/selected_data"

    # Read configuration parameters
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    runtime_json_names = config["data"]["runtime_json_names"]
    ros_rec_json_names = config["data"]["ros_rec_json_names"]
    n_idx_ignore = config["data"]["n_idx_ignore"]

    plot_settings = config["plot_settings"]
    n_rows_plot = plot_settings["n_rows"]
    n_cols_plot = plot_settings["n_cols"]
    plot_stage_idx_at_ax_idx = plot_settings["plot_stage_idx_at_ax_idx"]

    # Print warning if the number of runtime and ros recording json names do not match
    n_runtime_json_names = len(runtime_json_names)
    n_ros_rec_json_names = len(ros_rec_json_names)
    if n_runtime_json_names != n_ros_rec_json_names:
        raise ValueError(
            f"Number of runtime_json_names ({n_runtime_json_names}) and ros_rec_json_names ({n_ros_rec_json_names}) must match"
        )

    # Iterate over all runtime and ros recording json names
    for file_idx in range(n_runtime_json_names):
        runtime_json_name = runtime_json_names[file_idx]
        ros_rec_json_name = ros_rec_json_names[file_idx]

        # Read ROS runtime json data
        runtime_json_path = f"{runtime_json_dir}/{runtime_json_name}.json"
        log.warning(f"Selected runtime json file: {runtime_json_path}")
        if not path.exists(runtime_json_path):
            raise FileNotFoundError(f"Runtime json path {runtime_json_path} not found")
        with open(runtime_json_path, "r") as file:
            runtime_data = json.load(file)

        runtime_static_data = runtime_data["static_data"]
        stepsize_tmpc = runtime_static_data["stepsize"]
        steps_tmpc = runtime_static_data["steps"]
        P_delta = np.array(runtime_static_data["P_delta"])
        rho_c = np.array(runtime_static_data["rho_c"])

        # Read ROS recording json data
        ros_rec_json_path = f"{ros_rec_json_dir}/{ros_rec_json_name}.json"
        log.warning(f"Selected ROS recording json file: {ros_rec_json_path}")
        if not path.exists(ros_rec_json_path):
            raise FileNotFoundError(
                f"ROS recording json path {ros_rec_json_path} not found"
            )
        with open(ros_rec_json_path, "r") as file:
            ros_rec_data = json.load(file)

        time_precision = ros_rec_data["time_precision"]

        data_x_cur = ros_rec_data["/falcon/ground_truth/odometry"]
        t_x_cur = np.array(data_x_cur["t"])
        x_cur = np.array(data_x_cur["x"])

        data_x_cur_est = ros_rec_data["/mpc/rec/current_state"]
        t_x_cur_est = np.array(data_x_cur_est["t"])
        x_cur_est = np.array(data_x_cur_est["current_state"])

        data_pred_traj = ros_rec_data["/mpc/rec/predicted_trajectory/0"]
        t_pred_traj = np.array(data_pred_traj["t"])
        u_pred_traj = np.array(data_pred_traj["u_pred"])
        x_pred_traj = np.array(data_pred_traj["x_pred"])

        # Set times to a specific precision
        t_x_cur_est = np.round(t_x_cur_est, time_precision)
        t_pred_traj = np.round(t_pred_traj, time_precision)

        # Determine various parameters of runtime data
        n_tmpc = min(len(t_x_cur_est), len(t_pred_traj))
        N_tmpc = x_pred_traj.shape[1] - 1
        dt_tmpc = steps_tmpc * stepsize_tmpc

        # Align all data recorded in mpc
        t_x_cur_est = t_x_cur_est[:n_tmpc]
        x_cur_est = x_cur_est[:n_tmpc, :]
        x_cur_est = x_cur_est[:n_tmpc, :]
        t_pred_traj = t_pred_traj[:n_tmpc]
        u_pred_traj = u_pred_traj[:n_tmpc, :, :]
        x_pred_traj = x_pred_traj[:n_tmpc, :, :]

        # Ensure that all times are aligned
        if t_x_cur_est[-1] > t_x_cur[-1]:
            log.warning(
                "The estimated state has a recording after the ground truth state. Shrinking t_x_cur_est, x_cur_est, t_pred_traj, u_pred_traj, and x_pred_traj to the last time of t_x_cur"
            )
            t_x_cur_est = t_x_cur_est[t_x_cur_est <= t_x_cur[-1]]
            x_cur_est = x_cur_est[: len(t_x_cur_est)]
            t_pred_traj = t_pred_traj[t_pred_traj <= t_x_cur[-1]]
            u_pred_traj = u_pred_traj[: len(t_pred_traj)]
            x_pred_traj = x_pred_traj[: len(t_pred_traj)]
        print(f"t start: {t_x_cur_est[0]}")
        print(f"t end: {t_x_cur_est[-1]}")

        # Determine w_bar_c for all prediction stages at all time steps
        w_bar_c_all = np.zeros((n_tmpc - 1))
        for t in range(n_tmpc - 1):
            # t_x_cur_idx = np.abs(t_x_cur - t_x_cur_est[t + 1]).argmin()
            # x_err = x_cur[t_x_cur_idx, :] - x_pred_traj[t, 1, :]
            x_err = x_cur_est[t + 1, :] - x_pred_traj[t, 1, :]
            w_bar_c_all[t] = (
                np.sqrt(x_err.T @ P_delta @ x_err)
                * rho_c
                / (1 - math.exp(-rho_c * dt_tmpc))
            )
        w_bar_c_all = w_bar_c_all[n_idx_ignore:]
        w_bar_c = np.max(w_bar_c_all)
        print(f"{ros_rec_json_name} - w_bar_c: {w_bar_c}")

        # Determine epsilon at all time steps
        epsilon_all = np.zeros(n_tmpc)
        for t in range(n_tmpc):
            t_x_cur_idx = np.abs(t_x_cur - t_x_cur_est[t]).argmin()
            x_err = x_cur[t_x_cur_idx, :] - x_cur_est[t, :]
            epsilon_all[t] = np.sqrt(x_err.T @ P_delta @ x_err)
        epsilon_all = epsilon_all[n_idx_ignore:]
        epsilon = np.max(epsilon_all)
        print(f"{ros_rec_json_name} - epsilon: {epsilon}")

        # Set plotting properties
        helpers.set_plt_properties()
        props = helpers.set_fig_properties()

        # Create w_bar_c figure
        fig, ax = plt.subplots()
        fig.suptitle(f"{ros_rec_json_name} - computed w_bar_c over time")
        ax.plot(t_x_cur_est[n_idx_ignore + 1 :], w_bar_c_all)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\bar{w}^\mathrm{c}$")

        # Create w_bar_c sorted figure
        fig, ax = plt.subplots()
        fig.suptitle(f"{ros_rec_json_name} - computed w_bar_c sorted")
        ax.plot(np.arange(n_idx_ignore + 1, n_tmpc), sorted(w_bar_c_all))
        ax.set_xlabel("Index")
        ax.set_ylabel(r"$\bar{w}^\mathrm{c}$")

        # Create epsilon figure
        fig, ax = plt.subplots()
        fig.suptitle(f"{ros_rec_json_name} - computed epsilon over time")
        ax.plot(t_x_cur_est[n_idx_ignore:], epsilon_all)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\epsilon$")

        # Create epsilon sorted figure
        fig, ax = plt.subplots()
        fig.suptitle(f"{ros_rec_json_name} - computed epsilon sorted")
        ax.plot(np.arange(n_idx_ignore, n_tmpc), sorted(epsilon_all))
        ax.set_xlabel("Index")
        ax.set_ylabel("$\epsilon$")

    plt.show()
