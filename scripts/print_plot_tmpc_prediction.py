import argparse
import json
import logging
import math
import yaml

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
from os import path
from pathlib import Path

FLOAT_TOL = 1e-6


class Colors(Enum):
    GREY = (0.859375, 0.859375, 0.859375)
    ORANGE = (1, 0.6484375, 0)
    PURPLE = (0.5, 0, 0.5)
    VIRIDIS_0 = (0.21875, 0.347656, 0.546875)


class HandlerCircle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Circle(xy=center, radius=height / 2)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerEllipse(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width, height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def set_fig_properties():
    props = dict()
    props["titlepad"] = 4
    props["tickpad"] = 1
    props["xlabelpad"] = 0
    props["ylabelpad"] = 1
    props["zlabelpad"] = 0
    props["textsize"] = plt.rcParams["xtick.labelsize"]
    return props


def set_plt_properties():
    # Plot settings
    fontsize = 10
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["axes.labelsize"] = fontsize - 2
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize - 4
    plt.rcParams["ytick.labelsize"] = fontsize - 4
    plt.rcParams["legend.fontsize"] = fontsize - 4
    # Set dpi for saving figures
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300


np.set_printoptions(threshold=np.inf)


def compute_alpha(alpha_min, alpha_max, t, t_end):
    return alpha_max - (alpha_max - alpha_min) * (t / t_end) if t_end > 0 else alpha_max


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

    plot_settings = config["plot_settings"]
    r_tmpc_ref = plot_settings["r_tmpc_ref"]
    r_tmpc = plot_settings["r_tmpc"]
    r_x0 = plot_settings["r_x0"]
    r_x = plot_settings["r_x"]

    c_tmpc_ref = mcolors.CSS4_COLORS["red"]
    c_tmpc = Colors.ORANGE.value
    c_x0 = Colors.PURPLE.value
    c_x = mcolors.CSS4_COLORS["green"]
    c_xf = mcolors.CSS4_COLORS["black"]
    alpha_max = 1
    alpha_min = 0.5

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

    time_precision = ros_rec_data["time_precision"]

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
    t_x_cur = np.round(t_x_cur, time_precision)
    t_x_cur_est = np.round(t_x_cur_est, time_precision)
    t_ref_traj = np.round(t_ref_traj, time_precision)
    t_pred_traj = np.round(t_pred_traj, time_precision)

    # Check t_to_print and t_to_plot
    t_min = min(np.min(t_x_cur_est), np.min(t_ref_traj), np.min(t_pred_traj))
    t_max = max(np.max(t_x_cur_est), np.max(t_ref_traj), np.max(t_pred_traj))
    if t_to_print != -1 and (t_to_print < t_min or t_to_print > t_max):
        raise ValueError(
            f"t_to_print ({t_to_print}) should be either -1 or be contained within bounds [{t_min}, {t_max}]"
        )

    # Determine various parameters of runtime data
    n_tmpc = min(len(t_x_cur_est), len(t_ref_traj), len(t_pred_traj))
    N_tmpc = x_ref_traj.shape[1] - 1
    ts_tmpc = np.round(t_ref_traj[1] - t_ref_traj[0], time_precision)
    ts_sim = np.round(t_x_cur[1] - t_x_cur[0], time_precision)
    print(f"n_tmpc: {n_tmpc}, N_tmpc: {N_tmpc}, ts_tmpc: {ts_tmpc}, ts_sim: {ts_sim}")

    # Select data to print
    if t_to_print < 0:
        t_x_cur_print_idx = -1
        t_x_cur_est_print_idx = -1
        t_ref_traj_print_idx = -1
        t_pred_traj_print_idx = -1
    else:
        t_x_cur_print_idx = np.abs(t_x_cur - t_to_print).argmin()
        t_x_cur_est_print_idx = np.abs(t_x_cur_est - t_to_print).argmin()
        t_ref_traj_print_idx = np.abs(t_ref_traj - t_to_print).argmin()
        t_pred_traj_print_idx = np.abs(t_pred_traj - t_to_print).argmin()

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

    # Compute objective values
    if t_x_cur_est_print_idx < 0:
        pobj_comp = np.zeros(n_tmpc)
        for t_idx in range(n_tmpc):
            pobj_comp[t_idx] = 0
            for k in range(N_tmpc):
                pobj_comp[t_idx] += (
                    (x_pred_traj[t_idx, k, :] - x_ref_traj[t_idx, k, :])
                    @ Q
                    @ (x_pred_traj[t_idx, k, :] - x_ref_traj[t_idx, k, :])
                )
                pobj_comp[t_idx] += (
                    (u_pred_traj[t_idx, k, :] - u_ref_traj[t_idx, k, :])
                    @ R
                    @ (u_pred_traj[t_idx, k, :] - u_ref_traj[t_idx, k, :])
                )
    else:
        pobj_comp = 0
        for k in range(N_tmpc):
            pobj_comp += (
                (
                    x_pred_traj[t_pred_traj_print_idx, k, :]
                    - x_ref_traj[t_ref_traj_print_idx, k, :]
                )
                @ Q
                @ (
                    x_pred_traj[t_pred_traj_print_idx, k, :]
                    - x_ref_traj[t_ref_traj_print_idx, k, :]
                )
            )
            pobj_comp += (
                (
                    u_pred_traj[t_pred_traj_print_idx, k, :]
                    - u_ref_traj[t_ref_traj_print_idx, k, :]
                )
                @ R
                @ (
                    u_pred_traj[t_pred_traj_print_idx, k, :]
                    - u_ref_traj[t_ref_traj_print_idx, k, :]
                )
            )
    # print(f"pobj_comp: {pobj_comp}")

    # Select data to plot
    if t_to_plot == -1:
        raise ValueError("t_to_plot must be a non-negative value")
    else:
        t_x_cur_plot_start_idx = np.abs(t_x_cur - t_to_plot).argmin()
        t_x_cur_plot_end_idx = np.abs(
            t_x_cur - np.round(t_to_plot + N_tmpc * ts_tmpc, time_precision)
        ).argmin()
        t_x_cur_est_plot_idx = np.abs(t_x_cur_est - t_to_plot).argmin()
        t_ref_traj_plot_idx = np.abs(t_ref_traj - t_to_plot).argmin()
        t_pred_traj_plot_idx = np.abs(t_pred_traj - t_to_plot).argmin()

    print(f"t_x_cur_plot_start_idx: {t_x_cur_plot_start_idx}")
    print(f"t_x_cur_plot_end_idx: {t_x_cur_plot_end_idx}")

    # Create figure
    set_plt_properties()
    props = set_fig_properties()
    fig, ax = plt.subplots()

    # Add TMPC reference trajectory to plot
    handles_tmpc_ref = []
    for k in range(N_tmpc + 1):
        tmpc_ref = Circle(
            (
                x_ref_traj[t_ref_traj_plot_idx, k, 0],
                x_ref_traj[t_ref_traj_plot_idx, k, 1],
            ),
            r_tmpc_ref,
            facecolor=c_tmpc_ref,
            alpha=compute_alpha(alpha_min, alpha_max, k * ts_tmpc, N_tmpc * ts_tmpc),
            zorder=2,
            label="TMPC ref traj",
        )
        handles_tmpc_ref.append(ax.add_patch(tmpc_ref))

    # Add TMPC prediction to plot
    handle_x0 = ax.add_patch(
        Circle(
            (
                x_cur_est[t_x_cur_est_plot_idx, 0],
                x_cur_est[t_x_cur_est_plot_idx, 1],
            ),
            r_x0,
            facecolor=c_x0,
            alpha=compute_alpha(alpha_min, alpha_max, 0, N_tmpc * ts_tmpc),
            zorder=4,
            label="TMPC init state",
        )
    )
    handles_tmpc_pred = []
    for k in range(1, N_tmpc + 1):
        tmpc_pred = Circle(
            (
                x_pred_traj[t_pred_traj_plot_idx, k, 0],
                x_pred_traj[t_pred_traj_plot_idx, k, 1],
            ),
            r_tmpc,
            facecolor=c_tmpc,
            alpha=compute_alpha(alpha_min, alpha_max, k * ts_tmpc, N_tmpc * ts_tmpc),
            zorder=3,
            label="TMPC prediction",
        )
        handles_tmpc_pred.append(ax.add_patch(tmpc_pred))

    # TODO: add increasing tube size to plot

    # Add closed-loop state to plot
    handles_x = []
    for t_x_cur_plot_idx in range(t_x_cur_plot_start_idx, t_x_cur_plot_end_idx + 1):
        x = Circle(
            (x_cur[t_x_cur_plot_idx, 0], x_cur[t_x_cur_plot_idx, 1]),
            r_x,
            facecolor=c_x,
            alpha=compute_alpha(
                alpha_min,
                alpha_max,
                (t_x_cur_plot_idx - t_x_cur_plot_start_idx) * ts_sim,
                N_tmpc * ts_tmpc,
            ),
            zorder=5,
            label="Closed-loop state",
        )
        handles_x.append(ax.add_patch(x))

    # Add title, etc. to plot
    # ax.set_title(f"TMPC prediction", pad=props["titlepad"])
    ax.set_xlim(-0.01, 0.01)
    ax.set_ylim(0.025, 0.045)
    # ax.set_xlim(-1.166667, 1.166667)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_xlim(-0.8, 2)
    # ax.set_ylim(-0.6, 0.6)
    # ax.set_xlim(0, 1.166666)
    # ax.set_ylim(-0.5, 0)
    # edge_val = 4
    # ax.set_xlim(-edge_val, edge_val)
    # ax.set_ylim(-edge_val, edge_val)
    ax.set_xlabel("$p^x$ (m)")
    ax.set_ylabel("$p^y$ (m)")
    ax.xaxis.labelpad = props["xlabelpad"]
    ax.yaxis.labelpad = props["ylabelpad"]
    ax.tick_params(pad=props["tickpad"])
    ax.set_axisbelow(True)
    handles = [
        handles_tmpc_ref[0],
        handle_x0,
        handles_tmpc_pred[0],
        handles_x[0],
    ]
    ax.legend(
        handles=handles,
        handler_map={
            Circle: HandlerCircle(),
            Ellipse: HandlerEllipse(),
        },
        loc="upper right",
    )
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    plt.show()
