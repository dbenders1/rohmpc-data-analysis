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
from matplotlib.patches import Circle, Ellipse, Patch, Polygon
from os import path
from pathlib import Path
from rmpc_data_analysis import helpers

FLOAT_TOL = 1e-6
NU = 4
NX = 12


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


def compute_pobj(u_pred_traj, x_pred_traj, u_ref_traj, x_ref_traj, N, Q_dt, R_dt, P_dt):
    pobj = 0
    for k in range(N):
        pobj += (
            (x_pred_traj[k, :] - x_ref_traj[k, :])
            @ Q_dt
            @ (x_pred_traj[k, :] - x_ref_traj[k, :])
        )
        pobj += (
            (u_pred_traj[k, :] - u_ref_traj[k, :])
            @ R_dt
            @ (u_pred_traj[k, :] - u_ref_traj[k, :])
        )
    pobj += (
        (x_pred_traj[N, :] - x_ref_traj[N, :])
        @ P_dt
        @ (x_pred_traj[N, :] - x_ref_traj[N, :])
    )
    return pobj


def compute_track_err(
    u_pred_traj, x_pred_traj, u_ref_traj, x_ref_traj, N, u_idc, x_idc
):
    Q_dt = np.zeros((NX, NX))
    Q_dt[x_idc, x_idc] = 1
    R_dt = np.zeros((NU, NU))
    R_dt[u_idc, u_idc] = 1
    P_dt = np.zeros((NX, NX))
    P_dt[x_idc, x_idc] = 1
    track_err = compute_pobj(
        u_pred_traj, x_pred_traj, u_ref_traj, x_ref_traj, N, Q_dt, R_dt, P_dt
    )
    return track_err


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

    print_settings = config["print_settings"]
    t_print = print_settings["t"]
    check_gt_est_states = print_settings["check_gt_est_states"]
    print_pobj_comp = print_settings["print_pobj_comp"]
    print_track_err = print_settings["print_track_err"]

    do_plot_settings = config["do_plot"]
    do_plot_obs = do_plot_settings["obs"]
    do_plot_tmpc_ref = do_plot_settings["tmpc_ref"]
    do_plot_rpi_tube = do_plot_settings["rpi_tube"]
    do_plot_tmpc_pred = do_plot_settings["tmpc_pred"]
    do_plot_growing_tube = do_plot_settings["growing_tube"]
    do_plot_x = do_plot_settings["x"]

    plot_settings = config["plot_settings"]
    t_plot = plot_settings["t"]
    k_ref = plot_settings["k_ref"]
    k_pred = plot_settings["k_pred"]
    t_div_tube = plot_settings["t_div_tube"]
    linewidth_rpi_tube = plot_settings["linewidth_rpi_tube"]
    r_tmpc_ref = plot_settings["r_tmpc_ref"]
    r_tmpc = plot_settings["r_tmpc"]
    r_x0 = plot_settings["r_x0"]
    r_x = plot_settings["r_x"]

    c_obs_inflated = Colors.GREY.value
    c_rpi_tube = mcolors.CSS4_COLORS["blue"]
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
    P = np.array(runtime_static_data["P"])
    alpha = runtime_static_data["alpha"]
    c_o = runtime_static_data["c_o"]
    robot_radius = runtime_static_data["robot_radius"]
    stepsize_tmpc = runtime_static_data["stepsize"]
    steps_tmpc = runtime_static_data["steps"]

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

    obs_len = 0.6
    obs_dim = np.array([2, 1]) * obs_len
    data_obs_0 = ros_rec_data["/grid/obs/rec/rectangle2x1_0"]
    obs_0_p = np.array(data_obs_0["p"])
    obs_0_eul = np.array(data_obs_0["eul"])
    obs_0 = {"p": obs_0_p, "eul": obs_0_eul, "dim": obs_dim}

    data_obs_1 = ros_rec_data["/grid/obs/rec/rectangle2x1_1"]
    obs_1_p = np.array(data_obs_1["p"])
    obs_1_eul = np.array(data_obs_1["eul"])
    obs_1 = {"p": obs_1_p, "eul": obs_1_eul, "dim": obs_dim}
    obs_list = [obs_0, obs_1]

    data_x_cur = ros_rec_data["/falcon/ground_truth/odometry"]
    t_x_cur = np.array(data_x_cur["t"])
    x_cur = np.array(data_x_cur["x"])

    data_x_cur_est = ros_rec_data["/mpc/rec/current_state"]
    t_x_cur_est = np.array(data_x_cur_est["t"])
    x_cur_est = np.array(data_x_cur_est["current_state"])

    data_ref_traj = ros_rec_data["/mpc/rec/reference_trajectory/0"]
    t_ref_traj = np.array(data_ref_traj["t"])
    u_ref_traj = np.array(data_ref_traj["u_ref"])
    x_ref_traj = np.array(data_ref_traj["x_ref"])

    data_pred_traj = ros_rec_data["/mpc/rec/predicted_trajectory/0"]
    t_pred_traj = np.array(data_pred_traj["t"])
    u_pred_traj = np.array(data_pred_traj["u_pred"])
    x_pred_traj = np.array(data_pred_traj["x_pred"])

    # Computed vertices of obstacle and vertices required to create inflated rectangular obstacles
    obs_verts = helpers.compute_inflated_obstacle_vertices(obs_list, robot_radius)

    # Set times to a specific precision
    t_x_cur = np.round(t_x_cur, time_precision)
    t_x_cur_est = np.round(t_x_cur_est, time_precision)
    t_ref_traj = np.round(t_ref_traj, time_precision)
    t_pred_traj = np.round(t_pred_traj, time_precision)

    # Determine various parameters of runtime data
    n_tmpc = min(len(t_x_cur_est), len(t_ref_traj), len(t_pred_traj))
    N_tmpc = x_ref_traj.shape[1] - 1
    dt_tmpc = steps_tmpc * stepsize_tmpc
    ts_tmpc = np.round(t_ref_traj[1] - t_ref_traj[0], time_precision)
    ts_sim = np.round(t_x_cur[1] - t_x_cur[0], time_precision)
    Q_dt = Q * dt_tmpc
    R_dt = R * dt_tmpc

    # Ensure that all times are aligned
    if t_x_cur_est[-1] > t_x_cur[-1]:
        log.warning(
            "The estimated state has a recording after the ground truth state. Shrinking (t_x_cur_est, x_cur_est), (t_ref_traj, u_ref_traj, x_ref_traj), and (t_pred_traj, u_pred_traj, x_pred_traj) to the last time of t_x_cur"
        )
        t_x_cur_est = t_x_cur_est[t_x_cur_est <= t_x_cur[-1]]
        x_cur_est = x_cur_est[: len(t_x_cur_est)]
        t_ref_traj = t_ref_traj[t_ref_traj <= t_x_cur[-1]]
        u_ref_traj = u_ref_traj[: len(t_ref_traj)]
        x_ref_traj = x_ref_traj[: len(t_ref_traj)]
        t_pred_traj = t_pred_traj[t_pred_traj <= t_x_cur[-1]]
        u_pred_traj = u_pred_traj[: len(t_pred_traj)]
        x_pred_traj = x_pred_traj[: len(t_pred_traj)]

    # Check if the estimated state overlaps with the ground truth state
    if check_gt_est_states:
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

    # Determine min and max times for printing and plotting
    t_min = min(np.min(t_x_cur_est), np.min(t_ref_traj), np.min(t_pred_traj))
    t_max = max(np.max(t_x_cur_est), np.max(t_ref_traj), np.max(t_pred_traj))

    # Select data to print
    if t_print != -1 and (t_print < t_min or t_print > t_max):
        raise ValueError(
            f"t_print ({t_print}) should be either -1 or be contained within bounds [{t_min}, {t_max}]"
        )

    if t_print < 0:
        t_x_cur_est_idx = -1
        t_pred_traj_idx = -1
        t_ref_traj_idx = -1
    else:
        t_x_cur_est_idx = np.abs(t_x_cur_est - t_print).argmin()
        t_mpc_start = t_x_cur_est[t_x_cur_est_idx]
        t_x_cur_start_idx = np.abs(t_x_cur - t_mpc_start).argmin()
        t_x_cur_end_idx = np.abs(
            t_x_cur - np.round(t_mpc_start + N_tmpc * ts_tmpc, time_precision)
        ).argmin()
        t_pred_traj_idx = np.abs(t_pred_traj - t_print).argmin()
        t_ref_traj_idx = np.abs(t_ref_traj - t_print).argmin()

    # Compute objective values
    if print_pobj_comp:
        if t_x_cur_est_idx < 0:
            pobj_comp = np.zeros(n_tmpc)
            for t_idx in range(n_tmpc):
                pobj_comp[t_idx] = compute_pobj(
                    u_pred_traj[t_idx, :, :],
                    x_pred_traj[t_idx, :, :],
                    u_ref_traj[t_idx, :, :],
                    x_ref_traj[t_idx, :, :],
                    N_tmpc,
                    Q_dt,
                    R_dt,
                    P,
                )
        else:
            pobj_comp = compute_pobj(
                u_pred_traj[t_pred_traj_idx, :, :],
                x_pred_traj[t_pred_traj_idx, :, :],
                u_ref_traj[t_ref_traj_idx, :, :],
                x_ref_traj[t_ref_traj_idx, :, :],
                N_tmpc,
                Q_dt,
                R_dt,
                P,
            )
        print(f"pobj_comp: {pobj_comp}")

    # Compute position tracking error
    if print_track_err:
        for key in print_track_err:
            if key == "total":
                u_idc = np.arange(NU)
                x_idc = np.arange(NX)
            elif key == "u":
                u_idc = np.arange(NU)
                x_idc = []
            elif key == "x":
                u_idc = []
                x_idc = np.arange(NX)
            elif key == "p":
                u_idc = []
                x_idc = np.arange(3)
            elif key == "2d_p":
                u_idc = []
                x_idc = np.arange(2)
            else:
                raise ValueError(f"Unknown track error type '{key}'")
            if t_x_cur_est_idx < 0:
                track_err = np.zeros(n_tmpc)
                for t_idx in range(n_tmpc):
                    track_err[t_idx] = compute_track_err(
                        u_pred_traj[t_idx, :, :],
                        x_pred_traj[t_idx, :, :],
                        u_ref_traj[t_idx, :, :],
                        x_ref_traj[t_idx, :, :],
                        N_tmpc,
                        u_idc,
                        x_idc,
                    )
            else:
                track_err = compute_track_err(
                    u_pred_traj[t_pred_traj_idx, :, :],
                    x_pred_traj[t_pred_traj_idx, :, :],
                    u_ref_traj[t_ref_traj_idx, :, :],
                    x_ref_traj[t_ref_traj_idx, :, :],
                    N_tmpc,
                    u_idc,
                    x_idc,
                )
            print(f"{key}_track_err: {track_err}")

    # Select data to plot
    if len(t_plot) != 2:
        raise ValueError("t_plot must contain exactly two elements")
    elif (
        t_plot[0] < t_min or t_plot[0] > t_max or t_plot[1] < t_min or t_plot[1] > t_max
    ):
        raise ValueError(
            f"The interval specified by t_plot ({t_plot}) must be contained within bounds [{t_min}, {t_max}]"
        )
    t_total_plot = t_plot[1] - t_plot[0]
    mpc_idc_plot = np.where((t_x_cur_est >= t_plot[0]) & (t_x_cur_est <= t_plot[1]))[0]
    t_x_cur_est_plot = t_x_cur_est[mpc_idc_plot]
    x_cur_est_plot = x_cur_est[mpc_idc_plot]
    t_ref_traj_plot = t_ref_traj[mpc_idc_plot]
    u_ref_traj_plot = u_ref_traj[mpc_idc_plot]
    x_ref_traj_plot = x_ref_traj[mpc_idc_plot]
    t_pred_traj_plot = t_pred_traj[mpc_idc_plot]
    u_pred_traj_plot = u_pred_traj[mpc_idc_plot]
    x_pred_traj_plot = x_pred_traj[mpc_idc_plot]
    x_cur_idc_plot = np.where(
        (t_x_cur >= t_x_cur_est_plot[0]) & (t_x_cur <= t_x_cur_est_plot[-1])
    )[0]
    t_x_cur_plot = t_x_cur[x_cur_idc_plot]
    x_cur_plot = x_cur[x_cur_idc_plot]

    # Create figure
    helpers.set_plt_properties()
    props = helpers.set_fig_properties()
    fig, ax = plt.subplots()
    handles = []

    # Add obstacles to plot
    if do_plot_obs:
        handles_obs = []
        for obs_idx, verts in enumerate(obs_verts):
            if obs_idx == 0:
                obs_polygon = Polygon(
                    verts["orig"],
                    closed=True,
                    edgecolor="black",
                    facecolor="black",
                    label="Obstacle",
                )
            else:
                obs_polygon = Polygon(
                    verts["orig"], closed=True, edgecolor="black", facecolor="black"
                )
            handles_obs.append(ax.add_patch(obs_polygon))
            obs_polygon_long = Polygon(
                verts["long"], closed=True, color=c_obs_inflated, zorder=0
            )
            handles_obs.append(ax.add_patch(obs_polygon_long))
            obs_polygon_wide = Polygon(
                verts["wide"], closed=True, color=c_obs_inflated, zorder=0
            )
            handles_obs.append(ax.add_patch(obs_polygon_wide))
            for i in range(4):
                obs_polygon_circle = Circle(
                    verts["orig"][i, :],
                    robot_radius,
                    color=c_obs_inflated,
                    fill=True,
                    zorder=0,
                )
                handles_obs.append(ax.add_patch(obs_polygon_circle))
        handles.append(handles_obs[0])
        inflated_obs_patch = Patch(color=c_obs_inflated, label="$\\mathcal{R}$")
        handles.append(inflated_obs_patch)

    # Add TMPC reference trajectory to plot
    if do_plot_tmpc_ref:
        handles_tmpc_ref = []
        for t_idx in mpc_idc_plot:
            for k in k_ref:
                tmpc_ref = Circle(
                    (
                        x_ref_traj[t_idx, k, 0],
                        x_ref_traj[t_idx, k, 1],
                    ),
                    r_tmpc_ref,
                    facecolor=c_tmpc_ref,
                    alpha=compute_alpha(
                        alpha_min, alpha_max, k * ts_tmpc, t_total_plot
                    ),
                    zorder=2,
                    label="TMPC ref pos",
                )
                handles_tmpc_ref.append(ax.add_patch(tmpc_ref))
        handles.append(handles_tmpc_ref[0])

    # Compute RPI tube around reference trajectory and add to plot
    if do_plot_rpi_tube:
        ref_traj_pos = x_ref_traj_plot[:, 0, :2]
        pmpc_tightening = c_o * alpha
        rpi_tube = helpers.compute_tube(ref_traj_pos, pmpc_tightening)
        handles_rpi_tube = []
        for tube_idx in range(rpi_tube.shape[0]):
            tube_handle = ax.plot(
                rpi_tube[tube_idx, :, 0],
                rpi_tube[tube_idx, :, 1],
                color=c_rpi_tube,
                linewidth=linewidth_rpi_tube,
                zorder=2,
                label="RPI tube",
            )
            handles_rpi_tube.append(tube_handle)
        handles.append(handles_rpi_tube[0][0])

    # Add TMPC prediction to plot
    if do_plot_tmpc_pred:
        handles_tmpc_pred = []
        for t_idx in mpc_idc_plot:
            for k in k_pred:
                if k == 0:
                    facecolor = c_x0
                else:
                    facecolor = c_tmpc
                tmpc_pred = Circle(
                    (
                        x_pred_traj[t_idx, k, 0],
                        x_pred_traj[t_idx, k, 1],
                    ),
                    r_tmpc,
                    facecolor=facecolor,
                    alpha=compute_alpha(
                        alpha_min, alpha_max, k * ts_tmpc, t_total_plot
                    ),
                    zorder=3,
                    label="TMPC prediction",
                )
                handles_tmpc_pred.append(ax.add_patch(tmpc_pred))
        handles.append(handles_tmpc_pred[0])

    # Compute growing tubes around specific TMPC predictions and add to plot
    if do_plot_growing_tube:
        print(f"Growing tubes around TMPC predictions are not implemented yet")

    # Add closed-loop state to plot
    if do_plot_x:
        handles_x = []
        for t_idx in x_cur_idc_plot:
            x = Circle(
                (x_cur[t_idx, 0], x_cur[t_idx, 1]),
                r_x,
                facecolor=c_x,
                alpha=compute_alpha(
                    alpha_min,
                    alpha_max,
                    (t_idx - x_cur_idc_plot[0]) * ts_sim,
                    t_total_plot,
                ),
                zorder=5,
                label="Closed-loop pos",
            )
            handles_x.append(ax.add_patch(x))
        handles.append(handles_x[0])

    # Add title, etc. to plot
    # ax.set_title(f"TMPC prediction", pad=props["titlepad"])
    # xy_diff = 0.01
    # x_mid = x_pred_traj[t_pred_traj_idx, int(N_tmpc / 2), 0]
    # y_mid = x_pred_traj[t_pred_traj_idx, int(N_tmpc / 2), 1]
    # ax.set_xlim(x_mid - xy_diff, x_mid + xy_diff)
    # ax.set_ylim(y_mid - xy_diff, y_mid + xy_diff)
    edge_val = 2
    ax.set_xlim(-edge_val, edge_val)
    ax.set_ylim(-edge_val, edge_val)
    ax.set_xlabel("$p^x$ (m)")
    ax.set_ylabel("$p^y$ (m)")
    ax.xaxis.labelpad = props["xlabelpad"]
    ax.yaxis.labelpad = props["ylabelpad"]
    ax.tick_params(pad=props["tickpad"])
    ax.set_axisbelow(True)
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
