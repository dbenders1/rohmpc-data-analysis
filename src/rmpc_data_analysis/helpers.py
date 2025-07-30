import matplotlib.pyplot as plt
import numpy as np


def compute_rectangle_vertices(pos, yaw_angle, length, width):
    # Compute half-length and half-width
    half_length = length / 2
    half_width = width / 2

    # Define local coordinates of rectangle vertices in clockwise direction
    local_vertices = np.array(
        [
            [half_width, -half_width, -half_width, half_width],
            [half_length, half_length, -half_length, -half_length],
        ]
    )

    # Apply rotation and translation to get global coordinates
    global_vertices = compute_rotation_matrix_2d(yaw_angle) @ local_vertices + np.array(
        [[pos[0]], [pos[1]]]
    )

    return global_vertices


# Compute obstacle vertices, including robot region
# For rectangles, this means computing vertices of 2 more rectangles:
# - Rectangle with length = length + 2 * robot radius
# - Rectangle with width = width + 2 * robot radius
def compute_inflated_obstacle_vertices(obs_list, robot_radius):
    obs_verts = []
    for obs in obs_list:
        verts = dict()

        obs_pos = obs["p"]
        obs_att = obs["eul"]
        obs_dim = obs["dim"]

        verts["orig"] = compute_rectangle_vertices(
            obs_pos, obs_att[2], length=obs_dim[0], width=obs_dim[1]
        ).T
        verts["long"] = compute_rectangle_vertices(
            obs_pos, obs_att[2], length=obs_dim[0] + 2 * robot_radius, width=obs_dim[1]
        ).T
        verts["wide"] = compute_rectangle_vertices(
            obs_pos, obs_att[2], length=obs_dim[0], width=obs_dim[1] + 2 * robot_radius
        ).T

        obs_verts.append(verts)
    return obs_verts


def compute_rotation_matrix_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


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
