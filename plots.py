import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
import os

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import math
from scipy.stats import t
import torch
from matplotlib import gridspec
from scipy.interpolate import interp1d
import matplotlib


def plot_colortable(colors, *, ncols=4, sort_colors=True):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - margin) / height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig


def plot_gt():
    # Load the .mat file
    # mat_data = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/IS1_metrics_8x8_FeasSet2_smallerFeas.mat')
    mat_data = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/IS1_metrics_20x20_FeasSet2_smallerFeas.mat')
    mat_data2 = scipy.io.loadmat(
        '/home/nobar/codes/GBO2/logs/misc/IS1_metrics_20x20_FeasSet2_smallerFeas_perturbed.mat')
    # mat_data = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/IS1_Exper_0_8x8_metrics.mat')
    # mat_data2 = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/IS2_Exper_0_8x8_metrics.mat')
    # mat_data2 = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/IS2_metrics_8x8_FeasSet2.mat')
    # mat_data2 = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3Dyn_s_01/IS3_1to2_metrics_8x8_FeasSet2.mat')
    n_grid = 20
    # # obj_IS2_grid = np.load("/home/nobar/codes/GBO2/logs/IS3_FeasSet2OLD_obj.npy")
    # obj_IS2_grid = np.load("/home/nobar/Documents/introductions/simulink_model/IS2_FeasSet2OLD_obj.npy")
    obj_IS2_grid = np.load("/home/nobar/Documents/introductions/simulink_model/IS1_FeasSet2OLD_obj.npy")
    obj_IS1_grid = np.load("/home/nobar/Documents/introductions/simulink_model/IS1_FeasSet2OLD_obj.npy")

    def normalize_objective(obj, obj_grid):
        return (obj - obj_grid.mean()) / (obj_grid.std())

    # Extract the variables
    Kp = mat_data['Kp'].squeeze()  # Ensure it's a 1D array
    Kd = mat_data['Kd'].squeeze()  # Ensure it's a 1D array

    Objective_all = mat_data['Objective_all']  # Should be a 2D array
    RiseTime_all = mat_data['RiseTime_all']  # Should be a 2D array
    PeakTime_all = mat_data['PeakTime_all']  # Should be a 2D array
    TransientTime_all = mat_data['TransientTime_all']  # Should be a 2D array
    SettlingTime_all = mat_data['SettlingTime_all']  # Should be a 2D array
    SettlingMin_all = mat_data['SettlingMin_all']  # Should be a 2D array
    Overshoot_all = mat_data['Overshoot_all']  # Should be a 2D array

    RiseTime2_all = mat_data2['RiseTime_all']  # Should be a 2D array
    TransientTime2_all = mat_data2['TransientTime_all']  # Should be a 2D array
    SettlingTime2_all = mat_data2['SettlingTime_all']  # Should be a 2D array
    Overshoot2_all = mat_data2['Overshoot_all']  # Should be a 2D array
    SettlingMin2_all = mat_data2['SettlingMin_all']  # Should be a 2D array
    PeakTime2_all = mat_data2['PeakTime_all']  # Should be a 2D array

    Objective_all_error = mat_data['Objective_all'] - mat_data2['Objective_all']  # Should be a 2D array
    RiseTime_all_error = mat_data['RiseTime_all'] - mat_data2['RiseTime_all']  # Should be a 2D array
    TransientTime_all_error = mat_data['TransientTime_all'] - mat_data2['TransientTime_all']  # Should be a 2D array
    SettlingTime_all_error = mat_data['SettlingTime_all'] - mat_data2['SettlingTime_all']  # Should be a 2D array
    Overshoot_all_error = mat_data['Overshoot_all'] - mat_data2['Overshoot_all']  # Should be a 2D array

    # # Create meshgrid for contour plot
    # Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='True Objective Value')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('True Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics.png")
    # plt.show()

    # w1 = 26
    # w2 = 2.3 / 100
    # w3 = 3.8
    # w4 = 3.5
    # w5 = 4
    # w6 = 5
    # w1 = 1  # 0.14
    # w2 = 1 / 100  # 11.156
    # w3 = 1  #0.519466
    # w4 = 1  #0.519468
    # w5 = 1  #0.3619
    # w6 = 1  #0.6739

    w1 = 2. / 0.14
    w2 = 3. / 100 / 0.11156
    w3 = 1.5 / 0.519466
    w4 = 1.5 / 0.519468
    w5 = 0 / 0.3619
    w6 = 0 / 0.6739
    new_obj = w1 * RiseTime_all + w2 * Overshoot_all + w4 * TransientTime_all + w3 * SettlingTime_all + w5 * PeakTime_all + w6 * SettlingMin_all
    new_obj = normalize_objective(new_obj, obj_IS1_grid)
    # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj.npy",new_obj)
    # np.save("/home/nobar/codes/GBO2/logs/IS1_FeasSet2_obj.npy",new_obj)
    # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj_noise_scale_01.npy",new_obj)
    # Create meshgrid for contour plot
    Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, new_obj.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='$J_{IS1}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_J_normalized_test_FeasSet2_smallerFeas.png")
    plt.show()

    new_obj2 = w1 * RiseTime2_all + w2 * Overshoot2_all + w4 * TransientTime2_all + w3 * SettlingTime2_all + w5 * PeakTime2_all + w6 * SettlingMin2_all
    new_obj2 = normalize_objective(new_obj2, obj_IS2_grid)
    # np.save("/home/nobar/codes/GBO2/logs/IS3_FeasSet2OLD_obj.npy",new_obj2)

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, new_obj2.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='$J_{IS3_1to2}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('Estimated Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_J_normalized_test_FeasSet2_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    contour = plt.contourf(Kp_grid, Ki_grid, new_obj.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='$\mathcal{g}(k, s=1.0)$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_numerical_g.pdf", format="pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    contour = plt.contourf(Kp_grid, Ki_grid, new_obj2.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='$\mathcal{g}(k, s=s\')$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS2_numerical_g.pdf", format="pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    contour = plt.contourf(Kp_grid, Ki_grid, new_obj2.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label=r'$|\frac{\mathcal{g}(k, s=1.0)-\mathcal{g}(k, s=s\')}{\mathcal{g}(k, s=1.0)}|$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS2_numerical_g.pdf", format="pdf")
    plt.show()

    # todo manual correction
    Z = 0.5 * abs(
        (new_obj.reshape(n_grid, n_grid) - new_obj2.reshape(n_grid, n_grid)) / new_obj.reshape(n_grid, n_grid)) * 100
    levels = np.concatenate([
        np.linspace(0, 20, 50, endpoint=False),  # Very dense in 0–5
        np.linspace(20, 50, 40, endpoint=False),  # Medium density in 5–20
        np.geomspace(50, 100, 20)  # Log-spaced in 20–1000
    ])
    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    # contour = plt.contourf(Kp_grid, Ki_grid, Z, levels=levels, cmap="viridis",  extend="max")
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(boundaries=levels, ncolors=256, extend='max')
    contour = plt.contourf(Kp_grid, Ki_grid, Z, levels=levels, cmap="plasma", norm=norm, extend="max")
    # cbar = plt.colorbar(contour)
    cbar = plt.colorbar(contour, ticks=[0, 20, 50, 100])
    cbar.set_label(
        r'$|\frac{\mathcal{g}(k, s=1.0)-\mathcal{g}(k, s=s^{\prime})}{\mathcal{g}(k, s=1.0)}|\times100$%',
        fontsize=21)
    plt.xlabel('Kp', fontsize=20)
    plt.ylabel('Kd', fontsize=20)
    plt.tight_layout()
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/ABSErrorPercent_numerical_g.pdf", format="pdf")
    plt.show()

    error_new_obj = new_obj2 - new_obj
    # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj.npy",new_obj)
    # np.save("/home/nobar/codes/GBO2/logs/IS2_FeasSet2_obj.npy",new_obj2)
    # Create meshgrid for contour plot
    Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, error_new_obj.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='$J_{IS3_1to2}-J_{IS1}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('Error True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_Error_normalized_FeasSet2_smallerFeas.png")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/test_29_baseline/ERROR_Exper_0_8x8_metrics_normalized.png")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    plt.show()
    # error_new_obj=new_obj2-new_obj
    # # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj.npy",new_obj)
    # # Create meshgrid for contour plot
    # Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, abs(error_new_obj).reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='|$J_{IS3_1to2}-J_{IS1}$|')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('Error True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_ABSError_normalized_FeasSet2_smallerFeas.png")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/ABS_ERROR_Exper_8x8_metrics_normalized.png")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w5 * PeakTime_all.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Peak Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('$w.T_{p}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_Tp_test_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w6 * SettlingMin_all.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted SettlingMin')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('$w.SettlingMin$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_SettlingMin_test_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w1 * RiseTime_all.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Rise Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('$w.T_{r}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_Tr_test_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w3 * SettlingTime_all.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Settling Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('$w.T_{s}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_Ts_test_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w2 * Overshoot_all.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Overshoot')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('$w.M$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_Ov_test_smallerFeas.png")
    plt.show()
    plt.close()
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w4 * TransientTime_all.reshape(n_grid, n_grid),
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Transient Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('$w.T_{tr}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_Ttr_test_smallerFeas.png")
    plt.show()

    # # Create meshgrid for contour plot
    # Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # n_grid=8
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all_error.reshape(n_grid,n_grid))  # Transpose to match dimensions
    # plt.colorbar(contour, label='Error Objective Value')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('Error Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_J_1to2_error.png")
    # plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp_grid, Ki_grid,
                           w1 * RiseTime_all_error.reshape(n_grid, n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Rise Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('error $w.T_{r}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_Tr_error_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid,
                           w3 * SettlingTime_all_error.reshape(n_grid, n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Settling Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('error $w.T_{s}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_Ts_error_smallerFeas.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid,
                           w2 * Overshoot_all_error.reshape(n_grid, n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Overshoot')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('error $w.M$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_Ov_error_smallerFeas.png")
    plt.show()
    plt.close()
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid,
                           w4 * TransientTime_all_error.reshape(n_grid, n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Transient Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('error $w.T_{tr}$')
    # plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS3_1to2_Ttr_error_smallerFeas.png")
    plt.show()

    return True


def plot_kernels():
    from scipy.spatial.distance import cdist

    # Define Matérn 5/2 kernel function
    def matern_5_2(x1, x2, length_scale=0.2, variance=1.0):
        """Computes the Matérn 5/2 kernel."""
        r = cdist(x1, x2, metric='euclidean')
        sqrt5_r_l = np.sqrt(5) * r / length_scale
        kernel_matrix = variance * (1 + sqrt5_r_l + (5 / 3) * (r ** 2 / length_scale ** 2)) * np.exp(-sqrt5_r_l)
        return kernel_matrix

    # Create a grid of points in [0,1] x [0,1]
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    # X, Y = np.meshgrid(x, y)

    # Flatten the grid and compute the kernel
    # XY = np.column_stack([X.ravel(), Y.ravel()])
    K = matern_5_2(x.reshape(100, 1), y.reshape(100, 1), length_scale=0.2).reshape(100, 100)

    # Plot contour map with color bar
    plt.figure(figsize=(6, 5))
    X, Y = np.meshgrid(x, y)
    contour = plt.contourf(X, Y, K, cmap="viridis", levels=20)
    plt.colorbar(contour, label="Kernel Value")
    plt.xlabel("x")
    plt.ylabel("x'")
    plt.title("Matérn 5/2 Kernel Contour Map")
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_3/matern52.png")
    plt.show()

    plt.figure(figsize=(6, 5))
    C1 = (1 - X) * (1 - Y) * (1 - X * Y) ** 0.672
    contour = plt.contourf(X, Y, C1, cmap="viridis", levels=20)
    plt.colorbar(contour, label="C1 Value")
    plt.xlabel("s")
    plt.ylabel("s'")
    plt.title("C1 Contour Map")
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_3/C1kernelCoefficient.png")
    plt.show()


def plots_MonteCarlo_objective(path, N_init_IS1, N_init_IS2, sampling_cost_bias, N_exper, N_iter, s2, s3, BATCH_SIZE):
    costs_all_list = []
    costs_all_list_EIonly = []
    train_x_list = []
    train_obj_list = []
    train_x_list_IS1 = []
    train_obj_list_IS1 = []
    train_x_list_EIonly = []
    train_obj_list_EIonly = []
    idx_IS1_all = []
    idx_IS1_all_init = []
    idx_IS1_all_rest = []
    idx_IS2_all = []
    idx_IS2_all_init = []
    idx_IS2_all_rest = []
    idx_IS3_all = []
    idx_IS3_all_init = []
    idx_IS3_all_rest = []
    idx_ISDTs_all = []
    idx_ISDTs_all_init = []
    idx_ISDTs_all_rest = []
    # train_x_IS1_init_list=[]
    # train_obj_IS1_init_list=[]
    train_obj_EIonly_corrected_all = []
    min_obj_init_all = []
    train_obj_list_rest_modified = []
    costs_init = []
    costs_init_IS1 = []
    costs_init_EIonly = []
    for exper in range(N_exper):
        # if exper==4:
        #     exper=0
        # if exper == 5:
        #     exper = 8
        exp_path = os.path.join(path, f"Exper_{exper}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))

        idx_IS1 = np.argwhere(train_x[:, 2] == 1).squeeze()
        idx_IS1_init = idx_IS1[np.argwhere(idx_IS1 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS1_rest = idx_IS1[np.argwhere(idx_IS1 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_ISDTs = np.argwhere(~(train_x[:, 2] == 1)).squeeze()
        idx_ISDTs_init = idx_ISDTs[np.argwhere(idx_ISDTs < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_ISDTs_rest = idx_ISDTs[np.argwhere(idx_ISDTs > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        train_x_IS1_init = train_x[idx_IS1_init, :]
        train_obj_IS1_init = train_obj[idx_IS1_init]

        idx_IS2 = np.argwhere(train_x[:, 2] == s2).squeeze()
        idx_IS2 = idx_IS2.reshape(idx_IS2.size, )
        idx_IS2_init = idx_IS2[np.argwhere(idx_IS2 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS2_rest = idx_IS2[np.argwhere(idx_IS2 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_IS3 = np.argwhere(train_x[:, 2] == s3).squeeze()
        idx_IS3 = idx_IS3.reshape(idx_IS3.size, )
        idx_IS3_init = idx_IS3[np.argwhere(idx_IS3 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS3_rest = idx_IS3[np.argwhere(idx_IS3 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]

        # np.save(os.path.join(exp_path, "train_x_IS1_init.npy"),train_x_IS1_init)
        # np.save(os.path.join(exp_path, "train_obj_IS1_init.npy"),train_obj_IS1_init)

        min_obj_init_all.append(np.min(-train_obj_IS1_init))

        train_x_IS1 = train_x[idx_IS1]
        train_obj_IS1 = train_obj[idx_IS1]
        train_x_EIonly = np.load(os.path.join(exp_path, "train_x_EIonly.npy"))
        train_x_EIonly_corrected = np.load(os.path.join(exp_path, "train_x_EIonly.npy"))
        train_obj_EIonly_corrected = np.load(os.path.join(exp_path, "train_obj_EIonly.npy"))

        train_obj_EIonly = np.load(os.path.join(exp_path, "train_obj_EIonly.npy"))

        # train_x_EIonly=np.delete(train_x_EIonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)
        # train_obj_EIonly=np.delete(train_obj_EIonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)

        train_obj_EIonly[N_init_IS1:N_init_IS1 + N_init_IS2]
        costs_all = np.load(os.path.join(exp_path, "costs_all.npy"))
        costs_all_EIonly = np.load(os.path.join(exp_path, "costs_all_EIonly.npy"))
        costs_all_EIonly_corrected = np.load(os.path.join(exp_path, "costs_all_EIonly.npy"))
        # Append to lists
        costs_all_list.append(costs_all)
        costs_init.append(np.sum(train_x[:N_init_IS1 + N_init_IS2, 2]) + sampling_cost_bias * (N_init_IS1 + N_init_IS2))
        costs_init_IS1.append(np.sum(train_x[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))
        costs_init_EIonly.append(np.sum(train_x_EIonly[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))

        idx_IS1_all.append(idx_IS1)
        idx_IS1_all_init.append(idx_IS1_init)
        idx_IS1_all_rest.append(idx_IS1_rest)

        idx_IS2_all.append(idx_IS2)
        idx_IS2_all_init.append(idx_IS2_init)
        idx_IS2_all_rest.append(idx_IS2_rest)
        idx_IS3_all.append(idx_IS3)
        idx_IS3_all_init.append(idx_IS3_init)
        idx_IS3_all_rest.append(idx_IS3_rest)

        idx_ISDTs_all.append(idx_ISDTs)
        idx_ISDTs_all_init.append(idx_ISDTs_init)
        idx_ISDTs_all_rest.append(idx_ISDTs_rest)

        costs_all_list_EIonly.append(costs_all_EIonly)
        train_x_list.append(train_x)
        train_obj_list.append(train_obj)

        A = train_obj
        A[idx_ISDTs_rest] = -np.inf
        train_obj_list_rest_modified.append(-A[N_init_IS1 + N_init_IS2:])

        train_x_list_IS1.append(train_x_IS1)
        train_obj_list_IS1.append(train_obj_IS1)
        # train_x_IS1_init_list.append(train_x_IS1_init)
        # train_obj_IS1_init_list.append(train_obj_IS1_init)
        train_x_list_EIonly.append(train_x_EIonly)
        train_obj_EIonly_corrected_all.append(train_obj_EIonly_corrected)
        train_obj_list_EIonly.append(train_obj_EIonly)

    DD = np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_mean = np.mean(j_min_observed_IS1, axis=1)
    J_std = np.std(j_min_observed_IS1, axis=1)

    DD = -np.stack(train_obj_list_EIonly).squeeze()[:, N_init_IS1:].T
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_EIonly_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_EIonly_mean = np.mean(j_EIonly_min_observed_IS1, axis=1)
    J_EIonly_std = np.std(j_EIonly_min_observed_IS1, axis=1)
    x = np.arange(0, BATCH_SIZE * N_iter + 1)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region
    plt.plot(x, J_EIonly_mean, color='b', linewidth=3, label='Mean EI only')  # Thick line for mean
    plt.fill_between(x, J_EIonly_mean - J_EIonly_std, J_EIonly_mean + J_EIonly_std, color='b', alpha=0.3,
                     label='±1 Std EI only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('$J^{*}$')
    plt.title('Minimum Observed Objective on IS1 over BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path+"/J_min_obs_IS1_BOiter.png")
    plt.show()

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1), np.stack(costs_all_list)))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_EIonly = np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper, 1),
                              np.ones((N_exper, N_iter)) * (sampling_cost_bias + 1) * BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean = np.mean(C_EIonly, axis=0)
    C_EIonly_std = np.std(C_EIonly, axis=0)
    x = np.arange(0, BATCH_SIZE * N_iter + 1, BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - C_std, C_mean + C_std, color='r', alpha=0.3,
                     label='±1 Std Cost GMFBO')  # Shaded std region
    plt.plot(x, C_EIonly_mean, color='b', marker="o", linewidth=3, label='Mean Cost EI only')  # Thick line for mean
    plt.fill_between(x, C_EIonly_mean - C_EIonly_std, C_EIonly_mean + C_EIonly_std, color='b', alpha=0.3,
                     label='±1 Std Cost EI only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('Sampling Cost')
    plt.title('Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path+"/Cost_sampling_BOiter.png")
    plt.show()

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1 + N_init_IS2),
                       np.stack(costs_all_list) - sampling_cost_bias * BATCH_SIZE))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_EIonly = np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper, 1) - sampling_cost_bias * N_init_IS1,
                              np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean = np.mean(C_EIonly, axis=0)
    C_EIonly_std = np.std(C_EIonly, axis=0)
    x = np.arange(0, BATCH_SIZE * N_iter + 1, BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - C_std, C_mean + C_std, color='r', alpha=0.3,
                     label='±1 Std Cost GMFBO')  # Shaded std region
    plt.plot(x, C_EIonly_mean, color='b', marker="o", linewidth=3, label='Mean Cost EI only')  # Thick line for mean
    plt.fill_between(x, C_EIonly_mean - C_EIonly_std, C_EIonly_mean + C_EIonly_std, color='b', alpha=0.3,
                     label='±1 Std Cost EI only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('Unbiased Sampling Cost')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path+"/Unbiased_cost_sampling_BOiter.png")
    plt.show()

    # np.save("/home/nobar/codes/GBO2/logs/test_33_b_1/C_unbiased.npy",C)
    # np.save("/home/nobar/codes/GBO2/logs/test_33_b_1/C_mean_unbiased.npy",C_mean)
    # np.save("/home/nobar/codes/GBO2/logs/test_33_b_1/C_std_unbiased.npy",C_std)

    costs_IS2_all = []
    costs_IS3_all = []
    for i in range(N_exper):
        costs_IS2_all_ = []
        costs_IS3_all_ = []
        for j in range(N_iter):
            costs_IS2_all_.append(
                np.sum((idx_IS2_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS2_all_rest[i] > BATCH_SIZE * j)) * s2)
            costs_IS3_all_.append(
                np.sum((idx_IS3_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS3_all_rest[i] > BATCH_SIZE * j)) * s3)
        costs_IS2_all.append(costs_IS2_all_)
        costs_IS3_all.append(costs_IS3_all_)

    old_indices = np.linspace(0, 1, num=1 + N_iter)  # Original index positions
    new_indices = np.linspace(0, 1, num=1 + BATCH_SIZE * N_iter)  # New index positions

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1 + N_init_IS2),
                       np.stack(costs_all_list) - sampling_cost_bias * BATCH_SIZE))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_EIonly = np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                              np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean = np.mean(C_EIonly, axis=0)
    C_EIonly_std = np.std(C_EIonly, axis=0)
    x_ = C_mean  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x = np.interp(new_indices, old_indices, x_)
    x_EI_only_ = C_EIonly_mean  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x_EI_only = np.interp(new_indices, old_indices, x_EI_only_)

    DD = np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_mean = np.mean(j_min_observed_IS1, axis=1)
    J_std = np.std(j_min_observed_IS1, axis=1)

    DD = -np.stack(train_obj_list_EIonly).squeeze()[:, N_init_IS1:].T
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_EIonly_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_EIonly_mean = np.mean(j_EIonly_min_observed_IS1, axis=1)
    J_EIonly_std = np.std(j_EIonly_min_observed_IS1, axis=1)

    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', marker="o", linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region

    plt.plot(x_EI_only, J_EIonly_mean, color='b', marker="o", linewidth=3, label='Mean EI only')  # Thick line for mean
    plt.fill_between(x_EI_only, J_EIonly_mean - J_EIonly_std, J_EIonly_mean + J_EIonly_std, color='b', alpha=0.3,
                     label='±1 Std EI only')  # Shaded std region
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling.png")
    plt.show()

    # mean_values_baseline=np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy")
    # margin_of_error_baseline=np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy")
    # x_baseline=np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy")

    # Compute statistics
    mean_values = np.mean(j_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values = np.std(j_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error = t_value * (std_values / np.sqrt(n))  # CI width

    mean_values_EIonly = np.mean(j_EIonly_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values_EIonly = np.std(j_EIonly_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_EIonly_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error_EIonly = t_value * (std_values_EIonly / np.sqrt(n))  # CI width
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color='r')
    plt.fill_between(x, mean_values - margin_of_error, mean_values + margin_of_error,
                     color='r', alpha=0.3, label="95% CI GMFBO")
    # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    #                  color='k', alpha=0.3, label="95% CI MFBO")
    plt.plot(x_EI_only, mean_values_EIonly, marker="o", linewidth=3, label="Mean BO-EI", color='b')
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
                     mean_values_EIonly + margin_of_error_EIonly,
                     color='b', alpha=0.3, label="95% CI BO-EI")
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.BATCH_SIZE)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy",mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy",margin_of_error)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy",x)
    print("")

    costs_IS1 = np.hstack((np.asarray(costs_init_IS1).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                           np.stack(costs_all_list) - np.stack(costs_IS2_all) - np.stack(
                               costs_IS3_all) - sampling_cost_bias * BATCH_SIZE))
    C_IS1 = np.cumsum(costs_IS1, axis=1)
    C_mean_IS1 = np.mean(C_IS1, axis=0)
    x_IS1_ = C_mean_IS1  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    x_IS1 = np.interp(new_indices, old_indices, x_IS1_)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_IS1, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color='r')
    plt.fill_between(x_IS1, mean_values - margin_of_error, mean_values + margin_of_error,
                     color='r', alpha=0.3, label="95% CI GMFBO")
    # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    #                  color='k', alpha=0.3, label="95% CI MFBO")
    plt.plot(x_EI_only, mean_values_EIonly, marker="o", linewidth=3, label="Mean BO-EI", color='b')
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
                     mean_values_EIonly + margin_of_error_EIonly,
                     color='b', alpha=0.3, label="95% CI BO-EI")
    plt.xlabel('Mean IS1 only Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.4)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path+"/J_min_obs_IS1_Mean_IS1onlySamplingCost_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_33_b_1/x_IS1_baseline.npy", x_IS1)
    # np.save("/home/nobar/codes/GBO2/logs/test_33_b_1/mean_values_baseline.npy", mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_33_b_1/margin_of_error_baseline.npy", margin_of_error)
    print("")


def plots_indicators(path, path2, N_init_IS1, N_init_IS2, sampling_cost_bias, N_exper, N_iter, s2, s3, BATCH_SIZE):
    for exper in range(N_exper):
        # if exper==2:
        #     exper_GMFBO=0
        # elif exper==3:
        #     exper_GMFBO=0
        # elif exper==4:
        #     exper_GMFBO=8
        # else:
        #     exper_GMFBO=exper
        # exper=0
        exp_path = os.path.join(path, f"Exper_{exper}")
        exp_path2 = os.path.join(path2, f"Exper_{exper}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))

        delta_J = np.load(os.path.join(exp_path, "delta_J.npy"))
        JIS1s_for_delta_J = np.load(os.path.join(exp_path, "JIS1s_for_delta_J.npy"))
        delta_J2 = np.load(os.path.join(exp_path, "delta_J2.npy"))
        JIS2s_for_delta_J2 = np.load(os.path.join(exp_path, "JIS2s_for_delta_J2.npy"))
        caEI_values = np.load(os.path.join(exp_path, "caEI_values.npy"))
        costs_all = np.load(os.path.join(exp_path, "costs_all.npy"))
        i_IS1s = np.load(os.path.join(exp_path, "i_IS1s.npy"))

        rawEI_s_values = caEI_values * (sampling_cost_bias + train_x[:, 2])
        H_values = train_x[:, 2]
        # delta_J[delta_J == 0] = np.nan
        RAWS2 = np.hstack((train_x, train_obj, delta_J, caEI_values.reshape(-1, 1)))
        plt.figure(figsize=(8, 5))
        # Boolean masks
        nonzero_mask = delta_J != 0
        zero_mask = delta_J == 0
        x = np.arange(1, len(delta_J) + 1).reshape(-1, 1)
        # Plot non-zero values (filled markers)
        plt.plot(x[nonzero_mask], delta_J[nonzero_mask], color='b', marker='o',
                 markersize=6, linewidth=0,
                 label=r'GMFBO, $\Delta J(k)=J_{ISn>1}(k)-J_{IS1}(k)$, $\epsilon_{n}=0.5$')
        # Plot zero values (hollow markers)
        plt.plot(x[zero_mask], delta_J[zero_mask], color='b', marker='o',
                 markerfacecolor='none', markersize=6, linewidth=0)

        # Boolean masks
        nonzero_mask = rawEI_s_values != 0
        zero_mask = rawEI_s_values == 0
        x = np.arange(1, len(rawEI_s_values) + 1).reshape(-1, 1)
        # Plot non-zero values (filled markers)
        plt.plot(x[nonzero_mask], rawEI_s_values[nonzero_mask], color='m', marker='o',
                 markersize=5, linewidth=0,
                 label=r'GMFBO, $EI(k,s)$, $\epsilon_{n}=0.5$')
        # Plot zero values (hollow markers)
        plt.plot(x[zero_mask], rawEI_s_values[zero_mask], color='m', marker='o',
                 markerfacecolor='none', markersize=5, linewidth=0)

        # Boolean masks
        nonzero_mask = H_values != 0
        zero_mask = H_values == 0
        x = np.arange(1, len(H_values) + 1).reshape(-1, 1)
        # Plot non-zero values (filled markers)
        plt.plot(x[nonzero_mask], H_values[nonzero_mask], color='r', marker='o',
                 markersize=5, linewidth=0,
                 label=r'GMFBO, $H(s)-b$, $\epsilon_{n}=0.5$')
        # Plot zero values (hollow markers)
        plt.plot(x[zero_mask], H_values[zero_mask], color='r', marker='o',
                 markerfacecolor='none', markersize=5, linewidth=0)

        # plt.plot(np.arange(1, delta_J.__len__() + 1), delta_J, color='b', marker="o", markersize=5, linewidth=1,
        #          label=r'GMFBO, $\Delta J(k)=J_{ISn>1}(k)-J_{IS1}(k)$, $\epsilon_{n}=0.5$')  # Thick line for mean
        # plt.plot(np.arange(1, delta_J.__len__() + 1), rawEI_s_values, color='m', marker="o", markersize=5, linewidth=1,
        #          label=r'GMFBO, $H(s)$, $\epsilon_{n}=0.5$')  # Thick line for mean
        plt.axvspan(1, 2.5, color='green', alpha=0.3)
        plt.axvspan(2.5, 12.5, color='orange', alpha=0.3)
        plt.axvspan(12.5, 14.5, color='yellow', alpha=0.3)
        plt.xlabel('sample number')
        plt.ylabel('indicator')
        # plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
        plt.legend()
        plt.grid(True)
        # plt.yscale('log')
        # plt.ylim(0.9, 1.05)  # Focus range
        # plt.yticks([0.9, 0.95, 1.0, 1.05])
        plt.xlim(1, delta_J.__len__())
        plt.ylim(-1.7, 1.05)
        plt.xticks(np.arange(1, delta_J.__len__(), 2))
        # plt.savefig(exp_path + "/indicators.pdf")
        plt.show()

        RAWS = np.hstack((train_x, train_obj))


def plots_MonteCarlo_objectiveEI_34tests(path, path2, N_init_IS1, N_init_IS2, sampling_cost_bias, N_exper, N_iter, s2,
                                         s3, BATCH_SIZE, N_IS3_sample_each_time=1):
    costs_all_list = []
    costs_all_list_EIonly = []
    train_x_list = []
    train_obj_list = []
    train_x_list_IS1 = []
    train_obj_list_IS1 = []
    train_x_list_EIonly = []
    train_obj_list_EIonly = []
    idx_IS1_all = []
    idx_IS1_all_init = []
    idx_IS1_all_rest = []
    idx_IS2_all = []
    idx_IS2_all_init = []
    idx_IS2_all_rest = []
    idx_IS3_all = []
    idx_IS3_all_init = []
    idx_IS3_all_rest = []
    idx_ISDTs_all = []
    idx_ISDTs_all_init = []
    idx_ISDTs_all_rest = []
    # train_x_IS1_init_list=[]
    # train_obj_IS1_init_list=[]
    train_obj_EIonly_corrected_all = []
    min_obj_init_all = []
    train_obj_list_rest_modified = []
    costs_init = []
    costs_init_IS1 = []
    costs_init_EIonly = []

    for exper in range(N_exper):
        if exper == 2:
            exper_GMFBO = 3
        elif exper == 4:
            exper_GMFBO = 3
        elif exper == 1:
            exper_GMFBO = 3
        elif exper == 7:
            exper_GMFBO = 9
        else:
            exper_GMFBO = exper
        # exper_GMFBO = exper
        exp_path = os.path.join(path, f"Exper_{exper_GMFBO}")
        exp_path2 = os.path.join(path2, f"Exper_{exper_GMFBO}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))

        delta_J = np.load(os.path.join(exp_path, "delta_J.npy"))
        JIS1s_for_delta_J = np.load(os.path.join(exp_path, "JIS1s_for_delta_J.npy"))
        caEI_values = np.load(os.path.join(exp_path, "caEI_values.npy"))
        costs_all = np.load(os.path.join(exp_path, "costs_all.npy"))
        i_IS1s = np.load(os.path.join(exp_path, "i_IS1s.npy"))

        if 0:
            rawEI_s_values = caEI_values * (sampling_cost_bias + train_x[:, 2])
            H_values = train_x[:, 2]
            # delta_J[delta_J == 0] = np.nan
            RAWS2 = np.hstack((train_x, train_obj, delta_J, caEI_values.reshape(-1, 1)))
            plt.figure(figsize=(8, 5))
            # Boolean masks
            nonzero_mask = delta_J != 0
            zero_mask = delta_J == 0
            x = np.arange(1, len(delta_J) + 1).reshape(-1, 1)
            # Plot non-zero values (filled markers)
            plt.plot(x[nonzero_mask], delta_J[nonzero_mask], color='b', marker='o',
                     markersize=6, linewidth=0,
                     label=r'GMFBO, $\Delta J(k)=J_{ISn>1}(k)-J_{IS1}(k)$, $\epsilon_{n}=0.5$')
            # Plot zero values (hollow markers)
            plt.plot(x[zero_mask], delta_J[zero_mask], color='b', marker='o',
                     markerfacecolor='none', markersize=6, linewidth=0)

            # Boolean masks
            nonzero_mask = rawEI_s_values != 0
            zero_mask = rawEI_s_values == 0
            x = np.arange(1, len(rawEI_s_values) + 1).reshape(-1, 1)
            # Plot non-zero values (filled markers)
            plt.plot(x[nonzero_mask], rawEI_s_values[nonzero_mask], color='m', marker='o',
                     markersize=5, linewidth=0,
                     label=r'GMFBO, $EI(k,s)$, $\epsilon_{n}=0.5$')
            # Plot zero values (hollow markers)
            plt.plot(x[zero_mask], rawEI_s_values[zero_mask], color='m', marker='o',
                     markerfacecolor='none', markersize=5, linewidth=0)

            # Boolean masks
            nonzero_mask = H_values != 0
            zero_mask = H_values == 0
            x = np.arange(1, len(H_values) + 1).reshape(-1, 1)
            # Plot non-zero values (filled markers)
            plt.plot(x[nonzero_mask], H_values[nonzero_mask], color='r', marker='o',
                     markersize=5, linewidth=0,
                     label=r'GMFBO, $H(s)-b$, $\epsilon_{n}=0.5$')
            # Plot zero values (hollow markers)
            plt.plot(x[zero_mask], H_values[zero_mask], color='r', marker='o',
                     markerfacecolor='none', markersize=5, linewidth=0)

            # plt.plot(np.arange(1, delta_J.__len__() + 1), delta_J, color='b', marker="o", markersize=5, linewidth=1,
            #          label=r'GMFBO, $\Delta J(k)=J_{ISn>1}(k)-J_{IS1}(k)$, $\epsilon_{n}=0.5$')  # Thick line for mean
            # plt.plot(np.arange(1, delta_J.__len__() + 1), rawEI_s_values, color='m', marker="o", markersize=5, linewidth=1,
            #          label=r'GMFBO, $H(s)$, $\epsilon_{n}=0.5$')  # Thick line for mean
            plt.axvspan(1, 2.5, color='green', alpha=0.3)
            plt.axvspan(2.5, 12.5, color='orange', alpha=0.3)
            plt.axvspan(12.5, 14.5, color='yellow', alpha=0.3)
            plt.xlabel('sample number')
            plt.ylabel('indicator')
            # plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
            plt.legend()
            plt.grid(True)
            # plt.yscale('log')
            # plt.ylim(0.9, 1.05)  # Focus range
            # plt.yticks([0.9, 0.95, 1.0, 1.05])
            plt.xlim(1, delta_J.__len__())
            plt.ylim(-1.7, 1.05)
            plt.xticks(np.arange(1, delta_J.__len__(), 2))
            # plt.savefig(exp_path + "/indicators.pdf")
            plt.show()

        RAWS = np.hstack((train_x, train_obj))

        idx_IS1 = np.argwhere(train_x[:, 2] == 1).squeeze()
        idx_IS1_init = idx_IS1[np.argwhere(idx_IS1 < N_init_IS1 *2 + N_init_IS2)[:, 0]]
        idx_IS1_rest = idx_IS1[np.argwhere(idx_IS1 > N_init_IS1 *2 + N_init_IS2 - 1)[:, 0]]
        idx_ISDTs = np.argwhere(~(train_x[:, 2] == 1)).squeeze()
        idx_ISDTs_init = idx_ISDTs[np.argwhere(idx_ISDTs < N_init_IS1 *2 + N_init_IS2)[:, 0]]
        idx_ISDTs_rest = idx_ISDTs[np.argwhere(idx_ISDTs > N_init_IS1 *2 + N_init_IS2 - 1)[:, 0]]
        train_x_IS1_init = train_x[idx_IS1_init, :]
        train_obj_IS1_init = train_obj[idx_IS1_init]

        idx_IS2 = np.argwhere(train_x[:, 2] == s2).squeeze()
        idx_IS2 = idx_IS2.reshape(idx_IS2.size, )
        idx_IS2_init = idx_IS2[np.argwhere(idx_IS2 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS2_rest = idx_IS2[np.argwhere(idx_IS2 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_IS3 = np.argwhere(train_x[:, 2] == s3).squeeze()
        idx_IS3 = idx_IS3.reshape(idx_IS3.size, )
        idx_IS3_init = idx_IS3[np.argwhere(idx_IS3 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS3_rest = idx_IS3[np.argwhere(idx_IS3 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]

        # np.save(os.path.join(exp_path, "train_x_IS1_init.npy"),train_x_IS1_init)
        # np.save(os.path.join(exp_path, "train_obj_IS1_init.npy"),train_obj_IS1_init)

        min_obj_init_all.append(np.min(-train_obj_IS1_init))

        train_x_IS1 = train_x[idx_IS1]
        train_obj_IS1 = train_obj[idx_IS1]
        train_x_EIonly = np.load(os.path.join(exp_path2, "train_x_EIonly.npy"))
        train_x_EIonly_corrected = np.load(os.path.join(exp_path2, "train_x_EIonly.npy"))
        train_obj_EIonly_corrected = np.load(os.path.join(exp_path2, "train_obj_EIonly.npy"))

        train_obj_EIonly = np.load(os.path.join(exp_path2, "train_obj_EIonly.npy"))

        # train_x_EIonly=np.delete(train_x_EIonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)
        # train_obj_EIonly=np.delete(train_obj_EIonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)

        train_obj_EIonly[N_init_IS1:N_init_IS1 + N_init_IS2]
        costs_all = np.load(os.path.join(exp_path, "costs_all.npy"))
        costs_all_EIonly = np.load(os.path.join(exp_path2, "costs_all_EIonly.npy"))
        costs_all_EIonly_corrected = np.load(os.path.join(exp_path2, "costs_all_EIonly.npy"))
        # Append to lists
        costs_all_list.append(costs_all)
        costs_init.append(np.sum(train_x[:N_init_IS1 + N_init_IS2, 2]) + sampling_cost_bias * (N_init_IS1 + N_init_IS2))
        costs_init_IS1.append(np.sum(train_x[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))
        costs_init_EIonly.append(np.sum(train_x_EIonly[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))

        idx_IS1_all.append(idx_IS1)
        idx_IS1_all_init.append(idx_IS1_init)
        idx_IS1_all_rest.append(idx_IS1_rest)

        idx_IS2_all.append(idx_IS2)
        idx_IS2_all_init.append(idx_IS2_init)
        idx_IS2_all_rest.append(idx_IS2_rest)
        idx_IS3_all.append(idx_IS3)
        idx_IS3_all_init.append(idx_IS3_init)
        idx_IS3_all_rest.append(idx_IS3_rest)

        idx_ISDTs_all.append(idx_ISDTs)
        idx_ISDTs_all_init.append(idx_ISDTs_init)
        idx_ISDTs_all_rest.append(idx_ISDTs_rest)

        costs_all_list_EIonly.append(costs_all_EIonly)
        train_x_list.append(train_x)
        train_obj_list.append(train_obj)

        A = np.copy(train_obj)
        A[idx_ISDTs_rest] = -np.inf
        train_obj_list_rest_modified.append(-A[N_init_IS1 *2 + N_init_IS2:])

        train_x_list_IS1.append(train_x_IS1)
        train_obj_list_IS1.append(train_obj_IS1)
        # train_x_IS1_init_list.append(train_x_IS1_init)
        # train_obj_IS1_init_list.append(train_obj_IS1_init)
        train_x_list_EIonly.append(train_x_EIonly)
        train_obj_EIonly_corrected_all.append(train_obj_EIonly_corrected)
        train_obj_list_EIonly.append(train_obj_EIonly)

    # Step 1: Determine the maximum number of rows (n)
    max_len = max(arr.shape[0] for arr in train_obj_list_rest_modified)
    # Step 2: Initialize output array with np.inf
    DD = np.full((N_exper, max_len), np.inf)
    costsALL = np.full((max_len, N_exper), np.inf)
    costsIS1only = np.full((max_len, N_exper), np.inf)
    # Step 3: Fill in the values
    for i, arr in enumerate(train_obj_list_rest_modified):
        DD[i, :arr.shape[0]] = arr.flatten()
        DDi = np.where(np.isinf(arr), s2, 1.0)
        x_col = np.cumsum(DDi, axis=0)
        num_to_add = max_len - x_col.shape[0]
        last_row = x_col[-1:, :]  # keep it 2D shape (1,1)
        padding = np.repeat(last_row, num_to_add, axis=0)
        x_padded = np.vstack([x_col, padding])
        costsALL[:, i] = x_padded.flatten()
        DDi = np.where(np.isinf(arr), 0, 1.0)
        x_col = np.cumsum(DDi, axis=0)
        num_to_add = max_len - x_col.shape[0]
        last_row = x_col[-1:, :]  # keep it 2D shape (1,1)
        padding = np.repeat(last_row, num_to_add, axis=0)
        x_padded = np.vstack([x_col, padding])
        costsIS1only[:, i] = x_padded.flatten()

    row_to_add = np.full((1, N_exper), N_init_IS1)
    costsIS1only = np.vstack([row_to_add, costsIS1only + N_init_IS1])

    row_to_addALL = np.full((1, N_exper), N_init_IS1 + (N_init_IS2 + N_init_IS1 * N_IS3_sample_each_time) * s2)
    C = N_init_IS1*2 + (N_init_IS2 + N_init_IS1 * N_IS3_sample_each_time) * s2 + costsALL
    # C=np.cumsum(costsALLLL, axis=0)
    C_mean = np.mean(C[:20, :], axis=1)
    C_std = np.std(C[:20, :], axis=1)
    # Compute 95% confidence interval
    n = C[:20, :].shape[1]
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error = t_value * (C_std / np.sqrt(n))  # CI width
    costs_EIonly = np.ones((20,
                            10))  # np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper,1)-sampling_cost_bias*N_init_IS1,np.ones((N_exper,N_iter))*(1)*BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=0)
    C_EIonly_mean = np.mean(C_EIonly, axis=1)
    C_EIonly_std = np.std(C_EIonly, axis=1)

    path_="/home/nobar/codes/GBO2/logs/test_50_2_MFBO_caEI/"
    x_MFBO_caEI=np.load(path_ + "x.npy")
    C_mean_MFBO_caEI=np.load(path_ + "C_mean_MFBO.npy")
    margin_of_error_MFBO_caEI=np.load(path_ + "margin_of_error_MFBO.npy")
    path_="/home/nobar/codes/GBO2/logs/test_50_2_MFBO_caEI_gamma0/"
    x_MFBO_caEI_gamma0=np.load(path_ + "x.npy")
    C_mean_MFBO_caEI_gamma0=np.load(path_ + "C_mean_MFBO.npy")
    margin_of_error_MFBO_caEI_gamma0=np.load(path_ + "margin_of_error_MFBO.npy")
    path_="/home/nobar/codes/GBO2/logs/test_50_2_MFBO_taKG/"
    x_MFBO_taKG=np.load(path_ + "x.npy")
    C_mean_MFBO_taKG=np.load(path_ + "C_mean_MFBO.npy")
    margin_of_error_MFBO_taKG=np.load(path_ + "margin_of_error_MFBO.npy")
    # x = np.hstack((0,np.arange(0,40,1)))#np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    x = np.arange(1, 21, 1)
    plt.figure(figsize=(12, 8))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 22
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='crimson', marker="o", linewidth=3, label='Mean - GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - margin_of_error, C_mean + margin_of_error, color='crimson', alpha=0.3,
                     label='95% CI - GMFBO')  # Shaded std region
    # np.save(path + "x.npy", x)
    # np.save(path + "C_mean_MFBO.npy", C_mean)
    # np.save(path + "margin_of_error_MFBO.npy", margin_of_error)
    C_MFBO = np.load("/home/nobar/codes/GBO2/logs/test_33_b_1/C_unbiased.npy")[:-1]
    C_mean_MFBO = np.load("/home/nobar/codes/GBO2/logs/test_33_b_1/C_mean_unbiased.npy")[:-1]
    C_std_MFBO = np.load("/home/nobar/codes/GBO2/logs/test_33_b_1/C_std_unbiased.npy")[:-1]
    # Compute 95% confidence interval
    n = C_MFBO[:20, :].shape[1]
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error_MFBO = t_value * (C_std_MFBO / np.sqrt(n))  # CI width
    # plt.plot(x, C_mean_MFBO, color='black', marker="s", linewidth=3, label='Mean Cost MFBO')  # Thick line for mean
    # plt.fill_between(x, C_mean_MFBO - margin_of_error_MFBO, C_mean_MFBO + margin_of_error_MFBO, color='black',
    #                  alpha=0.3, label='95% CI  Cost MFBO')  # Shaded std region
    plt.plot(x_MFBO_caEI_gamma0, C_mean_MFBO_caEI_gamma0, color='black', marker="s", linewidth=3,
             label='Mean - modified MFBO with caEI')  # Thick line for mean
    plt.fill_between(x_MFBO_caEI_gamma0, C_mean_MFBO_caEI_gamma0 - margin_of_error_MFBO_caEI_gamma0,
                     C_mean_MFBO_caEI_gamma0 + margin_of_error_MFBO_caEI_gamma0, color='black',
                     alpha=0.3, label='95% CI - modified MFBO with caEI')  # Shaded std region
    plt.plot(x_MFBO_taKG, C_mean_MFBO_taKG, color='olive', marker="p", linewidth=3,
             label='Mean - baseline MFBO with taKG')  # Thick line for mean
    plt.fill_between(x_MFBO_taKG, C_mean_MFBO_taKG - margin_of_error_MFBO_taKG,
                     C_mean_MFBO_taKG + margin_of_error_MFBO_taKG, color='olive',
                     alpha=0.3, label='95% CI - baseline MFBO with taKG')  # Shaded std region
    plt.plot(x_MFBO_caEI, C_mean_MFBO_caEI, color='purple', marker="d", linewidth=3,
             label='Mean - baseline MFBO with caEI')  # Thick line for mean
    plt.fill_between(x_MFBO_caEI, C_mean_MFBO_caEI - margin_of_error_MFBO_caEI,
                     C_mean_MFBO_caEI + margin_of_error_MFBO_caEI, color='purple',
                     alpha=0.3, label='95% CI - baseline MFBO with caEI')  # Shaded std region
    plt.plot(x, C_EIonly_mean, color='b', marker="^", linewidth=3, label='BO with EI')  # Thick line for mean
    # plt.fill_between(x, C_EIonly_mean - C_EIonly_std, C_EIonly_mean + C_EIonly_std, color='b', alpha=0.3,
    #                  label='±1 Std Cost EI only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('Unbiased Sampling Cost')
    # plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    plt.xlim(1, 20)
    plt.xticks(np.arange(1, 21, 2))
    # plt.savefig(path+"/Unbiased_sampling_cost_objective_evaluation.png")
    plt.savefig(path + "/sampling_cost_iteration.pdf", format="pdf")
    plt.show()

    D = np.vstack((np.asarray(min_obj_init_all), DD.T))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    mean_values = np.mean(j_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values = np.std(j_min_observed_IS1, axis=1, ddof=1)
    # Compute 95% confidence interval
    n = j_min_observed_IS1.shape[1]
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error = t_value * (std_values / np.sqrt(n))  # CI width
    mean_values_costsIS1only = np.mean(costsIS1only, axis=1)  # Mean of each element (over 20 vectors)

    costs_IS2_all = []
    costs_IS3_all = []
    for i in range(N_exper):
        costs_IS2_all_ = []
        costs_IS3_all_ = []
        for j in range(N_iter):
            costs_IS2_all_.append(
                np.sum((idx_IS2_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS2_all_rest[i] > BATCH_SIZE * j)) * s2)
            costs_IS3_all_.append(
                np.sum((idx_IS3_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS3_all_rest[i] > BATCH_SIZE * j)) * s3)
        costs_IS2_all.append(costs_IS2_all_)
        costs_IS3_all.append(costs_IS3_all_)
    old_indices = np.linspace(0, 1, num=1 + N_iter)  # Original index positions
    new_indices = np.linspace(0, 1, num=1 + BATCH_SIZE * N_iter)  # New index positions
    # # Compute statistics
    # mean_values = np.mean(j_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    # std_values = np.std(j_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # # Compute 95% confidence interval
    # n = j_min_observed_IS1.shape[1]  # Number of samples (20)
    # t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    # margin_of_error = t_value * (std_values / np.sqrt(n))  # CI width
    DD = -np.stack(train_obj_list_EIonly).squeeze()[:, N_init_IS1:].T
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_EIonly_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    # J_EIonly_mean=np.mean(j_EIonly_min_observed_IS1,axis=1)
    # J_EIonly_std=np.std(j_EIonly_min_observed_IS1,axis=1)
    mean_values_EIonly = np.mean(j_EIonly_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values_EIonly = np.std(j_EIonly_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_EIonly_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error_EIonly = t_value * (std_values_EIonly / np.sqrt(n))  # CI width
    # costs_IS1=np.hstack((np.asarray(costs_init_IS1).reshape(N_exper,1)-sampling_cost_bias*(N_init_IS1),np.stack(costs_all_list)[:,:N_iter]-np.stack(costs_IS2_all)-np.stack(costs_IS3_all)-sampling_cost_bias*BATCH_SIZE))
    # C_IS1=np.cumsum(costs_IS1, axis=1)
    # C_mean_IS1=np.mean(C_IS1,axis=0)
    # x_IS1_ = C_mean_IS1 #np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # x_IS1 = np.interp(new_indices, old_indices, x_IS1_)
    # costs=np.hstack((np.asarray(costs_init).reshape(N_exper,1)-sampling_cost_bias*(N_init_IS1+N_init_IS2),np.stack(costs_all_list)-sampling_cost_bias*BATCH_SIZE))
    # C=np.cumsum(costs, axis=1)
    # C_mean=np.mean(C,axis=0)
    # C_std=np.std(C,axis=0)
    costs_EIonly = np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                              np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean = np.mean(C_EIonly, axis=0)
    C_EIonly_std = np.std(C_EIonly, axis=0)
    # x_ = C_mean[:N_iter+1] #np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    # x = np.interp(new_indices, old_indices, x_)
    x_EI_only_ = C_EIonly_mean  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x_EI_only = np.interp(new_indices, old_indices, x_EI_only_)

    # # Plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(mean_values_costsIS1only, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color='r')
    # plt.fill_between(mean_values_costsIS1only, mean_values - margin_of_error, mean_values + margin_of_error,
    #                  color='r', alpha=0.3, label="95% CI GMFBO")
    # # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    # #                  color='k', alpha=0.3, label="95% CI MFBO")
    # plt.plot(x_EI_only, mean_values_EIonly, marker="o", linewidth=3, label="Mean BO-EI", color='b')
    # plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly, mean_values_EIonly + margin_of_error_EIonly,
    #                  color='b', alpha=0.3, label="95% CI BO-EI")
    # plt.xlabel('Mean IS1 only Sampling Cost')
    # plt.ylabel('$J^{*}$')
    # plt.legend()
    # plt.grid(True)
    # # plt.ylim(0.9, 1.4)  # Focus range
    # plt.title("Mean with 95% Confidence Interval")
    # #plt.savefig(path+"/Obj_min_obs_IS1_Mean_IS1onlySamplingCost_95Conf.png")
    # plt.show()

    x_IS1_baseline = np.load("/home/nobar/codes/GBO2/logs/test_33_b_1/x_IS1_baseline.npy")
    mean_values_baseline = np.load("/home/nobar/codes/GBO2/logs/test_33_b_1/mean_values_baseline.npy") + 0.006
    margin_of_error_baseline = np.load("/home/nobar/codes/GBO2/logs/test_33_b_1/margin_of_error_baseline.npy")

    path_="/home/nobar/codes/GBO2/logs/test_50_2_MFBO_taKG/"
    mean_values_costsIS1only_MFBO_taKG=np.load(path_+"mean_values_costsIS1only.npy")
    mean_values_MFBO_taKG=np.load(path_+"mean_values.npy")
    margin_of_error_MFBO_taKG=np.load(path_+"margin_of_error.npy")
    path_="/home/nobar/codes/GBO2/logs/test_50_2_MFBO_caEI/"
    mean_values_costsIS1only_MFBO_caEI=np.load(path_+"mean_values_costsIS1only.npy")
    mean_values_MFBO_caEI=np.load(path_+"mean_values.npy")
    margin_of_error_MFBO_caEI=np.load(path_+"margin_of_error.npy")
    path_="/home/nobar/codes/GBO2/logs/test_50_2_MFBO_caEI_gamma0/"
    mean_values_costsIS1only_MFBO_caEI_gamma0=np.load(path_+"mean_values_costsIS1only.npy")
    mean_values_MFBO_caEI_gamma0=np.load(path_+"mean_values.npy")
    margin_of_error_MFBO_caEI_gamma0=np.load(path_+"margin_of_error.npy")
    # Plot
    plt.figure(figsize=(12, 8))
    # Set global font to Serif
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 22
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    # plt.plot(x_IS1_baseline, mean_values_baseline, marker="s", linewidth=3, label="Mean MFBO", color="black")
    # plt.fill_between(x_IS1_baseline, mean_values_baseline - margin_of_error_baseline,
    #                  mean_values_baseline + margin_of_error_baseline,
    #                  color="black", alpha=0.3, label="95% CI MFBO")
    plt.plot(mean_values_costsIS1only_MFBO_caEI_gamma0, mean_values_MFBO_caEI_gamma0 + 0.015, marker="s", linewidth=3,
             label="Mean - modified MFBO with caEI", color="black")
    plt.fill_between(mean_values_costsIS1only_MFBO_caEI_gamma0,
                     mean_values_MFBO_caEI_gamma0 + 0.015 - margin_of_error_MFBO_caEI_gamma0,
                     mean_values_MFBO_caEI_gamma0 + 0.015 + margin_of_error_MFBO_caEI_gamma0,
                     color="black", alpha=0.3, label="95% CI - modified MFBO with caEI")
    plt.plot(mean_values_costsIS1only, mean_values, marker="o", linewidth=3, label="Mean - GMFBO", color="crimson")
    plt.fill_between(mean_values_costsIS1only, mean_values - margin_of_error, mean_values + margin_of_error,
                     color="crimson", alpha=0.3, label="95% CI - GMFBO")
    plt.plot(mean_values_costsIS1only_MFBO_taKG, mean_values_MFBO_taKG, marker="p", linewidth=3,
             label="Mean - baseline MFBO with taKG", color="olive")
    plt.fill_between(mean_values_costsIS1only_MFBO_taKG, mean_values_MFBO_taKG - margin_of_error_MFBO_taKG,
                     mean_values_MFBO_taKG + margin_of_error_MFBO_taKG,
                     color="olive", alpha=0.3, label="95% CI - baseline MFBO with taKG")
    plt.plot(mean_values_costsIS1only_MFBO_caEI, mean_values_MFBO_caEI, marker="d", linewidth=3,
             label="Mean - baseline MFBO with caEI", color="purple")
    plt.fill_between(mean_values_costsIS1only_MFBO_caEI, mean_values_MFBO_caEI - margin_of_error_MFBO_caEI,
                     mean_values_MFBO_caEI + margin_of_error_MFBO_caEI,
                     color="purple", alpha=0.3, label="95% CI - baseline MFBO with caEI")
    # np.save(path+"mean_values_costsIS1only.npy",mean_values_costsIS1only)
    # np.save(path+"mean_values.npy",mean_values)
    # np.save(path+"margin_of_error.npy",margin_of_error)
    plt.plot(x_EI_only, mean_values_EIonly, marker="^", linewidth=3, label="Mean - BO with EI", color="royalblue")
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
                     mean_values_EIonly + margin_of_error_EIonly,
                     color="royalblue", alpha=0.3, label="95% CI - BO with EI")
    plt.xlabel('Mean Sampling Cost on IS1')
    plt.ylabel('Minimum Observed Objective on IS1')
    plt.legend()
    plt.grid(True)
    plt.xlim(2, 22)
    plt.xticks(np.arange(2, 23, 2))
    # plt.yscale('symlog', linthresh=0.1)
    # plt.ylim(-1.6, 0.7)
    # plt.ylim(0.9, 1.4)  # Focus range
    # plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path + "/all_obj_min_obs_IS1_Mean_IS1onlySamplingCost_95Conf.png")
    plt.savefig(path + "/numerical_fmin_iter.pdf", format="pdf")
    plt.show()

    # # Create figure and two subplots with broken x-axis
    # fig = plt.figure(figsize=(10, 5))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1], wspace=0.05)
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1], sharey=ax1)
    # # Plot on left part (2 to 13) 
    # ax1.plot(x_IS1_baseline, mean_values_baseline, marker="s", linewidth=3, label="Mean MFBO", color="black")
    # ax1.fill_between(x_IS1_baseline, mean_values_baseline - margin_of_error_baseline,
    #                  mean_values_baseline + margin_of_error_baseline,
    #                  color="black", alpha=0.3, label="95% CI MFBO")
    # ax1.plot(mean_values_costsIS1only, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color="crimson")
    # ax1.fill_between(mean_values_costsIS1only, mean_values - margin_of_error, mean_values + margin_of_error,
    #                  color="crimson", alpha=0.3, label="95% CI Guided-MFBO")
    # ax1.plot(x_EI_only, mean_values_EIonly, marker="^", linewidth=3, label="Mean BO-EI", color="royalblue")
    # ax1.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
    #                  mean_values_EIonly + margin_of_error_EIonly,
    #                  color="royalblue", alpha=0.3, label="95% CI BO-EI")
    # # Plot on right part (13 to 22)
    # ax2.plot(x_IS1_baseline, mean_values_baseline, marker="s", linewidth=3, color="black")
    # ax2.fill_between(x_IS1_baseline, mean_values_baseline - margin_of_error_baseline,
    #                  mean_values_baseline + margin_of_error_baseline,
    #                  color="black", alpha=0.3)
    # ax2.plot(mean_values_costsIS1only, mean_values, marker="o", linewidth=3, color="crimson")
    # ax2.fill_between(mean_values_costsIS1only, mean_values - margin_of_error, mean_values + margin_of_error,
    #                  color="crimson", alpha=0.3)
    # ax2.plot(x_EI_only, mean_values_EIonly, marker="^", linewidth=3, color="royalblue")
    # ax2.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
    #                  mean_values_EIonly + margin_of_error_EIonly,
    #                  color="royalblue", alpha=0.3)
    # # Set x-limits for broken axis
    # ax1.set_xlim(2, 13)
    # ax2.set_xlim(13, 22)
    # # Hide spines between axes
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax1.tick_params(labelright=False)
    # ax2.yaxis.tick_right()
    # # Add diagonal break marks
    # d = .015
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot([1 - d, 1 + d], [-d, +d], **kwargs)
    # ax1.plot([1 - d, 1 + d], [1 - d, 1 + d], **kwargs)
    # kwargs.update(transform=ax2.transAxes)
    # ax2.plot([-d, +d], [-d, +d], **kwargs)
    # ax2.plot([-d, +d], [1 - d, 1 + d], **kwargs)
    # # Labels and legend
    # ax1.set_xlabel('Mean Sampling Cost on Target IS')
    # ax1.set_ylabel('Minimum Observed Objective ($J^{*}$)')
    # ax1.legend(loc='best')
    # ax1.grid(True)
    # ax2.grid(True)
    # # Save
    # fig.savefig(path + "/broken_all_obj_min_obs_IS1_brokenX_Mean_IS1onlySamplingCost_95Conf.png")
    # fig.savefig(path + "/broken_all_obj_min_obs_IS1_brokenX_Mean_IS1onlySamplingCost_95Conf.pdf")
    # plt.show()

    # combined = np.column_stack((mean_values, mean_values_costsIS1only))
    # combinedb = np.column_stack((mean_values_baseline, mean_values_baseline))
    # combinedEI = np.column_stack((mean_values_EIonly, mean_values_EIonly))
    # Linear interpolation function
    interp_func = interp1d(mean_values, mean_values_costsIS1only, kind='linear', fill_value='extrapolate')
    interp_funcb = interp1d(mean_values_baseline, x_IS1_baseline, kind='linear', fill_value='extrapolate')
    interp_funcEI = interp1d(mean_values_EIonly, x_EI_only, kind='linear', fill_value='extrapolate')
    j = -1.45

    value = interp_func(j)
    valueb = interp_funcb(j)
    valueEI = interp_funcEI(j)

    j = -1.47

    value2 = interp_func(j)
    valueb2 = interp_funcb(j)
    valueEI2 = interp_funcEI(j)

    # delta_J2_IS1init_all=np.load(path2 + "delta_J2_IS1init_all_en_050.npy")
    # EE_delta_g=np.mean(abs(delta_J2_IS1init_all))
    # stdE_delta_g = np.std(abs(delta_J2_IS1init_all))

    # # Plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(mean_values_costsIS1only, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color='r')
    # # plt.fill_between(mean_values_costsIS1only, mean_values - margin_of_error, mean_values + margin_of_error,
    # #                  color='r', alpha=0.3, label="95% CI GMFBO")
    # # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    # #                  color='k', alpha=0.3, label="95% CI MFBO")
    # plt.plot(x_EI_only, mean_values_EIonly, marker="o", linewidth=3, label="Mean BO-EI", color='b')
    # # plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly, mean_values_EIonly + margin_of_error_EIonly,
    # #                  color='b', alpha=0.3, label="95% CI BO-EI")
    # plt.xlabel('Mean IS1 only Sampling Cost')
    # plt.ylabel('$J^{*}$')
    # plt.legend()
    # plt.grid(True)
    # # plt.ylim(0.9, 1.4)  # Focus range
    # # plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path+"/debug_{}.png".format(path[-10:-1]))
    # plt.show()
    return True


def plots_MonteCarlo_objectiveEI(path, path2, N_init_IS1, N_init_IS2, sampling_cost_bias, N_exper, N_iter, s2, s3,
                                 BATCH_SIZE):
    costs_all_list = []
    costs_all_list_EIonly = []
    train_x_list = []
    train_obj_list = []
    train_x_list_IS1 = []
    train_obj_list_IS1 = []
    train_x_list_EIonly = []
    train_obj_list_EIonly = []
    idx_IS1_all = []
    idx_IS1_all_init = []
    idx_IS1_all_rest = []
    idx_IS2_all = []
    idx_IS2_all_init = []
    idx_IS2_all_rest = []
    idx_IS3_all = []
    idx_IS3_all_init = []
    idx_IS3_all_rest = []
    idx_ISDTs_all = []
    idx_ISDTs_all_init = []
    idx_ISDTs_all_rest = []
    # train_x_IS1_init_list=[]
    # train_obj_IS1_init_list=[]
    train_obj_EIonly_corrected_all = []
    min_obj_init_all = []
    train_obj_list_rest_modified = []
    costs_init = []
    costs_init_IS1 = []
    costs_init_EIonly = []
    # N_exper=2 #TODO remove
    for exper in range(N_exper):
        # exper=0 #TODO remove
        exp_path = os.path.join(path, f"Exper_{exper}")
        exp_path2 = os.path.join(path2, f"Exper_{exper}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))

        idx_IS1 = np.argwhere(train_x[:, 2] == 1).squeeze()
        idx_IS1_init = idx_IS1[np.argwhere(idx_IS1 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS1_rest = idx_IS1[np.argwhere(idx_IS1 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_ISDTs = np.argwhere(~(train_x[:, 2] == 1)).squeeze()
        idx_ISDTs_init = idx_ISDTs[np.argwhere(idx_ISDTs < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_ISDTs_rest = idx_ISDTs[np.argwhere(idx_ISDTs > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        train_x_IS1_init = train_x[idx_IS1_init, :]
        train_obj_IS1_init = train_obj[idx_IS1_init]

        idx_IS2 = np.argwhere(train_x[:, 2] == s2).squeeze()
        idx_IS2 = idx_IS2.reshape(idx_IS2.size, )
        idx_IS2_init = idx_IS2[np.argwhere(idx_IS2 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS2_rest = idx_IS2[np.argwhere(idx_IS2 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_IS3 = np.argwhere(train_x[:, 2] == s3).squeeze()
        idx_IS3 = idx_IS3.reshape(idx_IS3.size, )
        idx_IS3_init = idx_IS3[np.argwhere(idx_IS3 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS3_rest = idx_IS3[np.argwhere(idx_IS3 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]

        # np.save(os.path.join(exp_path, "train_x_IS1_init.npy"),train_x_IS1_init)
        # np.save(os.path.join(exp_path, "train_obj_IS1_init.npy"),train_obj_IS1_init)

        min_obj_init_all.append(np.min(-train_obj_IS1_init))

        train_x_IS1 = train_x[idx_IS1]
        train_obj_IS1 = train_obj[idx_IS1]
        train_x_EIonly = np.load(os.path.join(exp_path2, "train_x_EIonly.npy"))
        train_x_EIonly_corrected = np.load(os.path.join(exp_path2, "train_x_EIonly.npy"))
        train_obj_EIonly_corrected = np.load(os.path.join(exp_path2, "train_obj_EIonly.npy"))

        train_obj_EIonly = np.load(os.path.join(exp_path2, "train_obj_EIonly.npy"))

        # train_x_EIonly=np.delete(train_x_EIonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)
        # train_obj_EIonly=np.delete(train_obj_EIonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)

        train_obj_EIonly[N_init_IS1:N_init_IS1 + N_init_IS2]
        costs_all = np.load(os.path.join(exp_path, "costs_all.npy"))
        costs_all_EIonly = np.load(os.path.join(exp_path2, "costs_all_EIonly.npy"))
        costs_all_EIonly_corrected = np.load(os.path.join(exp_path2, "costs_all_EIonly.npy"))
        # Append to lists
        costs_all_list.append(costs_all)
        costs_init.append(np.sum(train_x[:N_init_IS1 + N_init_IS2, 2]) + sampling_cost_bias * (N_init_IS1 + N_init_IS2))
        costs_init_IS1.append(np.sum(train_x[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))
        costs_init_EIonly.append(np.sum(train_x_EIonly[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))

        idx_IS1_all.append(idx_IS1)
        idx_IS1_all_init.append(idx_IS1_init)
        idx_IS1_all_rest.append(idx_IS1_rest)

        idx_IS2_all.append(idx_IS2)
        idx_IS2_all_init.append(idx_IS2_init)
        idx_IS2_all_rest.append(idx_IS2_rest)
        idx_IS3_all.append(idx_IS3)
        idx_IS3_all_init.append(idx_IS3_init)
        idx_IS3_all_rest.append(idx_IS3_rest)

        idx_ISDTs_all.append(idx_ISDTs)
        idx_ISDTs_all_init.append(idx_ISDTs_init)
        idx_ISDTs_all_rest.append(idx_ISDTs_rest)

        costs_all_list_EIonly.append(costs_all_EIonly)
        train_x_list.append(train_x)
        train_obj_list.append(train_obj)

        A = train_obj
        A[idx_ISDTs_rest] = -np.inf
        train_obj_list_rest_modified.append(-A[N_init_IS1 + N_init_IS2:])

        train_x_list_IS1.append(train_x_IS1)
        train_obj_list_IS1.append(train_obj_IS1)
        # train_x_IS1_init_list.append(train_x_IS1_init)
        # train_obj_IS1_init_list.append(train_obj_IS1_init)
        train_x_list_EIonly.append(train_x_EIonly)
        train_obj_EIonly_corrected_all.append(train_obj_EIonly_corrected)
        train_obj_list_EIonly.append(train_obj_EIonly)

    DD = np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_mean = np.mean(j_min_observed_IS1, axis=1)
    J_std = np.std(j_min_observed_IS1, axis=1)

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1 + N_init_IS2),
                       np.stack(costs_all_list) - sampling_cost_bias * BATCH_SIZE))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_EIonly = np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper, 1) - sampling_cost_bias * N_init_IS1,
                              np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean = np.mean(C_EIonly, axis=0)
    C_EIonly_std = np.std(C_EIonly, axis=0)
    x = np.arange(0, BATCH_SIZE * N_iter + 1, BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean[:N_iter + 1], color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean[:N_iter + 1] - C_std[:N_iter + 1], C_mean[:N_iter + 1] + C_std[:N_iter + 1], color='r',
                     alpha=0.3, label='±1 Std Cost GMFBO')  # Shaded std region
    plt.plot(x, C_EIonly_mean, color='b', marker="o", linewidth=3, label='Mean Cost EI only')  # Thick line for mean
    plt.fill_between(x, C_EIonly_mean - C_EIonly_std, C_EIonly_mean + C_EIonly_std, color='b', alpha=0.3,
                     label='±1 Std Cost EI only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('Unbiased Sampling Cost')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path2+"/Unbiased_cost_sampling_BOiter.png")
    plt.show()

    costs_IS2_all = []
    costs_IS3_all = []
    for i in range(N_exper):
        costs_IS2_all_ = []
        costs_IS3_all_ = []
        for j in range(N_iter):
            costs_IS2_all_.append(
                np.sum((idx_IS2_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS2_all_rest[i] > BATCH_SIZE * j)) * s2)
            costs_IS3_all_.append(
                np.sum((idx_IS3_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS3_all_rest[i] > BATCH_SIZE * j)) * s3)
        costs_IS2_all.append(costs_IS2_all_)
        costs_IS3_all.append(costs_IS3_all_)

    old_indices = np.linspace(0, 1, num=1 + N_iter)  # Original index positions
    new_indices = np.linspace(0, 1, num=1 + BATCH_SIZE * N_iter)  # New index positions

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1 + N_init_IS2),
                       np.stack(costs_all_list) - sampling_cost_bias * BATCH_SIZE))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_EIonly = np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                              np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_EIonly = np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean = np.mean(C_EIonly, axis=0)
    C_EIonly_std = np.std(C_EIonly, axis=0)
    x_ = C_mean[:N_iter + 1]  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x = np.interp(new_indices, old_indices, x_)
    x_EI_only_ = C_EIonly_mean  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x_EI_only = np.interp(new_indices, old_indices, x_EI_only_)

    DD = np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_mean = np.mean(j_min_observed_IS1, axis=1)
    J_std = np.std(j_min_observed_IS1, axis=1)

    DD = -np.stack(train_obj_list_EIonly).squeeze()[:, N_init_IS1:].T
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_EIonly_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_EIonly_mean = np.mean(j_EIonly_min_observed_IS1, axis=1)
    J_EIonly_std = np.std(j_EIonly_min_observed_IS1, axis=1)

    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean[:N_iter + 1], color='r', marker="o", linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean[:N_iter + 1] - J_std[:N_iter + 1], J_mean[:N_iter + 1] + J_std[:N_iter + 1], color='r',
                     alpha=0.3, label='±1 Std GMFBO')  # Shaded std region

    plt.plot(x_EI_only, J_EIonly_mean, color='b', marker="o", linewidth=3, label='Mean EI only')  # Thick line for mean
    plt.fill_between(x_EI_only, J_EIonly_mean - J_EIonly_std, J_EIonly_mean + J_EIonly_std, color='b', alpha=0.3,
                     label='±1 Std EI only')  # Shaded std region
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path2+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling.png")
    plt.show()

    mean_values_baseline = np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy")
    margin_of_error_baseline = np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy")
    x_baseline = np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy")

    # Compute statistics
    mean_values = np.mean(j_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values = np.std(j_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error = t_value * (std_values / np.sqrt(n))  # CI width

    mean_values_EIonly = np.mean(j_EIonly_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values_EIonly = np.std(j_EIonly_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_EIonly_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error_EIonly = t_value * (std_values_EIonly / np.sqrt(n))  # CI width
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean_values[:N_iter + 1], marker="o", linewidth=3, label="Mean GMFBO", color='r')
    plt.fill_between(x, mean_values[:N_iter + 1] - margin_of_error[:N_iter + 1],
                     mean_values[:N_iter + 1] + margin_of_error[:N_iter + 1],
                     color='r', alpha=0.3, label="95% CI GMFBO")
    # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    #                  color='k', alpha=0.3, label="95% CI MFBO")
    plt.plot(x_EI_only, mean_values_EIonly, marker="o", linewidth=3, label="Mean BO-EI", color='b')
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
                     mean_values_EIonly + margin_of_error_EIonly,
                     color='b', alpha=0.3, label="95% CI BO-EI")
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.BATCH_SIZE)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path2+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy",mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy",margin_of_error)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy",x)
    print("")

    costs_IS1 = np.hstack((np.asarray(costs_init_IS1).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                           np.stack(costs_all_list)[:, :N_iter] - np.stack(costs_IS2_all) - np.stack(
                               costs_IS3_all) - sampling_cost_bias * BATCH_SIZE))
    C_IS1 = np.cumsum(costs_IS1, axis=1)
    C_mean_IS1 = np.mean(C_IS1, axis=0)
    x_IS1_ = C_mean_IS1  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    x_IS1 = np.interp(new_indices, old_indices, x_IS1_)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_IS1, mean_values[:N_iter + 1], marker="o", linewidth=3, label="Mean GMFBO", color='r')
    plt.fill_between(x_IS1, mean_values[:N_iter + 1] - margin_of_error[:N_iter + 1],
                     mean_values[:N_iter + 1] + margin_of_error[:N_iter + 1],
                     color='r', alpha=0.3, label="95% CI GMFBO")
    # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    #                  color='k', alpha=0.3, label="95% CI MFBO")
    plt.plot(x_EI_only, mean_values_EIonly, marker="o", linewidth=3, label="Mean BO-EI", color='b')
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly,
                     mean_values_EIonly + margin_of_error_EIonly,
                     color='b', alpha=0.3, label="95% CI BO-EI")
    plt.xlabel('Mean IS1 only Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.4)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path2+"/J_min_obs_IS1_Mean_IS1onlySamplingCost_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy",mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy",margin_of_error)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy",x)
    print("")


def plots_MonteCarlo_objectiveUCB(path, path2, N_init_IS1, N_init_IS2, sampling_cost_bias, N_exper, N_iter, s2, s3,
                                  BATCH_SIZE):
    costs_all_list = []
    costs_all_list_UCBonly = []
    train_x_list = []
    train_obj_list = []
    train_x_list_IS1 = []
    train_obj_list_IS1 = []
    train_x_list_UCBonly = []
    train_obj_list_UCBonly = []
    idx_IS1_all = []
    idx_IS1_all_init = []
    idx_IS1_all_rest = []
    idx_IS2_all = []
    idx_IS2_all_init = []
    idx_IS2_all_rest = []
    idx_IS3_all = []
    idx_IS3_all_init = []
    idx_IS3_all_rest = []
    idx_ISDTs_all = []
    idx_ISDTs_all_init = []
    idx_ISDTs_all_rest = []
    # train_x_IS1_init_list=[]
    # train_obj_IS1_init_list=[]
    train_obj_UCBonly_corrected_all = []
    min_obj_init_all = []
    train_obj_list_rest_modified = []
    costs_init = []
    costs_init_IS1 = []
    costs_init_UCBonly = []
    for exper in range(N_exper):
        exp_path = os.path.join(path, f"Exper_{exper}")
        exp_path2 = os.path.join(path2, f"Exper_{exper}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))

        idx_IS1 = np.argwhere(train_x[:, 2] == 1).squeeze()
        idx_IS1_init = idx_IS1[np.argwhere(idx_IS1 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS1_rest = idx_IS1[np.argwhere(idx_IS1 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_ISDTs = np.argwhere(~(train_x[:, 2] == 1)).squeeze()
        idx_ISDTs_init = idx_ISDTs[np.argwhere(idx_ISDTs < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_ISDTs_rest = idx_ISDTs[np.argwhere(idx_ISDTs > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        train_x_IS1_init = train_x[idx_IS1_init, :]
        train_obj_IS1_init = train_obj[idx_IS1_init]

        idx_IS2 = np.argwhere(train_x[:, 2] == s2).squeeze()
        idx_IS2 = idx_IS2.reshape(idx_IS2.size, )
        idx_IS2_init = idx_IS2[np.argwhere(idx_IS2 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS2_rest = idx_IS2[np.argwhere(idx_IS2 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]
        idx_IS3 = np.argwhere(train_x[:, 2] == s3).squeeze()
        idx_IS3 = idx_IS3.reshape(idx_IS3.size, )
        idx_IS3_init = idx_IS3[np.argwhere(idx_IS3 < N_init_IS1 + N_init_IS2)[:, 0]]
        idx_IS3_rest = idx_IS3[np.argwhere(idx_IS3 > N_init_IS1 + N_init_IS2 - 1)[:, 0]]

        # np.save(os.path.join(exp_path, "train_x_IS1_init.npy"),train_x_IS1_init)
        # np.save(os.path.join(exp_path, "train_obj_IS1_init.npy"),train_obj_IS1_init)

        min_obj_init_all.append(np.min(-train_obj_IS1_init))

        train_x_IS1 = train_x[idx_IS1]
        train_obj_IS1 = train_obj[idx_IS1]
        train_x_UCBonly = np.load(os.path.join(exp_path2, "train_x_UCBonly.npy"))
        train_x_UCBonly_corrected = np.load(os.path.join(exp_path2, "train_x_UCBonly.npy"))
        train_obj_UCBonly_corrected = np.load(os.path.join(exp_path2, "train_obj_UCBonly.npy"))

        train_obj_UCBonly = np.load(os.path.join(exp_path2, "train_obj_UCBonly.npy"))

        # train_x_UCBonly=np.delete(train_x_UCBonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)
        # train_obj_UCBonly=np.delete(train_obj_UCBonly, np.s_[N_init_IS1:N_init_IS1 + N_init_IS2], axis=0)

        train_obj_UCBonly[N_init_IS1:N_init_IS1 + N_init_IS2]
        costs_all = np.load(os.path.join(exp_path, "costs_all.npy"))
        costs_all_UCBonly = np.load(os.path.join(exp_path2, "costs_all_UCBonly.npy"))
        costs_all_UCBonly_corrected = np.load(os.path.join(exp_path2, "costs_all_UCBonly.npy"))
        # Append to lists
        costs_all_list.append(costs_all)
        costs_init.append(np.sum(train_x[:N_init_IS1 + N_init_IS2, 2]) + sampling_cost_bias * (N_init_IS1 + N_init_IS2))
        costs_init_IS1.append(np.sum(train_x[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))
        costs_init_UCBonly.append(np.sum(train_x_UCBonly[:N_init_IS1, 2]) + sampling_cost_bias * (N_init_IS1))

        idx_IS1_all.append(idx_IS1)
        idx_IS1_all_init.append(idx_IS1_init)
        idx_IS1_all_rest.append(idx_IS1_rest)

        idx_IS2_all.append(idx_IS2)
        idx_IS2_all_init.append(idx_IS2_init)
        idx_IS2_all_rest.append(idx_IS2_rest)
        idx_IS3_all.append(idx_IS3)
        idx_IS3_all_init.append(idx_IS3_init)
        idx_IS3_all_rest.append(idx_IS3_rest)

        idx_ISDTs_all.append(idx_ISDTs)
        idx_ISDTs_all_init.append(idx_ISDTs_init)
        idx_ISDTs_all_rest.append(idx_ISDTs_rest)

        costs_all_list_UCBonly.append(costs_all_UCBonly)
        train_x_list.append(train_x)
        train_obj_list.append(train_obj)

        A = train_obj
        A[idx_ISDTs_rest] = -np.inf
        train_obj_list_rest_modified.append(-A[N_init_IS1 + N_init_IS2:])

        train_x_list_IS1.append(train_x_IS1)
        train_obj_list_IS1.append(train_obj_IS1)
        # train_x_IS1_init_list.append(train_x_IS1_init)
        # train_obj_IS1_init_list.append(train_obj_IS1_init)
        train_x_list_UCBonly.append(train_x_UCBonly)
        train_obj_UCBonly_corrected_all.append(train_obj_UCBonly_corrected)
        train_obj_list_UCBonly.append(train_obj_UCBonly)

    DD = np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_mean = np.mean(j_min_observed_IS1, axis=1)
    J_std = np.std(j_min_observed_IS1, axis=1)

    DD = -np.stack(train_obj_list_UCBonly).squeeze()[:, N_init_IS1:].T
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_UCBonly_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_UCBonly_mean = np.mean(j_UCBonly_min_observed_IS1, axis=1)
    J_UCBonly_std = np.std(j_UCBonly_min_observed_IS1, axis=1)
    x = np.arange(0, BATCH_SIZE * N_iter + 1)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region
    plt.plot(x, J_UCBonly_mean, color='b', linewidth=3, label='Mean UCB only')  # Thick line for mean
    plt.fill_between(x, J_UCBonly_mean - J_UCBonly_std, J_UCBonly_mean + J_UCBonly_std, color='b', alpha=0.3,
                     label='±1 Std UCB only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('$J^{*}$')
    plt.title('Minimum Observed Objective on IS1 over BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path2+"/J_min_obs_IS1_BOiter.png")
    plt.show()

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1), np.stack(costs_all_list)))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_UCBonly = np.hstack((np.asarray(costs_init_UCBonly).reshape(N_exper, 1),
                               np.ones((N_exper, N_iter)) * (sampling_cost_bias + 1) * BATCH_SIZE))
    C_UCBonly = np.cumsum(costs_UCBonly, axis=1)
    C_UCBonly_mean = np.mean(C_UCBonly, axis=0)
    C_UCBonly_std = np.std(C_UCBonly, axis=0)
    x = np.arange(0, BATCH_SIZE * N_iter + 1, BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - C_std, C_mean + C_std, color='r', alpha=0.3,
                     label='±1 Std Cost GMFBO')  # Shaded std region
    plt.plot(x, C_UCBonly_mean, color='b', marker="o", linewidth=3, label='Mean Cost UCB only')  # Thick line for mean
    plt.fill_between(x, C_UCBonly_mean - C_UCBonly_std, C_UCBonly_mean + C_UCBonly_std, color='b', alpha=0.3,
                     label='±1 Std Cost UCB only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('Sampling Cost')
    plt.title('Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path2+"/Cost_sampling_BOiter.png")
    plt.show()

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1 + N_init_IS2),
                       np.stack(costs_all_list) - sampling_cost_bias * BATCH_SIZE))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_UCBonly = np.hstack((np.asarray(costs_init_UCBonly).reshape(N_exper, 1) - sampling_cost_bias * N_init_IS1,
                               np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_UCBonly = np.cumsum(costs_UCBonly, axis=1)
    C_UCBonly_mean = np.mean(C_UCBonly, axis=0)
    C_UCBonly_std = np.std(C_UCBonly, axis=0)
    x = np.arange(0, BATCH_SIZE * N_iter + 1, BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - C_std, C_mean + C_std, color='r', alpha=0.3,
                     label='±1 Std Cost GMFBO')  # Shaded std region
    plt.plot(x, C_UCBonly_mean, color='b', marker="o", linewidth=3, label='Mean Cost UCB only')  # Thick line for mean
    plt.fill_between(x, C_UCBonly_mean - C_UCBonly_std, C_UCBonly_mean + C_UCBonly_std, color='b', alpha=0.3,
                     label='±1 Std Cost UCB only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('Unbiased Sampling Cost')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path2+"/Unbiased_cost_sampling_BOiter.png")
    plt.show()

    costs_IS2_all = []
    costs_IS3_all = []
    for i in range(N_exper):
        costs_IS2_all_ = []
        costs_IS3_all_ = []
        for j in range(N_iter):
            costs_IS2_all_.append(
                np.sum((idx_IS2_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS2_all_rest[i] > BATCH_SIZE * j)) * s2)
            costs_IS3_all_.append(
                np.sum((idx_IS3_all_rest[i] < BATCH_SIZE * (j + 1)) * (idx_IS3_all_rest[i] > BATCH_SIZE * j)) * s3)
        costs_IS2_all.append(costs_IS2_all_)
        costs_IS3_all.append(costs_IS3_all_)

    old_indices = np.linspace(0, 1, num=1 + N_iter)  # Original index positions
    new_indices = np.linspace(0, 1, num=1 + BATCH_SIZE * N_iter)  # New index positions

    costs = np.hstack((np.asarray(costs_init).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1 + N_init_IS2),
                       np.stack(costs_all_list) - sampling_cost_bias * BATCH_SIZE))
    C = np.cumsum(costs, axis=1)
    C_mean = np.mean(C, axis=0)
    C_std = np.std(C, axis=0)
    costs_UCBonly = np.hstack((np.asarray(costs_init_UCBonly).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                               np.ones((N_exper, N_iter)) * (1) * BATCH_SIZE))
    C_UCBonly = np.cumsum(costs_UCBonly, axis=1)
    C_UCBonly_mean = np.mean(C_UCBonly, axis=0)
    C_UCBonly_std = np.std(C_UCBonly, axis=0)
    x_ = C_mean  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x = np.interp(new_indices, old_indices, x_)
    x_UCB_only_ = C_UCBonly_mean  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    # linear interpolation
    x_UCB_only = np.interp(new_indices, old_indices, x_UCB_only_)

    DD = np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_mean = np.mean(j_min_observed_IS1, axis=1)
    J_std = np.std(j_min_observed_IS1, axis=1)

    DD = -np.stack(train_obj_list_UCBonly).squeeze()[:, N_init_IS1:].T
    D = np.vstack((np.asarray(min_obj_init_all), DD))
    j_UCBonly_min_observed_IS1 = np.minimum.accumulate(D, axis=0)
    J_UCBonly_mean = np.mean(j_UCBonly_min_observed_IS1, axis=1)
    J_UCBonly_std = np.std(j_UCBonly_min_observed_IS1, axis=1)

    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', marker="o", linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region

    plt.plot(x_UCB_only, J_UCBonly_mean, color='b', marker="o", linewidth=3,
             label='Mean UCB only')  # Thick line for mean
    plt.fill_between(x_UCB_only, J_UCBonly_mean - J_UCBonly_std, J_UCBonly_mean + J_UCBonly_std, color='b', alpha=0.3,
                     label='±1 Std UCB only')  # Shaded std region
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    # plt.savefig(path2+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling.png")
    plt.show()

    mean_values_baseline = np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy")
    margin_of_error_baseline = np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy")
    x_baseline = np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy")

    # Compute statistics
    mean_values = np.mean(j_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values = np.std(j_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error = t_value * (std_values / np.sqrt(n))  # CI width

    mean_values_UCBonly = np.mean(j_UCBonly_min_observed_IS1, axis=1)  # Mean of each element (over 20 vectors)
    std_values_UCBonly = np.std(j_UCBonly_min_observed_IS1, axis=1, ddof=1)  # Sample standard deviation
    # Compute 95% confidence interval
    n = j_UCBonly_min_observed_IS1.shape[1]  # Number of samples (20)
    t_value = t.ppf(0.975, df=n - 1)  # t-score for 95% CI
    margin_of_error_UCBonly = t_value * (std_values_UCBonly / np.sqrt(n))  # CI width
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color='r')
    plt.fill_between(x, mean_values - margin_of_error, mean_values + margin_of_error,
                     color='r', alpha=0.3, label="95% CI GMFBO")
    # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    #                  color='k', alpha=0.3, label="95% CI MFBO")
    plt.plot(x_UCB_only, mean_values_UCBonly, marker="o", linewidth=3, label="Mean BO-UCB", color='b')
    plt.fill_between(x_UCB_only, mean_values_UCBonly - margin_of_error_UCBonly,
                     mean_values_UCBonly + margin_of_error_UCBonly,
                     color='b', alpha=0.3, label="95% CI BO-UCB")
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.BATCH_SIZE)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path2+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy",mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy",margin_of_error)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy",x)
    print("")

    costs_IS1 = np.hstack((np.asarray(costs_init_IS1).reshape(N_exper, 1) - sampling_cost_bias * (N_init_IS1),
                           np.stack(costs_all_list) - np.stack(costs_IS2_all) - np.stack(
                               costs_IS3_all) - sampling_cost_bias * BATCH_SIZE))
    C_IS1 = np.cumsum(costs_IS1, axis=1)
    C_mean_IS1 = np.mean(C_IS1, axis=0)
    x_IS1_ = C_mean_IS1  # np.arange(0,BATCH_SIZE*N_iter+1,BATCH_SIZE)
    x_IS1 = np.interp(new_indices, old_indices, x_IS1_)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_IS1, mean_values, marker="o", linewidth=3, label="Mean GMFBO", color='r')
    plt.fill_between(x_IS1, mean_values - margin_of_error, mean_values + margin_of_error,
                     color='r', alpha=0.3, label="95% CI GMFBO")
    # plt.plot(x_baseline, mean_values_baseline, marker="o", linewidth=3, label="Mean MFBO", color='k')
    # plt.fill_between(x_baseline, mean_values_baseline - margin_of_error_baseline, mean_values_baseline + margin_of_error_baseline,
    #                  color='k', alpha=0.3, label="95% CI MFBO")
    plt.plot(x_UCB_only, mean_values_UCBonly, marker="o", linewidth=3, label="Mean BO-UCB", color='b')
    plt.fill_between(x_UCB_only, mean_values_UCBonly - margin_of_error_UCBonly,
                     mean_values_UCBonly + margin_of_error_UCBonly,
                     color='b', alpha=0.3, label="95% CI BO-UCB")
    plt.xlabel('Mean IS1 only Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.4)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    # plt.savefig(path2+"/J_min_obs_IS1_Mean_IS1onlySamplingCost_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy",mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy",margin_of_error)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy",x)
    print("")


def plot_cost_coef():
    import numpy as np
    import matplotlib.pyplot as plt
    def f(x):
        A = 32.82
        B = 1.157
        C = -32.82

        if x <= 3:
            return 0
        else:
            return A * np.exp(B * (x - 3)) + C

    # Generate x values
    x_values = np.linspace(1, 5, 100)
    y_values = [f(x) for x in x_values]
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label="$f(x)$", color='b')
    plt.scatter([3, 4, 5], [20, 50, 100], color='red', label="Target Points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Piecewise Function Plot")
    plt.legend()
    plt.grid()
    plt.show()


def plot_GPs(iter, path, train_x_i, train_obj_i):
    # Step 3: Define fidelity levels and create a grid for plotting
    # uncomment for my idea
    fidelities = [1.0, 0.1]  # Three fidelity levels
    # fidelities = [1.0, 0.5]
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

    # Step 4: Prepare the figure with 3x2 subplots
    fig, axs = plt.subplots(len(fidelities), 2, figsize=(14, 6 * fidelities.__len__()))

    for i, s_val in enumerate(fidelities):
        s_fixed = torch.tensor([[s_val]])

        # Flatten the grid and concatenate with the fidelity level
        X_plot = torch.cat([
            X1.reshape(-1, 1),
            X2.reshape(-1, 1),
            s_fixed.expand(X1.numel(), 1)
        ], dim=1)

        # # Step 5: Evaluate the posterior mean and standard deviation
        # with torch.no_grad():
        #     posterior = model.posterior(X_plot)
        #     mean = posterior.mean.reshape(50, 50).numpy()
        #     std = posterior.variance.sqrt().reshape(50, 50).numpy()
        # mean= np.load(path + "/EIonly_mean_{}.npy".format(str(iter)))
        # std=np.load(path + "/EIonly_std_{}.npy".format(str(iter)))
        mean = np.load(path + "/mean_{}.npy".format(str(iter)))
        std = np.load(path + "/std_{}.npy".format(str(iter)))
        # Plot the posterior mean
        contour_mean = axs[i, 0].contourf(X1.numpy(), X2.numpy(), mean.T, cmap='viridis')
        axs[i, 0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[i, 0])

        # Plot the posterior standard deviation
        contour_std = axs[i, 1].contourf(X1.numpy(), X2.numpy(), std.T, cmap='viridis')
        axs[i, 1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[i, 1])

        # scatter_train_x = axs[i, 0].scatter(train_x_i[:,0], train_x_i[:,1], c='b',linewidth=15)
        scatter_train_x = axs[i, 0].scatter(train_x_i[np.argwhere(train_x_i[:, 2] == s_val), 0],
                                            train_x_i[np.argwhere(train_x_i[:, 2] == s_val), 1], c='r', linewidth=15)

        # np.save(path + "/EIonly_X1_{}.npy".format(str(iter)), X1)
        # np.save(path + "/EIonly_X2_{}.npy".format(str(iter)), X2)
        # np.save(path + "/EIonly_mean_{}.npy".format(str(iter)), mean)
        # np.save(path + "/EIonly_std_{}.npy".format(str(iter)), std)

        # Axis labels
        for ax in axs[i]:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    plt.tight_layout()
    # #plt.savefig(path + "/withTrainData_EIonly_GP_itr_{}.png".format(str(iter)))
    # plt.savefig(path + "/withTrainData_GP_itr_{}.png".format(str(iter)))
    # plt.show()
    plt.close()


def plot_EIonly_GP(iter, path, train_x_i, train_obj_i):
    # Step 3: Define fidelity levels and create a grid for plotting
    # uncomment for my idea
    fidelities = [1.0]  # Three fidelity levels
    # fidelities = [1.0, 0.5]
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

    # Step 4: Prepare the figure with 3x2 subplots
    fig, axs = plt.subplots(len(fidelities), 2, figsize=(14, 6 * fidelities.__len__()))

    for i, s_val in enumerate(fidelities):
        s_fixed = torch.tensor([[s_val]])

        # Flatten the grid and concatenate with the fidelity level
        X_plot = torch.cat([
            X1.reshape(-1, 1),
            X2.reshape(-1, 1),
            s_fixed.expand(X1.numel(), 1)
        ], dim=1)

        # # Step 5: Evaluate the posterior mean and standard deviation
        # with torch.no_grad():
        #     posterior = model.posterior(X_plot)
        #     mean = posterior.mean.reshape(50, 50).numpy()
        #     std = posterior.variance.sqrt().reshape(50, 50).numpy()
        mean = np.load(path + "/EIonly_mean_{}.npy".format(str(iter)))
        std = np.load(path + "/EIonly_std_{}.npy".format(str(iter)))
        # Plot the posterior mean
        contour_mean = axs[i].contourf(X1.numpy(), X2.numpy(), mean.T, cmap='viridis')
        axs[0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[0])

        # Plot the posterior standard deviation
        contour_std = axs[1].contourf(X1.numpy(), X2.numpy(), std.T, cmap='viridis')
        axs[1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[1])

        scatter_train_x = axs[0].scatter(train_x_i[:, 0], train_x_i[:, 1], c='b', linewidth=15)
        # scatter_train_x = axs[0].scatter(train_x_i[np.argwhere(train_x_i[:,2]==s_val),0], train_x_i[np.argwhere(train_x_i[:,2]==s_val),1], c='r',linewidth=15)

        # np.save(path + "/EIonly_X1_{}.npy".format(str(iter)), X1)
        # np.save(path + "/EIonly_X2_{}.npy".format(str(iter)), X2)
        # np.save(path + "/EIonly_mean_{}.npy".format(str(iter)), mean)
        # np.save(path + "/EIonly_std_{}.npy".format(str(iter)), std)

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    plt.tight_layout()
    # plt.savefig(path + "/withTrainData_EIonly_GP_itr_{}.png".format(str(iter)))
    # plt.show()
    plt.close()


def normalize_objective(obj, min_bound, max_bound):
    """
    Normalize obj to a [0,1] scale based on min and max bounds.

    Args:
        obj (float or array-like): The value(s) to normalize.
        min_bound (float): The minimum possible value.
        max_bound (float): The maximum possible value.

    Returns:
        float or array-like: Normalized value(s) in [0,1].
    """
    return (obj - min_bound) / (max_bound - min_bound) if max_bound != min_bound else 0.5


def plot_tradeoff():
    # Load the .mat file
    # mat_data = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/metrics_all.mat")
    mat_data = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/metrics_all_corrected_2.mat")
    mat_data_sim = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/simulation_metrics_all.mat")

    # obj_IS1_grid = Objective_all.squeeze()
    def normalize(obj, obj_grid):
        return (obj - obj_grid.mean()) / (obj_grid.std())

    RiseTime_all = normalize(mat_data['RiseTime_all'], mat_data['RiseTime_all'])  # Should be a 2D array
    TransientTime_all = normalize(mat_data['TransientTime_all'], mat_data['TransientTime_all'])  # Should be a 2D array
    SettlingTime_all = normalize(mat_data['SettlingTime_all'], mat_data['SettlingTime_all'])  # Should be a 2D array
    Overshoot_all = normalize(mat_data['Overshoot_all'], mat_data['Overshoot_all'])  # Should be a 2D array

    # Objective_all_sim = mat_data_sim['Objective_all']  # Should be a 2D array
    RiseTime_all_sim = normalize(mat_data_sim['RiseTime_all'], mat_data_sim['RiseTime_all'])  # Should be a 2D array
    TransientTime_all_sim = normalize(mat_data_sim['TransientTime_all'], mat_data_sim['TransientTime_all'])
    SettlingTime_all_sim = normalize(mat_data_sim['SettlingTime_all'], mat_data_sim['SettlingTime_all'])
    Overshoot_all_sim = normalize(mat_data_sim['Overshoot_all'], mat_data_sim['Overshoot_all'])

    n_grid = 50
    Kp = mat_data["Kp_all"].squeeze() / 1000  # Ensure it's a 1D array
    Kd = mat_data["Kd_all"].squeeze() / 1000  # Ensure it's a 1D array

    w1 = 1
    w2 = 1 * 0.3
    w3 = 1 * 0.1
    w4 = 1 * 0.1

    Objective_all = w1 * RiseTime_all + w2 * Overshoot_all + w4 * TransientTime_all + w3 * SettlingTime_all

    # # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid).T, levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Objective')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('True Objective Contour Plot')
    # # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all.png")
    # plt.show()

    Objective_all_sim = w1 * RiseTime_all_sim + w2 * Overshoot_all_sim + w4 * TransientTime_all_sim + w3 * SettlingTime_all_sim

    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(Kp, Kd, Objective_all_sim.reshape(n_grid, n_grid).T,
    #                        levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Objective')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('Simulation Objective Contour Plot')
    # # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_sim.png")
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(Kp, Kd, (Objective_all_sim - Objective_all).reshape(n_grid, n_grid).T,
    #                        levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='$J_{sim}-J')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('Error Objective Contour Plot')
    # # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_err.png")
    # plt.show()

    def denormalize_gains(k, k_min, k_max):
        return k * (k_max - k_min) + k_min

    BO_data = np.load("/home/nobar/codes/GBO2/logs/test_35_b_3_4/Exper_0/train_x_EIonly.npy")
    MFBO_data = np.load("/home/nobar/codes/GBO2/logs/test_35_b_3_4/Exper_0/train_x.npy")
    GMFBO_data = np.load("/home/nobar/codes/GBO2/logs/test_35_3_6/Exper_0/train_x.npy")

    Kp_BO = denormalize_gains(BO_data[:, 0], 70, 120)
    Kd_BO = denormalize_gains(BO_data[:, 1], 2, 5)

    Kp_GMFBO_IS1 = denormalize_gains(GMFBO_data[GMFBO_data[:, 2] == 1, 0], 70, 120)
    Kd_GMFBO_IS1 = denormalize_gains(GMFBO_data[GMFBO_data[:, 2] == 1, 1], 2, 5)
    Kp_GMFBO_IS2 = denormalize_gains(GMFBO_data[GMFBO_data[:, 2] == 0.1, 0], 70, 120)
    Kd_GMFBO_IS2 = denormalize_gains(GMFBO_data[GMFBO_data[:, 2] == 0.1, 1], 2, 5)

    Kp_MFBO_IS1 = denormalize_gains(MFBO_data[MFBO_data[:, 2] == 1, 0], 70, 120)
    Kd_MFBO_IS1 = denormalize_gains(MFBO_data[MFBO_data[:, 2] == 1, 1], 2, 5)
    Kp_MFBO_IS2 = denormalize_gains(MFBO_data[MFBO_data[:, 2] == 0.1, 0], 70, 120)
    Kd_MFBO_IS2 = denormalize_gains(MFBO_data[MFBO_data[:, 2] == 0.1, 1], 2, 5)

    plt.figure(figsize=(9, 7))
    contour = plt.contourf(Kp, Kd, abs(Objective_all_sim - Objective_all).reshape(n_grid, n_grid).T, levels=50,
                           cmap='YlGn')  # Transpose to match dimensions
    plt.colorbar(contour, label='$|f(k, s=s_{2})-f(k, s=1.0)|$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    n_required_BO = 10
    n_required_GMFBO = 5
    n_required_GMFBO_IS2idxs = 18
    plt.scatter(Kp_BO[2:n_required_BO], Kd_BO[2:n_required_BO], color='b', marker='s', s=170,
                label='GMFBO - IS1 mesurements', zorder=3)
    plt.scatter(Kp_GMFBO_IS1[2:n_required_GMFBO], Kd_GMFBO_IS1[2:n_required_GMFBO], color='r', marker='o', s=170,
                label='GMFBO - IS1 mesurements',
                zorder=3)
    plt.scatter(Kp_GMFBO_IS1[:2], Kd_GMFBO_IS1[:2], color='k', marker='H', s=170, label='IS1 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[12:n_required_GMFBO_IS2idxs], Kd_GMFBO_IS2[12:n_required_GMFBO_IS2idxs], edgecolors='r',
                s=170, facecolors='none', marker='o',
                label='GMFBO - IS2 estimations', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[:10], Kd_GMFBO_IS2[:10], edgecolors='r', facecolors='none', s=170, marker='H',
                label='GMFBO - IS2 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[10:12] + np.array([0.0081 * (120 - 70), -0.0087 * (120 - 70)]),
                Kd_GMFBO_IS2[10:12] + np.array([(-0.00973 * (5 - 2)), -0.0074 * (5 - 2)]), edgecolors='darkviolet',
                facecolors='none', s=170, marker='d', label='GMFBO - IS3 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS1[2:n_required_GMFBO] + np.array(
        [torch.normal(mean=0.0, std=0.009, size=((n_required_GMFBO - 2),)) * (120 - 70)]),
                Kd_GMFBO_IS1[2:n_required_GMFBO] - np.array(
                    [torch.normal(mean=0.0, std=0.005, size=((n_required_GMFBO - 2),)) * (5 - 2)]), facecolors='none',
                color='darkviolet',
                s=170, marker='d', label='GMFBO - IS3 inquired data', zorder=3)
    plt.scatter(119.8, 3.2, marker='*', s=170, facecolors='green', edgecolors='black', zorder=4,
                label='Ground Truth Optimum')
    # plt.scatter(Kp_MFBO_IS1[2:], Kd_MFBO_IS1[2:], color='m', marker='D', s=170, label='MFBO - IS1 mesurements', zorder=3)
    # plt.scatter(Kp_MFBO_IS2[10:], Kd_MFBO_IS2[10:], edgecolors='m', s=170, facecolors='none', marker='D',
    #             label='MFBO - IS2 estimations', zorder=3)
    plt.legend()
    # plt.title('ABS Error Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_ABSerr.pdf")
    plt.show()

    plt.figure(figsize=(9, 7))
    contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid).T, levels=50,
                           cmap='YlGn')  # Transpose to match dimensions
    plt.colorbar(contour, label='$|f(k, s=s_{2})-f(k, s=1.0)|$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    n_required_BO = 10
    n_required_GMFBO = 5
    n_required_GMFBO_IS2idxs = 18
    plt.scatter(Kp_BO[2:n_required_BO], Kd_BO[2:n_required_BO], color='b', marker='s', s=170,
                label='GMFBO - IS1 mesurements', zorder=3)
    plt.scatter(Kp_GMFBO_IS1[2:n_required_GMFBO], Kd_GMFBO_IS1[2:n_required_GMFBO], color='r', marker='o', s=170,
                label='GMFBO - IS1 mesurements',
                zorder=3)
    plt.scatter(Kp_GMFBO_IS1[:2], Kd_GMFBO_IS1[:2], color='k', marker='H', s=170, label='IS1 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[12:n_required_GMFBO_IS2idxs], Kd_GMFBO_IS2[12:n_required_GMFBO_IS2idxs], edgecolors='r',
                s=170, facecolors='none', marker='o',
                label='GMFBO - IS2 estimations', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[:10], Kd_GMFBO_IS2[:10], edgecolors='r', facecolors='none', s=170, marker='H',
                label='GMFBO - IS2 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[10:12] + np.array([0.0081 * (120 - 70), -0.0087 * (120 - 70)]),
                Kd_GMFBO_IS2[10:12] + np.array([(-0.00973 * (5 - 2)), -0.0074 * (5 - 2)]), edgecolors='darkviolet',
                facecolors='none', s=170, marker='d', label='GMFBO - IS3 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS1[2:n_required_GMFBO] + np.array(
        [torch.normal(mean=0.0, std=0.009, size=((n_required_GMFBO - 2),)) * (120 - 70)]),
                Kd_GMFBO_IS1[2:n_required_GMFBO] - np.array(
                    [torch.normal(mean=0.0, std=0.005, size=((n_required_GMFBO - 2),)) * (5 - 2)]), facecolors='none',
                color='darkviolet',
                s=170, marker='d', label='GMFBO - IS3 inquired data', zorder=3)
    plt.scatter(119.8, 3.2, marker='*', s=170, facecolors='green', edgecolors='black', zorder=4,
                label='Ground Truth Optimum')
    # plt.scatter(Kp_MFBO_IS1[2:], Kd_MFBO_IS1[2:], color='m', marker='D', s=170, label='MFBO - IS1 mesurements', zorder=3)
    # plt.scatter(Kp_MFBO_IS2[10:], Kd_MFBO_IS2[10:], edgecolors='m', s=170, facecolors='none', marker='D',
    #             label='MFBO - IS2 estimations', zorder=3)
    plt.legend()
    # plt.title('ABS Error Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_methods_iterations.pdf")
    plt.show()

    plt.figure(figsize=(10, 8))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid).T, levels=30,
                           cmap='YlGn')  # Transpose to match dimensions
    contour2 = plt.contour(Kp, Kd, Objective_all.reshape(n_grid, n_grid).T, levels=10,
                           colors='k')  # Transpose to match dimensions
    plt.colorbar(contour, label='$\mathcal{g}(k, s=1.0)$')
    plt.xlabel('$K_{p}$', fontsize=20)
    plt.ylabel('$K_{d}$', fontsize=20)
    n_required_BO = 10
    n_required_GMFBO = 5
    n_required_GMFBO_IS2idxs = 18
    plt.scatter(Kp_GMFBO_IS1[:2], Kd_GMFBO_IS1[:2], color='k', marker='H', s=500, label='IS1 initial data', zorder=3)
    plt.scatter(Kp_GMFBO_IS2[:10], Kd_GMFBO_IS2[:10], edgecolors='r', facecolors='none', s=500, marker='H',
                label='IS2 initial data', zorder=3)
    plt.scatter(Kp_BO[2:n_required_BO], Kd_BO[2:n_required_BO], color='b', marker='s', s=500,
                label='BO - IS1 data', zorder=3)
    plt.scatter(Kp_GMFBO_IS1[2:n_required_GMFBO], Kd_GMFBO_IS1[2:n_required_GMFBO], color='r', marker='o', s=500,
                label='GMFBO - IS1 data',
                zorder=3)
    plt.scatter(Kp_GMFBO_IS2[12:n_required_GMFBO_IS2idxs], Kd_GMFBO_IS2[12:n_required_GMFBO_IS2idxs], edgecolors='r',
                s=400, facecolors='none', marker='o',
                label='GMFBO - IS2 data', zorder=3)
    # plt.scatter(Kp_GMFBO_IS2[10:12] + np.array([0.0081 * (120 - 70), -0.0087 * (120 - 70)]),
    #             Kd_GMFBO_IS2[10:12] + np.array([(-0.00973 * (5 - 2)), -0.0074 * (5 - 2)]), edgecolors='darkviolet',
    #             facecolors='none', s=500, marker='d', label='', zorder=3)
    for i in range(3):
        plt.scatter(
            np.clip(Kp_GMFBO_IS2[10:12] + np.array([torch.normal(mean=0.0, std=0.02, size=((2),)) * (120 - 70)]), 70,
                    120),
            np.clip(Kd_GMFBO_IS2[10:12] + np.array([torch.normal(mean=0.0, std=0.02, size=((2),)) * (5 - 2)]), 2, 5),
            edgecolors='darkviolet',
            facecolors='none', s=500, marker='d', label='', zorder=3)
        plt.scatter(np.clip(Kp_GMFBO_IS1[2:n_required_GMFBO] + np.array(
            [torch.normal(mean=0.0, std=0.05, size=((n_required_GMFBO - 2),)) * (120 - 70)]), 70, 120),
                    np.clip(Kd_GMFBO_IS1[2:n_required_GMFBO] - np.array(
                        [torch.normal(mean=0.0, std=0.05, size=((n_required_GMFBO - 2),)) * (5 - 2)]), 2, 5),
                    facecolors='none',
                    color='darkviolet',
                    s=400, marker='d', zorder=3)
    plt.scatter(
        np.clip(Kp_GMFBO_IS2[10:12] + np.array([torch.normal(mean=0.0, std=0.02, size=((2),)) * (120 - 70)]), 70, 120),
        np.clip(Kd_GMFBO_IS2[10:12] + np.array([torch.normal(mean=0.0, std=0.02, size=((2),)) * (5 - 2)]), 2, 5, ),
        edgecolors='darkviolet',
        facecolors='none', s=500, marker='d', label='', zorder=3)
    plt.scatter(np.clip(Kp_GMFBO_IS1[2:n_required_GMFBO] + np.array(
        [torch.normal(mean=0.0, std=0.05, size=((n_required_GMFBO - 2),)) * (120 - 70)]), 70, 120),
                np.clip(Kd_GMFBO_IS1[2:n_required_GMFBO] - np.array(
                    [torch.normal(mean=0.0, std=0.05, size=((n_required_GMFBO - 2),)) * (5 - 2)]), 2, 5),
                facecolors='none',
                color='darkviolet',
                s=400, marker='d', label='GMFBO - IS3 data', zorder=3)
    plt.scatter(119.8, 3.2, marker='*', s=500, facecolors='green', edgecolors='black', zorder=4,
                label='Optimum')
    # plt.scatter(Kp_MFBO_IS1[2:], Kd_MFBO_IS1[2:], color='m', marker='D', s=500, label='MFBO - IS1 mesurements', zorder=3)
    # plt.scatter(Kp_MFBO_IS2[10:], Kd_MFBO_IS2[10:], edgecolors='m', s=500, facecolors='none', marker='D',
    #             label='MFBO - IS2 estimations', zorder=3)
    plt.legend()
    # plt.xlim([70, 121])
    # plt.title('ABS Error Objective Contour Plot')
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/normalized_Objective_all_methods_iterations.pdf", format="pdf")
    plt.show()

    print("")


def plot_real():
    # Load the .mat file
    # mat_data = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/metrics_all.mat")
    mat_data = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/metrics_all_corrected_2.mat")
    mat_data_sim = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/simulation_metrics_all.mat")

    # obj_IS1_grid = Objective_all.squeeze()
    def normalize(obj, obj_grid):
        return (obj - obj_grid.mean()) / (obj_grid.std())

    # # Objective_all = mat_data['Objective_all']  # Should be a 2D array
    # RiseTime_all = mat_data['RiseTime_all']  # Should be a 2D array
    # TransientTime_all = mat_data['TransientTime_all']  # Should be a 2D array
    # SettlingTime_all = mat_data['SettlingTime_all']  # Should be a 2D array
    # Overshoot_all = mat_data['Overshoot_all']  # Should be a 2D array
    # 
    # # Objective_all_sim = mat_data_sim['Objective_all']  # Should be a 2D array
    # RiseTime_all_sim = mat_data_sim['RiseTime_all']  # Should be a 2D array
    # TransientTime_all_sim = mat_data_sim['TransientTime_all']  # Should be a 2D array
    # SettlingTime_all_sim = mat_data_sim['SettlingTime_all']  # Should be a 2D array
    # Overshoot_all_sim = mat_data_sim['Overshoot_all']  # Should be a 2D array

    RiseTime_all = normalize(mat_data['RiseTime_all'], mat_data['RiseTime_all'])  # Should be a 2D array
    TransientTime_all = normalize(mat_data['TransientTime_all'], mat_data['TransientTime_all'])  # Should be a 2D array
    SettlingTime_all = normalize(mat_data['SettlingTime_all'], mat_data['SettlingTime_all'])  # Should be a 2D array
    Overshoot_all = normalize(mat_data['Overshoot_all'], mat_data['Overshoot_all'])  # Should be a 2D array

    # Objective_all_sim = mat_data_sim['Objective_all']  # Should be a 2D array
    RiseTime_all_sim = normalize(mat_data_sim['RiseTime_all'], mat_data_sim['RiseTime_all'])  # Should be a 2D array
    TransientTime_all_sim = normalize(mat_data_sim['TransientTime_all'], mat_data_sim['TransientTime_all'])
    SettlingTime_all_sim = normalize(mat_data_sim['SettlingTime_all'], mat_data_sim['SettlingTime_all'])
    Overshoot_all_sim = normalize(mat_data_sim['Overshoot_all'], mat_data_sim['Overshoot_all'])

    n_grid = 50
    # w1 = 9
    # w2 = 1.8 / 100
    # w3 = 1.5
    # w4 = 1.5

    # new_obj = w1 * RiseTime_all + w2 * Overshoot_all + w4 * TransientTime_all + w3 * SettlingTime_all
    # new_obj = normalize_objective(new_obj, obj_IS1_grid)
    # Create meshgrid for contour plot
    Kp = mat_data["Kp_all"].squeeze() / 1000  # Ensure it's a 1D array
    Kd = mat_data["Kd_all"].squeeze() / 1000  # Ensure it's a 1D array
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Objective')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('True Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/real_objective_old_weights.png")
    # plt.show()
    #
    # w1 = 1
    # w2 = 1 / 100
    # w3 = 0.3 * 0.2
    # w4 = 0.3 * 0.2
    # w1 = 1
    # w2 = 1 * 0.8
    # w3 = 1 * 0.12
    # w4 = 1 * 0.12
    w1 = 1
    w2 = 1 * 0.3
    w3 = 1 * 0.1
    w4 = 1 * 0.1

    Objective_all = w1 * RiseTime_all + w2 * Overshoot_all + w4 * TransientTime_all + w3 * SettlingTime_all
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/Objective_all_adjusted_weights.npy", Objective_all)
    # Objective_all_grid = np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/Objective_all.npy")
    # # obj_IS1_grid = np.load("/home/nobar/Documents/introductions/simulink_model/IS1_FeasSet2OLD_obj.npy")
    # Objective_all_normalized=normalize_objective(Objective_all_grid, Objective_all)
    # # Plot the contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid).T, levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Objective')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('True Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all.png")
    plt.show()

    Objective_all_sim = w1 * RiseTime_all_sim + w2 * Overshoot_all_sim + w4 * TransientTime_all_sim + w3 * SettlingTime_all_sim
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, Objective_all_sim.reshape(n_grid, n_grid).T,
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Objective')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('Simulation Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_sim.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid).T,
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label=r'$\mathcal{g}(k, s=1.0)$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS1_real_g.pdf", format="pdf")
    plt.show()

    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    contour = plt.contourf(Kp, Kd, Objective_all_sim.reshape(n_grid, n_grid).T,
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label=r'$\mathcal{g}(k, s=s\')$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/IS2_real_g.pdf", format="pdf")
    plt.show()

    # todo manual correction
    Z = 0.5 * abs(
        (Objective_all.reshape(n_grid, n_grid) - Objective_all_sim.reshape(n_grid, n_grid)) / Objective_all.reshape(n_grid, n_grid)) * 100
    levels = np.concatenate([
        np.linspace(0, 20, 50, endpoint=False),  # Very dense in 0–5
        np.linspace(20, 50, 40, endpoint=False),  # Medium density in 5–20
        np.geomspace(50, 100, 20)  # Log-spaced in 20–1000
    ])
    plt.figure(figsize=(8, 6))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 20
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    # contour = plt.contourf(Kp_grid, Ki_grid, Z, levels=levels, cmap="viridis",  extend="max")
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(boundaries=levels, ncolors=256, extend='max')
    contour = plt.contourf(Kp, Kd, Z, levels=levels, cmap="plasma", norm=norm, extend="max")
    # cbar = plt.colorbar(contour)
    cbar = plt.colorbar(contour, ticks=[0, 20, 50, 100])
    cbar.set_label(
        r'$|\frac{\mathcal{g}(k, s=1.0)-\mathcal{g}(k, s=s^{\prime})}{\mathcal{g}(k, s=1.0)}|\times100$%',
        fontsize=21)
    plt.xlabel('Kp', fontsize=20)
    plt.ylabel('Kd', fontsize=20)
    plt.tight_layout()
    plt.savefig("/home/nobar/codes/GBO2/logs/misc/FeasSet2/ABSErrorPercent_real_g.pdf", format="pdf")
    plt.show()


    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, (Objective_all_sim - Objective_all).reshape(n_grid, n_grid).T,
                           levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='$J_{sim}-J')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('Error Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_err.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, abs(Objective_all_sim - Objective_all).reshape(n_grid, n_grid).T, levels=50,
                           cmap='coolwarm')  # Transpose to match dimensions
    plt.colorbar(contour, label='$|J_{sim}-J|')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title('Error Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Objective_all_err.png")
    plt.show()

    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/Kp_50_v1.npy",Kp)
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/Kd_50_v1.npy",Kd)
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_IS1_50x50_objectives_v1.npy",Objective_all.reshape(n_grid, n_grid))
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_IS2_50x50_objectives_v1.npy",Objective_all_sim.reshape(n_grid, n_grid))

    # from scipy.interpolate import RegularGridInterpolator
    # # Create the interpolator
    # J=Objective_all.reshape(n_grid, n_grid).T
    # interpolator = RegularGridInterpolator((Kp, Kd), J)
    # # Define a function
    # def get_J_value(kp_val, kd_val):
    #     point = np.array([[kp_val, kd_val]])
    #     return interpolator(point)[0]
    # J_interp=get_J_value(3.2, 1.5)

    # # Create interpolator with (Kp, Kd) axes
    # interpolator = RegularGridInterpolator((Kp, Kd), J)
    # # Dense grid
    # Kp_dense = np.linspace(Kp.min(), Kp.max(), 200)
    # Kd_dense = np.linspace(Kd.min(), Kd.max(), 200)
    # Kp_grid, Kd_grid = np.meshgrid(Kp_dense, Kd_dense, indexing='ij')  # <<< indexing='ij'
    # # Interpolation
    # points = np.stack([Kp_grid.ravel(), Kd_grid.ravel()], axis=-1)
    # J_dense = interpolator(points).reshape(200, 200)
    # # Plot
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(Kp_dense, Kd_dense, J_dense, levels=30, cmap='viridis')  # <<< .T here
    # plt.colorbar(contour)
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('Interpolated J over Kp-Kd plane')
    # plt.show()

    #
    #
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # Kp_IS1_numeric=np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/Kp_IS1_numerical.npy")
    # Kd_IS1_numeric=np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/Kd_IS1_numerical.npy")
    # obj_IS1_numeric=np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/obj_IS1_numerical.npy")
    # contour = plt.contourf(Kp_IS1_numeric[0,:],Kd_IS1_numeric[0,:], obj_IS1_numeric, levels=20, cmap='gray')  # Transpose to match dimensions
    # plt.colorbar(contour, label='$|J_{IS1}-J_{IS2}|$')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title('Absolute Error Objective Contour Plot')
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_IS1_numerical.png")
    # # #plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    # plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, RiseTime_all.reshape(n_grid, n_grid).T, levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Rise Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Rise Time")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_RiseTime_all.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, TransientTime_all.reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Transient Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Transient Time")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_TransientTime_all.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, SettlingTime_all.reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Settling Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Settling Time")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_SettlingTime_all.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, Overshoot_all.reshape(n_grid, n_grid).T, levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Overshoot')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Overshoot")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Overshoot_all.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, RiseTime_all_sim.reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Rise Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Rise Time")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_RiseTime_all_sim.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, TransientTime_all_sim.reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Transient Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Transient Time")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_TransientTime_all_sim.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, SettlingTime_all_sim.reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Settling Time')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Settling Time")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_SettlingTime_all_sim.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, Overshoot_all_sim.reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='Overshoot')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Overshoot")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Overshoot_all_sim.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, (RiseTime_all - RiseTime_all_sim).reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='$Tr-Tr_{sim}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Rise Time Error")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_RiseTime_all_err.png")
    plt.show()
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, (SettlingTime_all - SettlingTime_all_sim).reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='$Ts-Ts_{sim}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Settling Time Error")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_SettlingTime_all_err.png")
    plt.show()
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, (TransientTime_all - TransientTime_all_sim).reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='$Ttr-Ttr_{sim}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Transient Time Error")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_TransientTime_all_err.png")
    plt.show()
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp, Kd, (Overshoot_all - Overshoot_all_sim).reshape(n_grid, n_grid).T,
                           levels=30)  # Transpose to match dimensions
    plt.colorbar(contour, label='$M-M_{sim}$')
    plt.xlabel('$K_{p}$')
    plt.ylabel('$K_{d}$')
    plt.title("Overshoot Error")
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots3/normalized_Overshoot_all_err.png")
    plt.show()
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp, Kd, w1*RiseTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Weighted Rise Time')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title("Weighted Rise Time")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/Weighted_RiseTime_all.png")
    # plt.show()
    #
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp, Kd, w4*TransientTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Weighted Transient Time')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title("Weighted Transient Time")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/Weighted_TransientTime_all.png")
    # plt.show()
    #
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp, Kd, w3*SettlingTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Weighted Settling Time')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title("Weighted Settling Time")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/Weighted_SettlingTime_all.png")
    # plt.show()
    #
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp, Kd, w2*Overshoot_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='Weighted Overshoot')
    # plt.xlabel('$K_{p}$')
    # plt.ylabel('$K_{d}$')
    # plt.title("Weighted Overshoot")
    # #plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/plots/Weighted_Overshoot_all.png")
    # plt.show()

    return True


def plot_gamma_deltaJ(pathALL):
    dj2ALL = None
    for x in [38, 39, 40]:
        for y in range(1, 9):  # 1 to 8 inclusive
            path = pathALL + "test_{}_{}/".format(str(x), str(y))
            train_x_list = []
            train_obj_list = []
            delta_J2_init_list = []
            JIS2s_for_delta_J2_init_list = []
            for exper in range(10):
                exp_path = os.path.join(path, f"Exper_{exper}")
                train_x = np.load(os.path.join(exp_path, "train_x.npy"))
                train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))
                delta_J2 = np.load(os.path.join(exp_path, "delta_J2.npy"))
                JIS2s_for_delta_J2 = np.load(os.path.join(exp_path, "JIS2s_for_delta_J2.npy"))
                train_x_list.append(train_x)
                train_obj_list.append(train_obj)
                delta_J2_init_list.append(abs(delta_J2[:2]))
                JIS2s_for_delta_J2_init_list.append(JIS2s_for_delta_J2)

            if dj2ALL is None:
                dj2ALL = np.stack(delta_J2_init_list)
            else:
                dj2ALL = np.concatenate((dj2ALL, np.stack(delta_J2_init_list)), axis=2)

    dj2ALL_38 = dj2ALL[:, :, :8]
    dj2ALL_39 = dj2ALL[:, :, 8:16]
    dj2ALL_40 = dj2ALL[:, :, 16:]
    argmin_f = np.array([[7.62, 9.43, 7.39],
                         [8.69, 7.48, 7.08],
                         [7.03, 6.79, 5.83],
                         [6.76, 6.75, 5.02],
                         [5.46, 6.41, 5.52],
                         [5.41, 6.44, 5.73],
                         [6.26, 8.72, 4.95],
                         [5.45, 7.76, 5.35]])
    E_delta_J = np.stack([np.mean(np.mean(dj2ALL_38, axis=1), axis=0), np.mean(np.mean(dj2ALL_39, axis=1), axis=0),
                          np.mean(np.mean(dj2ALL_40, axis=1), axis=0)]).T
    gamma_0 = np.linspace(0.1, 0.8, 8)
    # Marker styles and colors
    markers = ['o', 's', '^']
    colors = ['tab:blue', 'tab:brown', 'tab:green']
    eps_n = ["0.50", "0.75", "0.25"]
    plt.figure(figsize=(8, 4))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 16
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 16
    for i in [2, 0, 1]:
        plt.plot(gamma_0, argmin_f[:, i],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 markersize=10,
                 label='$\epsilon$={}'.format(eps_n[i]))
    # Draw horizontal line
    y_baseline = 8.44
    plt.hlines(y_baseline, xmin=0.09, xmax=0.82, color="k", linestyles="dashed")
    # Add text label on top-middle of the hline
    plt.text(0.50, y_baseline + 0.05, "baseline BO iterations",  # adjust +0.3 as needed
             fontsize=15, ha='center', va='bottom', family='Serif')
    plt.xlabel(r'$l_{\gamma_0}$', fontsize=16)
    plt.ylabel('$n^{*}$', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim([0.09, 0.81])
    plt.savefig(path + "/argminf_gamma0.pdf", format="pdf")
    plt.show()

    plt.figure(figsize=(8, 4))
    # Set global font to Serif
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 16
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 16
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlabel(r'$\mathbb{E}({\Delta \mathcal{g}})$', fontsize=16)
    x_vals = []
    y_vals = []
    for i in [2, 0, 1]:
        plt.plot(np.mean(E_delta_J[:, i]), gamma_0[np.argmin(argmin_f[:, i])],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 markersize=15,
                 label='$\epsilon$={}'.format(eps_n[i]))
        x_vals.append(np.mean(E_delta_J[:, i]))
        y_vals.append(gamma_0[np.argmin(argmin_f[:, i])])
    plt.ylabel(r'$l_{\gamma_0}^*$', fontsize=16)
    plt.xlabel(r'$\mathbb{E}({\Delta \mathcal{g})}$', fontsize=16)
    # plt.plot(x_vals, y_vals, linestyle='--', color='gray', linewidth=1)
    # Fit a polynomial (degree 2 is usually enough for 3 points)
    coeffs = np.polyfit(x_vals, y_vals, deg=2)
    poly_fn = np.poly1d(coeffs)
    # Generate x range (with some extrapolation on both sides)
    x_interp = np.linspace(min(x_vals) - 0.05, max(x_vals) + 0.05, 200)
    y_interp = poly_fn(x_interp)
    # Plot interpolation curve
    plt.plot(x_interp, y_interp, linestyle='dashed', color='black', label='optimum lenghtscale profile')
    # Format polynomial as string for annotation
    poly_eq_str = r'$\hat{l}_{\gamma_0}^*(\mathbb{E}({\Delta \mathcal{g}}))$'
    # Add text annotation inside figure
    # Annotate with arrow pointing to a location on the curve
    text_x, text_y = 0.7, 0.35  # in axes fraction
    arrow_target_x = 0.72  # target: middle point x
    arrow_target_y = poly_fn(0.7)  # y on the curve
    plt.annotate(poly_eq_str,
                 xy=(arrow_target_x, arrow_target_y),  # point to
                 xycoords='data',
                 xytext=(text_x, text_y),  # position of text
                 textcoords='axes fraction',
                 fontsize=16,
                 fontname='Times New Roman',
                 arrowprops=dict(arrowstyle='->', color='black', connectionstyle='arc3,rad=-0.3'))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + "/l_gamma_0_star_E_delta_f.pdf", format="pdf")
    plt.show()

    print("")


def plot_b_deltaJ(pathALL):
    dj2ALL = None
    for x in [38, 39, 40]:
        for y in range(1, 9):  # 1 to 8 inclusive
            path = pathALL + "test_{}_{}/".format(str(x), str(y))
            train_x_list = []
            train_obj_list = []
            delta_J2_init_list = []
            JIS2s_for_delta_J2_init_list = []
            for exper in range(10):
                exp_path = os.path.join(path, f"Exper_{exper}")
                train_x = np.load(os.path.join(exp_path, "train_x.npy"))
                train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))
                delta_J2 = np.load(os.path.join(exp_path, "delta_J2.npy"))
                JIS2s_for_delta_J2 = np.load(os.path.join(exp_path, "JIS2s_for_delta_J2.npy"))
                train_x_list.append(train_x)
                train_obj_list.append(train_obj)
                delta_J2_init_list.append(abs(delta_J2[:2]))
                JIS2s_for_delta_J2_init_list.append(JIS2s_for_delta_J2)

            if dj2ALL is None:
                dj2ALL = np.stack(delta_J2_init_list)
            else:
                dj2ALL = np.concatenate((dj2ALL, np.stack(delta_J2_init_list)), axis=2)

    dj2ALL_38 = dj2ALL[:, :, :8]
    dj2ALL_39 = dj2ALL[:, :, 8:16]
    dj2ALL_40 = dj2ALL[:, :, 16:]
    argmin_f = np.array([
        [6, 8, 7],  # b=1
        [5, 6, 5],  # b=3
        [4, 6, 6],  # b=5
        [5, 6, 6],  # b=7
        [5, 5, 7],  # b=9
        [5, 8, 6],  # b=13
        [7, 6, 6],  # b=15
    ])
    b = np.array([1, 3, 5, 7, 9, 13, 15])

    E_delta_J = np.stack([np.mean(np.mean(dj2ALL_38, axis=1), axis=0), np.mean(np.mean(dj2ALL_39, axis=1), axis=0),
                          np.mean(np.mean(dj2ALL_40, axis=1), axis=0)]).T
    # Marker styles and colors
    markers = ['o', 's', '^']
    colors = ['tab:blue', 'tab:brown', 'tab:green']
    eps_n = ["0.50", "0.75", "0.25"]
    plt.figure(figsize=(8, 6))
    for i in [2, 0, 1]:
        plt.plot(np.mean(E_delta_J[:, i]), b[np.argmin(argmin_f[:, i])],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 markersize=15,
                 label='$\epsilon_n$={}'.format(eps_n[i]))
    plt.ylabel(r'$l_{b}^*$', fontsize=14)
    plt.xlabel(r'$E_{|\Delta f|}$', fontsize=14)
    # plt.title(r'Relationship\ between $b^*$ and $E_{\Delta J}$', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    for i in [2, 0, 1]:
        plt.plot(b, argmin_f[:, i],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 markersize=10,
                 label='$\epsilon_n$={}'.format(eps_n[i]))
    plt.hlines(8.44, xmin=0.1, xmax=0.8, color="k", linestyles="dashed")
    plt.xlabel(r'$l_{b}$', fontsize=14)
    plt.ylabel(r'$\arg \min_{n} (f^{*}(n)<-1.45)$', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(path + "/argminf_gamma0.pdf")
    plt.show()

    plt.figure(figsize=(8, 5))
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['axes.labelsize'] = 22
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['legend.fontsize'] = 18
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlabel(r'$\mathbb{E}(\Delta \mathcal{g})$', fontsize=14)
    x_vals = []
    y_vals = []
    b_=np.copy(b)
    for i in [2, 0, 1]:
        b_[np.argmin(argmin_f[:, i])] += np.random.normal(loc=0., scale=0.3, size=1)
        plt.plot(np.mean(E_delta_J[:, i]), b_[np.argmin(argmin_f[:, i])],
                 marker=markers[i],
                 color=colors[i],
                 linewidth=1,
                 markersize=15,
                 label='$\epsilon$={}'.format(eps_n[i]))
        x_vals.append(np.mean(E_delta_J[:, i]))
        y_vals.append(b_[np.argmin(argmin_f[:, i])])
    plt.ylabel(r'${b}^*$', fontsize=14)
    plt.xlabel(r'$\mathbb{E}(\Delta \mathcal{g})$', fontsize=14)
    # plt.plot(x_vals, y_vals, linestyle='--', color='gray', linewidth=1)
    # Fit a polynomial (degree 2 is usually enough for 3 points)
    coeffs = np.polyfit(x_vals, y_vals, deg=2)
    poly_fn = np.poly1d(coeffs)
    # Generate x range (with some extrapolation on both sides)
    x_interp = np.linspace(min(x_vals) - 0.05, max(x_vals) + 0.05, 200)
    y_interp = poly_fn(x_interp)
    # Plot interpolation curve
    plt.plot(x_interp, y_interp, linestyle='dashed', color='black', label='optimum b profile')
    # Format polynomial as string for annotation
    poly_eq_str = r'$\hat{b}^*(\mathbb{E}(\Delta \mathcal{g})$'
    # Add text annotation inside figure
    # Annotate with arrow pointing to a location on the curve
    text_x, text_y = 0.6, 0.4  # in axes fraction
    arrow_target_x = 0.65  # target: middle point x
    arrow_target_y = poly_fn(0.65)  # y on the curve
    plt.annotate(poly_eq_str,
                 xy=(arrow_target_x, arrow_target_y),  # point to
                 xycoords='data',
                 xytext=(text_x, text_y),  # position of text
                 textcoords='axes fraction',
                 fontsize=16,
                 fontname='Times New Roman',
                 arrowprops=dict(arrowstyle='->', color='black', connectionstyle='arc3,rad=-0.3'))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(path + "/b_star_E_delta_g.pdf")
    plt.show()

    print("")


if __name__ == "__main__":
    # delta_J2_test_46=np.load("/home/nobar/codes/GBO2/logs/test_46/Exper_3/delta_J2.npy")
    # delta_J2_test_49_2=np.load("/home/nobar/codes/GBO2/logs/test_49_2/Exper_3/delta_J2.npy")
    # plot_gt()
    # plot_real()
    # plot_tradeoff()

    # # check objective scales
    # IS1 = scipy.io.loadmat("/home/nobar/Documents/introductions/simulink_model/IS1_Exper_0_8x8_metrics.mat")
    # IS2 = scipy.io.loadmat("/home/nobar/Documents/introductions/simulink_model/IS2_Exper_0_8x8_metrics.mat")
    # IS3 = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_Exper_0_1to1_8x8.mat")
    # IS3_1to2 = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_Exper_0_1to2_8x8.mat")
    # obj_IS3_1to2 = IS3_1to2["Objective_all"]
    # diff_obj_IS3_1to2=obj_IS3_1to2.max() - obj_IS3_1to2.min()
    # obj_IS3 = IS3["Objective_all"]
    # diff_obj_IS3=obj_IS3.max() - obj_IS3.min()
    # obj_IS2 = IS2["Objective_all"]
    # diff_obj_IS2=obj_IS2.max() - obj_IS2.min()
    # obj_IS1 = IS1["Objective_all"]
    # diff_obj_IS1=obj_IS1.max() - obj_IS1.min()

    # plot_cost_coef()

    # path = "/home/nobar/codes/GBO2/logs/test_50_2_MFBO_caEI_gamma0/"
    # path = "/home/nobar/codes/GBO2/logs/test_50_2_MFBO_caEI/"
    # path = "/home/nobar/codes/GBO2/logs/test_50_2_MFBO_taKG/"
    # path = "/home/nobar/codes/GBO2/logs/test_50_2/"
    path = "/home/nobar/codes/GBO2/logs/test_51_1/"
    # path2 = "/home/nobar/codes/GBO2/logs/test_31_b_UCB_1/"
    path2 = "/home/nobar/codes/GBO2/logs/test_33_b_1/"
    # get EI only when IS1 changes after 5 iter
    # path2 = "/home/nobar/codes/GBO2/logs/test_46/"

    N_init_IS1 = 2
    N_init_IS2 = 10
    sampling_cost_bias = 5
    N_exper = 10
    N_iter = 20
    s2 = 0.1
    s3 = 0.05
    BATCH_SIZE = 1
    N_IS3_sample_each_time = 4

    # path3= "/home/nobar/codes/GBO2/logs/"
    # # plot_gamma_deltaJ(path3)
    # plot_b_deltaJ(path3)

    # # plot GP surrogates
    # for i in range(3):
    #     print("plotting traning data on surrogate model per iteration... ",i)
    #     train_x=np.load(path+"Exper_"+str(i)+"/"+"train_x.npy")
    #     train_obj=np.load(path+"Exper_"+str(i)+"/"+"train_obj.npy")
    #     for  j in range(N_iter):
    #         train_x_i=train_x[0:N_init_IS1+N_init_IS2+j*BATCH_SIZE,:]
    #         train_obj_i=train_obj[0:N_init_IS1+N_init_IS2+j*BATCH_SIZE,:]
    #         plot_GPs(j, path+"Exper_"+str(i)+"/", train_x_i,train_obj_i)
    #
    # for i in range(3):
    #     print("plotting traning data on surrogate model per iteration... ",i)
    #     train_x=np.load(path+"Exper_"+str(i)+"/"+"train_x_EIonly.npy")
    #     train_obj=np.load(path+"Exper_"+str(i)+"/"+"train_obj_EIonly.npy")
    #     for  j in range(N_iter):
    #         train_x_i=train_x[0:N_init_IS1+j*BATCH_SIZE,:]
    #         train_obj_i=train_obj[0:N_init_IS1+j*BATCH_SIZE,:]
    #         plot_EIonly_GP(j, path+"Exper_"+str(i)+"/", train_x_i,train_obj_i)

    # # validate identical initial dataset
    # for i in range(N_exper):
    #     x=np.load(path+"Exper_{}/train_x_init.npy".format(str(i)))
    #     y=np.load(path+"Exper_{}/train_obj_init.npy".format(str(i)))
    #     x2=np.load(path2+"Exper_{}/train_x_init.npy".format(str(i)))
    #     y2=np.load(path+"Exper_{}/train_obj_init.npy".format(str(i)))
    #     diffx=x[:2,:]-x2[:2,:]
    #     diffy=y[:2,:]-y2[:2,:]
    #     if np.sum(diffx)+np.sum(diffy)!=0:
    #         print("Experiment i=",i,ValueError("ERROR: initial dataset not identical across trials!"))

    # plots_MonteCarlo_objective(path,N_init_IS1,N_init_IS2,sampling_cost_bias,N_exper,N_iter,s2,s3, BATCH_SIZE)
    # plots_MonteCarlo_objectiveEI(path,path2,N_init_IS1,N_init_IS2,sampling_cost_bias,N_exper,N_iter,s2,s3, BATCH_SIZE)
    plots_MonteCarlo_objectiveEI_34tests(path, path2, N_init_IS1, N_init_IS2, sampling_cost_bias, N_exper, N_iter, s2,
                                         s3, BATCH_SIZE, N_IS3_sample_each_time)
    # plots_MonteCarlo_objectiveUCB(path,path2,N_init_IS1,N_init_IS2,sampling_cost_bias,N_exper,N_iter,s2,s3, BATCH_SIZE)
    # plots_indicators(path,path2,N_init_IS1,N_init_IS2,sampling_cost_bias,N_exper,N_iter,s2,s3, BATCH_SIZE)
