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
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
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
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig
def plot_gt():
    # Load the .mat file
    mat_data = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/misc/IS2_8x8_metrics_noise_scale_01.mat')
    mat_data2 = scipy.io.loadmat('/home/nobar/codes/GBO2/logs/test_23_test/Exper_0/IS1_Exper_0_8x8_metrics.mat')
    n_grid=8
    obj_IS2_grid = np.load("/home/nobar/Documents/introductions/simulink_model/IS2_new_1_obj.npy")
    obj_IS1_grid = np.load("/home/nobar/Documents/introductions/simulink_model/IS1_new_1_obj.npy")
    def normalize_objective(obj, obj_grid):
        return (obj - obj_grid.mean()) / (obj_grid.std())

    # Extract the variables
    Kp = mat_data['Kp'].squeeze()  # Ensure it's a 1D array
    Kd = mat_data['Kd'].squeeze()  # Ensure it's a 1D array
    
    Objective_all = mat_data['Objective_all']  # Should be a 2D array
    RiseTime_all = mat_data['RiseTime_all']  # Should be a 2D array
    TransientTime_all = mat_data['TransientTime_all']  # Should be a 2D array
    SettlingTime_all = mat_data['SettlingTime_all']  # Should be a 2D array
    Overshoot_all = mat_data['Overshoot_all']  # Should be a 2D array

    RiseTime2_all = mat_data2['RiseTime_all']  # Should be a 2D array
    TransientTime2_all = mat_data2['TransientTime_all']  # Should be a 2D array
    SettlingTime2_all = mat_data2['SettlingTime_all']  # Should be a 2D array
    Overshoot2_all = mat_data2['Overshoot_all']  # Should be a 2D array


    Objective_all_error = mat_data['Objective_all']-mat_data2['Objective_all']  # Should be a 2D array
    RiseTime_all_error = mat_data['RiseTime_all']-mat_data2['RiseTime_all']  # Should be a 2D array
    TransientTime_all_error = mat_data['TransientTime_all']-mat_data2['TransientTime_all']  # Should be a 2D array
    SettlingTime_all_error = mat_data['SettlingTime_all']-mat_data2['SettlingTime_all']  # Should be a 2D array
    Overshoot_all_error = mat_data['Overshoot_all']-mat_data2['Overshoot_all']  # Should be a 2D array


    # # Create meshgrid for contour plot
    # Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='True Objective Value')
    # plt.xlabel('Kp')
    # plt.ylabel('Kd')
    # plt.title('True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics.png")
    # plt.show()

    w1=9
    w2=1.8/100
    w3=1.5
    w4=1.5
    new_obj=w1*RiseTime_all+w2*Overshoot_all+w4*TransientTime_all+w3*SettlingTime_all
    # new_obj=normalize_objective(new_obj, obj_IS2_grid)
    # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj.npy",new_obj)
    # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj_noise_scale_01.npy",new_obj)
    # Create meshgrid for contour plot
    Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, new_obj.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='True Objective Value')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/IS1_numerical.png")
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/Kp_IS1_numerical.npy",Kp_grid)
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/Kd_IS1_numerical.npy",Kp_grid)
    # np.save("/home/nobar/codes/GBO2/logs/50x50_dataset/obj_IS1_numerical.npy",new_obj.reshape(n_grid,n_grid))
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    plt.show()

    # new_obj2=w1*RiseTime2_all+w2*Overshoot2_all+w4*TransientTime2_all+w3*SettlingTime2_all
    # new_obj2=normalize_objective(new_obj2, obj_IS2_grid)
    # error_new_obj=new_obj2-new_obj
    # # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj.npy",new_obj)
    # # Create meshgrid for contour plot
    # Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp_grid, Ki_grid, error_new_obj.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='$J_{IS2}-J_{IS1}$')
    # plt.xlabel('Kp')
    # plt.ylabel('Kd')
    # plt.title('Error True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_29_baseline/ERROR_Exper_0_8x8_metrics_normalized.png")
    # # plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    # plt.show()
    # error_new_obj=new_obj2-new_obj
    # # np.save("/home/nobar/codes/GBO2/logs/IS2_new_1_obj.npy",new_obj)
    # # Create meshgrid for contour plot
    # Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    # # Plot the contour
    # plt.figure(figsize=(8, 6))
    # # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    # contour = plt.contourf(Kp_grid, Ki_grid, abs(error_new_obj).reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    # plt.colorbar(contour, label='|$J_{IS2}-J_{IS1}$|')
    # plt.xlabel('Kp')
    # plt.ylabel('Kd')
    # plt.title('Error True Objective Contour Plot')
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_29_baseline/ABS_ERROR_Exper_0_8x8_metrics_normalized.png")
    # # plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    # plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w1*RiseTime_all.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Rise Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('$w.T_{r}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/IS1_numeric_Rise.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w3*SettlingTime_all.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Settling Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('$w.T_{s}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/IS1_numeric_Settling.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w2*Overshoot_all.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Overshoot')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('$w.M$')
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/IS1_numeric_Overshoot.png")
    plt.show()
    plt.close()
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w4*TransientTime_all.reshape(n_grid,n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Transient Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('$w.T_{tr}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/IS1_numeric_Transiet.png")
    plt.show()



    # Create meshgrid for contour plot
    Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)
    n_grid=8
    # Plot the contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp_grid, Ki_grid, Objective_all_error.reshape(n_grid,n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Objective Value')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('Error Objective Contour Plot')
    plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_J_1to1_error.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Kp_grid, Ki_grid, w1*RiseTime_all_error.reshape(n_grid,n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Rise Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('error $w.T_{r}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_Tr_1to1_error.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w3*SettlingTime_all_error.reshape(n_grid,n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Settling Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('error $w.T_{s}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_Ts_1to1_error.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w2*Overshoot_all_error.reshape(n_grid,n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Overshoot')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('error $w.M$')
    plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_Ov_1to1_error.png")
    plt.show()
    plt.close()
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp_grid, Ki_grid, w4*TransientTime_all_error.reshape(n_grid,n_grid))  # Transpose to match dimensions
    plt.colorbar(contour, label='Error Weighted Transient Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('error $w.T_{tr}$')
    plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS3_Ttr_1to1_error.png")
    plt.show()



    print("")

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
    K = matern_5_2(x.reshape(100,1),y.reshape(100,1), length_scale=0.2).reshape(100, 100)

    # Plot contour map with color bar
    plt.figure(figsize=(6, 5))
    X, Y = np.meshgrid(x, y)
    contour = plt.contourf(X, Y, K, cmap="viridis", levels=20)
    plt.colorbar(contour, label="Kernel Value")
    plt.xlabel("x")
    plt.ylabel("x'")
    plt.title("Matérn 5/2 Kernel Contour Map")
    plt.savefig("/home/nobar/codes/GBO2/logs/test_3/matern52.png")
    plt.show()

    plt.figure(figsize=(6, 5))
    C1=(1-X)*(1-Y)*(1-X*Y)**0.672
    contour = plt.contourf(X, Y, C1, cmap="viridis", levels=20)
    plt.colorbar(contour, label="C1 Value")
    plt.xlabel("s")
    plt.ylabel("s'")
    plt.title("C1 Contour Map")
    plt.savefig("/home/nobar/codes/GBO2/logs/test_3/C1kernelCoefficient.png")
    plt.show()

def plots_MonteCarlo_objective(path,    N_init_IS1,N_init_IS2,    sampling_cost_bias,N_exper,N_iter, s2, s3, BATCH_SIZE):
    costs_all_list = []
    costs_all_list_EIonly = []
    train_x_list = []
    train_obj_list = []
    train_x_list_IS1 = []
    train_obj_list_IS1 = []
    train_x_list_EIonly = []
    train_obj_list_EIonly = []
    idx_IS1_all=[]
    idx_IS1_all_init=[]
    idx_IS1_all_rest=[]
    idx_IS2_all=[]
    idx_IS2_all_init=[]
    idx_IS2_all_rest=[]
    idx_IS3_all=[]
    idx_IS3_all_init=[]
    idx_IS3_all_rest=[]
    idx_ISDTs_all=[]
    idx_ISDTs_all_init=[]
    idx_ISDTs_all_rest=[]
    # train_x_IS1_init_list=[]
    # train_obj_IS1_init_list=[]
    train_obj_EIonly_corrected_all=[]
    min_obj_init_all=[]
    train_obj_list_rest_modified=[]
    costs_init=[]
    costs_init_IS1=[]
    costs_init_EIonly=[]
    for exper in range(N_exper):
        exp_path = os.path.join(path, f"Exper_{exper}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))

        idx_IS1 = np.argwhere(train_x[:, 2] == 1).squeeze()
        idx_IS1_init=idx_IS1[np.argwhere(idx_IS1<N_init_IS1+N_init_IS2)[:,0]]
        idx_IS1_rest=idx_IS1[np.argwhere(idx_IS1>N_init_IS1+N_init_IS2-1)[:,0]]
        idx_ISDTs = np.argwhere(~(train_x[:, 2] == 1)).squeeze()
        idx_ISDTs_init=idx_ISDTs[np.argwhere(idx_ISDTs<N_init_IS1+N_init_IS2)[:,0]]
        idx_ISDTs_rest=idx_ISDTs[np.argwhere(idx_ISDTs>N_init_IS1+N_init_IS2-1)[:,0]]
        train_x_IS1_init=train_x[idx_IS1_init, :]
        train_obj_IS1_init=train_obj[idx_IS1_init]

        idx_IS2 = np.argwhere(train_x[:, 2] == s2).squeeze()
        idx_IS2 = idx_IS2.reshape(idx_IS2.size, )
        idx_IS2_init=idx_IS2[np.argwhere(idx_IS2<N_init_IS1+N_init_IS2)[:,0]]
        idx_IS2_rest=idx_IS2[np.argwhere(idx_IS2>N_init_IS1+N_init_IS2-1)[:,0]]
        idx_IS3 = np.argwhere(train_x[:, 2] == s3).squeeze()
        idx_IS3=idx_IS3.reshape(idx_IS3.size, )
        idx_IS3_init=idx_IS3[np.argwhere(idx_IS3<N_init_IS1+N_init_IS2)[:,0]]
        idx_IS3_rest=idx_IS3[np.argwhere(idx_IS3>N_init_IS1+N_init_IS2-1)[:,0]]

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
        costs_all=np.load(os.path.join(exp_path, "costs_all.npy"))
        costs_all_EIonly=np.load(os.path.join(exp_path, "costs_all_EIonly.npy"))
        costs_all_EIonly_corrected=np.load(os.path.join(exp_path, "costs_all_EIonly.npy"))
        # Append to lists
        costs_all_list.append(costs_all)
        costs_init.append(np.sum(train_x[:N_init_IS1+N_init_IS2,2])+sampling_cost_bias*(N_init_IS1+N_init_IS2))
        costs_init_IS1.append(np.sum(train_x[:N_init_IS1,2])+sampling_cost_bias*(N_init_IS1))
        costs_init_EIonly.append(np.sum(train_x_EIonly[:N_init_IS1,2])+sampling_cost_bias*(N_init_IS1))

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

        A=train_obj
        A[idx_ISDTs_rest]=-np.inf
        train_obj_list_rest_modified.append(-A[N_init_IS1+N_init_IS2:])

        train_x_list_IS1.append(train_x_IS1)
        train_obj_list_IS1.append(train_obj_IS1)
        # train_x_IS1_init_list.append(train_x_IS1_init)
        # train_obj_IS1_init_list.append(train_obj_IS1_init)
        train_x_list_EIonly.append(train_x_EIonly)
        train_obj_EIonly_corrected_all.append(train_obj_EIonly_corrected)
        train_obj_list_EIonly.append(train_obj_EIonly)

    DD=np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D=np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1=np.minimum.accumulate(D, axis=0)
    J_mean=np.mean(j_min_observed_IS1,axis=1)
    J_std=np.std(j_min_observed_IS1,axis=1)

    DD=-np.stack(train_obj_list_EIonly).squeeze()[:,N_init_IS1:].T
    D=np.vstack((np.asarray(min_obj_init_all), DD))
    j_EIonly_min_observed_IS1=np.minimum.accumulate(D, axis=0)
    J_EIonly_mean=np.mean(j_EIonly_min_observed_IS1,axis=1)
    J_EIonly_std=np.std(j_EIonly_min_observed_IS1,axis=1)
    x = np.arange(0,41)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region
    plt.plot(x, J_EIonly_mean, color='b', linewidth=3, label='Mean EI only')  # Thick line for mean
    plt.fill_between(x, J_EIonly_mean - J_EIonly_std, J_EIonly_mean + J_EIonly_std, color='b', alpha=0.3, label='±1 Std EI only')  # Shaded std region
    plt.xlabel('BO Iteration')
    plt.ylabel('$J^{*}$')
    plt.title('Minimum Observed Objective on IS1 over BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    plt.savefig(path+"/J_min_obs_IS1_BOiter.png")
    plt.show()

    costs=np.hstack((np.asarray(costs_init).reshape(N_exper,1),np.stack(costs_all_list)))
    C=np.cumsum(costs, axis=1)
    C_mean=np.mean(C,axis=0)
    C_std=np.std(C,axis=0)
    costs_EIonly=np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper,1),np.ones((N_exper,N_iter))*(sampling_cost_bias+1)*BATCH_SIZE))
    C_EIonly=np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean=np.mean(C_EIonly,axis=0)
    C_EIonly_std=np.std(C_EIonly,axis=0)
    x = np.arange(0,41,BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - C_std, C_mean + C_std, color='r', alpha=0.3, label='±1 Std Cost GMFBO')  # Shaded std region
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
    plt.savefig(path+"/Cost_sampling_BOiter.png")
    plt.show()


    costs=np.hstack((np.asarray(costs_init).reshape(N_exper,1)-sampling_cost_bias*(N_init_IS1+N_init_IS2),np.stack(costs_all_list)-sampling_cost_bias*BATCH_SIZE))
    C=np.cumsum(costs, axis=1)
    C_mean=np.mean(C,axis=0)
    C_std=np.std(C,axis=0)
    costs_EIonly=np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper,1)-sampling_cost_bias*N_init_IS1,np.ones((N_exper,N_iter))*(1)*BATCH_SIZE))
    C_EIonly=np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean=np.mean(C_EIonly,axis=0)
    C_EIonly_std=np.std(C_EIonly,axis=0)
    x = np.arange(0,41,BATCH_SIZE)
    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, C_mean, color='r', marker="o", linewidth=3, label='Mean Cost GMFBO')  # Thick line for mean
    plt.fill_between(x, C_mean - C_std, C_mean + C_std, color='r', alpha=0.3, label='±1 Std Cost GMFBO')  # Shaded std region
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
    plt.savefig(path+"/Unbiased_cost_sampling_BOiter.png")
    plt.show()

    costs_IS2_all=[]
    costs_IS3_all=[]
    for i in range(N_exper):
        costs_IS2_all_=[]
        costs_IS3_all_=[]
        for j in range(N_iter):
            costs_IS2_all_.append(np.sum((idx_IS2_all_rest[i] < BATCH_SIZE * (j+1)) * (idx_IS2_all_rest[i] > BATCH_SIZE * j)) * s2)
            costs_IS3_all_.append(np.sum((idx_IS3_all_rest[i] < BATCH_SIZE * (j+1)) * (idx_IS3_all_rest[i] > BATCH_SIZE * j)) * s3)
        costs_IS2_all.append(costs_IS2_all_)
        costs_IS3_all.append(costs_IS3_all_)

    old_indices = np.linspace(0, 1, num=1+N_iter)  # Original index positions
    new_indices = np.linspace(0, 1, num=1+BATCH_SIZE*N_iter)  # New index positions

    costs=np.hstack((np.asarray(costs_init).reshape(N_exper,1)-sampling_cost_bias*(N_init_IS1+N_init_IS2),np.stack(costs_all_list)-sampling_cost_bias*BATCH_SIZE))
    C=np.cumsum(costs, axis=1)
    C_mean=np.mean(C,axis=0)
    C_std=np.std(C,axis=0)
    costs_EIonly=np.hstack((np.asarray(costs_init_EIonly).reshape(N_exper,1)-sampling_cost_bias*(N_init_IS1),np.ones((N_exper,N_iter))*(1)*BATCH_SIZE))
    C_EIonly=np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean=np.mean(C_EIonly,axis=0)
    C_EIonly_std=np.std(C_EIonly,axis=0)
    x_ = C_mean #np.arange(0,41,BATCH_SIZE)
    # linear interpolation
    x = np.interp(new_indices, old_indices, x_)
    x_EI_only_ = C_EIonly_mean #np.arange(0,41,BATCH_SIZE)
    # linear interpolation
    x_EI_only = np.interp(new_indices, old_indices, x_EI_only_)

    DD=np.stack(train_obj_list_rest_modified, axis=1).squeeze()
    D=np.vstack((np.asarray(min_obj_init_all), DD))
    j_min_observed_IS1=np.minimum.accumulate(D, axis=0)
    J_mean=np.mean(j_min_observed_IS1,axis=1)
    J_std=np.std(j_min_observed_IS1,axis=1)

    DD=-np.stack(train_obj_list_EIonly).squeeze()[:,N_init_IS1:].T
    D=np.vstack((np.asarray(min_obj_init_all), DD))
    j_EIonly_min_observed_IS1=np.minimum.accumulate(D, axis=0)
    J_EIonly_mean=np.mean(j_EIonly_min_observed_IS1,axis=1)
    J_EIonly_std=np.std(j_EIonly_min_observed_IS1,axis=1)

    plt.figure(figsize=(10, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', marker="o", linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region

    plt.plot(x_EI_only, J_EIonly_mean, color='b', marker="o", linewidth=3, label='Mean EI only')  # Thick line for mean
    plt.fill_between(x_EI_only, J_EIonly_mean - J_EIonly_std, J_EIonly_mean + J_EIonly_std, color='b', alpha=0.3, label='±1 Std EI only')  # Shaded std region
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.title('Unbiased Cumulative Sampling Cost vs BO Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])
    plt.savefig(path+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling.png")
    plt.show()

    mean_values_baseline=np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy")
    margin_of_error_baseline=np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy")
    x_baseline=np.load("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy")

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
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly, mean_values_EIonly + margin_of_error_EIonly,
                     color='b', alpha=0.3, label="95% CI BO-EI")
    plt.xlabel('Mean Unbiased Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.BATCH_SIZE)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    plt.savefig(path+"/J_min_obs_IS1_Mean_Unbiased_cost_sampling_95Conf.png")
    plt.show()
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/mean_values.npy",mean_values)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/margin_of_error.npy",margin_of_error)
    # np.save("/home/nobar/codes/GBO2/logs/test_6_11_baseline/x.npy",x)
    print("")


    costs_IS1=np.hstack((np.asarray(costs_init_IS1).reshape(N_exper,1)-sampling_cost_bias*(N_init_IS1),np.stack(costs_all_list)-np.stack(costs_IS2_all)-np.stack(costs_IS3_all)-sampling_cost_bias*BATCH_SIZE))
    C_IS1=np.cumsum(costs_IS1, axis=1)
    C_mean_IS1=np.mean(C_IS1,axis=0)
    x_IS1_ = C_mean_IS1 #np.arange(0,41,BATCH_SIZE)
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
    plt.fill_between(x_EI_only, mean_values_EIonly - margin_of_error_EIonly, mean_values_EIonly + margin_of_error_EIonly,
                     color='b', alpha=0.3, label="95% CI BO-EI")
    plt.xlabel('Mean IS1 only Sampling Cost')
    plt.ylabel('$J^{*}$')
    plt.legend()
    plt.grid(True)
    # plt.ylim(0.9, 1.4)  # Focus range
    plt.title("Mean with 95% Confidence Interval")
    plt.savefig(path+"/J_min_obs_IS1_Mean_IS1onlySamplingCost_95Conf.png")
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


def plot_GPs(iter, path, train_x_i,train_obj_i):
    # Step 3: Define fidelity levels and create a grid for plotting
    # uncomment for my idea
    fidelities = [1.0, 0.1]  # Three fidelity levels
    # fidelities = [1.0, 0.5]
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

    # Step 4: Prepare the figure with 3x2 subplots
    fig, axs = plt.subplots(len(fidelities), 2, figsize=(14, 6*fidelities.__len__()))

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
        mean= np.load(path + "/mean_{}.npy".format(str(iter)))
        std=np.load(path + "/std_{}.npy".format(str(iter)))
        # Plot the posterior mean
        contour_mean = axs[i, 0].contourf(X1.numpy(), X2.numpy(), mean.T, cmap='viridis')
        axs[i, 0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[i, 0])

        # Plot the posterior standard deviation
        contour_std = axs[i, 1].contourf(X1.numpy(), X2.numpy(), std.T, cmap='viridis')
        axs[i, 1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[i, 1])

        # scatter_train_x = axs[i, 0].scatter(train_x_i[:,0], train_x_i[:,1], c='b',linewidth=15)
        scatter_train_x = axs[i, 0].scatter(train_x_i[np.argwhere(train_x_i[:,2]==s_val),0], train_x_i[np.argwhere(train_x_i[:,2]==s_val),1], c='r',linewidth=15)

        # np.save(path + "/EIonly_X1_{}.npy".format(str(iter)), X1)
        # np.save(path + "/EIonly_X2_{}.npy".format(str(iter)), X2)
        # np.save(path + "/EIonly_mean_{}.npy".format(str(iter)), mean)
        # np.save(path + "/EIonly_std_{}.npy".format(str(iter)), std)

        # Axis labels
        for ax in axs[i]:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    plt.tight_layout()
    # plt.savefig(path + "/withTrainData_EIonly_GP_itr_{}.png".format(str(iter)))
    plt.savefig(path + "/withTrainData_GP_itr_{}.png".format(str(iter)))
    # plt.show()
    plt.close()

def plot_EIonly_GP(iter, path, train_x_i,train_obj_i):
    # Step 3: Define fidelity levels and create a grid for plotting
    # uncomment for my idea
    fidelities = [1.0]  # Three fidelity levels
    # fidelities = [1.0, 0.5]
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

    # Step 4: Prepare the figure with 3x2 subplots
    fig, axs = plt.subplots(len(fidelities), 2, figsize=(14, 6*fidelities.__len__()))

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
        mean= np.load(path + "/EIonly_mean_{}.npy".format(str(iter)))
        std=np.load(path + "/EIonly_std_{}.npy".format(str(iter)))
        # Plot the posterior mean
        contour_mean = axs[i].contourf(X1.numpy(), X2.numpy(), mean.T, cmap='viridis')
        axs[0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[0])

        # Plot the posterior standard deviation
        contour_std = axs[1].contourf(X1.numpy(), X2.numpy(), std.T, cmap='viridis')
        axs[1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[1])

        scatter_train_x = axs[0].scatter(train_x_i[:,0], train_x_i[:,1], c='b',linewidth=15)
        # scatter_train_x = axs[0].scatter(train_x_i[np.argwhere(train_x_i[:,2]==s_val),0], train_x_i[np.argwhere(train_x_i[:,2]==s_val),1], c='r',linewidth=15)

        # np.save(path + "/EIonly_X1_{}.npy".format(str(iter)), X1)
        # np.save(path + "/EIonly_X2_{}.npy".format(str(iter)), X2)
        # np.save(path + "/EIonly_mean_{}.npy".format(str(iter)), mean)
        # np.save(path + "/EIonly_std_{}.npy".format(str(iter)), std)


        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    plt.tight_layout()
    plt.savefig(path + "/withTrainData_EIonly_GP_itr_{}.png".format(str(iter)))
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


def plot_real():
    # Load the .mat file
    mat_data = scipy.io.loadmat("/home/nobar/codes/GBO2/logs/50x50_dataset/metrics_all.mat")
    Objective_all = mat_data['Objective_all']  # Should be a 2D array
    obj_IS1_grid = Objective_all.squeeze()

    def normalize_objective(obj, obj_grid):
        return (obj - obj_grid.mean()) / (obj_grid.std())

    RiseTime_all = mat_data['RiseTime_all']  # Should be a 2D array
    TransientTime_all = mat_data['TransientTime_all']  # Should be a 2D array
    SettlingTime_all = mat_data['SettlingTime_all']  # Should be a 2D array
    Overshoot_all = mat_data['Overshoot_all']  # Should be a 2D array
    n_grid=50
    w1 = 9
    w2 = 1.8 / 100
    w3 = 1.5
    w4 = 1.5
    # new_obj = w1 * RiseTime_all + w2 * Overshoot_all + w4 * TransientTime_all + w3 * SettlingTime_all
    # new_obj = normalize_objective(new_obj, obj_IS1_grid)
    # Create meshgrid for contour plot
    Kp = mat_data["Kp_all"].squeeze()  # Ensure it's a 1D array
    Kd = mat_data["Kp_all"].squeeze()  # Ensure it's a 1D array
    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, Objective_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Objective')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('True Objective Contour Plot')
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/real_objective.png")
    plt.show()


    # Plot the contour
    plt.figure(figsize=(8, 6))
    Kp_IS1_numeric=np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/Kp_IS1_numerical.npy")
    Kd_IS1_numeric=np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/Kd_IS1_numerical.npy")
    obj_IS1_numeric=np.load("/home/nobar/codes/GBO2/logs/50x50_dataset/obj_IS1_numerical.npy")
    contour = plt.contourf(Kp_IS1_numeric[0,:],Kd_IS1_numeric[0,:], obj_IS1_numeric, levels=20, cmap='gray')  # Transpose to match dimensions
    plt.colorbar(contour, label='$|J_{IS1}-J_{IS2}|$')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('Absolute Error Objective Contour Plot')
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/IS1_numerical.png")
    # plt.savefig("/home/nobar/codes/GBO2/logs/test_23_8_test/Exper_0/IS2_Exper_0_8x8_metrics_NEW.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, RiseTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Rise Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Rise Time")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/RiseTime_all.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, TransientTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Transient Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Transient Time")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/TransientTime_all.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, SettlingTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Settling Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Settling Time")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/SettlingTime_all.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, Overshoot_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Overshoot')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Overshoot")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/Overshoot_all.png")
    plt.show()



    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, w1*RiseTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Rise Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Weighted Rise Time")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/Weighted_RiseTime_all.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, w4*TransientTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Transient Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Weighted Transient Time")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/Weighted_TransientTime_all.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, w3*SettlingTime_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Settling Time')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Weighted Settling Time")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/Weighted_SettlingTime_all.png")
    plt.show()

    # Plot the contour
    plt.figure(figsize=(8, 6))
    # levs=[0.9,0.95,1,1.05,1.1,1.2,1.3,1.4,1.5,2,3]
    # contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    contour = plt.contourf(Kp, Kd, w2*Overshoot_all.reshape(n_grid, n_grid), levels=20)  # Transpose to match dimensions
    plt.colorbar(contour, label='Weighted Overshoot')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title("Weighted Overshoot")
    plt.savefig("/home/nobar/codes/GBO2/logs/50x50_dataset/Weighted_Overshoot_all.png")
    plt.show()

    print("")

if __name__ == "__main__":

    # plot_gt()
    # plot_real()

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

    path = "/home/nobar/codes/GBO2/logs/test_31_b_1/"
    path2 = "/home/nobar/codes/GBO2/logs/test_29_baseline_6/"
    N_init_IS1=2
    N_init_IS2=0
    sampling_cost_bias=5
    N_exper=10
    N_iter=40
    s2 = 0.1
    s3 = 0.05
    BATCH_SIZE=1

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


    # validate identical initial dataset
    for i in range(N_exper):
        x=np.load(path+"Exper_{}/train_x_init.npy".format(str(i)))
        y=np.load(path+"Exper_{}/train_obj_init.npy".format(str(i)))
        x2=np.load(path2+"Exper_{}/train_x_init.npy".format(str(i)))
        y2=np.load(path+"Exper_{}/train_obj_init.npy".format(str(i)))
        diffx=x[:2,:]-x2[:2,:]
        diffy=y[:2,:]-y2[:2,:]
        if np.sum(diffx)+np.sum(diffy)!=0:
            print(ValueError("ERROR: initial dataset not identical across trials!"))

    plots_MonteCarlo_objective(path,N_init_IS1,N_init_IS2,sampling_cost_bias,N_exper,N_iter,s2,s3, BATCH_SIZE)

