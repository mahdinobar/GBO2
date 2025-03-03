import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
import os

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import math

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
    mat_data = scipy.io.loadmat('/home/nobar/Documents/introductions/simulink_model/ground_truth_1_20by20.mat')

    # Extract the variables
    Kp = mat_data['Kp'].squeeze()  # Ensure it's a 1D array
    Kd = mat_data['Kd'].squeeze()  # Ensure it's a 1D array
    Objective_all = mat_data['Objective_all']  # Should be a 2D array

    # Create meshgrid for contour plot
    Kp_grid, Ki_grid = np.meshgrid(Kp, Kd)

    # Plot the contour
    plt.figure(figsize=(8, 6))
    levs=[0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,2,3]
    contour = plt.contourf(Kp_grid, Ki_grid, Objective_all.reshape(20,20),levs)  # Transpose to match dimensions
    plt.colorbar(contour, label='True Objective Value')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.title('True Objective Contour Plot')
    plt.savefig("/home/nobar/codes/GBO2/logs/test_3/true_objective_contourf_grid20by20.png")
    plt.show()

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

def plots_MonteCarlo_objective(path):
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
    # train_x_IS1_init_list=[]
    # train_obj_IS1_init_list=[]
    train_obj_EIonly_corrected_all=[]
    min_obj_init_all=[]
    train_obj_list_rest_modified=[]
    costs_init=[]
    N_init=2
    sampling_cost_bias=1
    for exper in range(20):
        exp_path = os.path.join(path, f"Exper_{exper}")
        # Load files
        train_x = np.load(os.path.join(exp_path, "train_x.npy"))
        train_obj = np.load(os.path.join(exp_path, "train_obj.npy"))


        idx_IS1 = np.argwhere(train_x[:, 2] == 1).squeeze()
        idx_IS1_init=idx_IS1[np.argwhere(idx_IS1<N_init)[:,0]]
        idx_IS1_rest=idx_IS1[np.argwhere(idx_IS1>N_init-1)[:,0]]
        idx_IS2 = np.argwhere(train_x[:, 2] == 0.5).squeeze()
        idx_IS2_init=idx_IS2[np.argwhere(idx_IS2<N_init)[:,0]]
        idx_IS2_rest=idx_IS2[np.argwhere(idx_IS2>N_init-1)[:,0]]
        train_x_IS1_init=train_x[idx_IS1_init, :]
        train_obj_IS1_init=train_obj[idx_IS1_init]

        # np.save(os.path.join(exp_path, "train_x_IS1_init.npy"),train_x_IS1_init)
        # np.save(os.path.join(exp_path, "train_obj_IS1_init.npy"),train_obj_IS1_init)

        min_obj_init_all.append(np.min(-train_obj_IS1_init))

        train_x_IS1 = train_x[idx_IS1]
        train_obj_IS1 = train_obj[idx_IS1]
        train_x_EIonly = np.load(os.path.join(exp_path, "train_x_EIonly_corrected.npy"))
        train_x_EIonly_corrected = np.load(os.path.join(exp_path, "train_x_EIonly_corrected.npy"))
        train_obj_EIonly_corrected = np.load(os.path.join(exp_path, "train_obj_EIonly_corrected.npy"))

        train_obj_EIonly = np.load(os.path.join(exp_path, "train_obj_EIonly_corrected.npy"))
        costs_all=np.load(os.path.join(exp_path, "costs_all.npy"))
        costs_all_EIonly=np.load(os.path.join(exp_path, "costs_all_EIonly_corrected.npy"))
        costs_all_EIonly_corrected=np.load(os.path.join(exp_path, "costs_all_EIonly_corrected.npy"))
        # Append to lists
        costs_all_list.append(costs_all)
        costs_init.append(np.sum(train_x[:N_init,2])+sampling_cost_bias*N_init)

        idx_IS1_all.append(idx_IS1)
        idx_IS1_all_init.append(idx_IS1_init)
        idx_IS1_all_rest.append(idx_IS1_rest)

        idx_IS2_all.append(idx_IS2)
        idx_IS2_all_init.append(idx_IS2_init)
        idx_IS2_all_rest.append(idx_IS2_rest)

        costs_all_list_EIonly.append(costs_all_EIonly)
        train_x_list.append(train_x)
        train_obj_list.append(train_obj)

        A=train_obj
        A[idx_IS2_rest]=-np.inf
        train_obj_list_rest_modified.append(-A[N_init:])

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

    DD=-np.stack(train_obj_list_EIonly).squeeze()[:,N_init:].T
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
    plt.savefig("/home/nobar/codes/GBO2/logs/test_5/J_min_obs_IS1_BOiter.png")
    plt.show()

    costs=np.hstack((np.asarray(costs_init).reshape(20,1),np.stack(costs_all_list)))
    C=np.cumsum(costs, axis=1)
    C_mean=np.mean(C,axis=0)
    C_std=np.std(C,axis=0)
    costs_EIonly=np.hstack((np.asarray(costs_init).reshape(20,1),np.ones((20,10))*(sampling_cost_bias+1)*4))
    C_EIonly=np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean=np.mean(C_EIonly,axis=0)
    C_EIonly_std=np.std(C_EIonly,axis=0)
    x = np.arange(0,41,4)
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
    plt.savefig("/home/nobar/codes/GBO2/logs/test_5/Cost_sampling_BOiter.png")
    plt.show()


    costs=np.hstack((np.asarray(costs_init).reshape(20,1)-sampling_cost_bias*N_init,np.stack(costs_all_list)-sampling_cost_bias*4))
    C=np.cumsum(costs, axis=1)
    C_mean=np.mean(C,axis=0)
    C_std=np.std(C,axis=0)
    costs_EIonly=np.hstack((np.asarray(costs_init).reshape(20,1)-sampling_cost_bias*N_init,np.ones((20,10))*(1)*4))
    C_EIonly=np.cumsum(costs_EIonly, axis=1)
    C_EIonly_mean=np.mean(C_EIonly,axis=0)
    C_EIonly_std=np.std(C_EIonly,axis=0)
    x = np.arange(0,41,4)
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
    plt.savefig("/home/nobar/codes/GBO2/logs/test_5/Unbiased_cost_sampling_BOiter.png")
    plt.show()





    train_obj_all_EIonly = -np.stack(train_obj_list_EIonly, axis=1).squeeze()
    train_obj_all_sorted_EIonly = np.minimum.accumulate(train_obj_all_EIonly, axis=0)
    J_EIonly=np.vstack((np.min(train_obj_all_sorted_EIonly[0:17], axis=0),train_obj_all_sorted_EIonly[17:]))

    for idx, arr in enumerate(train_obj_list):
        start_idx=np.argwhere(idx_IS1_all[idx] > 15)[0][0]
        # arr=arr[start_idx:]
        x = np.arange(1, len(arr)-start_idx+1)  # x-values: 1 to n, where n is the length of the array

        y = -1*arr.flatten()  # Flatten the array to make it 1D for plotting
        y = np.minimum.accumulate(y, axis=0)
        plt.plot(x, y[start_idx-1:-1], color='r')

        arr2= train_obj_EIonly_corrected_all[idx]
        y2 = -1*arr2.flatten()  # Flatten the array to make it 1D for plotting
        y2 = np.minimum.accumulate(y2, axis=0)
        plt.plot(x, y2[start_idx-1:start_idx+x.__len__()-1], color='b')

        # y_EIonly=J_EIonly[0:len(arr)-start_idx, idx]
        # plt.plot(x, y_EIonly, color='b')

    # Labels and legend
    plt.xlabel('Iteration')
    plt.ylabel('$J^{*}$')
    plt.title('Min Observed J(x,s=1) vs Iteration on IS1')
    # plt.legend()
    plt.grid(True)
    # Show plot
    plt.savefig("/home/nobar/codes/GBO2/logs/test_5/JIS1_vs_IterIS1.png")
    plt.show()


    # Stack along a new dimension (first dimension)
    # train_x_all = np.stack(train_x_list, axis=1)
    # ix_IS2=np.argwhere(train_x_all[:, :, 2] == 0.5)
    train_obj_all = -np.stack(train_obj_list, axis=1).squeeze()
    # train_obj_all[ix_IS2[:, 0], ix_IS2[:, 1]] = np.nan
    costs_all = np.stack(costs_all_list, axis=1)
    cumul_costs_all = np.cumsum(costs_all, axis=0)
    cumul_costs_all_EIonly = np.cumsum(costs_all_EIonly, axis=0)
    cumul_costs_all_mean = np.mean(cumul_costs_all,axis=1)
    cumul_costs_all_std = np.std(cumul_costs_all,axis=1)
    cumul_costs_all_EIonly_mean = np.mean(cumul_costs_all_EIonly,axis=1)
    cumul_costs_all_EIonly_std = np.std(cumul_costs_all_EIonly,axis=1)
    costs_all_EIonly = np.stack(costs_all_list_EIonly, axis=1)
    train_x_all_EIonly = np.stack(train_x_list_EIonly, axis=1)
    train_obj_all_EIonly = -np.stack(train_obj_list_EIonly, axis=1).squeeze()

    train_obj_all_sorted = np.minimum.accumulate(train_obj_all, axis=0)
    J=np.vstack((np.min(train_obj_all_sorted[0:17], axis=0),train_obj_all_sorted[17:]))
    J_mean=np.mean(J,axis=1)
    J_std=np.std(J,axis=1)
    train_obj_all_sorted_EIonly = np.minimum.accumulate(train_obj_all_EIonly, axis=0)
    J_EIonly=np.vstack((np.min(train_obj_all_sorted_EIonly[0:17], axis=0),train_obj_all_sorted_EIonly[17:]))
    J_EIonly_mean=np.mean(J_EIonly,axis=1)
    J_EIonly_std=np.std(J_EIonly,axis=1)

    N_iter=np.shape(J)[0]
    N_batch=4

    # Plot
    x = np.arange(1, N_iter+1)
    plt.figure(figsize=(8, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, J_mean, color='r', linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, J_mean - J_std, J_mean + J_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region
    plt.plot(x, J_EIonly_mean, color='b', linewidth=3, label='Mean EIonly')  # Thick line for mean
    plt.fill_between(x, J_EIonly_mean - J_EIonly_std, J_EIonly_mean + J_EIonly_std, color='b', alpha=0.3, label='±1 Std EIonly')  # Shaded std region
    plt.xlabel('Iteration')
    plt.ylabel('$J^{*}$')
    plt.title('Minimum Observed Objective over Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])

    plt.savefig("/home/nobar/codes/GBO2/logs/test_5/J_vs_Iter.png")
    plt.show()


    # Plot
    x = np.arange(1, N_iter/N_batch+1)
    plt.figure(figsize=(8, 5))
    # plot_colortable(mcolors.CSS4_COLORS)
    # mcolors.CSS4_COLORS['blueviolet']
    plt.plot(x, cumul_costs_all_mean, color='r', linewidth=3, label='Mean GMFBO')  # Thick line for mean
    plt.fill_between(x, cumul_costs_all_mean - cumul_costs_all_std, cumul_costs_all_mean + cumul_costs_all_std, color='r', alpha=0.3, label='±1 Std GMFBO')  # Shaded std region
    plt.plot(x, cumul_costs_all_EIonly_mean, color='b', linewidth=3, label='Mean EIonly')  # Thick line for mean
    plt.fill_between(x,cumul_costs_all_EIonly_mean - cumul_costs_all_EIonly_std, cumul_costs_all_EIonly_mean + cumul_costs_all_EIonly_std, color='b', alpha=0.3, label='±1 Std EIonly')  # Shaded std region
    plt.xlabel('Iteration')
    plt.ylabel('cost')
    plt.title('Cumulative Sampling Cost over Iterations')
    plt.legend()
    plt.grid(True)
    # plt.yscale('log')
    # plt.ylim(0.9, 1.05)  # Focus range
    # plt.yticks([0.9, 0.95, 1.0, 1.05])

    plt.savefig("/home/nobar/codes/GBO2/logs/test_5/SamplingCost_vs_Iter.png")
    plt.show()

    print("")

if __name__ == "__main__":
    # plot_gt()

    # train_x = np.load("/home/nobar/codes/GBO2/logs/test_3/train_x.npy")
    # train_obj = np.load("/home/nobar/codes/GBO2/logs/test_3/train_obj.npy")
    # train_x_MFBOonly = np.load("/home/nobar/codes/GBO2/logs/test_3_MFBOonly/train_x.npy")
    # train_obj_MFBOonly = np.load("/home/nobar/codes/GBO2/logs/test_3_MFBOonly/train_obj.npy")
    # D1=np.hstack((train_x, train_obj))
    # D2=np.hstack((train_x_MFBOonly, train_obj_MFBOonly))
    # print("placeholder")


    # plot_kernels()

    path = "/home/nobar/codes/GBO2/logs/test_5/"
    plots_MonteCarlo_objective(path)
