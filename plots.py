import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker

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


if __name__ == "__main__":
    # plot_gt()

    # train_x = np.load("/home/nobar/codes/GBO2/logs/test_3/train_x.npy")
    # train_obj = np.load("/home/nobar/codes/GBO2/logs/test_3/train_obj.npy")
    # train_x_MFBOonly = np.load("/home/nobar/codes/GBO2/logs/test_3_MFBOonly/train_x.npy")
    # train_obj_MFBOonly = np.load("/home/nobar/codes/GBO2/logs/test_3_MFBOonly/train_obj.npy")
    # D1=np.hstack((train_x, train_obj))
    # D2=np.hstack((train_x_MFBOonly, train_obj_MFBOonly))
    # print("placeholder")


    plot_kernels()
