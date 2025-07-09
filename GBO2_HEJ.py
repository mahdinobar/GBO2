"""
Author: Mahdi Nobar, mnobar@ethz.ch
All rights reserved.
"""

import os
import torch
import matplotlib.pyplot as plt

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
print("SMOKE_TEST is ", SMOKE_TEST)
"""Purpose of SMOKE_TEST

    Quick Validation: When SMOKE_TEST is set to True, it reduces the computational workload by:
        Using fewer iterations or data points.
        Running the code with smaller configurations.
        Reducing fidelity levels or simplifying the problem.

This helps to:

    Quickly validate whether the script runs without errors.
    Test the setup, model, and environment in a minimal setting."""
from botorch.test_functions.multi_fidelity import AugmentedHartmann, HEJ
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement, AnalyticAcquisitionFunction, ExpectedImprovement, \
    AcquisitionFunction
from botorch.utils.sampling import draw_sobol_samples  # Quasi-random sampling
from botorch.utils import t_batch_mode_transform


def set_seed(seed: int):
    """Set seed for reproducibility in Python, NumPy, and PyTorch."""
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA operations, if using GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Ensure deterministic behavior in PyTorch (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the seed for NumPy
    np.random.seed(seed)


def generate_initial_data(n_IS1, n_IS2):
    n = n_IS1 + n_IS2
    # # generate torch.rand based training data
    train_x = torch.hstack(
        ((bounds[1, 0].item() - bounds[0, 0].item()) * torch.rand(n, 1, **tkwargs) + bounds[0, 0].item(),
         (bounds[1, 1].item() - bounds[0, 1].item()) * torch.rand(n, 1, **tkwargs) + bounds[0, 1].item()))
    # # generate with sobol latin hypercube initial gains
    # train_x = draw_sobol_samples(bounds=bounds[:,:2], n=n, q=1, seed=seed).squeeze(1)
    # # TODO: uncomment for my idea
    # train_f = fidelities[torch.randint(1,3, (n, 1))]
    # # TODO: uncomment for deterministic initial dataset of just IS1
    # train_f = fidelities[torch.randint(2,3, (n, 1))]
    # # TODO: uncomment for n_IS1 and n_IS2 data
    train_f = fidelities[(torch.cat([torch.ones((n_IS1, 1)) * 1, torch.ones((n_IS2, 1)) * 0])).to(torch.int)] #Attention correct wrt fidelity dims
    # train_f = fidelities[torch.randint(2, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj


def initialize_model(train_x, train_obj,l_g_0):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension index 2 corrosponding to the fidelity random variable in state space, as in [2]
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[2],l_g_0=l_g_0,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# Normalize function
def normalize(X, lower, upper):
    return (X - lower) / (upper - lower)


def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


def get_mfkg(model):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[1],  # here we fix fidelity to 1.0 (highest fidelity) to compute the current value
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


class ExpectedImprovementWithCost(AcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a decay
    factor that reduces or increases the emphasis of the cost model c(x).
    """

    def __init__(self, model, cost_model, best_f_s1, best_f_s2, alpha=1):
        super().__init__(model=model)
        self.model = model
        self.cost_model = cost_model
        # self.ei_s1 = qExpectedImprovement(model=model, best_f=best_f_s1)
        # self.ei_s2 = qExpectedImprovement(model=model, best_f=best_f_s2)
        self.best_f_s1 = best_f_s1
        self.best_f_s2 = best_f_s2
        # self.best_f_s3 = best_f_s3
        self.alpha = alpha
        self.X_pending = None

    @t_batch_mode_transform()
    def forward(self, X):
        # for i in range(X[:, :, -1].__len__()):
        fidelities = X[:, :, -1].squeeze(-1)
        # best_f_s = torch.where(fidelities == 1, self.best_f_s1, torch.where(fidelities == 0.1, self.best_f_s2, self.best_f_s3))
        best_f_s = torch.where(fidelities == 1, self.best_f_s1, self.best_f_s2)
        self.ei = qExpectedImprovement(model=model, best_f=best_f_s)
        return self.ei(X) / torch.pow(self.cost_model(X)[:, 0], self.alpha).squeeze()


def get_cost_aware_ei(model, cost_model, best_f_s1, best_f_s2, alpha):
    eipu = ExpectedImprovementWithCost(
        model=model,
        cost_model=cost_model,
        best_f_s1=best_f_s1,
        best_f_s2=best_f_s2,
        alpha=alpha,
    )
    return eipu


def optimize_caEI_and_get_observation(caEI,change_IS1):
    candidates, caEI_value = optimize_acqf_mixed(acq_function=caEI, bounds=bounds, fixed_features_list=[{2: 0.1}, {2: 1.0}],
                                        q=BATCH_SIZE,
                                        num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
                                        options={"batch_limit": 4, "maxiter": 50}, )
    # candidates, _ = optimize_acqf_mixed(acq_function=caEI, bounds=bounds, fixed_features_list=[{2: 0.1}, {2: 1.0}],
    #                                     q=BATCH_SIZE,
    #                                     num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
    #                                     options={"batch_limit": 4, "maxiter": 50}, )
    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()

    if change_IS1==True and new_x[:, 2]==1.0:
        print("IS1 changed!")
        new_x[:, 2]=111
        new_obj = problem(new_x).unsqueeze(-1)
        new_x[:, 2] = 1.0
    else:
        new_obj = problem(new_x).unsqueeze(-1)

    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost, caEI_value


def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    # uncomment for my idea
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0.1}, {2: 1.0}],
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 4, "maxiter": 50},
    )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


def plot_GP(model, iter, path, train_x):
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

        # Step 5: Evaluate the posterior mean and standard deviation
        with torch.no_grad():
            posterior = model.posterior(X_plot)
            mean = posterior.mean.reshape(50, 50).numpy()
            std = posterior.variance.sqrt().reshape(50, 50).numpy()

        # Plot the posterior mean
        contour_mean = axs[i, 0].contourf(X1.numpy(), X2.numpy(), mean, cmap='viridis')
        axs[i, 0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[i, 0])

        # Plot the posterior standard deviation
        contour_std = axs[i, 1].contourf(X1.numpy(), X2.numpy(), std, cmap='viridis')
        axs[i, 1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[i, 1])

        scatter_train_x = axs[i, 0].scatter(train_x[np.argwhere(train_x[:, 2] == s_val), 0],
                                            train_x[np.argwhere(train_x[:, 2] == s_val), 1], c='r', linewidth=15)

        np.save(path + "/X1_{}.npy".format(str(iter)), X1)
        np.save(path + "/X2_{}.npy".format(str(iter)), X2)
        np.save(path + "/mean_{}.npy".format(str(iter)), mean)
        np.save(path + "/std_{}.npy".format(str(iter)), std)

        # Axis labels
        for ax in axs[i]:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    plt.tight_layout()
    # plt.savefig(path+"/GP_itr_{}.pdf".format(str(iter)))  # Save as PDF
    plt.savefig(path + "/GP_itr_{}.png".format(str(iter)))
    # plt.show()
    plt.close()


def plot_EIonly_GP(model, iter, path, train_x):
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

        # Step 5: Evaluate the posterior mean and standard deviation
        with torch.no_grad():
            posterior = model.posterior(X_plot)
            mean = posterior.mean.reshape(50, 50).numpy()
            std = posterior.variance.sqrt().reshape(50, 50).numpy()

        # Plot the posterior mean
        contour_mean = axs[0].contourf(X1.numpy(), X2.numpy(), mean, cmap='viridis')
        axs[0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[0])

        # Plot the posterior standard deviation
        contour_std = axs[1].contourf(X1.numpy(), X2.numpy(), std, cmap='viridis')
        axs[1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[1])

        scatter_train_x = axs[0].scatter(train_x[:, 0], train_x[:, 1], c='b', linewidth=15)

        np.save(path + "/EIonly_X1_{}.npy".format(str(iter)), X1)
        np.save(path + "/EIonly_X2_{}.npy".format(str(iter)), X2)
        np.save(path + "/EIonly_mean_{}.npy".format(str(iter)), mean)
        np.save(path + "/EIonly_std_{}.npy".format(str(iter)), std)

        # Axis labels
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    plt.tight_layout()
    plt.savefig(path + "/EIonly_GP_itr_{}.png".format(str(iter)))
    # plt.show()
    plt.close()

def plot_UCBonly_GP(model, iter, path, train_x):
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

        # Step 5: Evaluate the posterior mean and standard deviation
        with torch.no_grad():
            posterior = model.posterior(X_plot)
            mean = posterior.mean.reshape(50, 50).numpy()
            std = posterior.variance.sqrt().reshape(50, 50).numpy()

        # Plot the posterior mean
        contour_mean = axs[0].contourf(X1.numpy(), X2.numpy(), mean, cmap='viridis')
        axs[0].set_title(f"Posterior Mean (s={s_val})")
        fig.colorbar(contour_mean, ax=axs[0])

        # Plot the posterior standard deviation
        contour_std = axs[1].contourf(X1.numpy(), X2.numpy(), std, cmap='viridis')
        axs[1].set_title(f"Posterior Standard Deviation (s={s_val})")
        fig.colorbar(contour_std, ax=axs[1])

        scatter_train_x = axs[0].scatter(train_x[:, 0], train_x[:, 1], c='b', linewidth=15)

        np.save(path + "/UCBonly_X1_{}.npy".format(str(iter)), X1)
        np.save(path + "/UCBonly_X2_{}.npy".format(str(iter)), X2)
        np.save(path + "/UCBonly_mean_{}.npy".format(str(iter)), mean)
        np.save(path + "/UCBonly_std_{}.npy".format(str(iter)), std)

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    plt.tight_layout()
    plt.savefig(path + "/UCBonly_GP_itr_{}.png".format(str(iter)))
    # plt.show()
    plt.close()


def get_recommendation(model, lower, upper):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[1],
        # here we fix fidelity to 1.0 (highest fidelity) to have the recommendation based on only target fidelity
    )
    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )
    final_rec = rec_acqf._construct_X_full(final_rec)
    objective_value = problem(final_rec)  # here final_rec is the normalized states

    def denormalize(x, lower, upper):
        return x * (upper - lower) + lower

    print(
        f"Final posterior optimized recommended point:\n{final_rec}\n\nrecommended point (unnormalized):\n{denormalize(final_rec, lower, upper)}\n\nobjective value:\n{objective_value}")
    return final_rec, objective_value


def get_recommendation_max_observed(train_x, train_obj, lower, upper):
    idx_s1 = np.argwhere(train_x[:, 2] == 1).squeeze()
    idx_ = np.argmax(np.asarray(train_obj[idx_s1].squeeze()))
    idx_max = idx_s1[idx_]
    final_rec = train_x[idx_max, :]
    objective_value = train_obj[idx_max]

    def denormalize(x, lower, upper):
        return x * (upper - lower) + lower

    print(
        f"Max observed recommended point:\n{final_rec}\n\nrecommended point (unnormalized):\n{denormalize(final_rec, lower, upper)}\n\nobjective value:\n{objective_value}")
    return final_rec, objective_value


def get_ei(model, best_f):
    return FixedFeatureAcquisitionFunction(
        acq_function=qExpectedImprovement(model=model, best_f=best_f),
        d=3,
        columns=[2],
        values=[1],
    )

def get_ucb(model, beta: float = 0.1):
    return FixedFeatureAcquisitionFunction(
        acq_function=qUpperConfidenceBound(model=model, beta=beta),
        d=3,                      # total dimension (2 params + 1 fidelity)
        columns=[2],              # fix the fidelity dimension
        values=[1.0],             # use full fidelity for UCB acquisition
    )

def optimize_ei_and_get_observation(ei_acqf, change_IS1=False):
    """Optimizes EI and returns a new candidate, observation, and cost."""

    candidates, _ = optimize_acqf(
        acq_function=ei_acqf,
        bounds=bounds[:, :-1],
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # add the fidelity parameter
    candidates = ei_acqf._construct_X_full(candidates)

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()

    if change_IS1==True and new_x[:, 2]==1.0:
        print("IS1 changed!")
        new_x[:, 2]=111
        new_obj = problem(new_x).unsqueeze(-1)
        new_x[:, 2] = 1.0
    else:
        new_obj = problem(new_x).unsqueeze(-1)

    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


# Set a global seed using torch.rand
seed = 10
# reset seed(here is where seed is reset to count 0)
np.random.seed(seed)
set_seed(seed)

problem = HEJ(negate=True).to(
    **tkwargs)  # Setting negate=True typically multiplies the objective values by -1, transforming a minimization objective (i.e., minimizing f(x)) into a maximization objective (i.e., maximizing −f(x)).

N_exper = 10
NUM_RESTARTS = 4 if not SMOKE_TEST else 2
RAW_SAMPLES = 64 if not SMOKE_TEST else 4
BATCH_SIZE = 1
N_init_IS1 = 2 if not SMOKE_TEST else 2
N_init_IS2 = 10 if not SMOKE_TEST else 2
N_ITER = 20 if not SMOKE_TEST else 1
N_change=21 # iteration that IS1 dynamics changes; disable by N_change>=N_ITER
l_g_0=0.4 # GP kernel gamma_0 lenghscale
N_IS3_sample_each_time=1 # number of repeat IS3 samples in vicinity of IS1 measurement each time

# # generate seed for sobol initial dataset
# sobol_seeds=torch.randint(1,10000,(N_exper,))

delta_J2_IS1init_all = np.zeros((N_exper,N_init_IS1))

for exper in range(N_exper):
    print("**********Experiment {}**********".format(exper))
    # /home/nobar/codes/GBO2
    # /cluster/home/mnobar/code/GBO2
    path = "/cluster/home/mnobar/code/GBO2/logs/test_50_1/Exper_{}".format(str(exper))
    # Check i<f the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # reset mismatch dataset per each experiment
    problem.X_GP_train = None
    problem.y_GP_train = None
    problem.t_GP_train = None

    # uncomment for my idea
    # fidelities = torch.tensor([0.7, 0.1, 1.0], **tkwargs)
    fidelities = torch.tensor([0.1, 1.0], **tkwargs)

    # Define the bounds
    original_bounds = torch.tensor([[30, 2, 0.0], [200, 10, 1.0]], **tkwargs)
    lower, upper = original_bounds[0], original_bounds[1]
    # Example input data
    X_original = torch.stack([lower, upper]).to(**tkwargs)
    # Normalize inputs to [0, 1]
    bounds = normalize(X_original, lower, upper)

    target_fidelities = {2: 1.0}

    cost_model = AffineFidelityCostModel(fidelity_weights={2: 1.0}, fixed_cost=5, fixed_cost_2=0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    torch.set_printoptions(precision=3, sci_mode=False)

    # train_x_init, train_obj_init = generate_initial_data(n_IS1=N_init_IS1,n_IS2=N_init_IS2, seed=int(sobol_seeds[exper]))
    #
    # train_x_init, train_obj_init = generate_initial_data(n_IS1=N_init_IS1, n_IS2=N_init_IS2)
    # np.save(path + "/train_obj_init.npy", train_obj_init)
    # np.save(path + "/train_x_init.npy", train_x_init)

    path_data_init = "/cluster/home/mnobar/code/GBO2/logs/test_33_b_1/Exper_{}".format(str(exper))
    train_x_init = torch.as_tensor(np.load(path_data_init+"/train_x_init.npy"))
    # train_obj_init =  problem(train_x_init).unsqueeze(-1)

    # train_x = train_x_init
    # train_obj = train_obj_init


    train_x=None
    train_obj=None
    i_IS1s=None
    delta_J=None
    JIS1s_for_delta_J=None
    delta_J2=None
    JIS2s_for_delta_J2=None
    caEI_values=None
    for i in range(train_x_init.__len__()):
        x = train_x_init[i, :].clone().reshape(1, 3)
        obj_x = problem(x).clone().unsqueeze(-1)
        if train_x is None:
            train_x=x
            train_obj=obj_x
        else:
            train_x = torch.cat([train_x, x])
            train_obj = torch.cat([train_obj, obj_x])
        if train_x_init[i,2]==1:
            if i_IS1s is None:
                i_IS1s = torch.tensor([i])
                JIS1s_for_delta_J = obj_x
                caEI_values = torch.tensor([0])
                delta_J = torch.tensor([[0]])

                x_ = x.clone().reshape(1, 3)
                x_[:, 2] = 0.1
                obj_x_IS2_ = problem(x_).clone().unsqueeze(-1)
                delta_J2 = obj_x_IS2_ - obj_x
                JIS2s_for_delta_J2 = obj_x_IS2_
            else:
                i_IS1s = torch.cat([i_IS1s, torch.tensor([i])])
                delta_J = torch.cat([delta_J, torch.tensor([[0]])])
                JIS1s_for_delta_J = torch.cat([JIS1s_for_delta_J, obj_x])
                caEI_values = torch.cat([caEI_values, torch.tensor([0])])
                x_ = x.clone().reshape(1, 3)
                x_[:, 2] = 0.1
                obj_x_IS2_ = problem(x_).clone().unsqueeze(-1)
                delta_J2 = torch.cat([delta_J2, obj_x_IS2_ - obj_x])
                JIS2s_for_delta_J2 = torch.cat([JIS2s_for_delta_J2, obj_x_IS2_])

        else:
            x_ = x.clone().reshape(1,3)
            x_[:, 2] = 1.0
            obj_x_IS1_ = problem(x_).clone().unsqueeze(-1)
            delta_J = torch.cat([delta_J, obj_x-obj_x_IS1_])
            JIS1s_for_delta_J = torch.cat([JIS1s_for_delta_J, obj_x_IS1_])
            caEI_values = torch.cat([caEI_values, torch.tensor([0])])

            delta_J2 = torch.cat([delta_J2, torch.tensor([[0]])])
            JIS2s_for_delta_J2 = torch.cat([JIS2s_for_delta_J2, obj_x])

    # delta_J2_IS1init_all[exper,:]=delta_J2.squeeze()
    # (my idea) add IS3 estimations to GP dataset + add IS3 data after IS1 and IS2 initial data
    # (my idea) add IS3 estimations to GP dataset
    # Now add IS3 data after initial dataset
    for i in i_IS1s:
        IS3_obj_new_x_ = None
        while IS3_obj_new_x_ is None:
            IS3_new_x = train_x_init[i,:].clone().reshape(1,3)
            # gains_vicinity_noise = torch.normal(mean=0.0, std=0.005, size=(1, 2))
            gains_vicinity_noise = np.array([torch.normal(mean=0.0, std=0.005, size=(1, 1)),
                                             torch.normal(mean=0.0, std=0.005, size=(1, 1))]).reshape(1, 2)
            IS3_new_x[:, :2] += gains_vicinity_noise
            # If value > 1, wrap it to 1 - (value - 1) = 2 - value
            IS3_new_x[:, :2] = torch.where(IS3_new_x[:, :2] > 1, 2 - IS3_new_x[:, :2], IS3_new_x[:, :2])
            # If value < 0, multiply by -1
            IS3_new_x[:, :2] = torch.where(IS3_new_x[:, :2] < 0, -IS3_new_x[:, :2], IS3_new_x[:, :2])
            IS3_new_x[:, :2] = torch.clamp(IS3_new_x[:, :2], 0.0, 1.0)
            IS3_new_x[:, 2] = 0.7
            IS3_obj_new_x_=problem(IS3_new_x)
        IS3_obj_new_x = IS3_obj_new_x_.clone().unsqueeze(-1)
        IS3_new_x[:, 2] = 0.1
        train_x = torch.cat([train_x, IS3_new_x])
        train_obj = torch.cat([train_obj, IS3_obj_new_x])
        # TODO
        delta_J2 = torch.cat([delta_J2, torch.tensor([[0]])])
        JIS2s_for_delta_J2 = torch.cat([JIS2s_for_delta_J2, IS3_obj_new_x])

        x_ = IS3_new_x.clone().reshape(1, 3)
        x_[:, 2] = 1.0
        obj_ = problem(x_).clone().unsqueeze(-1)
        delta_J = torch.cat([delta_J, IS3_obj_new_x - obj_])
        JIS1s_for_delta_J = torch.cat([JIS1s_for_delta_J, obj_])
        caEI_values = torch.cat([caEI_values, torch.tensor([0])])

    cumulative_cost = 0.0
    costs_all = np.zeros(N_ITER)
    for i in range(N_ITER):
        print("batch iteration=", i)
        cost_model.b_iter = i + 1  # batch iteration for adaptive cost
        mll, model = initialize_model(train_x, train_obj, l_g_0)
        # train the GP model
        fit_gpytorch_mll(mll)
        plot_GP(model, i, path, train_x)
        # mfkg_acqf = get_mfkg(model)
        # new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
        best_f_s1 = train_obj[np.argwhere(train_x[:, 2] == 1)].squeeze().max()
        # # Attention set initial best_f when no data in IS is still available
        # if sum(train_x[:, 2] == 0.1) == 0:
        #     best_f_s2 = torch.tensor([0], dtype=torch.float64)
        #     best_f_s3 = torch.tensor([0], dtype=torch.float64)
        # else:
        #     best_f_s2 = train_obj[np.argwhere(train_x[:, 2] == 0.1)].squeeze().max()
        #     # Attention set initial best_f when no data in IS is still available
        #     if sum(train_x[:, 2] == 0.7) == 0:
        #         best_f_s3 = torch.tensor([best_f_s2], dtype=torch.float64)
        #     else:
        #         best_f_s3 = train_obj[np.argwhere(train_x[:, 2] == 0.7)].squeeze().max()
        #
        # caEI = get_cost_aware_ei(model, cost_model,
        #                         best_f_s1=best_f_s1,
        #                         best_f_s2=best_f_s2,
        #                         best_f_s3=best_f_s3,
        #                         alpha=1)

        if sum(train_x[:, 2] == 0.1) == 0:
            best_f_s2 = torch.tensor([0], dtype=torch.float64)
        else:
            best_f_s2 = train_obj[np.argwhere(train_x[:, 2] == 0.1)].squeeze().max()

        caEI = get_cost_aware_ei(model, cost_model,
                                best_f_s1=best_f_s1,
                                best_f_s2=best_f_s2,
                                alpha=1)

        if np.sum(np.asarray(train_x[:,2])==1)>N_init_IS1+N_change:
            change_IS1=True
            l_g_0 = 0.3
            cost_model.fixed_cost_2=7.75
        else:
            change_IS1=False

        new_x, new_obj, cost, caEI_value = optimize_caEI_and_get_observation(caEI,change_IS1)

        # fixed_features_list = [
        #     {2: 1.0},  # Fix fidelity s = 1.0 (real)
        #     {2: 2.0},  # Fix fidelity s = 2.0 (simulation)
        # ]
        # a, b = optimize_acqf_mixed(acq_function=caEI, bounds=bounds, fixed_features_list=fixed_features_list,
        #                            q=BATCH_SIZE,
        #                            num_restarts=10, raw_samples=512, options={"batch_limit": 5, "maxiter": 200}, )

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])




        if new_x[:,-1]==1:
            delta_J = torch.cat([delta_J, torch.tensor([[0]])])
            JIS1s_for_delta_J = torch.cat([JIS1s_for_delta_J, new_obj])
            caEI_values = torch.cat([caEI_values, caEI_value.reshape(1)])
            x_ = new_x.clone().reshape(1, 3)
            x_[:, 2] = 0.1
            obj_x_IS2_ = problem(x_).clone().unsqueeze(-1)
            delta_J2 = torch.cat([delta_J2, obj_x_IS2_ - new_obj])
            JIS2s_for_delta_J2 = torch.cat([JIS2s_for_delta_J2, obj_x_IS2_])


            for i in range(N_IS3_sample_each_time):
                IS3_obj_new_x_=None
                while IS3_obj_new_x_==None:
                    # # (my idea) add IS3 estimations to GP dataset
                    IS3_new_x=new_x.clone()
                    # gains_vicinity_noise = torch.normal(mean=0.0, std=0.005, size=(1, 1))
                    gains_vicinity_noise = np.array([torch.normal(mean=0.0, std=0.005, size=(1, 1)),torch.normal(mean=0.0, std=0.005, size=(1, 1))]).reshape(1,2)
                    IS3_new_x[:, :2] += gains_vicinity_noise
                    # If value > 1, wrap it to 1 - (value - 1) = 2 - value
                    IS3_new_x[:, :2] = torch.where(IS3_new_x[:, :2] > 1, 2 - IS3_new_x[:, :2], IS3_new_x[:, :2])
                    # If value < 0, multiply by -1
                    IS3_new_x[:, :2] = torch.where(IS3_new_x[:, :2] < 0, -IS3_new_x[:, :2], IS3_new_x[:, :2])
                    IS3_new_x[:, :2] = torch.clamp(IS3_new_x[:, :2], 0.0, 1.0)
                    IS3_new_x[:,2]=0.7
                    IS3_obj_new_x_=problem(IS3_new_x)
                IS3_obj_new_x=problem(IS3_new_x).unsqueeze(-1)
                IS3_new_x[:, 2] = 0.1
                train_x = torch.cat([train_x, IS3_new_x])
                train_obj = torch.cat([train_obj, IS3_obj_new_x])
                x_ = IS3_new_x.clone().reshape(1,3)
                x_[:, 2] = 1.0
                new_obj_ = problem(x_).clone().unsqueeze(-1)
                delta_J = torch.cat([delta_J, IS3_obj_new_x-new_obj_])
                JIS1s_for_delta_J = torch.cat([JIS1s_for_delta_J, IS3_obj_new_x])
                caEI_values = torch.cat([caEI_values, torch.tensor([0])])
                # TODO
                delta_J2 = torch.cat([delta_J2, torch.tensor([[0]])])
                JIS2s_for_delta_J2 = torch.cat([JIS2s_for_delta_J2, new_obj])

        elif new_x[:,-1] == 0.1:
            x_ = new_x.clone().reshape(1,3)
            x_[:, 2] = 1.0
            obj_x_IS1_ = problem(x_).clone().unsqueeze(-1)
            delta_J = torch.cat([delta_J, new_obj-obj_x_IS1_])
            JIS1s_for_delta_J = torch.cat([JIS1s_for_delta_J, obj_x_IS1_])
            caEI_values = torch.cat([caEI_values, caEI_value.reshape(1)])
            # TODO
            delta_J2 = torch.cat([delta_J2, torch.tensor([[0]])])
            JIS2s_for_delta_J2 = torch.cat([JIS2s_for_delta_J2, new_obj])

        cumulative_cost += cost
        costs_all[i] = cost
        np.save(path + "/costs_all.npy", costs_all)
        np.save(path + "/train_x.npy", train_x)
        np.save(path + "/train_obj.npy", train_obj)

        np.save(path + "/i_IS1s.npy", i_IS1s)
        np.save(path + "/delta_J.npy", delta_J)
        np.save(path + "/JIS1s_for_delta_J.npy", JIS1s_for_delta_J)
        np.save(path + "/delta_J2.npy", delta_J2)
        np.save(path + "/JIS2s_for_delta_J2.npy", JIS2s_for_delta_J2)
        np.save(path + "/caEI_values.npy", caEI_values)

    final_rec, objective_value = get_recommendation(model, lower, upper)
    np.save(path + "/final_rec.npy", final_rec)
    np.save(path + "/objective_value.npy", objective_value)

    final_rec_max_observed, objective_value_max_observed = get_recommendation_max_observed(train_x, train_obj, lower,
                                                                                           upper)
    np.save(path + "/final_rec_max_observed.npy", final_rec_max_observed)
    np.save(path + "/objective_value_max_observed.npy", objective_value_max_observed)

    print(f"\nMFBO total cost: {cumulative_cost}\n")

    # np.save("/cluster/home/mnobar/code/GBO2/logs/test_46/delta_J2_IS1init_all_en_025.npy", delta_J2_IS1init_all)

    # ####################################################################################################################
    # ####################################################################################################################
    # ####################################################################################################################
    # # Baseline Single Fidelity BO with EI
    # cumulative_cost = 0.0
    # costs_all = np.zeros(N_ITER)
    # train_x = train_x_init[:N_init_IS1]
    #
    # train_obj_init = problem(train_x_init[:N_init_IS1]).unsqueeze(-1)
    #
    # train_obj = train_obj_init[:N_init_IS1]
    #
    # # path2="/cluster/home/mnobar/code/GBO2/logs/test_31_b_5*/Exper_{}".format(str(exper))
    # # train_obj_init=np.load(path2 + "/train_obj_init.npy")
    # # train_x_init=np.load(path2 + "/train_x_init.npy")
    # # cumulative_cost = 0.0
    # # costs_all = np.zeros(N_ITER)
    # # train_x = torch.as_tensor(train_x_init[:N_init_IS1])
    # # train_obj = torch.as_tensor(train_obj_init[:N_init_IS1])
    #
    # for i in range(N_ITER):
    #     print("BO-EI batch iteration=", i)
    #     mll, model = initialize_model(train_x, train_obj,l_g_0=None)
    #     fit_gpytorch_mll(mll)
    #     plot_EIonly_GP(model, i, path, train_x)
    #     ei_acqf = get_ei(model, best_f=train_obj.max())
    #     if np.sum(np.asarray(train_x[:,2])==1)>N_init_IS1+N_change:
    #         change_IS1=True
    #     else:
    #         change_IS1=False
    #     new_x, new_obj, cost = optimize_ei_and_get_observation(ei_acqf, change_IS1)
    #     train_x = torch.cat([train_x, new_x])
    #     train_obj = torch.cat([train_obj, new_obj])
    #     cumulative_cost += cost
    #     costs_all[i] = cost
    #     np.save(path + "/costs_all_EIonly.npy", costs_all)
    #     np.save(path + "/train_x_EIonly.npy", train_x)
    #     np.save(path + "/train_obj_EIonly.npy", train_obj)
    #
    # final_rec_EIonly, objective_value_EIonly = get_recommendation(model, lower, upper)
    # np.save(path + "/final_rec_EIonly.npy", final_rec_EIonly)
    # np.save(path + "/objective_value_EIonly.npy", objective_value_EIonly)
    #
    # final_rec_max_observed_EIonly, objective_value_max_observed_EIonly = get_recommendation_max_observed(train_x,
    #                                                                                                      train_obj,
    #                                                                                                      lower, upper)
    # np.save(path + "/final_rec_max_observed_EIonly.npy", final_rec_max_observed_EIonly)
    # np.save(path + "/objective_value_max_observed_EIonly.npy", objective_value_max_observed_EIonly)
    #
    # print(f"\nEI only total cost: {cumulative_cost}\n")


    # # ####################################################################################################################
    # ####################################################################################################################
    # #####################################################################################################################
    # # Baseline Single Fidelity BO with UCB
    # cumulative_cost = 0.0
    # costs_all = np.zeros(N_ITER)
    # train_x = train_x_init[:N_init_IS1]
    # train_obj = train_obj_init[:N_init_IS1]
    #
    # # path2="/cluster/home/mnobar/code/GBO2/logs/test_31_b_5*/Exper_{}".format(str(exper))
    # # train_obj_init=np.load(path2 + "/train_obj_init.npy")
    # # train_x_init=np.load(path2 + "/train_x_init.npy")
    # # cumulative_cost = 0.0
    # # costs_all = np.zeros(N_ITER)
    # # train_x = torch.as_tensor(train_x_init[:N_init_IS1])
    # # train_obj = torch.as_tensor(train_obj_init[:N_init_IS1])
    #
    # for i in range(N_ITER):
    #     print("BO-UCB batch iteration=", i)
    #     mll, model = initialize_model(train_x, train_obj)
    #     fit_gpytorch_mll(mll)
    #     plot_UCBonly_GP(model, i, path, train_x)
    #     ucb_acqf = get_ucb(model, beta=0.2)  # Tune beta as needed
    #     new_x, new_obj, cost = optimize_ei_and_get_observation(ucb_acqf)
    #     train_x = torch.cat([train_x, new_x])
    #     train_obj = torch.cat([train_obj, new_obj])
    #     cumulative_cost += cost
    #     costs_all[i] = cost
    #     np.save(path + "/costs_all_UCBonly.npy", costs_all)
    #     np.save(path + "/train_x_UCBonly.npy", train_x)
    #     np.save(path + "/train_obj_UCBonly.npy", train_obj)
