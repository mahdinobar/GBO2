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
print("SMOKE_TEST is ",SMOKE_TEST)
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
from botorch.acquisition import qExpectedImprovement

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

def generate_initial_data(n=16):
    # generate training data
    train_x = torch.hstack(((bounds[1,0].item()-bounds[0,0].item())*torch.rand(n, 1, **tkwargs)+bounds[0,0].item(),
                            (bounds[1,1].item()-bounds[0,1].item())*torch.rand(n, 1, **tkwargs)+bounds[0,1].item()))
    # # TODO: uncomment for my idea
    # train_f = fidelities[torch.randint(1,3, (n, 1))]
    # uncomment for deterministic initial dataset of just IS1
    train_f = fidelities[torch.randint(2,3, (n, 1))]
    # train_f = fidelities[torch.randint(2, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj


def initialize_model(train_x, train_obj):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension index 2 corrosponding to the fidelity random variable in state space, as in [2]
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[2]
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
        values=[1], # here we fix fidelity to 1.0 (highest fidelity) to compute the current value
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )

    # model: This is the surrogate model representing our current understanding of the objective function based on available data.
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    # uncomment for my idea
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0.2},{2: 0.5}, {2: 1.0}],
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 4, "maxiter": 50},
    )
    # candidates, _ = optimize_acqf_mixed(
    #     acq_function=mfkg_acqf,
    #     bounds=bounds,
    #     fixed_features_list=[{2: 0.5}, {2: 1.0}],
    #     q=BATCH_SIZE,
    #     num_restarts=NUM_RESTARTS,
    #     raw_samples=RAW_SAMPLES,
    #     # batch_initial_conditions=X_init,
    #     options={"batch_limit": 4, "maxiter": 50},
    # )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost

def plot_GP(model, iter, path):
    # Step 3: Define fidelity levels and create a grid for plotting
    # uncomment for my idea
    fidelities = [1.0, 0.5, 0.2]  # Three fidelity levels
    # fidelities = [1.0, 0.5]  # Three fidelity levels
    x1 = torch.linspace(0, 1, 50)
    x2 = torch.linspace(0, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

    # Step 4: Prepare the figure with 3x2 subplots
    fig, axs = plt.subplots(len(fidelities), 2, figsize=(14, 18))

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

        np.save(path+"/X1_{}.npy".format(str(iter)),X1)
        np.save(path+"/X2_{}.npy".format(str(iter)),X2)
        np.save(path+"/mean_{}.npy".format(str(iter)), mean)
        np.save(path+"/std_{}.npy".format(str(iter)),std)

        # Axis labels
        for ax in axs[i]:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

    plt.tight_layout()
    plt.savefig(path+"/GP_itr_{}.pdf".format(str(iter)))  # Save as PDF
    plt.savefig(path+"/GP_itr_{}.png".format(str(iter)))
    # plt.show()

def get_recommendation(model, lower, upper):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[1], # here we fix fidelity to 1.0 (highest fidelity) to have the recommendation based on only target fidelity
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
    objective_value = problem(final_rec) #here final_rec is the normalized states
    def denormalize(x, lower, upper):
        return x * (upper - lower) + lower
    print(f"Final posterior optimized recommended point:\n{final_rec}\n\nrecommended point (unnormalized):\n{denormalize(final_rec, lower, upper)}\n\nobjective value:\n{objective_value}")
    return final_rec, objective_value

def get_recommendation_max_observed(train_x, train_obj, lower, upper):
    idx_s1 = np.argwhere(train_x[:, 2] == 1).squeeze()
    idx_ = np.argmax(np.asarray(train_obj[idx_s1].squeeze()))
    idx_max = idx_s1[idx_]
    final_rec = train_x[idx_max, :]
    objective_value = train_obj[idx_max]
    def denormalize(x, lower, upper):
        return x * (upper - lower) + lower
    print(f"Max observed recommended point:\n{final_rec}\n\nrecommended point (unnormalized):\n{denormalize(final_rec, lower, upper)}\n\nobjective value:\n{objective_value}")
    return final_rec, objective_value


def get_ei(model, best_f):
    return FixedFeatureAcquisitionFunction(
        acq_function=qExpectedImprovement(model=model, best_f=best_f),
        d=3,
        columns=[2],
        values=[1],
    )

def optimize_ei_and_get_observation(ei_acqf):
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

N_exper=20
NUM_RESTARTS = 4 if not SMOKE_TEST else 2
RAW_SAMPLES = 64 if not SMOKE_TEST else 4
BATCH_SIZE = 4
N_init = 2 if not SMOKE_TEST else 2
N_ITER = 10 if not SMOKE_TEST else 1

for exper in range(N_exper):
    print("**********Experiment {}**********".format(exper))
    path = "/home/nobar/codes/GBO2/logs/test_5/Exper_{}".format(str(exper))
    # Check if the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # uncomment for my idea
    fidelities = torch.tensor([0.2, 0.5, 1.0], **tkwargs)
    # fidelities = torch.tensor([0.5, 1.0], **tkwargs)

    # Define the bounds
    original_bounds = torch.tensor([[70, 2, 0.0], [120, 5, 1.0]], **tkwargs)
    lower, upper = original_bounds[0], original_bounds[1]
    # Example input data
    X_original = torch.stack([lower, upper]).to(**tkwargs)
    # Normalize inputs to [0, 1]
    bounds = normalize(X_original, lower, upper)

    target_fidelities = {2: 1.0}

    cost_model = AffineFidelityCostModel(fidelity_weights={2: 1.0}, fixed_cost=1)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    # In[1]:
    torch.set_printoptions(precision=3, sci_mode=False)

    train_x_init, train_obj_init = generate_initial_data(n=N_init)
    train_x=train_x_init
    train_obj = train_obj_init
    # print("train_obj_init=",train_obj_init)
    np.save(path+"/train_obj_init.npy", train_obj_init)
    np.save(path+"/train_x_init.npy", train_x_init)

    cumulative_cost = 0.0
    costs_all = np.zeros(N_ITER)
    for i in range(N_ITER):
        print("iteration=",i)
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
        plot_GP(model, i, path)
        mfkg_acqf = get_mfkg(model)
        new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        cumulative_cost += cost
        costs_all[i]=cost
        np.save(path+"/costs_all.npy", costs_all)
        np.save(path+"/train_x.npy", train_x)
        np.save(path+"/train_obj.npy", train_obj)

    final_rec, objective_value = get_recommendation(model, lower, upper)
    np.save(path+"/final_rec.npy", final_rec)
    np.save(path+"/objective_value.npy", objective_value)

    final_rec_max_observed, objective_value_max_observed = get_recommendation_max_observed(train_x, train_obj, lower, upper)
    np.save(path+"/final_rec_max_observed.npy", final_rec_max_observed)
    np.save(path+"/objective_value_max_observed.npy", objective_value_max_observed)

    print(f"\nMFBO total cost: {cumulative_cost}\n")



    # In[1]:
    # N_ITER=5
    cumulative_cost = 0.0
    costs_all = np.zeros(N_ITER)
    train_x, train_obj = generate_initial_data(n=16)
    train_x=train_x_init
    train_obj = train_obj_init
    # exp_path_EIonly = "/home/nobar/codes/GBO2/logs/test_4/Exper_{}".format(str(exper))
    # train_x=np.load(os.path.join(exp_path_EIonly, "train_x_IS1_init.npy"))
    # train_obj=np.load(os.path.join(exp_path_EIonly, "train_obj_IS1_init.npy"))
    # train_x=torch.as_tensor(train_x)
    # train_obj=torch.as_tensor(train_obj)

    for i in range(N_ITER):
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)
        ei_acqf = get_ei(model, best_f=train_obj.max())
        new_x, new_obj, cost = optimize_ei_and_get_observation(ei_acqf)
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        cumulative_cost += cost
        costs_all[i]=cost
        np.save(path+"/costs_all_EIonly_corrected.npy", costs_all)
        np.save(path+"/train_x_EIonly_corrected.npy", train_x)
        np.save(path+"/train_obj_EIonly_corrected.npy", train_obj)

    # In[12]:
    final_rec_EIonly, objective_value_EIonly = get_recommendation(model, lower, upper)
    np.save(path+"/final_rec_EIonly_corrected.npy", final_rec_EIonly)
    np.save(path+"/objective_value_EIonly_corrected.npy", objective_value_EIonly)

    final_rec_max_observed_EIonly, objective_value_max_observed_EIonly = get_recommendation_max_observed(train_x, train_obj, lower, upper)
    np.save(path+"/final_rec_max_observed_EIonly_corrected.npy", final_rec_max_observed_EIonly)
    np.save(path+"/objective_value_max_observed_EIonly_corrected.npy", objective_value_max_observed_EIonly)

    print(f"\nEI only total cost: {cumulative_cost}\n")
