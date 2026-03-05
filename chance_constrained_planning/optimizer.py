import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from load_njsdf.inference import mu_sigma_grad_nn, BETA
import torch
import matplotlib.pyplot as plt
# DELTA = 0.1/10
# BETA = norm.ppf(1 - DELTA) # chance constraint
debug_step_counter = 0
last_debug_epoch = -1
def solve_step(p0, p_goal, obstacle_points, model, device, epoch):
    '''
    p0 is current position, p_goal is the next waypoint to track
    p0 and p_goal are not global start and goal points
    '''
    global debug_step_counter, last_debug_epoch
    if epoch != last_debug_epoch:
        debug_step_counter = 0
        last_debug_epoch = epoch
    step_id = debug_step_counter
    debug_step_counter += 1
    if torch.is_tensor(p0):
        p0 = p0.detach().cpu().numpy()

    if torch.is_tensor(p_goal):
        p_goal = p_goal.detach().cpu().numpy()
    mu, sigma, grad, obs_k = mu_sigma_grad_nn(
            p0, obstacle_points, model, device
        )
    print("\n--- Chance constraint debug ---")
    print("robot:", p0)
    print("goal:", p_goal)
    print("min(mu):", np.min(mu))
    print("min(risk):", np.min(mu - BETA * sigma))
    print("mean(sigma):", np.mean(sigma))
    print("grad norms:", np.linalg.norm(grad, axis=1))
    def objective(x):
        dp = x[:2]
        slack = x[2:]
        return dp @ dp + 100.0 * (slack @ slack)

    def chance_constraints(x):
        dp = x[:2]
        return mu + grad @ dp - BETA * sigma

    def tracking_constraint(x):
        dp = x[:2]
        slack = x[2:]
        return p0[:2] + dp - p_goal[:2] - slack

    cons = [
        {"type": "ineq", "fun": chance_constraints},
        {"type": "eq", "fun": tracking_constraint},
    ]

    res = minimize(objective, np.zeros(4), constraints=cons, method="SLSQP")
    p_next = p0.copy()
    p_next[:2] += res.x[:2]
    # plot_chance_debug(p0, obstacle_points, mu, sigma, grad, obs_k, "/antfields/chance_constrained_planning/debug", epoch, step_id)
    # p_next = torch.tensor(p_next, dtype=torch.float32, device=device)
    return p_next, mu, sigma
    # return p0 + res.x[:2], mu0, sigma0#, res

def plot_chance_debug(robot_xy, obstacle_points, mu, sigma, grad, obs_k, folder, epoch, step_id):
    import os

    epoch_folder = os.path.join(folder, f"epoch_{epoch:04d}")
    os.makedirs(epoch_folder, exist_ok=True)
     # convert tensors to numpy if needed
    if torch.is_tensor(obstacle_points):
        obstacle_points = obstacle_points.detach().cpu().numpy()

    if torch.is_tensor(obs_k):
        obs_k = obs_k.detach().cpu().numpy()

    if torch.is_tensor(robot_xy):
        robot_xy = robot_xy.detach().cpu().numpy()
    # ---------- Plot 1: gradient directions ----------
    plt.figure(figsize=(6,6))

    plt.scatter(obstacle_points[:,0], obstacle_points[:,1],
                s=2, alpha=0.15, label="all obstacles")

    plt.scatter(obs_k[:,0], obs_k[:,1],
                s=20, c="orange", label="K constraints")

    plt.scatter(robot_xy[0], robot_xy[1],
                c='red', s=80, label="robot")

    for i in range(len(mu)):
        g = grad[i]
        plt.arrow(robot_xy[0], robot_xy[1],
                  g[0]*0.02, g[1]*0.02,
                  head_width=0.002,
                  color="blue",
                  alpha=0.7)

    plt.legend()
    plt.title("Chance constraint gradients")

    # plt.savefig(f"{folder}/chance_grad_{epoch}.png")
    plt.savefig(f"{epoch_folder}/step_{step_id:05d}_grad.png")
    plt.close()


    # ---------- Plot 2: sigma values ----------
    # plt.figure(figsize=(6,6))

    # plt.scatter(obstacle_points[:,0], obstacle_points[:,1],
    #             s=2, alpha=0.15)

    # sc = plt.scatter(obs_k[:,0], obs_k[:,1],
    #                  c=sigma, cmap="plasma",
    #                  s=40)

    # plt.scatter(robot_xy[0], robot_xy[1],
    #             c='red', s=80)

    # for i in range(len(sigma)):
    #     plt.text(obs_k[i,0], obs_k[i,1],
    #              f"{sigma[i]:.3f}",
    #              fontsize=7)

    # plt.colorbar(sc, label="sigma")

    # plt.title("Sigma values of active constraints")

    # plt.savefig(f"{folder}/chance_sigma_{epoch}.png")
    # plt.close()