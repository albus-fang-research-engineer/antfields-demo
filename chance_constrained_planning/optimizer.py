import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from load_njsdf.inference import mu_sigma_grad_nn

DELTA = 0.05/10
BETA = norm.ppf(1 - DELTA) # chance constraint


def solve_step(p0, p_goal, obstacle_points, model, device):
    '''
    p0 is current position, p_goal is the next waypoint to track
    p0 and p_goal are not global start and goal points
    '''
    mu0, sigma0, grad0 = mu_sigma_grad_nn(
            p0, obstacle_points, model, device
        )
    def objective(x):
        dp = x[:2]
        slack = x[2:]
        return dp @ dp + 10.0 * (slack @ slack)

    def chance_constraint(x):
        dp = x[:2]
        return mu0 + grad0 @ dp - BETA * sigma0

    def tracking_constraint(x):
        dp = x[:2]
        slack = x[2:]
        return p0 + dp - p_goal - slack

    cons = [
        {"type": "ineq", "fun": chance_constraint},
        {"type": "eq", "fun": tracking_constraint},
    ]

    res = minimize(objective, np.zeros(4), constraints=cons, method="SLSQP")
    return p0 + res.x[:2], mu0, sigma0#, res