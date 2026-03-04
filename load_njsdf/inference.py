import numpy as np
from pathlib import Path
from load_njsdf.sdf.stochastic_robot_sdf import RobotSdfCollisionNet
import torch


def load_sdf_2d_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobotSdfCollisionNet(
        in_channels=4,
        out_channels=1,
        layers=[128] * 4,
        skips=[]
    ).model

    model_path = Path(__file__).parent / "models" / "sdf_2d.pt"

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()

    print(f"SDF model loaded from: {model_path}")
    print(f"Using device: {device}")

    return model, device
RADIUS = 0.105

def predict_mu_var(model, x):
    pred = model(x)
    mu, logvar = torch.chunk(pred, 2, dim=-1)
    logvar = torch.clamp(logvar, -20.0, 10.0)
    return mu.squeeze(-1), torch.exp(logvar).squeeze(-1)

def mu_sigma_grad_nn(robot_xy, obstacle_points, model, device):
    """
    robot_xy: (2,)
    obstacle_points: (N,2)

    returns:
        mu      -> scalar (closest mean distance - radius)
        sigma   -> scalar
        grad    -> (2,) gradient wrt robot position
    """
    # print("robot coordinates shape: ", robot_xy.shape)
    # print("obstalce_points.shape: ", obstacle_points.shape)
    robot_xy = robot_xy[:2]            # keep x,y only
    obstacle_points = obstacle_points[:, :2]
    # --- ensure tensors ---
    if not torch.is_tensor(robot_xy):
        robot_xy = torch.tensor(robot_xy, dtype=torch.float32, device=device)

    if not torch.is_tensor(obstacle_points):
        obstacle_points = torch.tensor(obstacle_points, dtype=torch.float32, device=device)

    robot_xy = robot_xy.to(device)
    obstacle_points = obstacle_points.to(device)

    N = obstacle_points.shape[0]

    # repeat robot position
    robot_rep = robot_xy.unsqueeze(0).repeat(N, 1)

    # concatenate robot + obstacle
    x = torch.cat([robot_rep, obstacle_points], dim=1)

    # we need gradient wrt robot position
    x.requires_grad_(True)

    mu, var = predict_mu_var(model, x)

    # find most critical obstacle
    idx = torch.argmin(mu)

    mu_min = mu[idx]
    sigma_min = torch.sqrt(var[idx])
    print("mu_min is ", mu_min)
    print("sigma_min is ", sigma_min)
    # gradient wrt robot position
    grad_full = torch.autograd.grad(mu_min, x, retain_graph=False)[0]

    grad_robot = grad_full[idx, 0:2]

    return (
        mu_min.item(),
        sigma_min.item(),
        grad_robot.detach().cpu().numpy()
    )

def mu_sigma_grad_nn_npy(robot_xy, obstacle_points, model, device):
    """
    robot_xy: (2,)
    obstacle_points: (N,2)

    returns:
        mu      -> scalar (closest mean distance - radius)
        sigma   -> scalar
        grad    -> (2,) gradient wrt robot position
    """

    N = obstacle_points.shape[0]

    robot_rep = np.repeat(robot_xy[None, :], N, axis=0)
    x_input = np.concatenate([robot_rep, obstacle_points], axis=1)

    x = torch.tensor(x_input, dtype=torch.float32, device=device, requires_grad=True)

    mu, var = predict_mu_var(model, x)

    # subtract robot radius (same as analytic version)
    # mu = mu - RADIUS

    # find most critical point
    idx = torch.argmin(mu)

    mu_min = mu[idx]
    sigma_min = torch.sqrt(var[idx])

    # gradient wrt robot position
    grad_full = torch.autograd.grad(mu_min, x, retain_graph=False)[0]

    grad_robot = grad_full[idx, 0:2]   # only d/d(robot_x, robot_y)

    return (
        mu_min.item(),
        sigma_min.item(),
        grad_robot.detach().cpu().numpy()
    )


def risk_distance_cvar(
    robot_xy,
    obstacle_points,
    model,
    device,
    *,
    alpha=0.30,      # mild VaR (not too conservative)
    cvar_tail=0.10,  # worst 10%
    min_dist=0.0002  # 0.2 mm floor
):
    """
    Returns a single scalar risk-aware distance.
    """

    N = obstacle_points.shape[0]

    robot_rep = np.repeat(robot_xy[None, :], N, axis=0)
    x_input = np.concatenate([robot_rep, obstacle_points], axis=1)

    x = torch.tensor(x_input, dtype=torch.float32, device=device)

    mu, var = predict_mu_var(model, x)
    sigma = torch.sqrt(torch.clamp(var, min=1e-12))

    # ---- lower-tail VaR (small distance = risky) ----
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    z = normal.icdf(torch.tensor(alpha, device=device))
    var_i = mu + z * sigma

    # ---- CVaR over smallest distances ----
    k = max(1, int(np.ceil(cvar_tail * N)))
    worst_vals = torch.topk(var_i, k, largest=False).values
    cvar = worst_vals.mean()

    # ---- floor at 0.2 mm ----
    cvar = torch.clamp(cvar, min=min_dist)

    return float(cvar.detach().cpu().item())