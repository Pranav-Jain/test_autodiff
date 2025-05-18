import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import polyscope as ps
import gpytoolbox as gpy
import torch
from tqdm import tqdm
import glob
import sys
import os
import torch.nn as nn

# Enable higher-order gradients
# torch.set_default_dtype(torch.float64)

def get_spherical_coordinates(v, c, r_):
    r = np.ones(v.shape[0]) * r_
    theta = np.arccos((v[:, 2] - c[2]) / r)
    phi = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])

    return np.stack([r, theta, phi], axis=1)

def get_spherical_coordinates_torch(v, c, r_):
    r = torch.ones(v.shape[0], device=v.device) * r_
    theta = torch.arccos((v[:, 2] - c[2]) / r)
    phi = torch.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])

    r = r.requires_grad_(True)
    theta = theta.requires_grad_(True)
    phi = phi.requires_grad_(True)

    return torch.stack([r, theta, phi], dim=1)

def get_cartesian_coords_torch(v, r, c):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]
    
    x = r*torch.sin(theta)*torch.cos(phi)
    y = r*torch.sin(theta)*torch.sin(phi)
    z = r*torch.cos(theta)

    return torch.stack([x, y, z], dim=1)


def f(v, const = 0.0):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[2] == "1":
        f = r**2 * (const - 0.5*np.cos(theta))
    elif sys.argv[2] == "2":
        f = r**2 * (const - 0.5*np.sin(theta)*np.cos(phi))
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")
        exit()

    return f.squeeze()

def f_torch(v, const = 0.0):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[2] == "1":
        f = r**2 * (const - 0.5*torch.cos(theta))
    elif sys.argv[2] == "2":
        f = r**2 * (const - 0.5*torch.sin(theta)*torch.cos(phi))
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")
        exit()

    return f.squeeze()

def laplacian_f(v):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[2] == "1":
        lap_f = torch.cos(theta)
    elif sys.argv[2] == "2":
        lap_f = torch.sin(theta)*torch.cos(phi)
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")
        exit()

    return lap_f.squeeze()

# Create a MLP model
class Sine(torch.nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class MLP(torch.nn.Module):
    def __init__(self, n=512, n_layers=3, in_dim=3):
        super(MLP, self).__init__()
        net = []
        first_layer = torch.nn.Linear(in_dim, n)
        # Initializing the weights with the Xavier initialization method
        torch.nn.init.xavier_uniform_(first_layer.weight)
        net.append(first_layer)
        # net.append(Relu3(w0=1.0))
        net.append(nn.Tanh())
        for _ in range(n_layers):
            layer = torch.nn.Linear(n, n)
            # Initializing the weights with the Xavier initialization method
            torch.nn.init.xavier_uniform_(layer.weight)
            net.append(layer)
            # net.append(Relu3(w0=1.0))
            net.append(nn.Tanh())
        net.append(torch.nn.Linear(n, 1))

        self.layers = torch.nn.Sequential(*net)
    
    def forward(self, x):
        return self.layers(x)

def compute_hessian_vector(S_theta, x):
    """Compute the Hessian matrix H for a vector-valued function S_theta."""
    y = S_theta(x)  # Output is (3,)
    
    H = torch.zeros(y.shape[0], x.shape[0], x.shape[0], device=x.device)  # Shape (3, 3, 3)
    J = torch.zeros(y.shape[0], x.shape[0], device=x.device)  # Shape (3, 3)
    
    for i in range(y.shape[0]):  # Loop over 3 output dimensions
        grad_y_i = torch.autograd.grad(y[i], x, create_graph=True)[0]  # First derivative (Jacobian row)
        J[i] = grad_y_i  # Store Jacobian row
        for j in range(x.shape[0]):  # Loop over 3 input dimensions
            H[i, j] = torch.autograd.grad(grad_y_i[j], x, create_graph=True)[0]  # Second derivative
        
    return J, H

def compute_hessian_scalar(S_theta, x):
    """Compute the Hessian matrix H for a scalar-valued function S_theta."""
    y = S_theta(x)  # Output is scalar
    
    grad_y = torch.autograd.grad(y, x, create_graph=True)[0]  # First derivative (Jacobian row)
    H = torch.zeros(x.shape[0], x.shape[0], device=x.device)  # Shape (3, 3)
    
    for i in range(x.shape[0]):  # Loop over 3 input dimensions
        H[i] = torch.autograd.grad(grad_y[i], x, create_graph=True)[0]  # Second derivative
        
    return H

def get_interpolated_values(f, v, v_mesh, f_mesh):
    f_v = np.zeros_like(v[:, 0])

    _, idx, bary = gpy.squared_distance(v, v_mesh, f_mesh, use_cpp=True)
    for i in range(v.shape[0]):
        f_v[i] = np.sum(f[f_mesh[idx[i]]] * bary[i])
    # f_v = np.sum(f[f_mesh[idx]] * bary[:, None], axis=1)

    return f_v

def sample_in_sphere(n, r):
    # Sample uniformly on the unit sphere
    vec = np.random.uniform(-1, 1, (n, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit length
    vec = vec * r  # Scale to radius r

    return vec


def get_surface_laplacian(model, v_cart, center, radius):
    def model_scalar(v_):
        return model(v_).view(-1).requires_grad_(True)  # Ensure scalar output

    center = torch.Tensor(center).to(v_cart.device).requires_grad_(True)
    n = v_cart - center
    n = n / torch.linalg.norm(n, dim=1, keepdim=True)  # Vectorized normalization
    n_unsqueeze = n.unsqueeze(-1)

    v_sp = get_spherical_coordinates_torch(v_cart, center, radius)
    v_sp = v_sp.requires_grad_(True)
    f_ = model_scalar(v_sp)
    
    grad_f = torch.autograd.grad(f_, v_cart, torch.ones_like(f_), create_graph=True, retain_graph=True)[0]

    # Surface gradient (tangential component)
    grad_f_surf = grad_f - torch.sum(grad_f * n, dim=1, keepdim=True) * n

    # Compute divergence of surface gradient
    def compute_divergence(grad_, v_):
        # Compute derivatives of each component of grad_f_surf
        div_x = torch.autograd.grad(grad_[:, 0], v_, torch.ones_like(grad_[:, 0]), create_graph=True, retain_graph=True)[0]
        div_x = div_x - torch.sum(div_x * n, dim=1, keepdim=True) * n
        div_y = torch.autograd.grad(grad_[:, 1], v_, torch.ones_like(grad_[:, 1]), create_graph=True, retain_graph=True)[0]
        div_y = div_y - torch.sum(div_y * n, dim=1, keepdim=True) * n
        div_z = torch.autograd.grad(grad_[:, 2], v_, torch.ones_like(grad_[:, 2]), create_graph=True, retain_graph=True)[0]
        div_z = div_z - torch.sum(div_z * n, dim=1, keepdim=True) * n
        
        # Build Hessian matrix
        hessian = torch.zeros(v_cart.shape[0], 3, 3).to(v_cart.device)
        hessian[:, 0, :] = div_x
        hessian[:, 1, :] = div_y
        hessian[:, 2, :] = div_z

        # Sum diagonal terms for divergence
        divF = hessian[:, 0, 0] + hessian[:, 1, 1] + hessian[:, 2, 2]

        return divF, hessian

    div_grad_f_surf, hessians = compute_divergence(grad_f_surf, v_cart)

    # Compute Hessian applied to normal: H n
    hessian_dot_n = torch.bmm(hessians, n.unsqueeze(-1)).squeeze()

    # Compute normal term: n^T (H n)
    normals_term = torch.sum(n * hessian_dot_n, dim=1)

    # Final Laplace-Beltrami operator
    lap_beltrami = div_grad_f_surf - normals_term

    return lap_beltrami

def test_poisson(model, center, r, device, n):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5000, verbose=True)

    losses = []
    for i in (pbar:= tqdm(range(2000000))):
        # Zero the gradients
        optimizer.zero_grad()

        # Sample points on the sphere
        v_mesh_rnd_cart = sample_in_sphere(n, r)
        v_mesh_rnd = get_spherical_coordinates(v_mesh_rnd_cart, center, r)

        v_mesh_rnd = torch.tensor(v_mesh_rnd, dtype=torch.float32).to(device=device)
        v_mesh_rnd_cart = torch.tensor(v_mesh_rnd_cart, dtype=torch.float32).to(device=device).requires_grad_(True)
        
        laplacian_pred = get_surface_laplacian(model, v_mesh_rnd_cart, center, r)
        f_v = laplacian_f(v_mesh_rnd)

        ## Loss with extra points ##
        if len(sys.argv) > 3 and sys.argv[3] == "exactvalue":
            extra_pts = v_mesh_rnd[:2]
            loss = (1/n)*torch.linalg.norm(laplacian_pred - f_v, 2)**2 + (100)*torch.linalg.norm(model(extra_pts).squeeze() - f_torch(extra_pts), 2)**2

        ## Loss st the function integrates to zero ##
        elif len(sys.argv) > 3 and sys.argv[3] == "zerointegral":
            extra_pts = sample_in_sphere(10000, r)
            extra_pts = get_spherical_coordinates(extra_pts, center, r)
            extra_pts = torch.tensor(extra_pts, dtype=torch.float32).to(device=device).requires_grad_(True)
            loss = torch.nn.functional.mse_loss(laplacian_pred, f_v) + 1/(extra_pts.shape[0])*torch.sum(model(extra_pts).squeeze())**2

        else:
            loss = torch.linalg.norm(laplacian_pred - f_v, 2)**2
        
        loss.backward()
        if not torch.isnan(loss):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step(loss)

            if scheduler.state_dict()["_last_lr"][0] < 1e-7:
                break

            loss_value = loss.item()
            pbar.set_description(f"Loss: {loss_value}")

    return model, losses

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_layers = [3]
    size_layer = 2**np.arange(4, 11, 1)

    center = np.array([0.0, 0.0, 0.0])
    r = 1.0

    V = sample_in_sphere(50000, 1.0)
    V = get_spherical_coordinates(V, center, r)

    for i in n_layers:
        for j in size_layer:
            print(f"Processing model with {i} layers of {j} size")

            model = MLP(n=j, n_layers=i)
            model.to(device=device)

            model, losses = test_poisson(model, center, r, device, n=1000)

            l2_loss = np.linalg.norm(f(V) - model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy(), 2) #* (1/V.shape[0])
            linf_loss = np.linalg.norm(f(V) - model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy(), np.inf) #* (1/V.shape[0])
            print(l2_loss, linf_loss)

            # plt.figure()
            # plt.plot(losses)
            # plt.title(f"Relative L2 Loss for Example{sys.argv[2]}, Model {j}x{i}")
            # plt.xlabel("Iteration")
            # plt.ylabel("Relative L2 Loss")
            # # plt.show()
            # plt.savefig(f"./loss_example_{sys.argv[2]}_{j}_{i}.png") 

            if len(sys.argv) > 3:
                torch.save(model.state_dict(), f"poisson_results_sphere_{sys.argv[3]}/example{sys.argv[2]}/model_{j}_{i}.pth")
            else:
                torch.save(model.state_dict(), f"poisson_results_sphere/example{sys.argv[2]}/model_{j}_{i}.pth")

def plot():
    try:
        if len(sys.argv) > 3:
            filenames = glob.glob(f"poisson_results_sphere_{sys.argv[3]}/example{sys.argv[2]}/*.pth")
        else:
            filenames = glob.glob(f"poisson_results_sphere/example{sys.argv[2]}/*.pth")
    except:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2] [|exactvalue|zerointegral]")
        exit()
        
    dof = []
    losses = []

    V_cart = sample_in_sphere(100000, 1.0)
    center = np.array([0.0, 0.0, 0.0])
    V = get_spherical_coordinates(V_cart, center, 1.0)
    true = f(V)

    for file in filenames:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_layers = int(file.split("_")[-1][:-4])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
        const = np.mean(true - pred)
        pred = pred + const

        l2_loss = np.linalg.norm(pred - true) * (1/V.shape[0])
        print(l2_loss, n_layers, size_layer)
        losses.append(l2_loss)
        dof.append(2*n_layers*size_layer)

    fem_loss = []
    fem_dof = []
    for i in range(2, 7):
        v_mesh, f_mesh = gpy.icosphere(i)
        M_mesh = gpy.massmatrix(v_mesh, f_mesh)
        M_mesh = sp.sparse.csc_matrix(M_mesh)
        v_mesh_sp = get_spherical_coordinates(v_mesh, center, 1.0)
        f_ = laplacian_f(torch.tensor(v_mesh_sp, dtype=torch.float32).to(device=device)).squeeze().detach().cpu().numpy()
        
        l = gpy.cotangent_laplacian(v_mesh, f_mesh)
        lap_mesh = sp.sparse.linalg.inv(M_mesh) @ l
        u = sp.sparse.linalg.lsmr(lap_mesh, f_)[0]
        pred = get_interpolated_values(u, V, v_mesh, f_mesh)

        const = np.mean(true - pred)
        pred = pred + const

        l2_loss = np.linalg.norm(true - pred, 2) * (1/v_mesh.shape[0])
        # l2_loss = np.linalg.norm(true - pred) / np.linalg.norm(true)
        fem_loss.append(l2_loss)
        fem_dof.append(v_mesh.shape[0])

        # diff = np.abs(pred - true)
        # ps.init()
        # ps.register_point_cloud("pc", V_cart)
        # ps.get_point_cloud("pc").add_scalar_quantity("error", diff, enabled=True)
        # ps.show()

        # if i == 6:
        #     pred = model(torch.Tensor(v_mesh_sp).to(device=device)).squeeze().detach().cpu().numpy()
        #     true = f(v_mesh_sp)
        #     const = np.mean(true - pred)
        #     pred = pred + const
        #     diff = np.abs(pred - true)
        #     ps.init()
        #     ps.register_surface_mesh("mesh", v_mesh, f_mesh)
        #     ps.get_surface_mesh("mesh").add_scalar_quantity("error", diff, enabled=True)
        #     ps.show()

    dof = np.array(dof)
    losses = np.array(losses)
    fem_loss = np.array(fem_loss)

    dof_sorted = np.sort(dof)
    losses_sorted = losses[np.argsort(dof)]

    x = dof_sorted
    y = losses_sorted

    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    plt.loglog(x, y, label="L2 Loss", marker="o")
    plt.loglog(x, 1/x, label="1/x", color="red")
    plt.loglog(x, (1/x)**2, label="1/x^2", color="green")
    plt.loglog(fem_dof, fem_loss, label="FEM", marker="o")

    plt.xlabel("DOF")
    plt.ylabel("||true - pred||_2 / N")
    plt.title(f"L2 Loss vs DOF on Sphere, Slope: {coeffs[0]}")
    plt.legend()
    # plt.show()
    if len(sys.argv) > 3:
        plt.savefig(f"convergence_example_{sys.argv[2]}_{sys.argv[3]}.png")
    else:
        plt.savefig(f"convergence_example_{sys.argv[2]}.png")

if __name__ == "__main__":

    if len(sys.argv) > 3 and (sys.argv[3] == "exactvalue" or sys.argv[3] == "zerointegral"):
        if sys.argv[1] == "train":
            # If directory doesn't exist, create it
            if not os.path.exists(f"poisson_results_sphere_{sys.argv[3]}/example{sys.argv[2]}"):
                os.makedirs(f"poisson_results_sphere_{sys.argv[3]}/example{sys.argv[2]}")
            train()
        elif sys.argv[1] == "plot":
            plot()
    elif len(sys.argv) == 3:
        if sys.argv[1] == "train":
            # If directory doesn't exist, create it
            if not os.path.exists(f"poisson_results_sphere/example{sys.argv[2]}"):
                os.makedirs(f"poisson_results_sphere/example{sys.argv[2]}")
            train()
        elif sys.argv[1] == "plot":
            plot()
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")