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

def get_spherical_coordinates(v, c):
    r = np.linalg.norm(v - c, ord=2, axis=1)
    theta = np.arccos((v[:, 2] - c[2]) / r)
    phi = np.sign(v[:, 1] - c[1]) * np.arccos((v[:, 0] - c[0]) / np.sqrt((v[:, 0] - c[0])**2 + (v[:, 1] - c[1])**2))

    return np.stack([r, theta, phi], axis=1)

def get_spherical_coordinates_torch(v, c):
    r = torch.linalg.norm(v - c, ord=2, dim=1)
    theta = torch.arccos((v[:, 2] - c[2]) / r)
    phi = torch.sign(v[:, 1] - c[1]) * torch.arccos((v[:, 0] - c[0]) / torch.sqrt((v[:, 0] - c[0])**2 + (v[:, 1] - c[1])**2))

    return torch.stack([r, theta, phi], dim=1)

def f(v, const = 0.0):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[2] == "1":
        f = r**2 * (const - 0.5*np.cos(theta))
    elif sys.argv[2] == "2":
        f = r**2 * (const - 0.5*np.sin(theta)*np.cos(phi))
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")
        exit()

    return f

def f_torch(v, const = 0.0):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[2] == "1":
        f = r**2 * (const - 0.5*torch.cos(theta))
    elif sys.argv[2] == "2":
        f = r**2 * (const - 0.5*torch.sin(theta)*torch.cos(phi))
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")
        exit()

    return f

def laplacian_f(v):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[2] == "1":
        lap_f = torch.cos(theta)
    elif sys.argv[2] == "2":
        lap_f = torch.sin(theta)*torch.cos(phi)
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")
        exit()

    return lap_f

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
        net.append(torch.nn.Linear(in_dim, n))
        net.append(Sine())
        for _ in range(n_layers):
            net.append(torch.nn.Linear(n, n))
            net.append(Sine())
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
    v = v.cpu().detach().numpy()
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


def get_surface_laplacian(model, v, center):
    v = v.requires_grad_(True)

    f_ = model(v).squeeze()

    grad_f = torch.autograd.grad(f_, v, torch.ones_like(f_), create_graph=True, retain_graph=True)[0] # [del_f/del_x, del_f/del_y, del_f/del_z]

    lap = torch.zeros_like(f_)
    center = torch.Tensor(center).to(v.device)

    n = v - center
    n = n / torch.linalg.norm(n, dim=1, keepdim=True)  # Vectorized normalization

    hessians = torch.func.vmap(torch.func.hessian(model))(v)[:, 0] # Batched Hessian
    grad_f_surf = grad_f - torch.sum(grad_f * n, dim=1, keepdim=True) * n
    div_grad_f_surf = torch.autograd.grad(grad_f_surf, v, torch.ones_like(grad_f_surf), create_graph=True, retain_graph=True)[0]
    div_grad_f_surf = torch.sum(div_grad_f_surf, dim=1)
    
    hessian_dot_n = torch.bmm(hessians, n.unsqueeze(-1))  # Batched matrix-vector multiplication
    hessian_dot_n = hessian_dot_n.squeeze(-1)  # Remove the last singleton dimension
    lap = div_grad_f_surf - torch.sum(n * hessian_dot_n, dim=1)
    
    return lap


def test_poisson(model, center, r, device, n):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True)

    for i in (pbar:= tqdm(range(10000))):
        optimizer.zero_grad()

        # v_mesh_rnd = gpy.random_points_on_mesh(v_mesh, f_mesh, n, return_indices=False)
        v_mesh_rnd = sample_in_sphere(n, r)
        v_mesh_rnd = get_spherical_coordinates(v_mesh_rnd, center)

        v_mesh_rnd = torch.tensor(v_mesh_rnd, dtype=torch.float).to(device=device).requires_grad_(True)
        
        laplacian_pred = get_surface_laplacian(model, v_mesh_rnd, center)
        f_v = laplacian_f(v_mesh_rnd)

        ## Loss with extra points ##
        if len(sys.argv) > 3 and sys.argv[3] == "exactvalue":
            extra_pts = v_mesh_rnd[:2]
            loss = (1/n)*torch.linalg.norm(laplacian_pred - f_v, 2)**2 + (100)*torch.linalg.norm(model(extra_pts).squeeze() - f_torch(extra_pts), 2)**2

        ## Loss st the function integrates to zero ##
        elif len(sys.argv) > 3 and sys.argv[3] == "zerointegral":
            extra_pts = sample_in_sphere(10000, r)
            extra_pts = get_spherical_coordinates(extra_pts, center)
            extra_pts = torch.tensor(extra_pts, dtype=torch.float).to(device=device).requires_grad_(True)
            loss = (1/n)*torch.linalg.norm(laplacian_pred - f_v, 2)**2 + (100)*torch.sum(model(extra_pts).squeeze())**2

        else:
            loss = (1/n)*torch.linalg.norm(laplacian_pred - f_v, 2)**2


        # ps.init()
        # # ps.register_surface_mesh("mesh", )
        # ps.register_point_cloud("points", v_emb_rnd)
        # ps.register_point_cloud("mesh", phi(v_emb_rnd_sp).cpu().detach().numpy())
        # ps.get_point_cloud("mesh").add_scalar_quantity("lap_pred", laplacian_pred.cpu().detach().numpy())
        # ps.get_point_cloud("mesh").add_scalar_quantity("f_v", f_v.cpu().detach().numpy())
        # ps.show()
        
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        # if scheduler.state_dict()["_last_lr"][0] < 1e-7:
        #     break

        loss_value = loss.item()
        pbar.set_description(f"Loss: {loss_value}")

    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # n_layers = np.arange(3, 7, 1)
    n_layers = [5]
    size_layer = 2**np.arange(6, 11, 1)
    # size_layer = [256]

    center = np.array([0.0, 0.0, 0.0])
    r = 1.0

    V = sample_in_sphere(10000, 1.0)
    V = get_spherical_coordinates(V, center)

    dof = []
    losses = []
    for i in n_layers:
        for j in size_layer:
            print(f"Processing model with {i} layers of {j} size")

            model = MLP(n=j, n_layers=i)
            model.to(device=device)

            model = test_poisson(model, center, r, device, n=1000)

            l2_loss = np.linalg.norm(f(V) - model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy(), 2) * (1/V.shape[0])
            print(l2_loss)
            losses.append(l2_loss)
            dof.append(2*i*j)

            if len(sys.argv) > 3:
                if sys.argv[2] == "1":
                    torch.save(model.state_dict(), f"poisson_results_sphere_{sys.argv[3]}/example1/model_{j}_{i}.pth")
                elif sys.argv[2] == "2":
                    torch.save(model.state_dict(), f"poisson_results_sphere_{sys.argv[3]}/example2/model_{j}_{i}.pth")
            else:
                if sys.argv[2] == "1":
                    torch.save(model.state_dict(), f"poisson_results_sphere/example1/model_{j}_{i}.pth")
                elif sys.argv[2] == "2":
                    torch.save(model.state_dict(), f"poisson_results_sphere/example2/model_{j}_{i}.pth")

def plot():
    if len(sys.argv) > 3:
        if sys.argv[2] == "1":
            filenames = glob.glob(f"poisson_results_sphere_{sys.argv[3]}/example1/*.pth")
        elif sys.argv[2] == "2":
            filenames = glob.glob(f"poisson_results_sphere_{sys.argv[3]}/example2/*.pth")
    else:
        if sys.argv[2] == "1":
            filenames = glob.glob("poisson_results_sphere/example1/*.pth")
        elif sys.argv[2] == "2":
            filenames = glob.glob("poisson_results_sphere/example2/*.pth")
        else:
            print("Usage: python test_autodiff_sphere.py [train|plot] [1|2] [|exactvalue|zerointegral]")
            exit()
    dof = []
    losses = []

    V = sample_in_sphere(10000, 1.0)
    center = np.mean(V, axis=0)
    V = get_spherical_coordinates(V, center)

    for file in filenames:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        n_layers = int(file.split("_")[-1][:-4])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
        true = f(V)
        const = np.mean(true - pred)
        pred = pred + const

        l2_loss = np.linalg.norm(pred - true) * (1/V.shape[0])
        print(l2_loss, n_layers, size_layer)
        losses.append(l2_loss)
        dof.append(2*n_layers*size_layer)

    fem_loss = []
    fem_dof = []
    for i in range(3, 6):
        v_mesh, f_mesh = gpy.icosphere(i)
        M_mesh = gpy.massmatrix(v_mesh, f_mesh)
        f_ = f(v_mesh)
        
        l = gpy.cotangent_laplacian(v_mesh, f_mesh)
        lap_mesh = sp.sparse.linalg.inv(M_mesh) @ l
        u = sp.sparse.linalg.lsmr(lap_mesh, f_)[0]
        u_interp = get_interpolated_values(u, torch.Tensor(V), v_mesh, f_mesh)
        l2_loss = np.linalg.norm(f(V) - u_interp, 2) * (1/v_mesh.shape[0])
        fem_loss.append(l2_loss)
        fem_dof.append(v_mesh.shape[0])

    dof = np.array(dof)
    losses = np.array(losses)
    fem_loss = np.array(fem_loss)

    dof_sorted = np.sort(dof)
    losses_sorted = losses[np.argsort(dof)]

    x = dof_sorted
    y = losses_sorted
    plt.loglog(x, y, label="L2 Loss")
    plt.loglog(x, 1/x, label="1/x", color="red")
    plt.loglog(x, (1/x)**2, label="1/x^2", color="green")
    plt.loglog(fem_dof, fem_loss, label="FEM")

    plt.xlabel("DOF")
    plt.ylabel("L2 Loss")
    plt.title("L2 Loss vs DOF for Poisson Equation on Sphere")
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
            if sys.argv[2] == "1":
                if not os.path.exists(f"poisson_results_sphere_{sys.argv[3]}/example1"):
                    os.makedirs(f"poisson_results_sphere_{sys.argv[3]}/example1")
            elif sys.argv[2] == "2":
                if not os.path.exists(f"poisson_results_sphere_{sys.argv[3]}/example2"):
                    os.makedirs(f"poisson_results_sphere_{sys.argv[3]}/example2")
            train()
        elif sys.argv[1] == "plot":
            plot()
    elif len(sys.argv) == 3:
        if sys.argv[1] == "train":
            # If directory doesn't exist, create it
            if sys.argv[2] == "1":
                if not os.path.exists("poisson_results_sphere/example1"):
                    os.makedirs("poisson_results_sphere/example1")
            elif sys.argv[2] == "2":
                if not os.path.exists("poisson_results_sphere/example2"):
                    os.makedirs("poisson_results_sphere/example2")
            train()
        elif sys.argv[1] == "plot":
            plot()
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")