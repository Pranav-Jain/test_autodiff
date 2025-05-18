import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import gpytoolbox as gpy
import torch
import glob
import sys


def get_spherical_coordinates(v, c):
    r = np.linalg.norm(v - c, ord=2, axis=1)
    theta = np.arccos((v[:, 2] - c[2]) / r)
    phi = np.sign(v[:, 1] - c[1]) * np.arccos((v[:, 0] - c[0]) / np.sqrt((v[:, 0] - c[0])**2 + (v[:, 1] - c[1])**2))

    return np.stack([r, theta, phi], axis=1)

def f(v, const = 0.0):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[1] == "1":
        f = r**2 * (const - 0.5*np.cos(theta))
    elif sys.argv[1] == "2":
        f = r**2 * (const - 0.5*np.sin(theta)*np.cos(phi))
    else:
        print("Usage: python plot_all.py [1|2]")
        exit()

    return f

def f_torch(v, const = 0.0):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[1] == "1":
        f = r**2 * (const - 0.5*torch.cos(theta))
    elif sys.argv[1] == "2":
        f = r**2 * (const - 0.5*torch.sin(theta)*torch.cos(phi))
    else:
        print("Usage: python plot_all.py [1|2]")
        exit()

    return f

def laplacian_f(v):
    r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

    if sys.argv[1] == "1":
        lap_f = torch.cos(theta)
    elif sys.argv[1] == "2":
        lap_f = torch.sin(theta)*torch.cos(phi)
    else:
        print("Usage: python plot_all.py [1|2]")
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

def plot():
    for iter, dirname in enumerate(["poisson_results_sphere/", "poisson_results_sphere_exactvalue/", "poisson_results_sphere_zerointegral/"]):
        if sys.argv[1] == "1":
            filenames = glob.glob("{}example1/*.pth".format(dirname))
        elif sys.argv[1] == "2":
            filenames = glob.glob("{}example2/*.pth".format(dirname))
        else:
            print("Usage: python plot_all.py [1|2]")
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

        dof = np.array(dof)
        losses = np.array(losses)

        dof_sorted = np.sort(dof)
        losses_sorted = losses[np.argsort(dof)]

        x = dof_sorted
        y = losses_sorted
        if iter == 0:
            plt.loglog(x, y, label="L2 Loss")
        elif iter == 1:
            plt.loglog(x, y, label="L2 Loss + exact value loss")
        elif iter == 2:
            plt.loglog(x, y, label="L2 Loss + zero integral loss")

        if iter == 0:
            plt.loglog(x, 1/x, label="1/x")
            plt.loglog(x, (1/x)**2, label="1/x^2")

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

    fem_loss = np.array(fem_loss)
    plt.loglog(fem_dof, fem_loss, label="FEM")

    plt.xlabel("DOF")
    plt.ylabel("L2 Loss")
    plt.title("L2 Loss vs DOF for Poisson Equation on Sphere")
    plt.legend()
    # plt.show()
    plt.savefig("convergence_example{}_combined.png".format(sys.argv[1]))

if __name__ == "__main__":
    plot()