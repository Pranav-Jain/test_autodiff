import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import glob
import os
import gpytoolbox as gpy
import scipy as sp

class Sine(torch.nn.Module):
    def __init__(self, w0 = 1.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class MLP(torch.nn.Module):
    def __init__(self, n=512, n_layers=4, in_dim=2):
        super(MLP, self).__init__()
        net = []
        net.append(torch.nn.Linear(in_dim, n))
        # net.append(Sine())
        net.append(torch.nn.Tanh())
        for _ in range(n_layers):
            net.append(torch.nn.Linear(n, n))
            # net.append(Sine())
            net.append(torch.nn.Tanh())
        net.append(torch.nn.Linear(n, 1))

        self.layers = torch.nn.Sequential(*net)
    
    def forward(self, x):
        return self.layers(x)

def f_2d(x, y):
    if sys.argv[2] == "1":
        z = torch.sin(x) + torch.cos(y)
    elif sys.argv[2] == "2":
        z = x**3 + y**3
    else:
        print("Usage: python test_autodiff_2d.py [train|plot] [1|2]")
        exit()

    return z.squeeze()

def laplacian_f_2d(x, y):
    if sys.argv[2] == "1":
        z = -torch.sin(x) - torch.cos(y)
    elif sys.argv[2] == "2":
        z = 6*x + 6*y
    else:
        print("Usage: python test_autodiff_2d.py [train|plot] [1|2]")
        exit()

    return z.squeeze()

def f_2d_numpy(x, y):
    if sys.argv[2] == "1":
        z = np.sin(x) + np.cos(y)
    elif sys.argv[2] == "2":
        z = x**3 + y**3
    else:
        print("Usage: python test_autodiff_2d.py [train|plot] [1|2]")
        exit()

    return z.squeeze()

def laplacian_f_2d_numpy(x, y):
    if sys.argv[2] == "1":
        z = -np.sin(x) - np.cos(y)
    elif sys.argv[2] == "2":
        z = 6*x + 6*y
    else:
        print("Usage: python test_autodiff_2d.py [train|plot] [1|2]")
        exit()

    return z.squeeze()

def test_FEM(nx, ny):
    V, F = gpy.regular_square_mesh(nx, ny)
    V = V * 10 - 5

    BV = gpy.boundary_vertices(F)

    L = gpy.cotangent_laplacian(V, F)
    M = gpy.massmatrix(V, F).tocsr()
    laplacian = sp.sparse.linalg.inv(M) @ L

    lap_f = laplacian_f_2d_numpy(V[:, 0], V[:, 1])

    # Apply Dirichlet Boundary
    for i in BV:
        laplacian[i, :] = 0.0
        laplacian[i, i] = 1.0
        lap_f[i] = f_2d_numpy(V[i, 0], V[i, 1])

    u = sp.sparse.linalg.lsmr(laplacian, lap_f)[0]
    
    f = f_2d_numpy(V[:, 0], V[:, 1])

    L2loss = np.linalg.norm(u - f, 2)
    Linfloss = np.linalg.norm(u - f, np.inf)

    print(f"L2 loss: {L2loss}")
    print(f"Linf loss: {Linfloss}")

    return L2loss * 1/(nx*ny)

def train_second_order(dim=1, tol=1e-3, max_iter=10000, size_layer=256, n_layers=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = MLP(n=size_layer, n_layers=n_layers, in_dim=dim)
    model.to(device=device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # create a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2000, verbose=True)

    losses = []
    for i in (pbar:= tqdm(range(int(max_iter)))):
        optimizer.zero_grad()
        prev_loss = 1.0

        x = torch.tensor(np.random.uniform(-5, 5, (10000, 2))).to(device=device).requires_grad_(True).float()
        b = torch.tensor(np.random.uniform(-5., 5., (10))).to(device=device).requires_grad_(True).float()
        b1 = torch.stack([b, torch.ones_like(b)*5], dim=1)
        b2 = torch.stack([b, torch.ones_like(b)*-5], dim=1)
        b3 = torch.stack([torch.ones_like(b)*5, b], dim=1)
        b4 = torch.stack([torch.ones_like(b)*-5, b], dim=1)
        true = laplacian_f_2d(x[:, 0], x[:, 1])

        pred = model(x).squeeze()
        grad_pred = torch.autograd.grad(pred, x, torch.ones_like(pred), create_graph=True)[0]
        laplacian_pred = torch.zeros_like(pred)
        for i in range(x.shape[1]):
            grad_grad_pred = torch.autograd.grad(grad_pred[:, i], x, torch.ones_like(grad_pred[:, i]), create_graph=True)[0]
            laplacian_pred += grad_grad_pred[:, i]

        pred_b1 = model(b1).squeeze()
        pred_b2 = model(b2).squeeze()
        pred_b3 = model(b3).squeeze()
        pred_b4 = model(b4).squeeze()
        loss = torch.linalg.norm(laplacian_pred - true, ord=2)**2 
        
        loss += 100*(torch.linalg.norm(pred_b1 - f_2d(b1[:, 0], b1[:, 1]), 2)**2 
                     + torch.linalg.norm(pred_b2 - f_2d(b2[:, 0], b2[:, 1]), 2)**2 
                     + torch.linalg.norm(pred_b3 - f_2d(b3[:, 0], b3[:, 1]), 2)**2 
                     + torch.linalg.norm(pred_b4 - f_2d(b4[:, 0], b4[:, 1]), 2)**2)

        temp_loss = torch.linalg.norm(laplacian_pred - true, ord=2)**2
        pbar.set_description(f"Loss: {temp_loss}")
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if i > 10 and scheduler._last_lr[0] < 1e-7:
            break
        else:
            prev_loss = temp_loss

    plt.figure()
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"dim = {dim}, order = 2, {size_layer} x {n_layers}")
    if sys.argv[2] == "1":
        plt.savefig(f"poisson_results_2d/example1/loss_{size_layer}_{n_layers}.png")
    elif sys.argv[2] == "2":
        plt.savefig(f"poisson_results_2d/example2/loss_{size_layer}_{n_layers}.png")

    x = torch.tensor(np.random.uniform(-5, 5, (10000, 2))).to(device=device).float()
    true = f_2d(x[:, 0], x[:, 1]).detach().cpu().numpy()
    pred = model(x).squeeze().detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], true, label="True", color='blue')
    plt.legend()
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], pred, label="Predicted", color='orange')
    plt.legend()
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], true, label="True", color='blue')
    ax.scatter(x[:, 0], x[:, 1], pred, label="Predicted", color='orange')
    plt.legend()
    if sys.argv[2] == "1":
        plt.savefig(f"poisson_results_2d/example1/pred_{size_layer}_{n_layers}.png")
    elif sys.argv[2] == "2":
        plt.savefig(f"poisson_results_2d/example2/pred_{size_layer}_{n_layers}.png")

    l2_loss = np.linalg.norm(true - pred, 2)

    if sys.argv[2] == "1":
        torch.save(model.state_dict(), f"poisson_results_2d/example1/model_{size_layer}_{n_layers}.pth")
    elif sys.argv[2] == "2":
        torch.save(model.state_dict(), f"poisson_results_2d/example2/model_{size_layer}_{n_layers}.pth")

    return l2_loss

def train():
    n_layers = [5]
    size_layer = 2**np.arange(5, 11, 1)

    for i in n_layers:
        for j in size_layer:
            loss = train_second_order(dim=2, max_iter=1e5, size_layer=j, n_layers=i)


def plot():
    if sys.argv[2] == "1":
        filenames = glob.glob("poisson_results_2d/example1/*.pth")
    elif sys.argv[2] == "2":
        filenames = glob.glob("poisson_results_2d/example2/*.pth")
    else:
        print("Usage: python test_autodiff_2d.py [train|plot] [1|2]")
        exit()

    dof = []
    losses = []

    fem_losses = []
    for file in filenames:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # get layer size and n_layers from filename
        n_layers = int(file.split("_")[-1][:-4])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        x = torch.tensor(np.random.uniform(-5, 5, (100000, 2))).to(device=device).float()
        true = f_2d(x[:, 0], x[:, 1]).detach().cpu().numpy()
        pred = model(x).squeeze().detach().cpu().numpy()

        loss_l2 = np.linalg.norm(true - pred, 2) / x.shape[0]
        loss_linf = np.linalg.norm(true - pred, np.inf)

        print(n_layers, size_layer)
        print(loss_l2, loss_linf)

        dof.append(size_layer*n_layers)
        losses.append(loss_l2)

        fem_dof = int(np.sqrt(size_layer*n_layers))
        fem_loss = test_FEM(fem_dof, fem_dof)
        fem_losses.append(fem_loss)


    dof = np.array(dof)
    losses = np.array(losses)
    fem_losses = np.array(fem_losses)
    
    dof_sorted = np.sort(dof)
    losses_sorted = losses[np.argsort(dof)]
    fem_losses_sorted = fem_losses[np.argsort(dof)]

    x = dof_sorted
    y = losses_sorted

    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    plt.figure()
    plt.loglog(x, y, label="L2 Loss", marker='o')
    plt.loglog(x, 1/x, color='red', label="1/x")
    plt.loglog(x, (1/x)**2, color='green', label="1/x^2")
    plt.loglog(x, fem_losses_sorted, label="FEM", marker='o')

    plt.xlabel("DOF")
    plt.ylabel("L2 Loss")
    plt.title(f"L2 Loss vs DOF in 2D, Slope: {coeffs[0]}")
    plt.legend()
    plt.savefig(f"convergence_{sys.argv[2]}.png")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[1] == "train":
            # If directory doesn't exist, create it
            if sys.argv[2] == "1":
                if not os.path.exists("poisson_results_2d/example1"):
                    os.makedirs("poisson_results_2d/example1")
            elif sys.argv[2] == "2":
                if not os.path.exists("poisson_results_2d/example2"):
                    os.makedirs("poisson_results_2d/example2")
            train()
        elif sys.argv[1] == "plot":
            plot()
    else:
        print("Usage: python test_autodiff_sphere.py [train|plot] [1|2]")