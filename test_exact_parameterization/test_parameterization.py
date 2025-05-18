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

usage_msg = "Usage: python test_parameterization_2d_hemisphere.py [train|plot] [1|2|3] [hemisphere|heightfield|ellipsoid] [noNN|withNN] [debug(optional)]"

def get_cartesian_coordinates(v):
    r = torch.ones_like(v[:, 0])
    theta = v[:, 1]
    phi = v[:, 2]

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

def get_spherical_coordinates(v):
    r = np.ones(v.shape[0])
    theta = np.arccos((v[:, 2]) / r)
    phi = np.arctan2(v[:, 1], v[:, 0])

    return np.stack([r, theta, phi], axis=1)

def get_spherical_coordinates_torch(v):
    r = torch.ones(v.shape[0], device=v.device)
    theta = torch.arccos((v[:, 2]) / r)
    if torch.isnan(theta).any():
        theta = torch.nan_to_num(theta)
    phi = torch.arctan2(v[:, 1], v[:, 0])

    r = r.requires_grad_(True)
    theta = theta.requires_grad_(True)
    phi = phi.requires_grad_(True)

    return torch.stack([r, theta, phi], dim=1)

def get_spherical_coordinates_ellipsoid_torch(v):
    a, b, c = 2.0, 3.0, 1.0
    r = torch.ones(v.shape[0], device=v.device)
    theta = torch.arccos((v[:, 2]) / (c*r))
    if torch.isnan(theta).any():
        theta = torch.nan_to_num(theta)
    phi = torch.arctan2(v[:, 1]/b, v[:, 0]/a)

    r = r.requires_grad_(True)
    theta = theta.requires_grad_(True)
    phi = phi.requires_grad_(True)

    return torch.stack([r, theta, phi], dim=1)

def get_rotation_matrix(theta, phi):
    R = torch.zeros(theta.shape[0], 3, 2).to(theta.device)
    R[:, 0, 0] = torch.cos(theta) * torch.cos(phi)
    R[:, 0, 1] = -torch.sin(phi)
    R[:, 1, 0] = torch.cos(theta) * torch.sin(phi)
    R[:, 1, 1] = torch.cos(phi)
    R[:, 2, 0] = -torch.sin(theta)

    return R

def f(v):
    if sys.argv[3] == "hemisphere":
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[2] == "1":
            f = (-r**2/2) * np.cos(theta)
        elif sys.argv[2] == "2":
            f = (-r**2/2) * np.sin(theta) * np.cos(phi)
        elif sys.argv[2] == "3":
            f = (r**2/12600)*(315*np.cos(theta) + 105*np.cos(3*theta) + 63*np.cos(5*theta) + 45*np.cos(7*theta) - 140*np.cos(9*theta))
        elif sys.argv[2] == "4":
            f = (-r**2/6) * np.sin(theta)**2 * np.cos(2*phi)

        else:
            raise NotImplementedError()
    
    elif sys.argv[3] == "heightfield":
        x, y = v[:, 0], v[:, 1]
        if sys.argv[2] == "1":
            f = (-1/2)*np.sin(x)*np.cos(y)
        elif sys.argv[2] == "2":
            f = (-1/(2*np.pi**2))*np.sin(np.pi*x)*np.cos(np.pi*y)
        elif sys.argv[2] == "3":
            f = (-1/(13*np.pi**2))*np.sin(2*np.pi*x)*np.cos(3*np.pi*y)
        elif sys.argv[2] == "4":
            f = (-1/(500*np.pi**2))*np.sin(10*np.pi*x)*np.cos(20*np.pi*y)
        elif sys.argv[2] == "5":
            f = (-1/(6100*np.pi**2))*np.sin(50*np.pi*x)*np.cos(60*np.pi*y)
        else:
            raise NotImplementedError()
        
    elif sys.argv[3] == "ellipsoid":
        f = get_interpolated_values(u_mesh, v, v_mesh, f_mesh)
    else:
        print(usage_msg)
        exit()

    return f.squeeze()

def f_torch(v):
    if sys.argv[3] == "hemisphere":
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]

        if sys.argv[2] == "1":
            f = (-r**2/2) * torch.cos(theta)
        elif sys.argv[2] == "2":
            f = (-r**2/2) * torch.sin(theta) * torch.cos(phi)
        elif sys.argv[2] == "3":
            f = (r**2/12600)*(315*torch.cos(theta) + 105*torch.cos(3*theta) + 63*torch.cos(5*theta) + 45*torch.cos(7*theta) - 140*torch.cos(9*theta))
        elif sys.argv[2] == "4":
            f = (-r**2/6) * torch.sin(theta)**2 * torch.cos(2*phi)
        else:
            raise NotImplementedError()
    
    elif sys.argv[3] == "heightfield":
        x, y = v[:, 0], v[:, 1]
        if sys.argv[2] == "1":
            f = (-1/2)*torch.sin(x)*torch.cos(y)
        elif sys.argv[2] == "2":
            f = (-1/(2*np.pi**2))*torch.sin(np.pi*x)*torch.cos(np.pi*y)
        elif sys.argv[2] == "3":
            f = (-1/(13*np.pi**2))*torch.sin(2*np.pi*x)*torch.cos(3*np.pi*y)
        elif sys.argv[2] == "4":
            f = (-1/(500*np.pi**2))*torch.sin(10*np.pi*x)*torch.cos(20*np.pi*y)
        elif sys.argv[2] == "5":
            f = (-1/(6100*np.pi**2))*torch.sin(50*np.pi*x)*torch.cos(60*np.pi*y)
        else:
            raise NotImplementedError()
    
    elif sys.argv[3] == "ellipsoid":
        v_numpy = v.detach().cpu().numpy()
        f = get_interpolated_values(u_mesh, v_numpy, v_mesh, f_mesh)
        f = torch.tensor(f, device=v.device)
        
    else:
        print(usage_msg)
        exit()

    return f.squeeze()

def laplacian_f(v):
    if sys.argv[3] == "hemisphere":
        r, theta, phi = v[:, 0], v[:, 1], v[:, 2]
        if sys.argv[2] == "1":
            lap_f = torch.cos(theta)
        elif sys.argv[2] == "2":
            lap_f = torch.sin(theta) * torch.cos(phi)
        elif sys.argv[2] == "3":
            lap_f = torch.cos(9*theta)
        elif sys.argv[2] == "4":
            lap_f = torch.sin(theta)**2 * torch.cos(2*phi)
        else:
            raise NotImplementedError()
    
    elif sys.argv[3] == "heightfield":
        x, y = v[:, 0], v[:, 1]
        if sys.argv[2] == "1":
            lap_f = torch.sin(x)*torch.cos(y)
        elif sys.argv[2] == "2":
            lap_f = torch.sin(np.pi*x)*torch.cos(np.pi*y)
        elif sys.argv[2] == "3":
            lap_f = torch.sin(2*np.pi*x)*torch.cos(3*np.pi*y)
        elif sys.argv[2] == "4":
            lap_f = torch.sin(10*np.pi*x)*torch.cos(20*np.pi*y)
        elif sys.argv[2] == "5":
            lap_f = torch.sin(50*np.pi*x)*torch.cos(60*np.pi*y)
        else:
            raise NotImplementedError()
        
    elif sys.argv[3] == "ellipsoid":
        if sys.argv[2] == "1":
            lap_f = v[:, 0]
        elif sys.argv[2] == "2":
            v_sp = get_spherical_coordinates_torch(v)
            r, theta, phi = v_sp[:, 0], v_sp[:, 1], v_sp[:, 2]
            lap_f = torch.cos(theta)
        elif sys.argv[2] == "3":
            v_sp = get_spherical_coordinates_torch(v)
            r, theta, phi = v_sp[:, 0], v_sp[:, 1], v_sp[:, 2]
            lap_f = torch.sin(theta)*torch.cos(phi)
        else:
            raise NotImplementedError()
    
    else:
        print(usage_msg)
        exit()

    return lap_f.squeeze()

class MLP(torch.nn.Module):
    def __init__(self, n=512, n_layers=3, in_dim=3, out_dim=1):
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
        net.append(torch.nn.Linear(n, out_dim))

        self.layers = torch.nn.Sequential(*net)
    
    def forward(self, x):
        return self.layers(x)

def get_random_points(v_mesh, f_mesh, v_emb, f_emb, n):
    v_emb_rdm, ind_emb_rdm, bary_emb_rdm  = gpy.random_points_on_mesh(v_emb, f_emb, n, return_indices=True)
    v_mesh_rdm = np.zeros_like(v_emb_rdm)
    for i in range(v_emb_rdm.shape[0]):
        v = v_mesh[f_emb[ind_emb_rdm[i]]]
        for j in range(v.shape[0]):
            v_mesh_rdm[i] += v[j] * bary_emb_rdm[i][j]

    return v_mesh_rdm, v_emb_rdm

def get_interpolated_values(f, v, v_mesh, f_mesh):
    f_v = np.zeros_like(v[:, 0])

    _, idx, bary = gpy.squared_distance(v, v_mesh, f_mesh, use_cpp=True)
    for i in range(v.shape[0]):
        f_v[i] = np.sum(f[f_mesh[idx[i]]] * bary[i])

    return f_v

def sample_in_domain(n):
    if sys.argv[3] == "hemisphere":
        # Sample uniformly on the unit sphere
        vec = np.random.uniform(-1, 1, (2*n, 3))
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit length

        # Keep only the points in the upper hemisphere (z >= 0)
        vec = vec[vec[:, 2] >= 0]
        return vec
    
    elif sys.argv[3] == "heightfield":
        vec = np.random.uniform(-1, 1, (n, 2))
        z = 0.5 * (vec[:, 0]**2 + vec[:, 1]**2)
        return np.stack([vec[:, 0], vec[:, 1], z], axis=1)
    
    elif sys.argv[3] == "ellipsoid":
        a, b, c = 2.0, 3.0, 1.0
        # Sample uniformly on the unit sphere
        vec = np.random.uniform(-1, 1, (2*n, 3))
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit length
        # Scale to ellipsoid
        vec[:, 0] *= a
        vec[:, 1] *= b
        vec[:, 2] *= c
        return vec

    else:
        print(usage_msg)
        exit()

def get_embedding_torch(v_cart):
    if sys.argv[3] == "hemisphere":
        v_emb = get_spherical_coordinates_torch(v_cart)

    elif sys.argv[3] == "heightfield":
        v_emb = v_cart.clone()
        v_emb[:, 2] = 0.0

    elif sys.argv[3] == "ellipsoid":
        v_emb = v_cart.clone()
        v_emb[:, 0] = v_cart[:, 0] / 2.0
        v_emb[:, 1] = v_cart[:, 1] / 3.0
        v_emb[:, 2] = v_cart[:, 2] / 1.0

    else:
        print(usage_msg)
        exit()

    return v_emb

# Given a point in the embedding space, return the corresponding point on the surface
def parameterize(v_e):
    if sys.argv[3] == "hemisphere":
        v_c = get_cartesian_coordinates(v_e)

    elif sys.argv[3] == "heightfield":
        z = 0.5 * (v_e[:, 0]**2 + v_e[:, 1]**2)
        v_c = torch.stack([v_e[:, 0], v_e[:, 1], z], dim=1)
    
    elif sys.argv[3] == "ellipsoid":
        v_c = v_e.clone()
        v_c[:, 0] = v_e[:, 0] * 2.0
        v_c[:, 1] = v_e[:, 1] * 3.0
        v_c[:, 2] = v_e[:, 2] * 1.0

    else:
        print(usage_msg)
        exit()

    return v_c

def get_surface_laplacian(model, v_cart, v_emb, S_theta=None):

    def model_scalar(v_):
        if len(sys.argv) == 6 and sys.argv[5] == "debug" and sys.argv[3] == "ellipsoid":
            return torch.Tensor(u_mesh).to(v_.device).requires_grad_(True)
        
        return model(v_).view(-1).requires_grad_(True)  # Ensure scalar output
    
    def compute_jacobian(v_e):
        v_c = parameterize(v_e)
        J = torch.zeros(v_c.shape[0], 3, 3).to(v_c.device)
        row1 = torch.autograd.grad(v_c[:, 0], v_e, torch.ones_like(v_c[:, 0]), create_graph=True, retain_graph=True)[0]
        row2 = torch.autograd.grad(v_c[:, 1], v_e, torch.ones_like(v_c[:, 1]), create_graph=True, retain_graph=True)[0]
        row3 = torch.autograd.grad(v_c[:, 2], v_e, torch.ones_like(v_c[:, 2]), create_graph=True, retain_graph=True)[0]
        J[:, 0, :] = row1
        J[:, 1, :] = row2
        J[:, 2, :] = row3
        return J.requires_grad_(True)
    
    if sys.argv[4] == "withNN":
        J = torch.vmap(torch.func.jacrev(S_theta))(v_emb)
    elif sys.argv[4] == "noNN":
        J = compute_jacobian(v_emb)
    else:
        print(usage_msg)
        exit()
    
    if sys.argv[3] == "hemisphere":
        f_ = model_scalar(v_emb)
        
        cross = torch.linalg.cross(J[:, :, 1], J[:, :, 2])
        n = cross / torch.linalg.norm(cross, dim=1, keepdim=True)

    elif sys.argv[3] == "heightfield":
        f_ = model_scalar(v_cart)

        cross = torch.linalg.cross(J[:, 0, :], J[:, 1, :])
        n = cross / torch.linalg.norm(cross, dim=1, keepdim=True)

    elif sys.argv[3] == "ellipsoid":
        v_sp = get_spherical_coordinates_torch(v_emb)
        f_ = model_scalar(v_cart)

        R = get_rotation_matrix(v_sp[:, 1], v_sp[:, 2])
        J_local = torch.bmm(J, R)
        cross = torch.linalg.cross(J_local[:, :, 0], J_local[:, :, 1])
        n = cross / torch.linalg.norm(cross, dim=1, keepdim=True)

        # n_true = 2*v_cart.clone()
        # n_true[:, 0] = v_cart[:, 0]/4.0
        # n_true[:, 1] = v_cart[:, 1]/9.0
        # n_true[:, 2] = v_cart[:, 2]/1.0
        # true_n = n_true / torch.linalg.norm(n_true, dim=1, keepdim=True)

    # ps.init()
    # ps.register_point_cloud("points", v_cart.cpu().detach().numpy())
    # ps.get_point_cloud("points").add_vector_quantity("n", n.cpu().detach().numpy(), enabled=True)
    # ps.get_point_cloud("points").add_vector_quantity("n_true", true_n.cpu().detach().numpy(), enabled=True)
    # ps.show()
    # exit()

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

def get_bdry_points(n, device):
    if sys.argv[3] == "hemisphere":
        phi = torch.rand(n, device=device) * 2 * np.pi - np.pi
        theta = torch.ones_like(phi, device=device) * np.pi/2
        theta[n//2:] = 0.0
        # theta = torch.zeros_like(phi, device=device)
        r = torch.ones_like(phi, device=device)
        bdry_points = torch.stack([r, theta, phi], dim=1)

    elif sys.argv[3] == "heightfield":
        b = torch.tensor(np.random.uniform(-1., 1., (n))).to(device=device).requires_grad_(True).float()
        b1 = torch.stack([b, torch.ones_like(b)*1], dim=1)
        b2 = torch.stack([b, torch.ones_like(b)*-1], dim=1)
        b3 = torch.stack([torch.ones_like(b)*1, b], dim=1)
        b4 = torch.stack([torch.ones_like(b)*-1, b], dim=1)
        bdry_points = torch.cat((b1, b2, b3, b4), dim=0)

        z = 0.5 * (bdry_points[:, 0]**2 + bdry_points[:, 1]**2)

        bdry_points = torch.stack([bdry_points[:, 0], bdry_points[:, 1], z], dim=1)

    return bdry_points.requires_grad_(True)


def test_poisson(l_model, device, n):
    optimizer = torch.optim.Adam(l_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5000)

    if sys.argv[4] == "withNN":
        S_theta = MLP(n=512, n_layers=5, in_dim=3, out_dim=3)
        S_theta.to(device=device)
        if sys.argv[3] == "hemisphere":
            S_theta.load_state_dict(torch.load("parameterization_data/model_2d_hemisphere.pth", weights_only=True, map_location=device))
        elif sys.argv[3] == "heightfield":
            S_theta.load_state_dict(torch.load("parameterization_data/model_2d_hf.pth", weights_only=True, map_location=device))
        elif sys.argv[3] == "ellipsoid":
            S_theta.load_state_dict(torch.load("parameterization_data/model_sphere_ellipsoid.pth", weights_only=True, map_location=device))
        S_theta.requires_grad_(True)
    elif sys.argv[4] == "noNN":
        S_theta = None
    else:
        print(usage_msg)
        exit()

    # losses = []
    for i in (pbar:= tqdm(range(1000000))):
        try:
            optimizer.zero_grad()

            v_cart = sample_in_domain(n)

            v_cart = torch.tensor(v_cart, dtype=torch.float32, device=device).requires_grad_(True)
            
            v_emb = get_embedding_torch(v_cart)

            # ps.init()
            # ps.register_point_cloud("points", v_cart.cpu().detach().numpy())
            # ps.register_point_cloud("points_emb", v_emb.cpu().detach().numpy())
            # ps.show()
            

            if len(sys.argv) == 6 and sys.argv[5] == "debug":
                laplacian_pred = get_surface_laplacian(f_torch, v_cart, v_emb, S_theta)
            else:
                laplacian_pred = get_surface_laplacian(l_model, v_cart, v_emb, S_theta)
            

            true_lap = laplacian_f(v_emb)

            loss = torch.linalg.norm(laplacian_pred - true_lap, 2)**2

            # diff = torch.abs(laplacian_pred - true_lap)
            # ps.init()
            # ps.register_point_cloud("points", v_cart.cpu().detach().numpy())
            # ps.register_point_cloud("points_emb", v_emb.cpu().detach().numpy())
            # ps.get_point_cloud("points").add_scalar_quantity("error", diff.cpu().detach().numpy(), enabled=True)
            # ps.get_point_cloud("points").add_scalar_quantity("true", true_lap.cpu().detach().numpy(), enabled=True)
            # ps.get_point_cloud("points").add_scalar_quantity("pred", laplacian_pred.cpu().detach().numpy(), enabled=True)
            # ps.show()
            
            if len(sys.argv) < 6 and sys.argv[3] != "ellipsoid":
                bdry_points = get_bdry_points(n//10, device)
                loss += 10*torch.linalg.norm(l_model(bdry_points).squeeze() - f_torch(bdry_points), 2)**2 # Dirichlet boundary condition

            # ps.init()
            # ps.register_point_cloud("points", v_cart.cpu().detach().numpy())
            # ps.register_point_cloud("points_emb", v_emb.cpu().detach().numpy())
            # ps.register_point_cloud("points_bdry", bdry_points.cpu().detach().numpy())
            # ps.show()
            # exit()
            
            if len(sys.argv) == 6 and sys.argv[5] == "debug":
                print(f"Loss: {loss.item()}")
                exit()
            
            loss.backward()
            if not torch.isnan(loss):
                torch.nn.utils.clip_grad_norm_(l_model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step(loss)

                loss_value = loss.item()
                pbar.set_description(f"Loss: {loss_value}")
                # losses.append(loss_value)

            if scheduler._last_lr[0] < 1e-6:
                break
        
        except KeyboardInterrupt:
            # exit()
            break
        # exit()
        # if loss_value < 1e-6:
        #     break

    return l_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_layers = [3]
    size_layer = 2**np.arange(4, 11, 1)

    V_cart = sample_in_domain(100000)
    if sys.argv[3] == "hemisphere":
        V = get_spherical_coordinates(V_cart)
    elif sys.argv[3] == "heightfield" or sys.argv[3] == "ellipsoid":
        V = V_cart
    true = f(V)

    dof = []
    for i in n_layers:
        for j in size_layer:
            print(f"Processing model with {i} layers of {j} size")

            lap_model = MLP(n=j, n_layers=i)
            lap_model.to(device=device)

            lap_model = test_poisson(lap_model, device, n=1000)

            pred = lap_model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
            # const = np.mean(true - pred)
            # pred = pred + const
            
            l2_loss = np.linalg.norm(true - pred, 2) #* (1/V.shape[0])
            print(l2_loss)
            dof.append(2*i*j)

            # diff = np.abs(pred - true)
            # ps.init()
            # ps.register_point_cloud("points", V_cart)
            # ps.get_point_cloud("points").add_scalar_quantity("error", diff, enabled=True)
            # ps.get_point_cloud("points").add_scalar_quantity("true", true, enabled=True)
            # ps.get_point_cloud("points").add_scalar_quantity("pred", pred, enabled=True)
            # ps.show()

            # Save the model
            if sys.argv[3] == "hemisphere" or sys.argv[3] == "heightfield":
                torch.save(lap_model.state_dict(), f"poisson_results_parameterization_2d_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}/model_{j}_{i}.pth")
            elif sys.argv[3] == "ellipsoid":
                torch.save(lap_model.state_dict(), f"poisson_results_parameterization_sphere_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}/model_{j}_{i}.pth")

def plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if sys.argv[3] == "hemisphere" or sys.argv[3] == "heightfield":
            filenames = glob.glob(f"poisson_results_parameterization_2d_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}/*.pth")
        elif sys.argv[3] == "ellipsoid":
            filenames = glob.glob(f"poisson_results_parameterization_sphere_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}/*.pth")
    except:
        print(usage_msg)
        exit()
    dof = []
    losses = []

    V_cart = sample_in_domain(100000)
    if sys.argv[3] == "hemisphere":
        V = get_spherical_coordinates(V_cart)
    elif sys.argv[3] == "heightfield" or sys.argv[3] == "ellipsoid":
        V = V_cart

    for file in filenames:
        n_layers = int(file.split("_")[-1][:-4])
        size_layer = int(file.split("_")[-2])

        model = MLP(n=size_layer, n_layers=n_layers, in_dim=3, out_dim=1)
        model.to(device=device)
        model.load_state_dict(torch.load(file, weights_only=True, map_location=device))

        pred = model(torch.Tensor(V).to(device=device)).squeeze().detach().cpu().numpy()
        true = f(V)
        # const = np.mean(true - pred)
        # pred = pred + const

        l2_loss = np.linalg.norm(pred - true, 2) * (1/V.shape[0])
        print(l2_loss, n_layers, size_layer)
        losses.append(l2_loss)
        dof.append(2*n_layers*size_layer)

        # diff = np.abs(pred - true)
        # ps.init()
        # ps.register_point_cloud("points", V_cart)
        # ps.get_point_cloud("points").add_scalar_quantity("error", diff, enabled=True)
        # ps.get_point_cloud("points").add_scalar_quantity("true", true, enabled=True)
        # ps.get_point_cloud("points").add_scalar_quantity("pred", pred, enabled=True)
        # ps.show()
        # exit()

    fem_loss = []
    fem_dof = []
    for iter in range(4):
        if sys.argv[3] == "hemisphere":
            v_mesh, f_mesh = gpy.read_mesh(f"parameterization_data/hemisphere_{iter}.obj")
            v_mesh_sp = get_spherical_coordinates(v_mesh)
            f_ = laplacian_f(torch.Tensor(v_mesh_sp).to(device=device)).detach().cpu().numpy()
        elif sys.argv[3] == "heightfield":
            v_mesh, f_mesh = gpy.read_mesh(f"parameterization_data/heightfield_{iter}.obj")
            v_mesh_sp = v_mesh
            f_ = laplacian_f(torch.Tensor(v_mesh_sp).to(device=device)).detach().cpu().numpy()
        elif sys.argv[3] == "ellipsoid":
            v_mesh, f_mesh = gpy.read_mesh(f"parameterization_data/ellipsoid_{iter}.obj")
            v_mesh_sp = v_mesh
            v_mesh_sp[:, 0] = v_mesh[:, 0] / 2.0
            v_mesh_sp[:, 1] = v_mesh[:, 1] / 3.0
            v_mesh_sp[:, 2] = v_mesh[:, 2] / 1.0
            f_ = laplacian_f(torch.Tensor(v_mesh_sp).to(device=device)).detach().cpu().numpy()

        M_mesh = gpy.massmatrix(v_mesh, f_mesh)
        M_mesh = sp.sparse.csc_matrix(M_mesh)

        BV = gpy.boundary_vertices(f_mesh)
        
        l = gpy.cotangent_laplacian(v_mesh, f_mesh)
        lap_mesh = sp.sparse.linalg.inv(M_mesh) @ l

        # Apply Dirichlet Boundary
        for i in BV:
            lap_mesh[i, :] = 0
            lap_mesh[i, i] = 1
        f_[BV] = f(v_mesh_sp[BV])

        u = sp.sparse.linalg.lsmr(lap_mesh, f_)[0]
        u_interp = get_interpolated_values(u, V, v_mesh, f_mesh)
        l2_loss = np.linalg.norm(f(V) - u_interp, 2) * (1/v_mesh.shape[0])
        fem_loss.append(l2_loss)
        fem_dof.append(v_mesh.shape[0])

        # if iter == 3:
        #     # diff = np.abs(model(torch.Tensor(v_mesh_sp).to(device=device)).squeeze().detach().cpu().numpy() - f(v_mesh_sp))
        #     diff = np.abs(f(v_mesh_sp) - u)
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

    #fit a line to the data
    coeffs = np.polyfit(np.log(x), np.log(y), 1)

    plt.loglog(x, y, label="L2 Loss", marker='o')
    plt.loglog(x, 1/x, label="1/x", color="red")
    plt.loglog(x, (1/x)**2, label="1/x^2", color="green")
    plt.loglog(fem_dof, fem_loss, label="FEM", marker='o')

    plt.xlabel("DOF")
    plt.ylabel("||pred - true||_2/n")
    plt.title(f"L2 Loss vs DOF, Slope: {coeffs[0]}")
    plt.legend()
    # plt.show()
    if sys.argv[3] == "hemisphere" or sys.argv[3] == "heightfield":
        plt.savefig(f"poisson_results_parameterization_2d_{sys.argv[3]}_{sys.argv[4]}/convergence_example_{sys.argv[2]}.png")
    elif sys.argv[3] == "ellipsoid":
        plt.savefig(f"poisson_results_parameterization_sphere_{sys.argv[3]}_{sys.argv[4]}/convergence_example_{sys.argv[2]}.png")

if __name__ == "__main__":
    if sys.argv[3] == "ellipsoid":
        print("Computing cotangent laplacian for ellipsoid ...")
        mesh_name = "parameterization_data/ellipsoid_4.obj"
        v_mesh, f_mesh = gpy.read_mesh(mesh_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        v_emb = np.copy(v_mesh)
        v_emb[:, 0] = v_mesh[:, 0] / 2.0
        v_emb[:, 1] = v_mesh[:, 1] / 3.0
        v_emb[:, 2] = v_mesh[:, 2] / 1.0
        lap_f_ = laplacian_f(torch.Tensor(v_emb).to(device=device)).detach().cpu().numpy()

        M_mesh = gpy.massmatrix(v_mesh, f_mesh)
        M_mesh = sp.sparse.csc_matrix(M_mesh)
        
        l = gpy.cotangent_laplacian(v_mesh, f_mesh)
        lap_mesh = sp.sparse.linalg.inv(M_mesh) @ l

        u_mesh = sp.sparse.linalg.lsmr(lap_mesh, -lap_f_)[0]
        print("Cotangent laplacian computed")
        print("Residual norm:", np.linalg.norm(lap_mesh @ u_mesh + lap_f_))

        # ps.init()
        # ps.register_surface_mesh("mesh", v_mesh, f_mesh)
        # ps.get_surface_mesh("mesh").add_scalar_quantity("lap", lap_f_, enabled=True)
        # ps.get_surface_mesh("mesh").add_scalar_quantity("u", u_mesh, enabled=True)
        # ps.show()


    if sys.argv[1] == "train":
        if sys.argv[3] == "hemisphere" or sys.argv[3] == "heightfield":
            # If directory doesn't exist, create it
            if not os.path.exists(f"poisson_results_parameterization_2d_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}"):
                    os.makedirs(f"poisson_results_parameterization_2d_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}")
        elif sys.argv[3] == "ellipsoid":
            # If directory doesn't exist, create it
            if not os.path.exists(f"poisson_results_parameterization_sphere_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}"):
                os.makedirs(f"poisson_results_parameterization_sphere_{sys.argv[3]}_{sys.argv[4]}/example{sys.argv[2]}")
        else:
            print(usage_msg)
            exit()
        train()
    elif sys.argv[1] == "plot":
        plot()
    else:
        print(usage_msg)
        exit()