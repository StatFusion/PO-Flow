import os
import math
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from torchdiffeq import odeint
from torch.nn.utils.stateless import functional_call
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import kagglehub


# === Define Flow Matching Network === #
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return x + self.block(x)

class FlowMatchingNet(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=2, hidden_dim=512):
        super().__init__()
        # 1) a small MLP to turn [t,a] → a hidden feature
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2) the “main” layers
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim // 2),
        ])
        self.output_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y, t, a):
        if a.dim() == 1:
            a = a.unsqueeze(1)
        # cond = [t, a] → [B,2]
        cond = torch.cat([t, a], dim=1)
        # cond_feat: [B, hidden_dim]
        cond_feat = self.cond_mlp(cond)
        x = torch.cat([y, cond], dim=1)              # [B, latent+2]
        x = self.input_layer(x)                     # [B, hidden]
        x = x + cond_feat                            # broadcast‑injected

        for block in self.res_blocks:
            x = block(x) + cond_feat                 # add in cond each block

        return self.output_layer(x)                 # [B, latent_dim]


def train_flow_matching(model, loader, optimizer, device, epochs=50):
    """
    Now with a proper progress bar that knows its own epoch.
    """
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for y, a in pbar:
        y, a = y.to(device), a.to(device)

        # sample t∼U(0,1), z∼N(0,I)
        t = torch.rand(y.size(0), 1, device=device)
        z = torch.randn_like(y)

        # compute y_t and target vector field v = z − y
        y_t = (1 - t) * y + t * z
        v_target = z - y

        # predict with conditioning
        v_pred = model(y_t, t, a)

        loss = F.mse_loss(v_pred, v_target, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pbar.set_postfix(loss=total_loss / len(loader.dataset))

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

# === flow_to_sample Function (ODE) === #
def flow_to_sample(model, y0, a, device, num_steps=50, return_full=True):
    """
    Given noise y0 at t=1, generate sample at t=0.
    Solves dy/ds = -v(y, t=1-s, a) with s ∈ [0, 1]
    
    If return_full is True, return states at every 10 steps.
    """
    model.eval()
    s = torch.linspace(0, 1, num_steps, device=device)
    N, d = y0.shape
    a = a.view(N, 1)

    def ode_func(s_scalar, y_flat):
        y = y_flat.view(N, d)
        t = torch.full((N, 1), 1 - s_scalar.item(), device=device)
        v = model(y, t, a)
        return -v.view(-1)

    with torch.no_grad():
        y0_flat = y0.view(-1)
        y_traj = odeint(ode_func, y0_flat, s, method='dopri5')  # shape: [num_steps, N*d]

        if return_full:
            # Extract every 10 steps (0, 10, 20, ..., last)
            indices = list(range(0, num_steps, 10)) + [num_steps - 1]
            y_selected = y_traj[indices]  # shape: [len(indices), N*d]
            y_selected = y_selected.view(len(indices), N, d)  # reshape to [T, N, d]
            return y_selected  # shape: [T (selected steps), N, d]
        else:
            return y_traj[-1].view(N, d)


def flow_to_noise(model, y_data, a, device, num_steps=50):
    """
    Given data y_data at t=0, map to noise at t=1.
    Solves dy/ds = v(y, t=s, a) with s ∈ [0, 1]
    """
    model.eval()
    N, d = y_data.shape
    a = a.view(N, 1)

    s = torch.linspace(0, 1, num_steps, device=device)

    def ode_func(s_scalar, y_flat):
        y = y_flat.view(N, d)
        t = torch.full((N, 1), s_scalar.item(), device=device)
        v = model(y, t, a)
        return v.view(-1)

    with torch.no_grad():
        y0_flat = y_data.view(-1)
        y_traj = odeint(ode_func, y0_flat, s, method='dopri5')
        return y_traj[-1].view(N, d)


# === Main Script === #
if __name__ == "__main__":
    latent_dim = 512
    perturb = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dir = "CelebA/latent_dataset"
    z_all = np.load(os.path.join(latent_dir, f"z_latents_{latent_dim}_9.npy"))        # (N, 128)
    a_all = np.load(os.path.join(latent_dir, f"gender_labels_{latent_dim}_9.npy"))    # (N, 1)
    tensor_z = torch.from_numpy(z_all).float()
    tensor_a = torch.from_numpy(a_all).float().unsqueeze(1)
    dataset = TensorDataset(tensor_z, tensor_a)
    loader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4, pin_memory=True)
    vae_path = f"pre_trained_VAE/vae_{latent_dim}_celebA_9.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained VAE
    vae_model = VAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()
    
    # Flow matching model
    model = FlowMatchingNet(latent_dim=latent_dim, cond_dim=2, hidden_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    for epoch in range(epochs):
        loss = train_flow_matching(model, loader, optimizer, device, epochs=50)
        print(f"Epoch {epoch+1} Loss: {loss:.6f}")
        
        if (epoch + 1) % 20 == 0:
            factual_data, factual_a = next(iter(loader))
            factual_data = factual_data.to(device)  # [B, 128]
            factual_a    = factual_a.to(device)     # [B, 1]
            mask = (factual_a.squeeze(1) == 0)
            factual_data = factual_data[mask]
            factual_a    = factual_a[mask]
            
            if epoch + 1 == epochs:
                factual_data = factual_data[:100]
                factual_a    = factual_a[:100]
            
            else:
                factual_data = factual_data[:10]  # [N≤10, 128]
                factual_a    = factual_a[:10]     # [N≤10, 1]

            # map factual → noise
            mapped_z = flow_to_noise(model, factual_data, factual_a, device, num_steps=50)
            
            # flip labels to get counterfactual a=1
            counter_a = 1 - factual_a         # [N,1]

            # reverse ODE → get intermediates every 10 steps
            counter_data = flow_to_sample(
                model, mapped_z, counter_a, device,
                num_steps=50, return_full=True
            )
            
            counter_data = torch.cat([factual_data.unsqueeze(0), counter_data], dim=0)
            
            if perturb:
                mapped_z_tilde_1 = mapped_z + 0.3 * torch.randn_like(mapped_z).to(device)
                mapped_z_tilde_2 = mapped_z + 0.3 * torch.randn_like(mapped_z).to(device)

                counter_data_1 = flow_to_sample(
                    model, mapped_z_tilde_1, counter_a, device,
                    num_steps=50, return_full=False
                )
            
                counter_data_2 = flow_to_sample(
                    model, mapped_z_tilde_2, counter_a, device,
                    num_steps=50, return_full=False
                )
                counter_data = torch.cat([
                    counter_data,
                    counter_data_1.unsqueeze(0),  # [1, N, 128]
                    counter_data_2.unsqueeze(0)
                ], dim=0)  # final shape: [T+3, N, 128]
            

            # decode and plot
            T1, N, d = counter_data.shape
            flat_z = counter_data.view(T1 * N, d)
            with torch.no_grad():
                recon_flat = vae_model.decode(flat_z)        # [T1*N, 3, 128, 128]
                recon_flat = (recon_flat * 0.5 + 0.5).clamp(0, 1)
            recon = recon_flat.view(T1, N, 3, 128, 128)      # [T1, N, 3, H, W]

            if epoch + 1 == epochs:
                save_dir = "counterfactual_results"
                for i in range(N):
                    for j in range(T1):
                        img = recon[j, i]
                        img_path = os.path.join(save_dir, f"{i}_{j}.png")
                        vutils.save_image(img, img_path)
                
            else:
                fig, axes = plt.subplots(N, T1, figsize=(T1 * 2, N * 2))
                for i in range(N):
                    for j in range(T1):
                        img = recon[j, i].permute(1, 2, 0).cpu().numpy()
                        axes[i][j].imshow(img)
                        axes[i][j].axis("off")
                plt.suptitle(f"Counterfactual Reconstructions @ Epoch {epoch+1} (only a=0 → a=1)")
                plt.tight_layout()
                plt.show()

    # Save trained model
    save_dir = "NN_params"
    torch.save(model.state_dict(), os.path.join(save_dir, f"CelebA_latent_flow_matching_{latent_dim}.pth"))
    print(f"Flow Matching model saved to {os.path.join(save_dir, f'CelebA_latent_flow_matching_{latent_dim}.pth')}")