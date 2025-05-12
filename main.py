import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch.nn.utils.stateless import functional_call
from itertools import cycle
from matplotlib.cm import get_cmap
import os
import math
from sklearn.preprocessing import LabelEncoder
import argparse
import yaml
from argparse import Namespace
import gc
from utils import CounterfactualDatasetBuilder, wasserstein_dist

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FiLM(nn.Module):
    def __init__(self, cond_dim, target_dim):
        super().__init__()
        self.gamma_layer = nn.Linear(cond_dim, target_dim)
        self.beta_layer = nn.Linear(cond_dim, target_dim)

    def forward(self, x, cond):
        gamma = self.gamma_layer(cond)
        beta = self.beta_layer(cond)
        return gamma * x + beta

    
class ResidualBlockFlow(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        )

    def forward(self, flow_input, cond):
        y = torch.cat([flow_input, cond], dim=1)
        out = self.mlp(y)
        gate, filt = out.chunk(2, dim=1)
        gated = torch.sigmoid(gate) * torch.tanh(filt)
        new_x = (flow_input + gated) / math.sqrt(2.0)
        skip = gated
        return new_x, skip

class FlowMatchingNet(nn.Module):
    def __init__(self, y_dim=1, x_dim=177, hidden_dim=128):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.cond_dim = x_dim + 1
        self.hidden_dim = hidden_dim

        self.y_embedding = nn.Linear(y_dim, self.cond_dim)
        self.film = FiLM(cond_dim=self.cond_dim, target_dim=self.cond_dim)

        self.residual_layers = nn.ModuleList([
            ResidualBlockFlow(hidden_dim=self.cond_dim, cond_dim=self.cond_dim)
            for _ in range(2)
        ])

        self.embed_projection = nn.Linear(self.cond_dim, hidden_dim)
        
        self.prediction_v0 = nn.Linear(hidden_dim, y_dim)
        self.prediction_v1 = nn.Linear(hidden_dim, y_dim)
        #self.output_projection = nn.Linear(hidden_dim, 2 * y_dim)

    def forward(self, y, t, a, x):
        if a.dim() == 1:
            a = a.unsqueeze(1)
        cond = torch.cat([x, a], dim=1)
        embed_y = self.y_embedding(y)
        flow_input = self.film(embed_y, cond)
        flow_input = F.relu(flow_input)

        skip_list = []
        for block in self.residual_layers:
            flow_input, skip = block(flow_input, cond)
            # flow_input, skip = block(flow_input, x)
            skip_list.append(skip)

        skip_sum = torch.stack(skip_list, dim=0).sum(dim=0) / math.sqrt(len(self.residual_layers))
        hidden = F.relu(self.embed_projection(skip_sum))
        # output = self.output_projection(hidden)
        # v0, v1 = output.chunk(2, dim=1)
        v0 = self.prediction_v0(hidden)
        v1 = self.prediction_v1(hidden)
        return v0, v1

    

def train_flow_matching(model, dataloader, optimizer, device, propnet=None):
    model.train()
    total_loss = 0.0
    count = 0

    for y_data, a, x in dataloader:
        y_data = y_data.to(device)
        a = a.to(device).float().unsqueeze(1)
        x = x.to(device)
        batch_size = y_data.size(0)
        count += batch_size

        z = torch.randn(batch_size, model.y_dim, device=device)
        t = torch.rand(batch_size, 1, device=device)

        y_t = (1 - t) * y_data + t * z
        v_target = z - y_data

        v0_pred, v1_pred = model(y_t, t, a, x)
        v_pred = torch.where(a == 0, v0_pred, v1_pred)

        # Propensity-based sample weighting
        weights = None
        if propnet is not None:
            with torch.no_grad():
                pi_hat = propnet(x).detach()
                
                weights = (a / pi_hat[:, 1:2]) + ((1 - a) / pi_hat[:, 0:1])
                weights = torch.clamp(weights, min=0.1, max=10.0)

        fm_loss = ((v_pred - v_target) ** 2)
        if weights is not None:
            fm_loss = (weights * fm_loss).mean()
        else:
            fm_loss = fm_loss.mean()
            
        loss = fm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

    avg_loss = total_loss / count
    print(f"Training Loss (dual): {avg_loss:.4f}")

def get_e_ls(y, num_e):
    """
    Pre‐draw num_e random noise vectors for the Hutchinson estimator.
    y: Tensor of shape [batch, 1]
    returns: list of Tensors, each same shape as y
    """
    return [torch.randn_like(y) for _ in range(num_e)]


def divergence_approx(v, y, e_ls, t, net):
    """
    Hutchinson approximation of tr(∂v/∂y) without brute force.
    v: Tensor [batch,1]
    y: Tensor [batch,1]
    e_ls: list of noise Tensors [batch,1]
    t: Tensor [batch,1]
    net: callable (y_in, t_in) -> v_pred [batch,1]
    returns: div_est [batch], jac_norm placeholder [batch]
    """
    approx_tr = []
    for e in e_ls:
        sigma = 0.01 / torch.sqrt(torch.tensor(y.numel(), dtype=torch.float32))
        v_e = net(y + sigma * e, t)
        e_dvdy = (v_e - v) / sigma
        approx_tr.append((e_dvdy * e).view(e.shape[0], -1).sum(dim=1, keepdim=True))
    approx_tr = torch.cat(approx_tr, dim=1).mean(dim=1)  # shape [batch]
    return approx_tr, torch.zeros_like(approx_tr)

def flow_to_sample(model, y0, a, x, device, train=True, nll=False, num_steps=50, num_e=2):
    """
    Reverse ODE + log‑density tracking with Hutchinson divergence_approx.
    Returns: (y_final [batch,1], logp_final [batch])
    """
    s = torch.linspace(0, 1, num_steps, device=device)
    N = y0.shape[0]
    a = a.view(N, 1)
    x = x.view(N, -1)

    if train:
        # same as before
        def ode_func(t_scalar, y_flat):
            y = y_flat.view(N,1)
            tt = torch.full((N,1), 1 - t_scalar.item(), device=device)
            v0, v1 = model(y, tt, a, x)
            v = torch.where(a==0, v0, v1)
            return -v.view(-1)

        y0_flat = y0.view(-1)
        y_traj = odeint(ode_func, y0_flat, s, method='dopri5')
        return y_traj[-1].view(N,1)

    if not nll:
        model.eval()
        with torch.no_grad():
            def ode_func(t, y):
                t_tensor = torch.full((N,1), 1 - t, device=device)
                v0, v1 = model(y, t_tensor, a, x)
                v = torch.where(a==0, v0, v1)
                return -v
            y_traj = odeint(ode_func, y0, s, method='dopri5')
        return y_traj[-1]

    model.eval()
    y_flat = y0.clone().detach().view(N,1).requires_grad_(True)
    logp_flat = -0.5 * (y_flat**2 + math.log(2*math.pi))
    e_ls = get_e_ls(y_flat, num_e)

    def net_v(yy, tt):
        vv0, vv1 = model(yy, tt, a, x)
        return torch.where(a==0, vv0, vv1)

    def ode_func(t_scalar, z_flat):
        y_part = z_flat[:N].view(N,1)
        logp_part = z_flat[N:]
        tt = torch.full((N,1), 1 - t_scalar.item(), device=device)

        v = net_v(y_part, tt)           # [N,1]
        dy = -v.view(-1)                # [N]

        div_est, _ = divergence_approx(v, y_part, e_ls, t=tt, net=net_v)
        dlogp = -div_est                # [N]

        return torch.cat([dy, dlogp], dim=0)

    z0 = torch.cat([y_flat.view(-1), logp_flat.view(-1)], dim=0)
    zT = odeint(ode_func, z0, s, method='dopri5')[-1]
    y_T = zT[:N].view(N,1)
    logp_T = zT[N:]
    return y_T, logp_T

def flow_to_noise(model, y_data, a, x, device, train=True, num_steps=50):
    """
    Given a data sample y_data (at t=0) and class label a,
    integrate the forward ODE:
        dy/ds = v(y, s, a, x)
    from s = 0 (t=0) to s = 1 (t=1).
    """
    s = torch.linspace(0, 1, num_steps, device=device)
    batch_size = y_data.shape[0]
    a = a.view(batch_size, 1)
    x = x.view(batch_size, -1)

    if train:
        def ode_func(s_scalar, y_flat):
            y = y_flat.view(batch_size, 1)
            t_tensor = torch.full((batch_size, 1), s_scalar.item(), device=device)

            v0, v1 = model(y, t_tensor, a, x)
            v = torch.where(a == 0, v0, v1)
            return v.view(-1)

        y0_flat = y_data.view(-1)
        y_traj_flat = odeint(ode_func, y0_flat, s, method='dopri5')
        y_traj = y_traj_flat[-1].view(batch_size, 1)
        
    else:
        model.eval()
        with torch.no_grad():
            def ode_func(s_scalar, y_flat):
                y = y_flat.view(batch_size, 1)
                t_tensor = torch.full((batch_size, 1), s_scalar.item(), device=device)

                v0, v1 = model(y, t_tensor, a, x)
                v = torch.where(a == 0, v0, v1)
                return v.view(-1)

            y0_flat = y_data.view(-1)
            y_traj_flat = odeint(ode_func, y0_flat, s, method='dopri5')
            y_traj = y_traj_flat[-1].view(batch_size, 1)

    return y_traj


parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--PO_Flow_config', default = 'ACIC2018.yaml', type=str, help='Path to the YAML file')

args_parsed = parser.parse_args()
with open(args_parsed.PO_Flow_config, 'r') as file:
    args_yaml = yaml.safe_load(file)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    data = args_yaml['data']['name']
    df = pd.read_csv(f"datasets/{data}_sample.csv")
    x_dim = df.shape[1] - 5
    y_dim = args_yaml['data']['Ydim']

    builder = CounterfactualDatasetBuilder(df)
    train_dataset, test_dataset = builder.build()

    batch_size = args_yaml['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    model = FlowMatchingNet(y_dim=y_dim,
                            x_dim=x_dim, hidden_dim=128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = args_yaml['training']['epochs']
    test_epochs = args_yaml['test']['epochs']
    
    conv_counterfactual_RMSE, conv_pehe, conv_po_RMSE, conv_wass = [], [], [], []
    
    best_counterfactual_RMSE, best_po_rmse = float('inf'), float('inf')

    
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}")
        train_flow_matching(model, train_loader, optimizer, device, propnet=None)
        
        if epoch % test_epochs == 0:
            print(f"########### Testing at epoch {epoch} ###########")
            model.eval()
            all_factual, all_est_counter, all_true_counter = [], [], []
            all_y0_hat, all_y1_hat, all_y0_true, all_y1_true = [], [], [], []
            all_log_y0_hat, all_log_y1_hat = [], []
            all_mu0_true, all_mu1_true = [], []
            kl0_all, kl1_all = [], []
            full_a_batch = []

            with torch.no_grad():
                for batch in test_loader:
                    y_true, a_batch, x_batch, counter_y, mu0, mu1 = [b.to(device) for b in batch]
                    counter_y = counter_y.unsqueeze(1)
                    a0, a1 = torch.zeros_like(a_batch), torch.ones_like(a_batch)
                    counter_a_batch = 1 - a_batch

                    # Counterfactuals Prediction
                    z_factual = flow_to_noise(model, y_true, a_batch, x_batch, device, train=False, num_steps=100)
                    mapped_counter_y = flow_to_sample(model, z_factual, counter_a_batch, x_batch, device, train=False, num_steps=100)

                    if epoch == args_yaml['test']['KL_metric_epoch'] and args_yaml['test']['if_test_KL_metric']:
                        print(f"########### Testing KL divergence at epoch {epoch} ###########")
                        print("This requires numerical integration of divergence of flow, which is time-consuming.")
                        print("Please be patient and wait for the results.")
                        B, n_samples = x_batch.size(0), 100
                        z = torch.randn(B * n_samples, 1, device=device)
                        x_rep = x_batch.repeat_interleave(n_samples, dim=0)
                        a0_rep = a0.repeat_interleave(n_samples).view(-1, 1)
                        a1_rep = a1.repeat_interleave(n_samples).view(-1, 1)
                        mu0_rep = mu0.unsqueeze(1).repeat(1, n_samples).view(-1, 1)
                        mu1_rep = mu1.unsqueeze(1).repeat(1, n_samples).view(-1, 1)

                        y0_hat_matrix, log_q_y0 = flow_to_sample(model, z, a0_rep, x_rep, device, train=False, nll=True)
                        y1_hat_matrix, log_q_y1 = flow_to_sample(model, z, a1_rep, x_rep, device, train=False, nll=True)

                        log_p_y0 = -0.5 * ((y0_hat_matrix - mu0_rep)**2 + np.log(2 * np.pi))
                        log_p_y1 = -0.5 * ((y1_hat_matrix - mu1_rep)**2 + np.log(2 * np.pi))

                        kl0 = (log_q_y0 - log_p_y0).view(B, n_samples).mean(dim=1)
                        kl1 = (log_q_y1 - log_p_y1).view(B, n_samples).mean(dim=1)

                        kl0_all.append(kl0)
                        kl1_all.append(kl1)

                        y0_hat = y0_hat_matrix.view(B, n_samples).mean(dim=1)
                        y1_hat = y1_hat_matrix.view(B, n_samples).mean(dim=1)
                        log_y0_hat = log_q_y0.view(B, n_samples).mean(dim=1)
                        log_y1_hat = log_q_y1.view(B, n_samples).mean(dim=1)
                    else:
                        z_rand = torch.randn(x_batch.size(0), model.y_dim, device=device)
                        y0_hat = flow_to_sample(model, z_rand, a0, x_batch, device, train=False)
                        y1_hat = flow_to_sample(model, z_rand, a1, x_batch, device, train=False)
                        log_y0_hat = torch.zeros(x_batch.size(0), device=device)
                        log_y1_hat = torch.zeros(x_batch.size(0), device=device)

                    a_float = a_batch.float().unsqueeze(1)
                    y0_true = y_true * (1 - a_float) + counter_y * a_float
                    y1_true = counter_y * (1 - a_float) + y_true * a_float

                    # Accumulate
                    full_a_batch.append(a_batch)
                    all_factual.append(y_true.view(-1, 1).cpu())
                    all_est_counter.append(mapped_counter_y.view(-1, 1).cpu())
                    all_true_counter.append(counter_y.view(-1, 1).cpu())
                    all_y0_hat.append(y0_hat.detach().cpu())
                    all_y1_hat.append(y1_hat.detach().cpu())
                    all_log_y0_hat.append(log_y0_hat.detach().cpu())
                    all_log_y1_hat.append(log_y1_hat.detach().cpu())
                    all_mu0_true.append(mu0.detach().cpu())
                    all_mu1_true.append(mu1.detach().cpu())
                    all_y0_true.append(y0_true.detach().cpu())
                    all_y1_true.append(y1_true.detach().cpu())
                    torch.cuda.empty_cache()

            # === Stack and Evaluate ===
            cat = lambda x: torch.cat(x, dim=0)
            est_cf, true_cf = cat(all_est_counter).to(device), cat(all_true_counter).to(device)
            y0_hat, y1_hat = cat(all_y0_hat).to(device), cat(all_y1_hat).to(device)
            mu0, mu1 = cat(all_mu0_true).to(device), cat(all_mu1_true).to(device)
            y0_true, y1_true = cat(all_y0_true).to(device), cat(all_y1_true).to(device)
            log_y0_hat, log_y1_hat = cat(all_log_y0_hat).to(device), cat(all_log_y1_hat).to(device)
            a_full = torch.cat(full_a_batch).float().to(device)

            # Counterfactuals
            l2 = F.mse_loss(est_cf, true_cf)
            counterfactual_RMSE = torch.sqrt(l2)

            # PEHE
            pehe = torch.sqrt(F.mse_loss((y1_hat - y0_hat).view(-1), (mu1 - mu0).view(-1)))

            # Potential Outcome RMSE/RRMSE
            y_hat_po = torch.where(a_full == 0, y0_hat, y1_hat)
            y_true_po = torch.where(a_full == 0, y0_true, y1_true)
            po_mse = F.mse_loss(y_hat_po.view(-1), y_true_po.view(-1))
            po_rmse = torch.sqrt(po_mse)

            # Other metrics
            wass_dist = 0.5 * (wasserstein_dist(y0_hat, mu0) + wasserstein_dist(y1_hat, mu1))

            # Print
            print(f"[Epoch {epoch}] Counterfactual RMSE: {counterfactual_RMSE:.4f}")
            print(f"PEHE: {pehe:.4f}")
            print(f"PO RMSE: {po_rmse:.4f}")
            print(f"Wasserstein: {wass_dist:.4f}")
            if args_yaml['test']['if_test_KL_metric']:
                if epoch == args_yaml['test']['KL_metric_epoch']:
                    print(f"KL(Q||P): y0 = {torch.cat(kl0_all).mean():.4f}, y1 = {torch.cat(kl1_all).mean():.4f}")

            # Append metrics
            it = epoch * len(train_dataset) // batch_size
            conv_counterfactual_RMSE.append((it, counterfactual_RMSE.item()))
            conv_pehe.append((it, pehe.item()))
            conv_po_RMSE.append((it, po_rmse.item()))
            conv_wass.append((it, wass_dist))

            # === Save Best Performance ===
            if counterfactual_RMSE.item() < best_counterfactual_RMSE:
                best_counterfactual_RMSE = counterfactual_RMSE.item()
            if po_rmse.item() < best_po_rmse:
                best_po_rmse = po_rmse.item()


            # === Histogram Plot ===
            # plt.figure(figsize=(8, 6))
            # plt.hist(all_est_counter.cpu().numpy(), bins=30, alpha=0.6, label='Estimated Counterfactual')
            # plt.hist(all_true_counter.cpu().numpy(), bins=30, alpha=0.6, label='True Counterfactual')
            # plt.title(f"Epoch {epoch}: Estimated vs True Counterfactuals")
            # plt.xlabel("Outcome")
            # plt.ylabel("Frequency")
            # plt.legend()
            # plt.grid(True)
            # plt.show()

        # === Save final metrics and model === 
        if epoch == n_epochs:
            os.makedirs("results", exist_ok=True)  # Ensure the results/ folder exists

            df = pd.DataFrame({
                "PEHE": [v for _, v in conv_pehe],
                "PO_RMSE": [v for _, v in conv_po_RMSE],
                "counterfactual_RMSE": [v for _, v in conv_counterfactual_RMSE],
                "W1_distance": [v for _, v in conv_wass],
            })
            out_path = f"results/{data}_metrics.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved metrics to: {out_path}")
    
    print(f"Counterfactual RMSE: {best_counterfactual_RMSE:.4f}")
    print(f"PO RMSE: {best_po_rmse:.4f}")

    # Save model
    save_dir = "NN_params"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{data}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")