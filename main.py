import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import plotly.graph_objects as go


class config:
    epochs = 20
    batch_size = 400
    iters_per_epoch = 1000
    lr = 1e-3

    num_steps = 40
    freq_len = 40


def swiss_roll(batch):
    """ Sample from 2D swiss roll
    Inputs:
        batch: int, sample size
    Outputs:
        fea: [B, 2] tensor(float), 2D swiss roll points
    """
    a = torch.rand(batch)
    b = torch.rand(batch)

    theta = a * 10.0
    r = (theta + 0.1 * b)/10.0

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return torch.stack([x, y], dim=-1)


class DiffusionDataset(Dataset):
    def __init__(self, batch, size):
        self.batch = batch
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return swiss_roll(self.batch).cuda()


def betas_for_alpha_bar(num_steps, max_beta=0.999):
    """ squaredcos_cap_v2 schedule
    Inputs:
        num_steps: int, number of diffusion process steps
        max_beta: float, upper bound for beta_t, important for beta_T
    Outputs:
        betas: tensor(float), sequence of diffusion betas
    """
    alpha_bar = np.cos((np.arange(num_steps+1)/num_steps + 0.008) / 1.008 * np.pi / 2) ** 2
    beta = np.minimum(1-alpha_bar[1:]/alpha_bar[:-1], max_beta)
    return torch.tensor(beta, dtype=torch.float32)


class Scheduler:
    def __init__(self, num_steps):
        alpha = 1-betas_for_alpha_bar(num_steps).cuda()
        self.alpha_bar = torch.cumprod(alpha, 0)
        self.timesteps = reversed(torch.arange(num_steps))

    def add_noise(self, x, eps, t):
        """ forward diffusion
        Inputs:
            x:   [B, C] tensor(float) original sample
            eps: [B, C] tensor(float) gaussian noise
            t:   [B]    tensor(int) difusion time step
        Outputs:
            xt:  [B, C] tensor(float) diffused sample
        """
        abar = self.get_alpha_bar(t).unsqueeze(-1)
        xt = torch.sqrt(abar) * x + torch.sqrt(1.0-abar) * eps
        return xt

    def step(self, x, eps, t):
        """ backward denoising
        Inputs:
            x:   [B, C] tensor(float) diffused sample at t
            eps: [B, C] tensor(float) predicted noise
            t:   [B]    tensor(int) difusion time step
        Outputs:
            x_prev:  [B, C] tensor(float) denoised sample
        """
        raise NotImplementedError

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]


class DDPMScheduler(Scheduler):
    """ Denoiser following https://arxiv.org/abs/2006.11239
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, x, eps, t):
        abar = self.get_alpha_bar(t)
        abar_prev = 1.0 if t==0 else self.get_alpha_bar(t-1)
        alpha = abar / abar_prev
        
        variance = (1-abar_prev)/(1-abar) * (1-alpha)
        std_dev_t = variance ** (0.5)
        noise = std_dev_t * torch.randn_like(x)
        
        x_prev = (x - (1-alpha)/(1-abar) ** (0.5) * eps)/alpha ** (0.5) + noise
        return x_prev


class DDIMScheduler(Scheduler):
    """ Denoiser following https://arxiv.org/abs/2010.02502
    """
    def __init__(self, eta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta

    def step(self, x, eps, t):
        abar = self.get_alpha_bar(t)
        abar_prev = 1.0 if t==0 else self.get_alpha_bar(t-1)

        # variance = ((1-abar_prev) / (1-abar)) * (1 - abar / abar_prev)
        variance = 0.0
        std_dev_t = self.eta * variance ** (0.5)
        noise = std_dev_t * torch.randn_like(x)

        x0_pred = (x - (1-abar) ** (0.5) * eps) / abar ** (0.5)
        x_prev = (abar_prev ** (0.5) * x0_pred
                  + (1 - abar_prev - std_dev_t**2) ** (0.5) * eps
                  + noise)
        return x_prev


class DiffusionModel(nn.Module):
    def __init__(self, freq_len, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6*freq_len, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2)
        )

        self.t_freq = torch.tensor(np.power(64, np.linspace(0, -1, freq_len))).float().cuda()
        self.x_freq = torch.tensor(np.power(64, np.linspace(0, -1, freq_len))).float().cuda()

    def forward(self, xt, t):
        """ predict noise epsilon
        Inputs:
            xt: [B, C] tensor(float) diffused sample at t
            t:  [B] tensor(int) diffusion time step
        Outputs:
            eps: [B, C] tensor(float) predicted noise
        """
        pos_enc = xt.unsqueeze(-1) * self.x_freq.expand(1, 1, -1)
        t_enc = t.unsqueeze(-1) * self.t_freq.expand(1, -1)

        fea = torch.concat([pos_enc.flatten(1, 2), t_enc], dim=-1) # [B, *]
        x = torch.concat([torch.cos(fea), torch.sin(fea)], dim=-1)
        return self.mlp(x)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = DiffusionModel(config.freq_len)

    def forward(self, x, scheduler):
        """
        Inputs:
            x: [B, C], original data
            scheduler: scheduler that provides add_noise()
        """
        eps = torch.randn_like(x).to(x.device)
        t = torch.randint(0, config.num_steps, (x.shape[0], )).to(x.device)
        xt = scheduler.add_noise(x, eps, t)
        eps_pred = self.model(xt, t)
        return F.mse_loss(eps, eps_pred)


def train(model, scheduler):
    dataset = DiffusionDataset(config.batch_size, config.iters_per_epoch)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    running_loss = 0.0
    running_weight = 0.0
    for epoch in range(config.epochs):
        for i in range(len(dataset)):
            inputs = dataset[i]
            optimizer.zero_grad()
            loss = model(inputs, scheduler)
            loss.backward()
            optimizer.step()

            running_loss = running_loss * 0.99 + loss.item()
            running_weight = running_weight * 0.99 + 1.0

            if (i+1) % 100 == 0:
                print(f'[epoch {epoch+1}, iter {i+1:5d}] loss: {running_loss / running_weight}')


    print("Finished Training")
    torch.save(model.state_dict(), "model.weights")


def infer(model, scheduler):
    model.load_state_dict(torch.load("model.weights"))

    fig = go.Figure()

    x = torch.randn([500, 2]).cuda()
    for t in scheduler.timesteps:
        tin = t.expand(x.shape[0]).to(x.device)
        eps = model.model(x, tin)

        x = scheduler.step(x, eps, t)

        if t % 10 == 0:
            pts = x.cpu().detach().numpy()
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=str(t)))


    gt = swiss_roll(500).cpu().detach().numpy()
    fig.add_trace(go.Scatter(x=gt[:, 0], y=gt[:, 1], mode="markers", name='gt'))

    fig.update_xaxes(range=[-2, 2])
    fig.update_yaxes(range=[-2, 2])
    fig.show()


def main(mode, schedule_mode):
    model = Model()
    model.cuda()

    if schedule_mode == "ddpm":
        scheduler = DDPMScheduler(num_steps=config.num_steps)
    elif schedule_mode == "ddim":
        scheduler = DDIMScheduler(num_steps=config.num_steps)
    else:
        raise NotImplementedError

    if mode == "train":
        train(model, scheduler)

    if mode == "infer":
        infer(model, scheduler)

if __name__=="__main__":
    import sys
    main(sys.argv[1], sys.argv[2])