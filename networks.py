import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# -----------------------------
# Shared CNN Encoder
# -----------------------------
class CNNEncoder(nn.Module):
    """
    Network to convert RGB image into latent code for further processing of actor and critic networks
    """
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Dummy forward to compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            self._flatten_size = self.conv(dummy).shape[1]
        self.fc = nn.Linear(self._flatten_size, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x


# -----------------------------
# Actor Network
# -----------------------------
class Actor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=6):
        super().__init__()
        self.fc_mu = nn.Linear(latent_dim, action_dim)
        self.fc_log_std = nn.Linear(latent_dim, action_dim)

    def forward(self, x):
        """
        forward step, 
        input: latent code
        output: action space, defined by mu(avg) and sigma(logarithmic standard deviation)
        """
        mu = self.fc_mu(x)           
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -5, 2)  # numeric stability
        return mu, log_std

    def sample(self, x):
        """
        Generates a action by calling self.forward()
        input: latent code
        output: action, log_prob
        """
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.rsample() #sample action from distribution
        action_bound = torch.tanh(action).clamp(-1 + 1e-6,1 - 1e-6) # map action to [-1, 1]
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        #tanh korrektur: log π(a|s) = log µ(u|s) − sum (i=1 -> D)log(1-tanh^2(u_i))
        log_prob -= torch.log(1 - action_bound.pow(2)).sum(-1, keepdim=True)
        return action_bound, log_prob

                # log_prob = dist.log_prob(x_t).sum(1)
                # log_prob -= (2*(np.log(2) - x_t - F.softplus(-2*x_t))).sum(1)


# -----------------------------
# Critic Network (Double Q)
# -----------------------------
class Critic(nn.Module):
    def __init__(self, latent_dim=256, action_dim=6):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, action):
        xa = torch.cat([x, action], dim=-1)
        q1 = self.q1(xa)
        q2 = self.q2(xa)
        return q1, q2

#TODO loss checken
# -----------------------------
# Full Actor-Critic with shared encoder
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_channels=3, action_dim=6, latent_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(input_channels, latent_dim)
        self.actor = Actor(latent_dim, action_dim)
        self.critic = Critic(latent_dim, action_dim)

    def forward(self, obs, action=None):
        z = self.encoder(obs)
        mu, log_std = self.actor(z)
        q1, q2 = None, None
        if action is not None:
            q1, q2 = self.critic(z, action)
        return mu, log_std, q1, q2
