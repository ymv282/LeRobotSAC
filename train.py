import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from networks import ActorCritic
from so101_env import SO101SimulationEnv


# --------------------------------
# Hyperparameter
# --------------------------------
gamma = 0.99
tau = 0.005
#alpha = 0.025
lr = 3e-4
batch_size = 64



def train_step(ac, ac_target, buffer, batch_size, 
               actor_optim, critic_optim, log_alpha, alpha_optim, target_alpha, train_alpha):
    
    if len(buffer.buffer) < batch_size:
        return

    s, a, r, s2, d = buffer.sample(batch_size)
    alpha = log_alpha.exp()

    # ----- Critic -----
    #bellman eq
    with torch.no_grad():
        z2 = ac_target.encoder(s2)
        a2, logp2 = ac_target.actor.sample(z2)
        q1_t, q2_t = ac_target.critic(z2, a2)
        q_target = r + gamma * (1 - d) * (torch.min(q1_t, q2_t)- alpha * logp2)

    z = ac.encoder(s)
    q1, q2 = ac.critic(z, a)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # ----- Actor -----
    z = ac.encoder(s)#.detach() encoder soll von actor und critic lernen
    a_new, logp = ac.actor.sample(z) #sample neue actions von actor
    q1_pi, q2_pi = ac.critic(z, a_new)
    actor_loss = (alpha.detach() * logp - torch.min(q1_pi, q2_pi)).mean()#ensure policy is differentiable
    # if len(buffer.buffer) % 1000 == 0:
    #     print(f"Q-mean: {torch.min(q1_pi, q2_pi).mean().item():.4f}, "
    #         f"Log-prob: {logp.mean().item():.4f}, "
    #       f"Actor loss: {actor_loss.item():.4f}")
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()
    if train_alpha:
        
        # ---- Alpha-Update ------
        # Alpha wird optimiert um target_entropy zu erreichen
        alpha_loss = -(log_alpha * (logp.detach() + target_alpha)).mean()
        
        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()
    
    # ----- Target Update -----
    with torch.no_grad():
        for p, pt in zip(ac.encoder.parameters(), ac_target.encoder.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
        for p, pt in zip(ac.critic.parameters(), ac_target.critic.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
        for p, pt in zip(ac.actor.parameters(), ac_target.actor.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
    
    return alpha.item() #fürs logging



