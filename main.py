
import faulthandler
faulthandler.enable()

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
#import imageio
#from torch.utils.tensorboard import SummaryWriter
#import time
import matplotlib.pyplot as plt
from collections import deque



# main.py
# https://github.com/adi3e08/SAC/blob/main/sac.py
# https://spinningup.openai.com/en/latest/algorithms/sac.html
#https://arxiv.org/pdf/1801.01290
# original paper: https://arxiv.org/pdf/1812.05905


from so101_env import SO101SimulationEnv
from networks import ActorCritic
from train import train_step
from logger import log_output
from replayBuffer import ReplayBuffer
import config
## physische daten zum gripper (zentral in config.py)
joint_limits = config.jointLimits
max_r = config.maxReachRadius
lr = config.learningRate
joint_min = joint_limits[:, 0]
joint_max = joint_limits[:, 1]
save_img = True
#Für skalierung
joint_range = (joint_max - joint_min) / 2
joint_center = (joint_max + joint_min) / 2


def SAC(
        scheduler,
        NUM_EPISODES,
        MAX_STEPS,
        LOG_INTERVAL,
        WARMUP_STEPS,
        BATCH_SIZE,
        SAVE_DIR,
        ENABLE_VIEWER=False,
        NO_ML=False,
        CHECKPOIN_INTERVAL=None
        ):
    import numpy as np
    log_output()
    os.makedirs(os.path.join(SAVE_DIR, "gifs"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf Device {device}")
    print(f'Anzahl Episoden: {NUM_EPISODES}|' 
          f'Scheduler: {scheduler}|'
          f'Max Episoden: {NUM_EPISODES}|')

    env = SO101SimulationEnv(enable_viewer=ENABLE_VIEWER)
    ac = ActorCritic(input_channels=config.inputChannels, action_dim=config.actionDim).to(device)
    ac_target = ActorCritic(input_channels=config.inputChannels, action_dim=config.actionDim).to(device)
    ac_target.load_state_dict(ac.state_dict())

    buffer = ReplayBuffer(device=device)
    episode_rewards = []
    global_step = 0
    save_path = config.modelSaveDir
    os.makedirs(save_path, exist_ok=True)
    best_avg = -np.inf 
    training = False
    gif_created = False
    #erreichbarer Arbeitsraum TODO: Überprüfen
    # r = np.random.uniform(0.15, 0.36)
    # phi = np.pi/4 #np.random.uniform(-np.pi/2, np.pi/2)
    # #0.2 0.0 0.025
    # x = 0.2#r * np.cos(phi)
    # y = 0.0#r * np.sin(phi)
    
    # z_ = 0.025#0 #np.random.uniform(0.02, 0.15)
    # erreichbarer Arbeitsraum 

    ### Optimizers ###
    actor_optim  = Adam(ac.actor.parameters(), lr=lr) 
    critic_optim = Adam(list(ac.encoder.parameters()) + list(ac.critic.parameters()), lr=lr)

    # Scheduler
    if scheduler == "reduceOnPlateau":
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            actor_optim, mode='max', factor=config.schedulerFactor, patience=config.schedulerPatience,
            min_lr=config.schedulerMinLr
        )
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            critic_optim, mode='max', factor=config.schedulerFactor, patience=config.schedulerPatience,
            min_lr=config.schedulerMinLr
        )
    elif scheduler == "cosineAnnealing":
        actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, NUM_EPISODES, eta_min=config.schedulerMinLr)
        critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optim, NUM_EPISODES, eta_min=config.schedulerMinLr)
    elif scheduler == "stepLR":
        actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optim, config.stepLrStepSize, config.stepLrGamma)
        critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optim, config.stepLrStepSize, config.stepLrGamma)
    else:
        actor_scheduler = "Static"
        critic_scheduler = "Static"
    action_dim = config.actionDim
    target_alpha = - action_dim
    alpha_init = config.alphaInit  # gewünschter Startwert
    log_alpha = torch.tensor(
    [np.log(alpha_init)],
    requires_grad=True,
    device=device
    )
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = Adam([log_alpha], lr=lr)
    success_fifo = deque(maxlen=config.successFifoMaxlen) 


    def success_rate(fifo):
        return sum(fifo) / len(fifo) if fifo else 0.0
    

    for ep in range(NUM_EPISODES):
        r = np.random.uniform(config.cubeSpawnRadiusMin, config.cubeSpawnRadiusMax)
        phi = np.random.uniform(config.cubeSpawnPhiMin, config.cubeSpawnPhiMax)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z_ = config.cubeSpawnZ
        

        obs = env.reset(x=x, y=y, z=z_)  # Arm konstant, ziel konstant
        
        obs = obs.transpose(2, 0, 1) / 255.0 #formatieren, normalisieren
        done = False
        ep_reward = 0.0
        reward = 0
        frames = []
        max_reward = 0
        for step in range(MAX_STEPS):
            global_step += 1
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                z = ac.encoder(obs_t)
                action, logp = ac.actor.sample(z)
                #print(f"Policy: mu={mu[0].cpu().numpy()}, std={logp.exp()[0].cpu().numpy()}")

            action = action.cpu().numpy()[0]  # in [-1,1]
            action_bound = joint_center + action * joint_range #auf min/maxwerte normalisieren TODO: wird nicht shcon im ENV angepasst??

            next_obs, reward, done, dist_gripper, dist_target, object_pos = env.step(action_bound) 
            next_obs_norm = next_obs.transpose(2, 0, 1) / 255.0
            buffer.add(obs, action, reward, next_obs_norm, done)
            if reward > max_reward:
                max_reward = reward

            ep_reward += reward
            obs = next_obs_norm

            if global_step > WARMUP_STEPS and global_step % config.trainEveryNSteps == 0: #fill replay buffer beofre starting actual training
                if global_step % config.alphaTrainEveryNSteps == 0:
                    train_alpha = True
                else:
                    train_alpha = False
                train_step(
                    ac, ac_target, buffer, BATCH_SIZE, 
                    actor_optim=actor_optim, 
                    critic_optim=critic_optim,
                    alpha_optim=alpha_optim,      
                    log_alpha=log_alpha,          
                    target_alpha=target_alpha,
                    train_alpha=train_alpha
                )
            
            if not gif_created:
                frames.append(env.get_gif_img())
            
                
            if global_step ==1:
                env.save_img()

            if done:                    
                success_fifo.append(1)
                if not gif_created:
                    os.makedirs(config.videoDir, exist_ok=True)
                    make_gif(frames, config.gifPath)
                    gif_created = True
                break
            else:
                success_fifo.append(0)


        episode_rewards.append(ep_reward)
        if ep % 10 == 0:
            avg_step_reward = ep_reward / step  # Durchschnitt pro Step
            avg_50_reward = np.mean(episode_rewards[-50:])
            current_alpha = log_alpha.exp().item()
            
            try:
                print(

                f"[Episode {ep}] Reward: {ep_reward:.2f} | "
                f"Avg(50): {avg_50_reward:.2f} | "
                #f'Step Reward: {avg_step_reward:.3f}|'
                f'Distance Gripper to Object:  {dist_gripper:.3f} | '
                f'Distance Object to Target: {dist_target:.3f}|'
                f'Object Poistion: {np.round(object_pos, 3)}\n'
                #f'Max Reward: {max_reward:.3f}'
                f'Done: {done}|'
                #f'Global Step: {global_step}|'
                #f'Buffer Size: {len(buffer.buffer)}'
                f'Current Alpha: {current_alpha:.2f}|'
                f'Success Rate: {success_rate(success_fifo)}|'
                
                #f'Entropy: {logp}|'
                #f"Actor grad: {ac.actor.fc_mu.weight.grad.abs().mean().item() if ac.actor.fc_mu.weight.grad is not None else 0}\n"
                #f"Action: {np.array2string(action, precision=3)}, Action_bound: {np.array2string(action_bound, precision=3)}|"
                
                )
            except: 
                print("Print failed")
            if len(episode_rewards) >= 50 and scheduler != "Static":
                actor_scheduler.step(avg_reward)
                critic_scheduler.step(avg_reward)
                current_actor_lr = actor_optim.param_groups[0]['lr']
                current_critic_lr = critic_optim.param_groups[0]['lr']
                print(f"LR - Actor: {current_actor_lr:.2e}, Critic: {current_critic_lr:.2e}|\n")
                                
        
            
        avg_reward = np.mean(episode_rewards[-50:])

        if avg_reward > best_avg:
            best_avg = avg_reward
            torch.save(ac.state_dict(), config.bestModelPath)
    env.close()
    return episode_rewards

def make_gif(rgb_images, path, fps=config.gifFps):
    try:
        from PIL import Image
        """
        rgb_images: numpy array oder Liste mit Shape (N, H, W, 3), dtype uint8
        path: Ausgabepfad, z.B. "output.gif"
        fps: Frames pro Sekunde
        """
        frames = [Image.fromarray(img.astype(np.uint8)) for img in rgb_images]
        duration = int(1000 / fps)  # ms pro Frame

        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
    except:
        print("Gif Creation not successful")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC Training für SO101")
    parser.add_argument("--episodes", type=int, default=config.numEpisodes)
    parser.add_argument("--maxsteps", type=int, default=config.maxSteps)
    parser.add_argument("--loginterval", type=int, default=config.logInterval)
    parser.add_argument("--warmup_steps", type=int, default=config.warmupSteps)
    parser.add_argument("--batchsize", type=int, default=config.batchSize)
    parser.add_argument("--save_dir", type=str, default=config.logFolder)
    parser.add_argument("--enable_viewer", action="store_true", default=False)
    parser.add_argument("--no_ml", action="store_true", default=False)
    parser.add_argument("--scheduler", type=str, default="none",
                         choices=config.schedulerChoices)
    args = parser.parse_args()

    ep_rewards = SAC(
        NUM_EPISODES=args.episodes,
        MAX_STEPS=args.maxsteps,
        LOG_INTERVAL=args.loginterval,
        WARMUP_STEPS=args.warmup_steps,
        BATCH_SIZE=args.batchsize,
        SAVE_DIR=args.save_dir,
        ENABLE_VIEWER=args.enable_viewer,
        NO_ML=args.no_ml,
        scheduler = args.scheduler
    )
    t_vec = np.arange(args.episodes)
    plt.figure()
    plt.plot(t_vec, ep_rewards)
    plt.title(f"SAC with {args.scheduler} LR Scheduler and {args.episodes} Episodes ")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")
    plt.grid()
    plt.savefig(config.rewardPlotPath)