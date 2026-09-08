# test.py
import os
import numpy as np
import torch
import time
from so101_env import SO101SimulationEnv
from networks import ActorCritic
import config

def test_trained_model(
    model_path=config.testModelPath,
    num_episodes=config.testNumEpisodes,
    max_steps=config.testMaxSteps,
    cube_x=config.defaultCubeX,
    cube_y=config.defaultCubeY,
    cube_z=config.defaultCubeZ
):
    """
    Testet ein trainiertes Modell im MuJoCo Viewer
    
    Args:
        model_path: Pfad zum gespeicherten Modell
        num_episodes: Anzahl Test-Episoden
        max_steps: Maximale Steps pro Episode
        cube_x, cube_y, cube_z: Feste Cube-Position
    """
    
    print("="*60)
    print("TESTING TRAINED SAC MODEL")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Joint limits (für Action-Denormalisierung, zentral in config.py)
    joint_limits = config.jointLimits
    
    joint_min = joint_limits[:, 0]
    joint_max = joint_limits[:, 1]
    joint_range = (joint_max - joint_min) / 2
    joint_center = (joint_max + joint_min) / 2
    
    # Environment mit Viewer starten
    print("\nStarting MuJoCo Viewer...")
    env = SO101SimulationEnv(enable_viewer=True)
    
    # Modell laden
    print(f"\nLoading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please train a model first or check the path.")
        env.close()
        return
    
    # Initialisiere Netzwerk
    ac = ActorCritic(input_channels=config.inputChannels, action_dim=config.actionDim).to(device)
    
    # Lade Weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Falls checkpoint ein Dict ist (mit zusätzlichen Infos)
        if isinstance(checkpoint, dict) and 'ac_state_dict' in checkpoint:
            ac.load_state_dict(checkpoint['ac_state_dict'])
            print("Loaded checkpoint with metadata")
        else:
            ac.load_state_dict(checkpoint)
            print("Loaded model weights")
            
    except Exception as e:
        print(f"ERROR loading model: {e}")
        env.close()
        return
    
    ac.eval()  # Evaluation mode (keine Dropout etc.)
    print("Model loaded successfully!")
    
    print(f"\nCube position: ({cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f})")
    print(f"\nRunning {num_episodes} test episodes...")
    print("Press Ctrl+C to stop early\n")
    
    # Statistiken
    success_count = 0
    total_rewards = []
    
    try:
        for ep in range(num_episodes):
            print("="*60)
            print(f"EPISODE {ep + 1}/{num_episodes}")
            print("="*60)
            
            # Reset Environment
            obs = env.reset(x=cube_x, y=cube_y, z=cube_z)
            obs = obs.transpose(2, 0, 1) / 255.0  # Normalize
            
            # Initial State
            gripper_pos = env.get_gripper_position()
            cube_pos = env.get_cube_position()
            target_pos = env.get_target_position()
            
            print(f"Initial State:")
            print(f"  Gripper: {gripper_pos}")
            print(f"  Cube:    {cube_pos}")
            print(f"  Target:  {target_pos}")
            
            reward, done, distGripper, distTarget, objectPos = env.compute_reward()
            print(f"  Initial Reward: {reward:.4f}")
            print(f"  Distance Gripper→Object: {distGripper:.4f} m")
            print(f"  Distance Object→Target:  {distTarget:.4f} m")
            print()
            
            ep_reward = 0.0
            step = 0
            
            while step < max_steps and env.viewer.is_running():
                step += 1
                
                # Forward pass durch Netzwerk
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    z = ac.encoder(obs_t)
                    action, _ = ac.actor.sample(z)
                
                # Action denormalisieren
                action = action.cpu().numpy()[0]
                action_bound = joint_center + action * joint_range
                
                # Step in Environment
                next_obs, reward, done, distGripper, distTarget, objectPos = env.step(action_bound)
                next_obs_norm = next_obs.transpose(2, 0, 1) / 255.0
                
                ep_reward += reward
                obs = next_obs_norm
                
                # Status alle 20 Steps
                if step % 20 == 0:
                    print(f"[Step {step:3d}] "
                          f"Reward: {reward:+.3f} | "
                          f"Total: {ep_reward:+.2f} | "
                          f"Dist G→O: {distGripper:.3f} | "
                          f"Dist O→T: {distTarget:.3f}")
                
                # Langsamer für Beobachtung
                time.sleep(config.testStepSleepSeconds)
                
                if done:
                    print(f"\n✓ SUCCESS! Object reached target in {step} steps!")
                    success_count += 1
                    break
            
            # Episode Summary
            print(f"\nEpisode {ep + 1} Summary:")
            print(f"  Total Reward: {ep_reward:.2f}")
            print(f"  Steps: {step}")
            print(f"  Success: {'YES ✓' if done else 'NO ✗'}")
            print(f"  Final Distance Object→Target: {distTarget:.4f} m")
            
            total_rewards.append(ep_reward)
            
            # Kurze Pause zwischen Episodes
            if ep < num_episodes - 1:
                print("\nNext episode in 3 seconds...")
                time.sleep(config.testEpisodePauseSeconds)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user (Ctrl+C)")
    
    finally:
        # Final Statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Episodes completed: {len(total_rewards)}")
        print(f"Success rate: {success_count}/{len(total_rewards)} ({100*success_count/max(len(total_rewards),1):.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.2f}")
        print(f"Best reward: {max(total_rewards):.2f}")
        print(f"Worst reward: {min(total_rewards):.2f}")
        print("="*60)
        
        env.close()
        print("\nViewer closed. Test complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained SAC model")
    parser.add_argument("--model", type=str, default=config.testModelPath,
                        help="Path to trained model (.pth file)")
    parser.add_argument("--episodes", type=int, default=config.testNumEpisodes,
                        help="Number of test episodes")
    parser.add_argument("--steps", type=int, default=config.testMaxSteps,
                        help="Max steps per episode")
    parser.add_argument("--cube_x", type=float, default=config.defaultCubeX,
                        help="Cube X position")
    parser.add_argument("--cube_y", type=float, default=config.defaultCubeY,
                        help="Cube Y position")
    parser.add_argument("--cube_z", type=float, default=config.defaultCubeZ,
                        help="Cube Z position")
    
    args = parser.parse_args()
    
    test_trained_model(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        cube_x=args.cube_x,
        cube_y=args.cube_y,
        cube_z=args.cube_z
    )