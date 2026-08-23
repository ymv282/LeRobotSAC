import numpy as np
import time
import json
from datetime import datetime
from so101_env import SO101SimulationEnv

# Physische Daten zum Gripper
joint_limits = np.array([
    [-1.9198621771937616,  1.9198621771937634],  # shoulder_pan
    [-1.7453292519943224,  1.7453292519943366],  # shoulder_lift
    [-1.69,                 1.69],               # elbow_flex
    [-1.6580628494556928,  1.6580627293335335],  # wrist_flex
    [-2.7438472969992493,  2.841206309382605],   # wrist_roll
    [-0.17453297762778586, 1.7453291995659765],  # gripper
])

joint_names = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", 
    "wrist_flex", "wrist_roll", "gripper"
]

class SessionLogger:
    def __init__(self):
        self.start_time = datetime.now()
        self.steps = []
        self.total_reward = 0.0
        self.max_reward = -np.inf
        self.min_reward = np.inf
        self.max_dist_gripper = -np.inf
        self.min_dist_gripper = np.inf
        self.max_dist_target = -np.inf
        self.min_dist_target = np.inf
    
    def log_step(self, step_num, angles, reward, dist_gripper, dist_target, done):
        self.steps.append({
            'step': step_num,
            'timestamp': (datetime.now() - self.start_time).total_seconds(),
            'joint_angles_rad': angles.tolist(),
            'joint_angles_deg': np.degrees(angles).tolist(),
            'reward': reward,
            'dist_gripper': dist_gripper,
            'dist_target': dist_target,
            'done': done
        })
        
        self.total_reward += reward
        self.max_reward = max(self.max_reward, reward)
        self.min_reward = min(self.min_reward, reward)
        self.max_dist_gripper = max(self.max_dist_gripper, dist_gripper)
        self.min_dist_gripper = min(self.min_dist_gripper, dist_gripper)
        self.max_dist_target = max(self.max_dist_target, dist_target)
        self.min_dist_target = min(self.min_dist_target, dist_target)
    
    def save_log(self, filename=None):
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"mujoco_session_{timestamp}.log"
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        log_data = {
            'session_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_steps': len(self.steps)
            },
            'statistics': {
                'reward': {
                    'total': self.total_reward,
                    'average': self.total_reward / max(len(self.steps), 1),
                    'max': self.max_reward,
                    'min': self.min_reward
                },
                'distance_gripper_to_object': {
                    'max': self.max_dist_gripper,
                    'min': self.min_dist_gripper,
                    'final': self.steps[-1]['dist_gripper'] if self.steps else None
                },
                'distance_object_to_target': {
                    'max': self.max_dist_target,
                    'min': self.min_dist_target,
                    'final': self.steps[-1]['dist_target'] if self.steps else None
                }
            },
            'joint_limits': {
                joint_names[i]: {
                    'min_rad': float(joint_limits[i, 0]),
                    'max_rad': float(joint_limits[i, 1]),
                    'min_deg': float(np.degrees(joint_limits[i, 0])),
                    'max_deg': float(np.degrees(joint_limits[i, 1]))
                }
                for i in range(6)
            },
            'steps': self.steps
        }
        
        # JSON speichern
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Zusätzlich lesbare Text-Datei
        txt_filename = filename.replace('.log', '.txt')
        with open(txt_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" MUJOCO SESSION LOG ".center(80, "=") + "\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Start Time:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration:    {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
            f.write(f"Total Steps: {len(self.steps)}\n\n")
            
            f.write("-"*80 + "\n")
            f.write(" STATISTICS ".center(80, "-") + "\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"Reward:\n")
            f.write(f"  Total:   {self.total_reward:10.4f}\n")
            f.write(f"  Average: {self.total_reward / max(len(self.steps), 1):10.4f}\n")
            f.write(f"  Max:     {self.max_reward:10.4f}\n")
            f.write(f"  Min:     {self.min_reward:10.4f}\n\n")
            
            f.write(f"Distance Gripper to Object:\n")
            f.write(f"  Max:     {self.max_dist_gripper:10.6f} m\n")
            f.write(f"  Min:     {self.min_dist_gripper:10.6f} m\n")
            if self.steps:
                f.write(f"  Final:   {self.steps[-1]['dist_gripper']:10.6f} m\n\n")
            
            f.write(f"Distance Object to Target:\n")
            f.write(f"  Max:     {self.max_dist_target:10.6f} m\n")
            f.write(f"  Min:     {self.min_dist_target:10.6f} m\n")
            if self.steps:
                f.write(f"  Final:   {self.steps[-1]['dist_target']:10.6f} m\n\n")
            
            f.write("-"*80 + "\n")
            f.write(" JOINT LIMITS ".center(80, "-") + "\n")
            f.write("-"*80 + "\n\n")
            
            for i, name in enumerate(joint_names):
                f.write(f"{name:15s}: [{joint_limits[i,0]:7.3f}, {joint_limits[i,1]:7.3f}] rad\n")
                f.write(f"{'':15s}  [{np.degrees(joint_limits[i,0]):7.2f}, {np.degrees(joint_limits[i,1]):7.2f}] deg\n\n")
            
            if len(self.steps) <= 100:  # Nur bei wenigen Steps alle ausgeben
                f.write("-"*80 + "\n")
                f.write(" DETAILED STEPS ".center(80, "-") + "\n")
                f.write("-"*80 + "\n\n")
                
                for step_data in self.steps:
                    f.write(f"Step {step_data['step']:4d} @ {step_data['timestamp']:7.2f}s:\n")
                    f.write(f"  Reward: {step_data['reward']:8.4f}\n")
                    f.write(f"  Dist Gripper: {step_data['dist_gripper']:.6f} m\n")
                    f.write(f"  Dist Target:  {step_data['dist_target']:.6f} m\n")
                    f.write(f"  Done: {step_data['done']}\n")
                    f.write(f"  Angles (rad): {step_data['joint_angles_rad']}\n\n")
        
        return filename, txt_filename

def main():
    """
    **Features:**

    1. **GUI-Steuerung:** Bewege den Arm direkt in MuJoCo GUI
    2. **Kontinuierliches Logging:** Alle Steps werden automatisch geloggt
    3. **Zwei Logfile-Formate:**
    - `.log` (JSON) - maschinenlesbar, alle Daten
    - `.txt` - menschenlesbar, Zusammenfassung + Details
    4. **Statistiken:** Min/Max/Average für Rewards und Distanzen
    5. **Timestamps:** Jeder Step mit Zeitstempel
    6. **Gelenkwinkel:** In Radiant und Grad gespeichert
    7. **Sauberes Beenden:** Strg+C erstellt automatisch Logfiles

    **Ausgabe-Beispiel (TXT):**
    ```
    ================================================================================
                            MUJOCO SESSION LOG                            
    ================================================================================

    Start Time:  2026-01-25 12:30:45
    End Time:    2026-01-25 12:35:12
    Duration:    267.43 seconds (4.46 minutes)
    Total Steps: 8023

    --------------------------------------------------------------------------------
                                STATISTICS                                
    --------------------------------------------------------------------------------

    Reward:
    Total:       -1234.5678
    Average:        -0.1539
    Max:             2.3456
    Min:           -15.6789
    ...
    """
    print("\n" + "="*80)
    print(" MUJOCO INTERACTIVE MODE ".center(80, "="))
    print("="*80)
    print("\n📋 Anleitung:")
    print("  1. Nutze die MuJoCo GUI um den Arm zu bewegen")
    print("  2. Die Simulation läuft kontinuierlich und loggt alle Steps")
    print("  3. Drücke Strg+C um zu beenden und das Logfile zu erstellen")
    print("\n" + "="*80 + "\n")
    
    # Target-Position konfigurieren
    x = float(input("Target X [default: 0.2]: ") or 0.2)
    y = float(input("Target Y [default: 0.0]: ") or 0.0)
    z = float(input("Target Z [default: 0.025]: ") or 0.025)
    
    # Simulation-Rate
    fps = int(input("Simulation FPS [default: 30]: ") or 30)
    dt = 1.0 / fps
    
    print(f"\n✓ Starte Simulation mit Target: x={x}, y={y}, z={z}")
    print(f"✓ Simulation läuft mit {fps} FPS")
    print("✓ Drücke Strg+C zum Beenden...\n")
    
    env = SO101SimulationEnv(enable_viewer=True)
    obs = env.reset(x=x, y=y, z=z)
    
    logger = SessionLogger()
    step_count = 0
    
    try:
        while True:
            # Hole aktuelle Gelenkwinkel aus MuJoCo
            # (Annahme: env hat Methode um aktuelle Winkel zu lesen)
            # Falls nicht vorhanden, kannst du env.model.data.qpos verwenden
            current_angles = np.array([
                env.data.joint(joint_names[i]).qpos[0] 
                for i in range(6)
            ])
            
            # Führe Step aus
            obs, reward, done, dist_gripper, dist_target, _object_pos = env.step(current_angles)
            
            # Logge Step
            logger.log_step(step_count, current_angles, reward, dist_gripper, dist_target, done)
            step_count += 1
            
            # Konsolen-Feedback (minimal)
            if step_count % 100 == 0:
                print(f"Step {step_count:5d} | Reward: {reward:7.4f} | "
                      f"Dist G: {dist_gripper:.4f} | Dist T: {dist_target:.4f}")
            
            # Warte für nächsten Frame
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Beende Simulation und erstelle Logfiles...\n")
    
    finally:
        # Speichere Logs
        json_file, txt_file = logger.save_log()
        
        print("="*80)
        print(" SESSION BEENDET ".center(80, "="))
        print("="*80)
        print(f"\n📊 Statistiken:")
        print(f"  Total Steps:      {len(logger.steps)}")
        print(f"  Total Reward:     {logger.total_reward:.4f}")
        print(f"  Average Reward:   {logger.total_reward / max(len(logger.steps), 1):.4f}")
        print(f"  Max Reward:       {logger.max_reward:.4f}")
        print(f"  Min Reward:       {logger.min_reward:.4f}")
        print(f"\n📁 Logfiles erstellt:")
        print(f"  JSON: {json_file}")
        print(f"  TXT:  {txt_file}")
        print("\n" + "="*80 + "\n")
        
        env.close()

if __name__ == "__main__":
    main()
