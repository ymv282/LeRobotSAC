import numpy as np
import mujoco
import mujoco.viewer
import cv2
import os
import matplotlib.pyplot as plt
import threading
import time
from collections import deque
import imageio
#TODO: Kamera, next state, another 

class SO101SimulationEnv:
    """Sim Environment für Mujoco"""
    arm_joints = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    def __init__(self, model_path=os.path.expanduser("~/LeRobot/SRC/SO-ARM100/Simulation/SO101/scene.xml"), camera_name="rgbd_camera", 
                 img_size=(84, 84), log_img_size=(512, 512), enable_viewer=False):
        self.img_size = img_size
        self.camera_name = camera_name
        self.log_img_size = log_img_size
        self.enable_viewer = enable_viewer
        self.viewer = None
        self.log_dir = '/home/elia/LeRobot/SRC/logs'

        
        # Lade MuJoCo Modell
        if model_path is None:
            model_path = os.path.expanduser("~/LeRobot/SRC/SO-ARM100/Simulation/SO101/scene.xml")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XML Datei nicht gefunden: {model_path}")
        
        print(f"Lade MuJoCo Modell von: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        # MuJoCo Viewer starten (optional)
        if enable_viewer:
            self._start_viewer()
        try:
            self.camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            print(f"\nVerwende Kamera: '{camera_name}' (ID: {self.camera_id})")
        except:
            print("Keine Kamera gefunden")
    def get_camera_image(self):
        """Holt RGB Kamerabild"""
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()
        rgb_small = cv2.resize(rgb, self.img_size)
        return rgb_small
    
    def get_gif_img(self):
        "Img fürs rendern des Gifs"
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()
        rgb_large = cv2.resize(rgb, self.log_img_size)
        return rgb_large
    def save_img(self):
        "Img fürs Rendern des Gifs und direkt im Log-Ordner speichern"
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()
        rgb_large = cv2.resize(rgb, self.log_img_size)

        os.makedirs(self.log_dir, exist_ok=True)
        filename = 'Testpic.png'#os.path.join(self.log_dir, f"frame_{self.step():06d}.png")
        cv2.imwrite(filename, cv2.cvtColor(rgb_large, cv2.COLOR_RGB2BGR))

        return rgb_large

    def _start_viewer(self):
        """Startet MuJoCo Viewer in separatem Thread"""
        def viewer_loop():
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                print("MuJoCo Viewer gestartet")
                while viewer.is_running():
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
        
        self.viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
        self.viewer_thread.start()
        time.sleep(1)  # Warte auf Viewer Start
    def sync_viewer(self):
        """Synchronisiert den Viewer mit dem aktuellen mjData"""
        if self.viewer is not None:
            self.viewer.sync()
    def reset(self, x, y, z):
        """
        Reset der Simulation:
        - Arm zufällig innerhalb der Gelenklimits
        - Cube (free joint) zufällig im erreichbaren Arbeitsraum platzieren
        - Vorwärtskinematik aktualisieren
        - Kamera-Observation zurückgeben
        """
        self.data = mujoco.MjData(self.model)

        # ----- Arm-Joints randomisieren -----
        

        for name in self.arm_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = self.model.jnt_qposadr[jid]
            qmin, qmax = self.model.jnt_range[jid]
            self.data.qpos[qadr] = 0#np.random.uniform(qmin, qmax)

        # ----- Cube (free joint) platzieren -----
        cube_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube"
        )
        qadr = self.model.jnt_qposadr[cube_jid]

        # free joint: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[qadr:qadr + 7] = np.array(
            [x, y, z, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

        # ----- Vorwärtskinematik -----
        mujoco.mj_forward(self.model, self.data)

        return self.get_camera_image()
    def move_cube(self):
         # ----- Cube (free joint) platzieren -----
        cube_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube"
        )
        qadr = self.model.jnt_qposadr[cube_jid]

        # erreichbarer Arbeitsraum TODO: Überprüfen
        r = 0.45 #np.random.uniform(0.15, 0.45)
        phi = np.random.uniform(-np.pi/2, np.pi/2)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.random.uniform(0.02, 0.15)

        # free joint: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[qadr:qadr + 7] = np.array(
            [x, y, z, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

        # ----- Vorwärtskinematik -----
        mujoco.mj_forward(self.model, self.data)

    
        



    def get_gripper_position(self):
        gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        # Endeffektor Position
        return self.data.xpos[gripper_id].copy()
    def get_gripper_angle(self):
        gripper_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        qadr = self.model.jnt_qposadr[gripper_jid]
        return self.data.qpos[qadr]
    def get_target_position(self):
        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        return self.data.xpos[target_id].copy()
    def get_cube_position(self):
        cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        return self.data.xpos[cube_id].copy()
    # In so101_env.py - compute_reward()

    def compute_reward(self):
        gripper_pos = self.get_gripper_position()
        object_pos = self.get_cube_position()
        target_pos = self.get_target_position()
        
        dist_gripper_object = np.linalg.norm(gripper_pos - object_pos)
        dist_object_target = np.linalg.norm(object_pos - target_pos)
        r_reach = -dist_gripper_object
        r_place = -dist_object_target
            # Adaptive Gewichtung basierend auf Phase
        if dist_gripper_object > 0.08:
            # Phase 1: Hauptfokus auf Reaching
            reward = 3.0 * r_reach + 0.5 * r_place
        else:
            # Phase 2: Hauptfokus auf Placing, aber Gripper-Kontakt behalten
            reward = 0.5 * r_reach + 5.0 * r_place
            
            # Bonus für Kontakt halten
            if dist_gripper_object < 0.12:
                reward += 0.5
    
        # Kleine Zeit-Strafe
        reward -= 0.02
        
        # # Phase 1: Gripper muss erst zum Object
        # if dist_gripper_object > 0.08:
        #     reward = -dist_gripper_object * 2.0
    
        
        # # Phase 2: Object zum Target schieben
        # else:
        #     reward = -dist_object_target * 5.0
        # # reward = - dist_object_target
        done = dist_object_target < 0.05
        if done:
            reward +=75.0  # Höherer Success-Bonus
        
        return reward, done, dist_gripper_object, dist_object_target, object_pos



    
    def move_joints(self,target_angles):
        for _ in range(1000):
            self.data.ctrl[:] = target_angles
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)
    def get_joint_positions(self, joint_names):
        positions = {}

        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[jid]
            positions[name] = self.data.qpos[qpos_adr].copy()

        return positions
    def set_joint_angles(self, angles):
        for i, name in enumerate(self.arm_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = angles[i]

        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        """
        action: 6-Tupel, Zielgelenkwinkel des Roboterarms
        returns: obs, reward, done, distance, info
        """
        self.data.ctrl[:] = action

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Observation: Kamerabild
        obs = self.get_camera_image()

        # Reward & Done
        reward, done, distance_object, distance_target, object_pos = self.compute_reward()

        info = {
            "gripper_pos": self.get_gripper_position(),
            "cube_pos": self.get_cube_position(),
            "target_pos": self.get_target_position(),
        }

        return obs, reward, done, distance_object, distance_target, object_pos

    def close(self):
        """Cleanup"""
        if self.viewer is not None:
            self.viewer.close()
if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SO101SimulationEnv(enable_viewer=False)
    targetAngles = [1.9198621771937634, 1.7453292519943366, 1.69, 1.6580627293335335, 2.841206309382605, 1.7453291995659765]
    env.move_joints(targetAngles)
    for _ in range(10):
            mujoco.mj_step(env.model, env.data)
    val = env.get_gripper_position
    print(f'Max Pos: {val}')

    
    
    """
    frames = []
    numTries = 3
    joints =[
        ["shoulder_pan",  -1.9198621771937616,  1.9198621771937634],
        ["shoulder_lift", -1.7453292519943224,  1.7453292519943366],
        ["elbow_flex",    -1.69,                1.69],
        ["wrist_flex",    -1.6580628494556928,  1.6580627293335335],
        ["wrist_roll",    -2.7438472969992493,  2.841206309382605],
        ["gripper",       -0.17453297762778586, 1.7453291995659765],
        ]
    joint_names = [j[0] for j in joints]
    target_angles = [0.0 for _ in joints]
    for j in range(1,5):
        env.reset()
        for i in range (numTries):
            for i in range(len(joints)):
                target_angles[i] = 0.5*random.uniform(joints[i][1], joints[i][2])
                print("target_angles:", target_angles)
                print("Gripper Position:", env.get_gripper_position())
                for i in range(5):
                    env.sync_viewer
                env.move_joints(target_angles)
                rewads, done = env.compute_reward()
                print(f'Reward: {rewads}, Done: {done}')
                frames.append(env.get_camera_image())
                time.sleep(0.02)
        
    imageio.mimsave('simulation.gif', frames, fps=10)
    env.close()
    """