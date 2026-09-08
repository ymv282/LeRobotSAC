"""
Zentrale Konfiguration für das SO101-SAC-Projekt.
Fasst alle Pfade und Parameter zusammen, die bisher verstreut/dupliziert
in so101_env.py, manualEnvironment.py, main.py, train.py, networks.py,
replayBuffer.py, test.py und logger.py hartcodiert waren.
"""

import os
import numpy as np

# =====================================================================
# Pfade
# =====================================================================

# so101_env.py: model_path Default
modelXmlPath = os.path.expanduser("~/LeRobot/SRC/SO-ARM100/Simulation/SO101/scene.xml")

# so101_env.py: self.log_dir (Bilder/Frames)
logDir = "/home/elia/LeRobot/SRC/logs"

# main.py: make_gif() Zielordner + konkrete Ausgabedatei (Original übergab
# nur den Ordner ohne Dateinamen an Image.save() -> Exception)
videoDir = os.path.join(logDir, "videos")
gifPath = os.path.join(videoDir, "training.gif")

# main.py: save_path für Modell-Checkpoints
modelSaveDir = "models"
bestModelFileName = "actor_critic_best.pth"
bestModelPath = os.path.join(modelSaveDir, bestModelFileName)

# main.py: plt.savefig() — im Original ein Verzeichnis ohne Dateinamen/
# Extension (fehlerhaft, wirft beim Speichern eine Exception). Hier mit
# konkretem Dateinamen versehen.
rewardPlotPath = os.path.join(logDir, "reward_plot.png")

# so101_env.py: save_img() Ausgabedatei
debugImagePath = "Testpic.png"

# logger.py: log_output() Defaults
logFileName = "log.txt"
logFolder = "logs"

# test.py: Default-Pfad zum zu testenden Modell
testModelPath = bestModelPath


# =====================================================================
# Physische Roboterdaten (SO-ARM100 / SO101)
# =====================================================================

jointNames = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# [min, max] in rad, Reihenfolge wie jointNames
jointLimits = np.array([
    [-1.9198621771937616,  1.9198621771937634],  # shoulder_pan
    [-1.7453292519943224,  1.7453292519943366],  # shoulder_lift
    [-1.69,                 1.69],               # elbow_flex
    [-1.6580628494556928,  1.6580627293335335],  # wrist_flex
    [-2.7438472969992493,  2.841206309382605],   # wrist_roll
    [-0.17453297762778586, 1.7453291995659765],  # gripper
])
jointMin = jointLimits[:, 0]
jointMax = jointLimits[:, 1]
jointRange = (jointMax - jointMin) / 2
jointCenter = (jointMax + jointMin) / 2

# main.py: max_r (erreichbarer Arbeitsraum, unbenutzt außer als Kommentarwert)
maxReachRadius = 0.36


# =====================================================================
# Environment / Simulation (so101_env.py)
# =====================================================================

cameraName = "rgb_camera"
observationImgSize = (84, 84)      # für CNN-Encoder-Input
logImgSize = (512, 512)            # für GIF/Debug-Frames
rendererWidth = 640
rendererHeight = 480

mjStepsPerAction = 10              # Physik-Substeps pro env.step()-Aufruf

# move_joints(): manuelles Anfahren einer Zielpose
moveJointsRampSteps = 1000
moveJointsSettleSteps = 500
moveJointsSleepSeconds = 0.002

# compute_reward(): Phasenschwelle, Erfolgsschwelle, Gewichte, Boni
reachPhaseThreshold = 0.08         # Wechsel Reach- -> Place-Phase
successDistThreshold = 0.05        # dist_object_target < ... => done
gripperContactThreshold = 0.12     # Bonusbedingung "Kontakt halten"
timePenalty = 0.02
successBonus = 75.0
contactBonus = 0.5
reachWeightPhase1 = 3.0
placeWeightPhase1 = 0.5
reachWeightPhase2 = 0.5
placeWeightPhase2 = 5.0

# Cube-Spawn pro Episode (main.py): ausschließlich vorderer sichtbarer
# Viertelkreis = mathematischer 1. Quadrant (x>=0, y>=0), Radius randomisiert
# statt fix, damit die Box innerhalb des von rgb_camera erfassten Bereichs
# variiert.
cubeSpawnRadiusMin = 0.12
cubeSpawnRadiusMax = 0.32          # < maxReachRadius, bleibt im Bildausschnitt
cubeSpawnPhiMin = 0.0
cubeSpawnPhiMax = np.pi / 2
cubeSpawnZ = 0.025

# so101_env.py move_cube(): identische Spawn-Logik wie oben (1. Quadrant,
# randomisierter Radius). Vorher inkonsistent: Halbkreis (-pi/2..pi/2) statt
# Viertelkreis, Radius hartcodiert statt randomisiert.
cubeSpawnRadiusAltMin = cubeSpawnRadiusMin
cubeSpawnRadiusAltMax = cubeSpawnRadiusMax
cubeSpawnPhiMinAlt = cubeSpawnPhiMin
cubeSpawnPhiMaxAlt = cubeSpawnPhiMax
cubeSpawnZMinAlt = 0.02
cubeSpawnZMaxAlt = 0.15

# manualEnvironment.py / test.py: feste Default-Cube-Position
defaultCubeX = 0.2
defaultCubeY = 0.0
defaultCubeZ = 0.025


# =====================================================================
# Netzwerk-Architektur (networks.py)
# =====================================================================

inputChannels = 3
latentDim = 256
actionDim = 6


# =====================================================================
# Replay Buffer (replayBuffer.py)
# =====================================================================

replayBufferSize = 100_000


# =====================================================================
# SAC-Training (train.py / main.py)
# =====================================================================

learningRate = 3e-4
gamma = 0.99
tau = 0.005
alphaInit = 0.1
batchSize = 256
successFifoMaxlen = 50

numEpisodes = 20000
maxSteps = 300
logInterval = 100
warmupSteps = 1000
trainEveryNSteps = 10
alphaTrainEveryNSteps = 100

# ReduceLROnPlateau
schedulerPatience = 100
schedulerFactor = 0.5
schedulerMinLr = 1e-6

# StepLR
stepLrStepSize = 6500
stepLrGamma = 0.1

schedulerChoices = ["none", "reduceOnPlateau", "cosineAnnealing", "stepLR"]

# GIF-Export
gifFps = 10


# =====================================================================
# Test / Evaluation (test.py)
# =====================================================================

testNumEpisodes = 5
testMaxSteps = 200
testStepSleepSeconds = 0.02
testEpisodePauseSeconds = 3
