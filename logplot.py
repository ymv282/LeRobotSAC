import re
import matplotlib.pyplot as plt

def plot_logfile(path):
    episodes = []
    rewards = []
    avg50 = []
    distances = []

    pattern = re.compile(
        r"\[Episode (\d+)\]\s+Reward:\s+([-.\d]+)\s+\|\s+Avg\(50\):\s+([-.\d]+)\s+\|\s+Distance Gripper to Object:\s+([-.\d]+)"
    )

    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                avg50.append(float(m.group(3)))
                distances.append(float(m.group(4)))

    plt.figure()
    plt.plot(episodes, rewards, label="Reward")
    plt.plot(episodes, avg50, label="Avg(50)")
    plt.plot(episodes, distances, label="Distance")
    plt.xlabel("Episode")
    plt.legend()
    plt.show()
