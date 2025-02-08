import numpy as np

from src.novel_swarms.world.simulate import main as simulate
from src.novel_swarms.behavior.AngularMomentum import AngularMomentumBehavior
from src.novel_swarms.behavior.AverageSpeed import AverageSpeedBehavior
from src.novel_swarms.behavior.GroupRotationBehavior import GroupRotationBehavior
from src.novel_swarms.behavior.RadialVariance import RadialVarianceBehavior
from src.novel_swarms.behavior.ScatterBehavior import ScatterBehavior
from src.novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from src.novel_swarms.config.WorldConfig import RectangularWorldConfig
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pandas import DataFrame

def simulate_and_ret(controller, stop_detection, max_steps=3000):
    SEED = 1

    sensors = SensorSet([
        BinaryLOSSensor(angle=0),
    ])

    agent_config = DiffDriveAgentConfig(
        controller=controller,
        sensors=sensors,
        seed=SEED,
    )

    behavior = [
        AverageSpeedBehavior(),
        AngularMomentumBehavior(),
        RadialVarianceBehavior(),
        ScatterBehavior(),
        GroupRotationBehavior(),
    ]

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        seed=SEED,
        behavior=behavior,
        agentConfig=agent_config,
        padding=15,
        stop_at=max_steps
    )

    world = simulate(world_config=world_config, stop_detection=stop_detection)
    if world.total_steps >= max_steps:
        return False, world
    return True, world


def stop_detection_method(world):
    EPSILON = 0.001
    if world.total_steps > 100 and world.behavior[2].out_average()[1] < EPSILON:
        return True
    return False


if __name__ == "__main__":
    v_l_1, v_r_1 = 1.0, 1.0
    output_A = np.zeros((21, 21))
    output_B = np.zeros((21, 21))

    x_tick_labels = [str(round(i * 0.1, 1)) for i in range(-10, 11, 1)]
    y_tick_labels = [str(round(i * 0.1, 1)) for i in range(-10, 11, 1)]

    for i in range(0, 21, 1):
        v_l_0 = round((i - 10) * 0.1, 1)
        for j in range(0, 21, 1):
            v_r_0 = round((j - 10) * 0.1, 1)
            controller = [v_l_0, v_r_0, v_l_1, v_r_1]
            cyclic, w = simulate_and_ret(controller, stop_detection_method, max_steps=2500)
            print(f"Detection? - {cyclic}")
            output_A[i][j] = w.total_steps
            output_B[i][j] = -w.behavior[2].out_average()[1]

    df = DataFrame(output_A)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=False, ax=ax, square=True)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("$V_{r0}$")
    ax.set_ylabel("$V_{l0}$")
    plt.title("Simulation Time (Max: 2500)")
    plt.tight_layout()
    df.to_csv("out/sim-time-cyclic.csv")

    plt.show()

    df = DataFrame(output_B)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=False, ax=ax, square=True)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("$V_{r0}$")
    ax.set_ylabel("$V_{l0}$")
    plt.title("Radial Variance")
    plt.tight_layout()
    df.to_csv("out/r-var-cyclic.csv")

    df = DataFrame(output_A + output_B)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=False, ax=ax, square=True)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("$V_{r0}$")
    ax.set_ylabel("$V_{l0}$")
    plt.title("Combined Score for Cycle Detection")
    plt.tight_layout()

    plt.show()