"""
DO NOT ALTER THIS FILE.
This file should remain a constant reference to a specific behavior.
Please create your own file for simulating or alter 'demo/simulation/playground.py' instead.
"""
from src.novel_swarms.world.simulate import main as simulate
from src.novel_swarms.metrics.AngularMomentum import AngularMomentumBehavior
from src.novel_swarms.metrics.AverageSpeed import AverageSpeedBehavior
from src.novel_swarms.metrics.GroupRotationBehavior import GroupRotationBehavior
from src.novel_swarms.metrics.RadialVariance import RadialVarianceMetric
from src.novel_swarms.metrics.ScatterBehavior import ScatterBehavior
from src.novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from src.novel_swarms.config.WorldConfig import RectangularWorldConfig

if __name__ == "__main__":

    DISPERSAL_GENOME = [0.2, 0.7, -0.5, -0.1]
    SEED = None

    sensors = SensorSet([
        BinaryLOSSensor(angle=0),
    ])

    agent_config = DiffDriveAgentConfig(
        controller=DISPERSAL_GENOME,
        sensors=sensors,
        seed=SEED,
    )

    behavior = [
        AverageSpeedBehavior(),
        AngularMomentumBehavior(),
        RadialVarianceMetric(),
        ScatterBehavior(),
        GroupRotationBehavior(),
    ]

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=30,
        seed=SEED,
        behavior=behavior,
        agentConfig=agent_config,
        padding=15
    )

    simulate(world_config=world_config)
