"""
Feel free to copy this file and explore configurations that lead to interesting results.

If you do not plan to make commits to the GitHub repository or if you can ensure that changes to this file
are not included in your commits, you may directly edit and run this file.
"""
from src.novel_swarms.metrics.AlgebraicConnectivity import AlgebraicConn
from src.novel_swarms.config.defaults import ConfigurationDefaults
from src.novel_swarms.novelty.GeneRule import GeneRule, GeneBuilder, GeneRuleContinuous
from src.novel_swarms.novelty.evolve import main as evolve
from src.novel_swarms.results.results import main as report
from src.novel_swarms.metrics.AngularMomentum import AngularMomentumBehavior
from src.novel_swarms.metrics.AverageSpeed import AverageSpeedBehavior
from src.novel_swarms.metrics.GroupRotationBehavior import GroupRotationBehavior
from src.novel_swarms.metrics.RadialVariance import RadialVarianceMetric
from src.novel_swarms.metrics.ScatterBehavior import ScatterBehavior
from src.novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from src.novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from src.novel_swarms.metrics.AngularMomentum import AngularMomentumBehavior
from src.novel_swarms.metrics.AverageSpeed import AverageSpeedBehavior
from src.novel_swarms.metrics.GroupRotationBehavior import GroupRotationBehavior
from src.novel_swarms.metrics.RadialVariance import RadialVarianceMetric
from src.novel_swarms.metrics.ScatterBehavior import ScatterBehavior
from src.novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from src.novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.sensors.GenomeDependentSensor import GenomeBinarySensor
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.config.AgentConfig import DiffDriveAgentConfig, UnicycleAgentConfig
from src.novel_swarms.config.WorldConfig import RectangularWorldConfig
from src.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
import numpy as np
from src.novel_swarms.metrics.SensorOffset import GeneElementDifference

if __name__ == "__main__":

    SEED = None

    BL = 15.1
    GUI_PADDING = 15
    N_AGENTS = 10
    WIDTH, HEIGHT = int(BL * 29.8), int(BL * 29.8)

    sensors = SensorSet([
        BinaryFOVSensor(
            theta=14 / 2,
            distance=(BL * 12.5),
            bias=-4,
            degrees=True,
            false_positive=0.1,
            false_negative=0.05,
            # Rectangle Representing Environment Boundaries
            walls=[[GUI_PADDING, GUI_PADDING], [GUI_PADDING + WIDTH, GUI_PADDING + HEIGHT]],
            wall_sensing_range=(BL * 6),
            time_step_between_sensing=2,
            store_history=True
        )
    ])

    agent_config = UnicycleAgentConfig(
        agent_radius=BL / 2,
        dt=0.13,  # 130ms sampling period
        sensors=sensors,
        seed=None,
        idiosyncrasies=True
    )

    behavior = [
        AverageSpeedBehavior(),
        AngularMomentumBehavior(),
        RadialVarianceMetric(),
        ScatterBehavior(),
        GroupRotationBehavior(),
        AlgebraicConn(),
    ]

    r = 50
    pi_slice = (2 * np.pi) / N_AGENTS
    center = (int(BL * 29.8) / 2, int(BL * 29.8) / 2)
    init_positions = [(r * np.cos(t * pi_slice), r * np.sin(t * pi_slice), t * pi_slice) for t in range(0, N_AGENTS)]
    init_positions = [(center[0] + x, center[1] + y, t) for x, y, t in init_positions]

    EVAL_TIL = None
    world_config = RectangularWorldConfig(
        size=(WIDTH + GUI_PADDING, HEIGHT + GUI_PADDING),
        n_agents=N_AGENTS,
        seed=SEED,
        behavior=behavior,
        agentConfig=agent_config,
        padding=GUI_PADDING,
        show_walls=True,
        agent_initialization=init_positions,
        stop_at=EVAL_TIL
    )

    gene_specifications = GeneBuilder(
        heuristic_validation=False,
        round_to_digits=1,
        rules=[
            GeneRuleContinuous(_max=20.0, _min=-20.0, mutation_step=3.0, round_digits=0),
            GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRuleContinuous(_max=20.0, _min=-20.0, mutation_step=3.0, round_digits=0),
            GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
        ]
    )

    novelty_config = GeneticEvolutionConfig(
        gene_builder=gene_specifications,
        phenotype_config=behavior,
        n_generations=50,
        n_population=100,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=1200,
        display_novelty=True,
        save_archive=True,
        show_gui=False,
        save_every=1,
    )

    # Novelty Search through Genetic Evolution
    archive = evolve(config=novelty_config)

    results_config = ConfigurationDefaults.RESULTS
    results_config.world = world_config
    results_config.archive = archive

    # Take Results from Evolution, reduce dimensionality, and present User with Clusters.
    report(config=results_config)
