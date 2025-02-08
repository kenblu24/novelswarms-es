"""
Feel free to copy this file and explore configurations that lead to interesting results.

If you do not plan to make commits to the GitHub repository or if you can ensure that changes to this file
are not included in your commits, you may directly edit and run this file.
"""
from src.novel_swarms.config.defaults import ConfigurationDefaults
from src.novel_swarms.novelty.GeneRule import GeneRule, GeneRuleContinuous
from src.novel_swarms.novelty.evolve import main as evolve
from src.novel_swarms.results.results import main as report
from src.novel_swarms.metrics.AngularMomentum import AngularMomentumBehavior
from src.novel_swarms.metrics.AverageSpeed import AverageSpeedBehavior
from src.novel_swarms.metrics.GroupRotationBehavior import GroupRotationBehavior
from src.novel_swarms.metrics.RadialVariance import RadialVarianceMetric
from src.novel_swarms.metrics.ScatterBehavior import ScatterBehavior
from src.novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from src.novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from src.novel_swarms.config.WorldConfig import RectangularWorldConfig
from src.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig

if __name__ == "__main__":

    SEED = None

    sensors = SensorSet([
        BinaryLOSSensor(angle=0),
        # BinaryLOSSensor(angle=45),
        # BinaryLOSSensor(angle=45)
        # BinaryFOVSensor(theta=14 / 2, distance=(20 * 13.25), degrees=True)
    ])

    agent_config = ConfigurationDefaults.DIFF_DRIVE_AGENT

    genotype = [
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=2),
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=2),
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=2),
        GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=2),
    ]

    phenotype = [
        AverageSpeedBehavior(),
        AngularMomentumBehavior(),
        RadialVarianceMetric(),
        ScatterBehavior(),
        GroupRotationBehavior(),
    ]

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=24,
        seed=SEED,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15
    )

    novelty_config = GeneticEvolutionConfig(
        gene_rules=genotype,
        phenotype_config=phenotype,
        n_generations=10,
        n_population=4,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=3,
        simulation_lifespan=1200,
        display_novelty=True,
        save_archive=True,
        show_gui=True,
        save_every=1,
        seed=None
    )

    # Novelty Search through Genetic Evolution
    archive = evolve(config=novelty_config)

    results_config = ConfigurationDefaults.RESULTS
    results_config.world = world_config
    results_config.archive = archive

    # Take Results from Evolution, reduce dimensionality, and present User with Clusters.
    report(config=results_config)
