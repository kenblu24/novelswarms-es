
import pygame
import time
import numpy as np
import random
from src.novel_swarms.world.simulate import main as simulate
from src.novel_swarms.novelty.GeneRule import GeneRule, GeneBuilder, GeneRuleContinuous
from src.novel_swarms.config.OutputTensorConfig import OutputTensorConfig
from src.novel_swarms.behavior.AngularMomentum import AngularMomentumBehavior
from src.novel_swarms.behavior.AverageSpeed import AverageSpeedBehavior
from src.novel_swarms.behavior.GroupRotationBehavior import GroupRotationBehavior
from src.novel_swarms.behavior.RadialVariance import RadialVarianceBehavior
from src.novel_swarms.behavior.ScatterBehavior import ScatterBehavior
from src.novel_swarms.sensors.BinaryLOSSensor import BinaryLOSSensor
from src.novel_swarms.sensors.SensorSet import SensorSet
from src.novel_swarms.config.AgentConfig import DiffDriveAgentConfig
from src.novel_swarms.config.WorldConfig import RectangularWorldConfig
from src.novel_swarms.config.EvolutionaryConfig import GeneticEvolutionConfig
from src.novel_swarms.novelty.BehaviorDiscovery import BehaviorDiscovery
import matplotlib.pyplot as plt

if __name__ == "__main__":

    SEED = 3
    random.seed(SEED)

    sensors = SensorSet([
        BinaryLOSSensor(angle=0, draw=False),
    ])

    agent_config = DiffDriveAgentConfig(
        sensors=sensors,
        agent_radius=12,
        wheel_radius=1,
    )

    genotype = GeneBuilder(
        heuristic_validation=True,
        round_to_digits=1,
        rules=[
            GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
            GeneRuleContinuous(_max=1.0, _min=-1.0, mutation_step=0.4, round_digits=1),
        ]
    )

    phenotype = [
        AverageSpeedBehavior(),
        AngularMomentumBehavior(),
        RadialVarianceBehavior(),
        ScatterBehavior(),
        GroupRotationBehavior(),
    ]

    world_config = RectangularWorldConfig(
        size=(500, 500),
        n_agents=18,
        seed=SEED,
        behavior=phenotype,
        agentConfig=agent_config,
        padding=15,
    )

    pygame.init()
    pygame.display.set_caption("Evolutionary Novelty Search")
    screen = pygame.display.set_mode((world_config.w, world_config.h))

    output_config = OutputTensorConfig(
        timeless=True,
        total_frames=90,
        steps_between_frames=2,
        screen=screen,
    )

    # Original Experiment was 50 gens, 50 pop
    novelty_config = GeneticEvolutionConfig(
        gene_builder=genotype,
        phenotype_config=phenotype,
        n_generations=50,
        n_population=50,
        crossover_rate=0.7,
        mutation_rate=0.15,
        world_config=world_config,
        k_nn=15,
        simulation_lifespan=8000,
        display_novelty=False,
        save_archive=True,
        show_gui=False,
        mutation_flip_chance=0.2,
        use_external_archive=False,
        # save_every=1
    )

    # initialize the pygame module
    pygame.init()
    pygame.display.set_caption("Evolutionary Novelty Search")

    # screen must be global so that other modules can access + draw to the window
    width = world_config.w
    screen = pygame.display.set_mode((width, world_config.h))

    # define a variable to control the main loop
    running = True
    paused = False

    # Create a trial folder in the specified location
    trial_name = f"{str(int(time.time()))}"

    # Initialize GA
    config = novelty_config
    gene_builder = config.gene_builder
    evolution = BehaviorDiscovery(
        generations=config.generations,
        population_size=config.population,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        world_config=config.world_config,
        lifespan=config.lifespan,
        k_neighbors=config.k,
        genome_builder=gene_builder,
        behavior_config=config.behavior_config,
        mutation_flip_chance=config.mutation_flip_chance,
        allow_external_archive=config.use_external_archive
    )

    TRIAL = "1_visuals"
    CONTROLLERS = [
        # [-0.7, -1.0, 1.0, -1.0],  # Aggregation
        # [-0.7, 0.3, 1.0, 1.0],  # Cyclic Pursuit
        # [0.2, 0.7, -0.5, -0.1],  # Dispersal
        [0.8, 1.0, 0.5, 0.6] , # Milling
        # [1.0, 0.93, 1.0, 1.0],
        # [0.8, 0.5, 0.6, -0.5, -0.5, -0.0, -0.2, 0.5, -(np.pi / 3)],
        # [-0.4, 0.8, 0.9, -0.2, 0.6, 1.0, 0.6, -0.0, np.pi/6]
    ]
    for i, controller in enumerate(CONTROLLERS):
        elem = str(i)
        output = evolution.runSinglePopulation(screen, save=True, genome=controller, seed=1, output_config=output_config)

        print(np.shape(output))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(output, cmap='Greys')
        plt.savefig("out/" + f'{TRIAL}_{elem}.png', bbox_inches='tight', transparent="True", pad_inches=0)