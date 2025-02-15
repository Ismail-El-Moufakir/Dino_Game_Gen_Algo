import numpy as np
import torch
import pygame
from torch import nn
import matplotlib.pyplot as plt
import random

# Assume these constants are defined in utils.py
from utils import POPULATION_SIZE, CROSSOVER_RATE, NBRE_GENERATION
import Env


class Agent(nn.Module):
    def __init__(self):
        """
        Initializes the Agent with a neural network and gameplay attributes.
        """
        super().__init__()
        self.score = 0
        self.y = 250  # initial y position of Agent Dino
        self.walk_state = 0  # 0: LEFT, 1: RIGHT
        self.collided = False
        self.is_jumping = False
        self.jump_velocity = 0

        # Load images only once for efficiency
        self.walk_left_img = pygame.image.load("assets/Walk_Left.png")
        self.walk_right_img = pygame.image.load("assets/Walk_Right.png")
        self.jump_img = pygame.image.load("assets/Jump.png")

        # Define a simple feed-forward neural network
        self.net = nn.Sequential(
            nn.Linear(3, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        return self.net(x)

    def act(self, env) -> int:
        """
        Determines and returns the agent's action based on environment inputs.
        Action 0 corresponds to a jump.
        """
        # Default block x-position
        block_position_x = 950
        if env.blocks:
            # Get the x-position of the last block (assuming order matters)
            block_positions = [pos[0] for pos in env.blocks.values()]
            block_position_x = block_positions[-1]

        # Create input features: block position, current y position, and environment speed
        x = torch.tensor([block_position_x, self.y, env.speed], dtype=torch.float32)
        output = self.forward(x)
        action = torch.argmax(output).item()
        return action

    def walk(self, screen: pygame.Surface) -> None:
        """
        Renders the walking animation.
        """
        dino_img = self.walk_left_img if self.walk_state == 0 else self.walk_right_img
        self.walk_state = 1 - self.walk_state  # Toggle walking state
        screen.blit(dino_img, (50, self.y))

    def jump(self, screen: pygame.Surface) -> None:
        """
        Initiates a jump action.
        """
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = -25  # Set initial jump velocity
            self.y += self.jump_velocity
            screen.blit(self.jump_img, (50, self.y))

    def update_jump(self) -> None:
        """
        Updates the agent's vertical position to simulate gravity.
        """
        if self.is_jumping:
            self.y += self.jump_velocity
            self.jump_velocity += 5  # Gravity effect
            if self.y >= 250:  # Landed on the ground
                self.y = 250
                self.is_jumping = False
                self.jump_velocity = 0

    def check_collision(self, env) -> bool:
        """
        Checks if the agent collides with any block in the environment.
        """
        dino_rect = pygame.Rect(50, self.y,
                                self.walk_right_img.get_width(),
                                self.walk_right_img.get_height())
        for block, position in env.blocks.items():
            block_rect = pygame.Rect(position[0], position[1],
                                     block.get_width(), block.get_height())
            if dino_rect.colliderect(block_rect):
                return True
        return False


class GeneticAlgorithm:
    def __init__(self):
        """
        Initializes the genetic algorithm with a population of agents.
        """
        self.population = [Agent() for _ in range(POPULATION_SIZE)]
        self.fitnesses = [0 for _ in range(len(self.population))]
        self.max_fitnesses = []

    def fitness_function(self, agent: Agent) -> float:
        """
        Returns the fitness of an agent.
        """
        return agent.score

    def mutation(self, agent: Agent) -> None:
        """
        Applies mutation to an agent's parameters.
        """
        for param in agent.parameters():
            mutation_tensor = torch.normal(0, 0.1, size=param.shape)
            param.data.add_(mutation_tensor)

    def crossover(self, agent1: Agent, agent2: Agent) -> Agent:
        """
        Performs crossover between two agents to produce a new child agent.
        """
        child = Agent()
        for param1, param2, child_param in zip(agent1.parameters(),
                                                 agent2.parameters(),
                                                 child.parameters()):
            # Flatten the parameters for crossover
            flat1 = param1.data.view(-1)
            flat2 = param2.data.view(-1)
            child_flat = child_param.data.view(-1)

            # Choose a random split point along the flattened tensor
            split_point = np.random.randint(0, flat1.numel())
            child_flat[:split_point] = flat1[:split_point]
            child_flat[split_point:] = flat2[split_point:]
        return child

    def selection(self) -> list:
        """
        Selects individuals from the current population using tournament selection.
        """
        tournament_size = 5
        selected = []
        for _ in range(len(self.population)):
            tournament = random.sample(self.population, tournament_size)
            best = max(tournament, key=self.fitness_function)
            selected.append(best)
        return selected

    def new_generation(self) -> None:
        """
        Creates a new generation using selection, crossover, and mutation.
        """
        new_population = []
        selected_population = self.selection()
        current_pop_size = len(self.population)

        while len(new_population) < current_pop_size:
            if np.random.rand() < CROSSOVER_RATE:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child = self.crossover(parent1, parent2)
                new_population.append(child)
            else:
                # Clone and mutate one individual from the selected population
                individual = random.choice(selected_population)
                # Reset relevant attributes before mutation
                individual.score = 0
                individual.collided = False
                individual.y = 250
                individual.is_jumping = False
                individual.jump_velocity = 0
                self.mutation(individual)
                new_population.append(individual)
        self.population = new_population
        self.fitnesses = [0 for _ in range(len(self.population))]

    def train(self) -> None:
        """
        Trains the population over multiple generations.
        A new game window is created for each generation and closed when the generation ends.
        The simulation now resets as soon as all agents have collided.
        """
        generation = 0
        training_running = True

        while training_running and generation < NBRE_GENERATION:
            # Initialize a new window for the current generation
            pygame.init()
            screen = pygame.display.set_mode((800, 600))
            clock = pygame.time.Clock()
            env = Env.Env()
            running = True

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        training_running = False
                        break

                # Check collisions for each agent
                for i, agent in enumerate(self.population):
                    if not agent.collided and agent.check_collision(env):
                        self.fitnesses[i] = agent.score
                        agent.collided = True
                        print(f"Agent {i} collided at score {agent.score}")

                # If all agents have collided, reset the simulation for this generation.
                if all(agent.collided for agent in self.population):
                    running = False

                # Update environment speed based on time
                if env.time >= 100:
                    env.speed += 0.05
                    env.time = 0
                env.time += 1

                # Process each agent's action if they haven't collided
                for agent in self.population:
                    if not agent.collided:
                        action = agent.act(env)
                        if action == 0 and not agent.is_jumping:
                            agent.jump(screen)
                        agent.update_jump()

                # Rendering
                screen.fill("white")
                env.Draw_Ground(screen)
                env.Draw_Blocks(screen)
                for agent in self.population:
                    if not agent.collided:
                        agent.walk(screen)
                        agent.score += 1

                pygame.display.update()
                clock.tick(60)

            # For agents that did not collide during the simulation, update their fitness
            for i, agent in enumerate(self.population):
                if not agent.collided:
                    self.fitnesses[i] = agent.score

            pygame.quit()

            if not training_running:
                break

            generation += 1
            best_score = max(self.fitnesses)
            print(f"Generation: {generation}, Best Score: {best_score}")
            self.max_fitnesses.append(best_score)
            print(f"Population size: {len(self.population)}")

            self.new_generation()

        # Plot the maximum fitness over generations after training is complete.
        plt.plot(self.max_fitnesses)
        plt.xlabel("Generation")
        plt.ylabel("Max Fitness")
        plt.title("Max Fitness per Generation")
        plt.show()

    def best_agent(self) -> Agent:
        """
        Returns the agent with the highest fitness.
        """
        best_idx = np.argmax(self.fitnesses)
        return self.population[best_idx]

    def play(self) -> None:
        """
        Uses the best agent to play the game.
        """
        agent = self.best_agent()
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()
        env = Env.Env()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if env.time >= 100:
                env.speed += 0.1
                env.time = 0
            env.time += 1

            if agent.check_collision(env):
                running = False
                break

            action = agent.act(env)
            if action == 0 and not agent.is_jumping:
                agent.jump(screen)
            agent.update_jump()

            screen.fill("white")
            env.Draw_Ground(screen)
            env.Draw_Blocks(screen)
            agent.walk(screen)
            agent.score += 1

            pygame.display.update()
            pygame.time.delay(100)
            clock.tick(60)

        pygame.quit()



