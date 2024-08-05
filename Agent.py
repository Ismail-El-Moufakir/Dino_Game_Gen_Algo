import numpy as np
import torch
import pygame
from utils import POPULATION_SIZE, CROSSOVER_RATE, NBRE_GENERATION
from torch import nn
import matplotlib.pyplot as plt
import Env

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = 0
        self.y = 250  # initial y position of Agent Dino

        self.WalkState = 0  # 0: LEFT, 1: RIGHT
        self.collided = False
        self.is_jumping = False
        self.jump_velocity = 0
        self.Walk_Left = "assets/Walk_Left.png"
        self.Walk_Right = "assets/Walk_Right.png"
        self.jump = "assets/Jump.png"
        self.flatten = nn.Flatten()
        self.Linear_Relu_Stack = nn.Sequential(
            nn.Linear(3, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        logits = self.Linear_Relu_Stack(x)
        return logits

    def action(self, env):
        block_position1 = 950
        if len(env.blocks) >= 1:
            block_position1 = [position[0] for position in env.blocks.values()][-1]
        
        x = torch.tensor([block_position1, self.y, env.speed], dtype=torch.float32)
        output = self.forward(x)
        action = torch.argmax(output).item()
        return action

    def Walk(self, screen):
        DinoSurface = pygame.image.load(self.Walk_Left if self.WalkState == 0 else self.Walk_Right)
        self.WalkState = 1 - self.WalkState
        screen.blit(DinoSurface, (50, self.y))

    def Jump(self, screen):
        if not self.is_jumping:
            DinoSurface = pygame.image.load(self.jump)
            self.is_jumping = True
            self.jump_velocity = -25  # Set initial jump velocity
            self.y += self.jump_velocity
            screen.blit(DinoSurface, (50, self.y))

    def Update_Jump(self):
        if self.is_jumping:
            self.y += self.jump_velocity
            self.jump_velocity += 5  # Simulating gravity
            if self.y >= 250:  # Check if on the ground
                self.y = 250
                self.is_jumping = False
                self.jump_velocity = 0

    def is_Collided(self, env):
        DinoSurface = pygame.image.load(self.Walk_Right)
        dino_rect = pygame.Rect(50, self.y, DinoSurface.get_width(), DinoSurface.get_height())
        for block, position in env.blocks.items():
            block_rect = pygame.Rect(position[0], position[1], block.get_width(), block.get_height())
            if dino_rect.colliderect(block_rect):
                return True
        return False

class Gene_Algo():
    def __init__(self):
        self.Population = [Agent() for _ in range(POPULATION_SIZE)]
        self.fitnesses = [0 for _ in range(POPULATION_SIZE)]
        self.max_fitnesses = []

    def fitness_function(self, agent):
        return agent.score

    def Mutation(self, agent):
        for param in agent.parameters():
            mutation_tensor = torch.normal(0, 0.1, size=param.shape)  # Adjust mutation strength
            param.data += mutation_tensor

    def crossOver(self, agent1, agent2):
        child = Agent()
        for param1, param2, child_param in zip(agent1.parameters(), agent2.parameters(), child.parameters()):
            split_point = int(len(param1) * np.random.rand())
            new_param = torch.cat((param1[:split_point], param2[split_point:]))
            child_param.data = new_param
        return child

    def Selection(self):
        tournament_size = 5  # Number of individuals in each tournament
        selected_population = []
        
        for _ in range(POPULATION_SIZE):
            tournament = np.random.choice(self.Population, size=tournament_size)
            best_in_tournament = max(tournament, key=self.fitness_function)
            selected_population.append(best_in_tournament)
        
        return selected_population

    def New_Generation(self):
        new_population = []
        selected_population = self.Selection()

        while len(new_population) < POPULATION_SIZE:
            if np.random.rand() < CROSSOVER_RATE:
                parent1  = selected_population[np.random.randint(0, len(selected_population))]
                parent2  = selected_population[np.random.randint(0, len(selected_population))]
                child = self.crossOver(parent1, parent2)
                new_population.append(child)
            else:
                individual = selected_population[np.random.randint(0, len(selected_population))]
                individual.score = 0
                individual.collided = False
                self.Mutation(individual)
                new_population.append(individual)
        self.Population = new_population

    def Train(self):
        generation = 0
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()

        for _ in range(NBRE_GENERATION):
            env = Env.Env()
            nbre_collided = 0
            is_running = True
            while is_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False

                for i in range(POPULATION_SIZE):
                    if self.Population[i].is_Collided(env) and not self.Population[i].collided:
                        self.fitnesses[i] = self.Population[i].score
                        self.Population[i].collided = True
                        nbre_collided += 1
                        print("total collided: ", nbre_collided)
                        if nbre_collided == POPULATION_SIZE:
                            is_running = False
                            break

                if env.time >= 100:
                    env.speed += 0.05
                    env.time = 0
                env.time += 1

                for i in range(POPULATION_SIZE):
                    action = self.Population[i].action(env)
                    if action == 0 and not self.Population[i].is_jumping:
                        self.Population[i].Jump(screen)
                    self.Population[i].Update_Jump()

                screen.fill("white")
                env.Draw_Ground(screen)
                env.Draw_Blocks(screen)
                for i in range(POPULATION_SIZE):
                    if not self.Population[i].collided:
                        self.Population[i].Walk(screen)
                        self.Population[i].score += 1

                pygame.display.update()
                clock.tick(60)

            generation += 1
            self.New_Generation()
            #all info
            print(f"Generation: {generation}, Best Score: {max(self.fitnesses)}")
            self.max_fitnesses.append(max(self.fitnesses))
            print(f"population size: {len(self.Population)}")
        pygame.quit()
        #plotting the max fitnesses over generations
        plt.plot(self.max_fitnesses)
        plt.xlabel("Generation")
        plt.ylabel("Max Fitness")
        plt.show()

    def Best_Agent(self):
        return self.Population[np.argmax(self.fitnesses)]

    def play(self):
        agent = self.Best_Agent()
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()
        env = Env.Env()
        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

            if env.time >= 100:
                env.speed += 0.1
                env.time = 0
            env.time += 1
            #check collision
            if agent.is_Collided(env):
                is_running = False
                break
            action = agent.action(env)
            if action == 0 and not agent.is_jumping:
                agent.Jump(screen)
            agent.Update_Jump()

            screen.fill("white")
            env.Draw_Ground(screen)
            env.Draw_Blocks(screen)
            agent.Walk(screen)
            agent.score += 1

            pygame.display.update()
            pygame.time.delay(100)
            clock.tick(60)
