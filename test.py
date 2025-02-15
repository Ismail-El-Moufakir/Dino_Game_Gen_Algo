import pygame
import torch
from Env import Env
from Dino import Dino
from Agent import Agent,GeneticAlgorithm



if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.train()
    # To run the best agent after training, uncomment the following line:
    ga.play()