import pygame
import torch
from Env import Env
from Dino import Dino
from Agent import Agent,Gene_Algo

Gene_Algo = Gene_Algo()
env = Env()
Gene_Algo.Train()
Gene_Algo.play()
