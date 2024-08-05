import pygame
import numpy as np
class Env():
    def __init__(self):
        self.ground = "assets\Ground.png"
        self.X1_Ground = 0
        self.X2_Ground = 1204
        self.blocks_Path = ["assets/block_1.png","assets/block_3.png"]
        self.blocks = {}
        self.speed = 0.15
        self.time = 0

    def Draw_Ground(self, Screen: pygame.display):
        Ground = pygame.image.load(self.ground)
        if self.X1_Ground <= -1200:
            self.X1_Ground =self.X2_Ground +1204
        if self.X2_Ground <= -1200:
            self.X2_Ground = self.X1_Ground+1204
        
        self.X1_Ground -= self.speed*200
        self.X2_Ground -= self.speed*200
       

        Screen.blit(Ground, (self.X1_Ground, 280))
        Screen.blit(Ground, (self.X2_Ground, 280))
    def Draw_Blocks(self, Screen: pygame.display):
        #number of obstacle to show:
        random_Num = np.random.randint(0,2)
        Random_Interval = 200*np.random.rand()
        if len(self.blocks) == 0:
            for i in range(random_Num):
                # Randomly select a block
                randomBlock = np.random.randint(0, 2)
                block = pygame.image.load(self.blocks_Path[randomBlock])
                self.blocks[block] = [950 + i*Random_Interval, 250]

           
            
           
        else:
            blocks_to_remove = []
            for block, position in self.blocks.items():
                if position[0] <= 0:
                    blocks_to_remove.append(block)
                else:
                    position[0] -= 200 * self.speed
                    Screen.blit(block, position)
            
            for block in blocks_to_remove:
                self.blocks.pop(block)

            

            

       


        
        

