import pygame
import Env
class Dino():
    def __init__(self):
        self.Walk_Left = "assets\Walk_Left.png"
        self.Walk_Right = "assets\Walk_Right.png"
        self.is_jumping = False
        self.y = 250
        self.jump = "assets\Jump.png"
        self.WalkState = 0 # 0 : LEFT 1: RIGHT
        self.score = 0


    def Walk(self,screen:pygame.display):
        DinoSurface = None
        if self.WalkState == 0 :
            DinoSurface = pygame.image.load(self.Walk_Left)
            self.WalkState = 1
        else:
            DinoSurface = pygame.image.load(self.Walk_Right)
            self.WalkState = 0
        screen.blit(DinoSurface,(50,self.y))
        pygame.time.delay(70)

    def Jump(self,screen:pygame.display):
        DinoSurface = pygame.image.load(self.jump)
        self.is_jumping = True
        self.y-=125
        screen.blit(DinoSurface,(50,self.y)) 
        
    def is_Collided(self, env: Env):
        DinoSurface = pygame.image.load(self.Walk_Right)

        for block, position in env.blocks.items():
            block_rect = block.get_rect(topleft=(position[0], position[1]))
            dino_rect = pygame.Rect(50, self.y, DinoSurface.get_width(), DinoSurface.get_height())  
            
            if dino_rect.colliderect(block_rect):
                return True
        return False

        
            
