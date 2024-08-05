import pygame
import Dino
import Env
# Initialize Pygame
pygame.init()

# Set up the screen, clock, and other necessary variables
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()


# Create instances of your environment and dinosaur
env = Env.Env()
dino = Dino.Dino()

exit_game = False


exit_game = False

while not exit_game:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_game = True
    #check Collisons 
    if dino.is_Collided(env):
        exit_game = True
    #updating the score
    dino.score+=10
    # Adapt the difficulty of the game
    if env.time >= 100:
        env.speed += 0.02
        env.time = 0
    
    env.time += 1
    
    # Getting keys and evaluating pressed keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE] and not dino.is_jumping:
        dino.Jump(screen)
        dino.is_jumping = True

    if dino.is_jumping:
        if dino.y >= 250:
            dino.is_jumping = False
        else:
            dino.y += 25  # Adjust the jump speed here if needed

    # Clear the screen
    screen.fill("white")
    
    # Draw environment and dinosaur
    env.Draw_Ground(screen)
    env.Draw_Blocks(screen)
    dino.Walk(screen)
    
    # Update the display
    pygame.display.update()
    
    # Tick the clock to limit the frame rate
    clock.tick(60)

pygame.quit()