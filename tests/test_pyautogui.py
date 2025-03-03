import pygame
import sys
import time

def main():
    # Initialisation de pygame
    pygame.init()
    
    # Créer une fenêtre visible, mais très petite
    window = pygame.display.set_mode((1, 1))
    pygame.display.set_caption("")  # Titre vide
    
    print("Écouteur de clavier actif. Utilisez les flèches, espace ou échap pour quitter.")
    print("La fenêtre doit avoir le focus - cliquez dessus si nécessaire")
    
    running = True
    while running:
        # Assurez-vous que des événements sont générés en rafraîchissant l'écran
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print("up")
                elif event.key == pygame.K_DOWN:
                    print("down")
                elif event.key == pygame.K_LEFT:
                    print("left")
                elif event.key == pygame.K_RIGHT:
                    print("right")
                elif event.key == pygame.K_SPACE:
                    print("space")
                elif event.key == pygame.K_ESCAPE:
                    print("escape - quitting")
                    running = False
                else:
                    # Pour voir toutes les touches
                    print(f"Autre touche: {pygame.key.name(event.key)}")
        
        # Petit délai pour réduire l'utilisation du CPU
        time.sleep(0.01)
        
    pygame.quit()
    print("Programme terminé")

if __name__ == "__main__":
    main()
