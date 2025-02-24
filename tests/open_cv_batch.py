import numpy as np
import cv2

# Créer un tableau de matrices
images = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)

# Convertir les images en HSV en mode batch
hsv_images = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

print(hsv_images.shape)  # Output : (10, 256, 256, 3)
