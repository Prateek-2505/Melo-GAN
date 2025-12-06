import numpy as np
import matplotlib.pyplot as plt
import os
import json

def plot_gan_losses(log_data):
    """
    Visualizes the 3 components of the GAN Loss.
    """
    epochs = np.arange(1, len(log_data['d_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, log_data['d_loss'], label='Critic Loss (WGAN)', color='purple', alpha=0.7)
    plt.plot(epochs, log_data['g_adv'], label='Generator Adv Loss', color='green', alpha=0.7)
    plt.plot(epochs, log_data['g_emo'], label='Generator Emotion Loss', color='orange', linewidth=2)
    
    plt.title('GAN Training Losses (WGAN-GP + Emotion Guidance)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gan_losses.png')
    plt.close()
   
    
if __name__ == "__main__":
    history_path = os.path.join("experiments/gan/logs", "gan_training_history.json")
    with open(history_path, 'r') as f:
        history = json.load(f)
    plot_gan_losses(history)
    
    
