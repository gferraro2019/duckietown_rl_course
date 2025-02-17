import os
import random
import time
from dataclasses import dataclass

import gym as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from duckietown_rl_course.duckietownrl.gym_duckietown import envs


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

### Algo specific arguments
    lambda_h: float = 0.1
    """The entropy regularization term"""
    learning_rate: float = 0.0003
    """learning rate of the model"""
    batch_size: int = 256
    """batch size"""
    num_epochs: int = 100
    """number of epochs"""




class Discriminator(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Discriminator, self).__init__()
        
        # Vérification des espaces
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)
        
        n_actions = action_space.n
        
        # Architecture CNN similaire à celle d'Atari
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calcul de la taille de sortie du CNN
        sample_input = torch.zeros((1, *observation_space.shape))
        conv_out_size = self.feature_extraction(sample_input).shape[1]
        
        # Couches fully connected
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Sort une seule valeur pour le discriminateur
        )
        
        # Fonction d'activation finale
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, obs, actions=None):
        """
        Args:
            obs: Tensor de forme (batch_size, 3, 32, 32)
            actions: Tensor de forme (batch_size,) - optionnel
        Returns:
            Probabilités que les paires état-action proviennent de l'expert
        """
        # Permutation des dimensions si nécessaire (NHWC -> NCHW)
        if obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)
            
        # Normalisation
        obs = obs.float() / 255.0
        
        # Extraction des features
        features = self.feature_extraction(obs)
        
        # Classification
        logits = self.classifier(features)
        
        # Probabilité finale
        probs = self.sigmoid(logits)
        
        return probs.squeeze()
    
    
    
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, env.single_action_space.n)

    def forward(self, x):
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x




if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    
    # ########## TRAINING ##########
    # Configuration
    batch_size = args.batch_size
    num_epochs = 100
    learning_rate = 1e-4

    # Création d'un environnement factice pour les dimensions
    observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)
    env = type('Env', (), {'single_observation_space': observation_space, 
                          'single_action_space': action_space})()

    # Création des modèles
    q_net = QNetwork(env).to(device)
    discriminator = Discriminator(observation_space, action_space).to(device)

    # Création d'un dataset random
    dataset_size = 10000
    fake_obs = torch.randint(0, 255, (dataset_size, 84, 84, 3), device=device)
    fake_actions = torch.randint(0, action_space.n, (dataset_size,), device=device)
    fake_labels = torch.randint(0, 2, (dataset_size,), device=device).float()  # 0: fake, 1: real

    # Optimizers
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss functions
    q_criterion = nn.MSELoss()
    disc_criterion = nn.BCELoss()

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        q_losses = []
        disc_losses = []
        
        for batch_obs, batch_actions, batch_labels in dataloader:
            # Entraînement du Q-Network
            q_optimizer.zero_grad()
            q_values = q_net(batch_obs)
            # Création de Q-values cibles factices pour l'exemple
            target_q_values = torch.randn_like(q_values)
            q_loss = q_criterion(q_values, target_q_values)
            q_loss.backward()
            q_optimizer.step()
            
            # Entraînement du Discriminateur
            disc_optimizer.zero_grad()
            disc_pred = discriminator(batch_obs)
            disc_loss = disc_criterion(disc_pred, batch_labels)
            disc_loss.backward()
            disc_optimizer.step()
            
            q_losses.append(q_loss.item())
            disc_losses.append(disc_loss.item())
        
        # Affichage des métriques
        avg_q_loss = sum(q_losses) / len(q_losses)
        avg_disc_loss = sum(disc_losses) / len(disc_losses)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average Q-Network Loss: {avg_q_loss:.4f}")
        print(f"Average Discriminator Loss: {avg_disc_loss:.4f}")
        print("-" * 50)


   