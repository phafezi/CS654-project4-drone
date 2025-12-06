import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    def __init__(self, in_channels, n_actions,  kin_dim):
        super().__init__()

        # Image feature extractor
        self.featureExtractor= nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )

        # MLP feature extractor
        self.MLP= nn.Sequential(
            nn.Linear(512 + kin_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Actor
        self.actor_head = nn.Linear(512, n_actions)
        self.logstd = nn.Parameter(torch.zeros(n_actions))

        # Critic
        self.value_head = nn.Linear(512, 1)

    def forward(self, img, kin):
        #image processing
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img/255.0
        img_feature = self.featureExtractor(img)
        img_feature = img_feature.view(img_feature.size(0), -1)
        #joint input
        x = torch.cat([img_feature, kin], dim=1)
  
        #MLP on the joint data
        features = self.MLP(x)

        # Actor
        mean = self.actor_head(features)
        std  = self.logstd.exp().clamp(1e-3, 50)
        dist = torch.distributions.Normal(mean, std)

        # Critic
        value = self.value_head(features)

        return dist, value