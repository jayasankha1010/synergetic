import torch
import torch.nn as nn
DROPOUT_RATE = 0.2

class SingleDiseaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64, 64, 64, 64, 64, 64, 32], dropout_rate=DROPOUT_RATE):
        """
        A standard MLP for a single disease (CP or DEE) with 5 hidden layers.
        """
        super(SingleDiseaseMLP, self).__init__()
        
        # Ensure we have exactly 5 hidden layers as requested
        # assert len(hidden_dims) == 5, "Please provide exactly 5 hidden dimensions."
        
        # Build the 5 hidden layers
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim)) # Added for training stability
            layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification head (Output is 1 probability score)
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        probability = self.classifier(features)
        return probability.squeeze()


class CombinedDiseaseMLP(nn.Module):
    def __init__(self, input_dim, shared_dims=[512, 256, 128], head_dims=[64, 64, 64, 64, 32], dropout_rate=DROPOUT_RATE):
        """
        A Multi-Task MLP with 3 shared layers and 2 separate branches (2 layers each) 
        for CP and DEE.
        """
        super(CombinedDiseaseMLP, self).__init__()
        
        # assert len(shared_dims) == 3, "Shared layers must be exactly 3."
        # assert len(head_dims) == 2, "Task-specific heads must have exactly 2 layers."
        
        # --- SHARED LAYERS (3 Hidden Layers) ---
        shared_layers = []
        current_dim = input_dim
        for h_dim in shared_dims:
            shared_layers.append(nn.Linear(current_dim, h_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(h_dim))
            shared_layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim
            
        self.shared_extractor = nn.Sequential(*shared_layers)
        
        # --- BRANCH 1: CP HEAD (2 Hidden Layers) ---
        self.cp_head = self._build_head(current_dim, head_dims)
        
        # --- BRANCH 2: DEE HEAD (2 Hidden Layers) ---
        self.dee_head = self._build_head(current_dim, head_dims)

    def _build_head(self, in_dim, head_dims, dropout_rate=DROPOUT_RATE):
        """Helper to build the 2-layer classification branches."""
        layers = []
        curr_dim = in_dim
        for h_dim in head_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = h_dim
            
        # Add the final classification node
        layers.append(nn.Linear(curr_dim, 1))
        # layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Pass input through shared layers
        shared_features = self.shared_extractor(x)
        
        # Pass shared features into both specific heads
        prob_cp = self.cp_head(shared_features)
        prob_dee = self.dee_head(shared_features)
        
        return prob_cp.squeeze(), prob_dee.squeeze()