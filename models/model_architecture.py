import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttentionFusionModel(nn.Module):
    def __init__(self, bert_dim, lex_dim, syn_dim, hidden_dim, num_classes):
        super().__init__()
        self.bert_proj = nn.Linear(bert_dim, hidden_dim)
        self.lex_proj = nn.Linear(lex_dim, hidden_dim)
        self.syn_proj = nn.Linear(syn_dim, hidden_dim)

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 3, 3)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, bert_vec, lex_vec, syn_vec):
        # Project all features to same dimension
        bert_feat = self.bert_proj(bert_vec)
        lex_feat = self.lex_proj(lex_vec)
        syn_feat = self.syn_proj(syn_vec)

        # Stack features for attention
        stacked = torch.stack([bert_feat, lex_feat, syn_feat], dim=1)

        # Compute attention weights
        concat = torch.cat([bert_feat, lex_feat, syn_feat], dim=1)
        attn_weights = F.softmax(self.attention(concat), dim=1).unsqueeze(2)

        # Apply attention and fuse features
        fused = torch.sum(attn_weights * stacked, dim=1)

        return self.classifier(fused)
