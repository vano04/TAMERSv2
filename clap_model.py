import torch, torch.nn.functional as F
from torch import nn

class ProjHead(nn.Module):
    def __init__(self, dim_in=768, dim_out=768, dropout=0.1):
        super().__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.fc(x)
        z = self.drop(z)
        return F.normalize(z, p=2, dim=-1)

class CLAP(nn.Module):
    def __init__(self, dim=768) -> None:
        super().__init__()

        self.audio_proj = ProjHead(dim, dim) # audio embedding projection
        self.text_proj = ProjHead(dim, dim) # text embedding projection

        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, a_emb, t_emb):
        z_a = self.audio_proj(a_emb) # (N, d)
        z_t = self.text_proj(t_emb) # (N, d)

        return None, z_a, z_t

    def info_nce(self, logits):
        N = logits.size(0)
        labels = torch.arange(N, device=logits.device)

        loss_t2a = F.cross_entropy(logits, labels) # text -> audio
        loss_a2t = F.cross_entropy(logits.t(), labels) # audio -> text

        return 0.5 * (loss_t2a + loss_a2t)