import torch
from torch import nn

class FrozenCLAPProjectors(nn.Module):
    """Loads pretrained CLAP projector heads and keeps them frozen."""

    def __init__(self, ckpt_path: str, in_dim: int, proj_dim: int) -> None:
        super().__init__()

        ckpt = torch.load(ckpt_path, map_location="cpu")
        expected = {"audio_proj", "text_proj"}
        if not expected.issubset(ckpt.keys()):
            missing = ", ".join(sorted(expected - set(ckpt.keys())))
            raise KeyError(f"Missing projector weights in checkpoint: {missing}")

        self.audio_proj = nn.Linear(in_dim, proj_dim, bias=True)
        self.text_proj = nn.Linear(in_dim, proj_dim, bias=True)

        audio_state = self._extract_linear_state(ckpt["audio_proj"])
        text_state = self._extract_linear_state(ckpt["text_proj"])

        self.audio_proj.load_state_dict(audio_state, strict=True)
        self.text_proj.load_state_dict(text_state, strict=True)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward_audio(self, x: torch.Tensor) -> torch.Tensor:
        z = self.audio_proj(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        z = self.text_proj(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _extract_linear_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "weight" in state_dict and "bias" in state_dict:
            return {"weight": state_dict["weight"], "bias": state_dict["bias"]}

        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("fc."):
                remapped[key.split(".", 1)[1]] = value
        if "weight" not in remapped or "bias" not in remapped:
            raise KeyError("Unable to find linear weights in projector checkpoint state dict")
        return remapped


class ProsodySLM(nn.Module):
    """Prosody-aware semantic head that gates prosody with CLAP semantics."""

    def __init__(
        self,
        clap_ckpt_path: str,
        in_dim: int,
        proj_dim: int,
        num_emotions: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.clap = FrozenCLAPProjectors(clap_ckpt_path, in_dim, proj_dim)
        d_model = proj_dim

        self.prosody_lin = nn.Linear(in_dim, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        fusion_dim = d_model * 5
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

    def train(self, mode: bool = True) -> "ProsodySLM":
        super().train(mode)
        self.clap.eval()
        return self

    def forward(self, x_a: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            s_a = self.clap.forward_audio(x_a)
            s_t = self.clap.forward_text(x_t)

        p = self.prosody_lin(x_a)
        Q = self.W_q(s_t)
        K = self.W_k(s_a)
        V = self.W_v(p)

        scale = Q.size(-1) ** 0.5
        attn_score = (Q * K).sum(dim=-1, keepdim=True) / scale
        attn_weight = attn_score.sigmoid()
        c = attn_weight * V

        feat = torch.cat(
            (c, s_a, s_t, s_a * s_t, (s_a - s_t).abs()),
            dim=-1,
        )
        return self.mlp(feat)
