import torch
import torch.nn.functional as F

from src.algorithms.dreamer.networks import Projector
from .dreamerv3 import DreamerV3


class R2Dreamer(DreamerV3):
    """R2Dreamer: DreamerV3 world model with Barlow Twins and/or InfoNCE auxiliary loss.

    Reconstruction is disabled by default (loss_scales.recon: 0).
    Both barlow and infonce are independently toggleable via loss_scales.
    """

    def _init_extra(self, config, shapes):
        super()._init_extra(config, shapes)  # builds decoder if recon > 0
        if (
            self._loss_scales.get("barlow", 0) > 0
            or self._loss_scales.get("infonce", 0) > 0
        ):
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            self.barlow_lambd = float(config.r2dreamer.lambd)

    def _build_module_dict(self) -> dict:
        mods = super()._build_module_dict()
        if hasattr(self, "prj"):
            mods["projector"] = self.prj
        return mods

    def _compute_rep_losses(
        self, post_stoch, post_deter, feat, embed, data, initial, B, T
    ) -> dict:
        losses = super()._compute_rep_losses(
            post_stoch, post_deter, feat, embed, data, initial, B, T
        )
        if not hasattr(self, "prj"):
            return losses

        # R2-Dreamer: project latent features and encoder embeddings into a
        # shared space, then compute cross-correlation between the two views.
        # Flatten batch/time dims so we get one large cross-correlation matrix.
        # (B, T, F) -> (B*T, proj_dim)
        x1 = self.prj(feat.reshape(B * T, -1))
        # (B, T, E) -> (B*T, E) — detach so gradients don't flow through embed
        x2 = embed.reshape(B * T, -1).detach()  # this detach is important

        if self._loss_scales.get("barlow", 0) > 0:
            losses["barlow"] = self._barlow_loss(x1, x2, B, T)
        if self._loss_scales.get("infonce", 0) > 0:
            losses["infonce"] = self._infonce_loss(x1, x2)
        return losses

    def _barlow_loss(self, x1, x2, B, T):
        """Barlow Twins redundancy-reduction loss between projected features and embeddings."""
        x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
        x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
        # Normalised cross-correlation matrix between the two views
        c = torch.mm(x1_norm.T, x2_norm) / (B * T)
        # Diagonal should be 1 — invariance between the two views
        invariance = (torch.diagonal(c) - 1.0).pow(2).sum()
        # Off-diagonal should be 0 — decorrelate latent dimensions
        off_diag = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
        redundancy = c[off_diag].pow(2).sum()
        return invariance + self.barlow_lambd * redundancy

    def _infonce_loss(self, x1, x2):
        """Contrastive InfoNCE loss between projected latent features and encoder embeddings."""
        logits = torch.matmul(x1, x2.T)
        # Numerically stable softmax via max subtraction
        logits = logits - logits.max(dim=1, keepdim=True).values
        labels = torch.arange(logits.shape[0], device=x1.device)
        return F.cross_entropy(logits, labels)
