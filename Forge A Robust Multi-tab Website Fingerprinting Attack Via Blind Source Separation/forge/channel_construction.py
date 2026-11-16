"""Module 1: Traffic Channel Construction via Time-Delay Embedding (Section 4.1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficChannelConstructor(nn.Module):
    """Maps a 1D traffic sequence x in R^(B x l) into an 8-channel state-space V in R^(B x 8 x l).

    Channels:
        v0: Count of outgoing packets (+1) in window [t - tau + 1 : t + 1]
        v1: Count of incoming packets (-1) in window [t - tau + 1 : t + 1]
        v2: Total inter-arrival span sum(ts_{i+1} - ts_i) for outgoing packets in window
        v3: Total inter-arrival span sum(ts'_{j+1} - ts'_j) for incoming packets in window
        v4: Number of outgoing bursts |B_up| in window
        v5: Number of incoming bursts |B_down| in window
        v6: Mean outgoing burst length in window
        v7: Mean incoming burst length in window
    """

    def __init__(self, interval: int = 20, seq_len: int = 8000):
        super().__init__()
        self.interval = interval
        self.seq_len = seq_len

    @staticmethod
    def align_length(x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Truncate or right-pad sequence x (B, L) to exact seq_len."""
        cur_len = x.size(-1)
        if cur_len > seq_len:
            return x[..., :seq_len]
        if cur_len < seq_len:
            return F.pad(x, (0, seq_len - cur_len), value=0.0)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 8-channel time-delay embedding on-the-fly.

        Args:
            x: Raw traffic sequence of shape (B, L), encoding direction * timestamp (or +/-1).
        Returns:
            V: State-vector time series of shape (B, 8, seq_len).
        """
        x = self.align_length(x.float(), self.seq_len)
        tau = self.interval

        d = torch.sign(x)
        ts = torch.abs(x)

        # Compatibility fallback: if trace contains pure direction {-1, 0, +1} without timestamps
        # (every non-zero magnitude == 1.0), assign monotonic index timestamps.
        nonzero = d != 0
        if nonzero.any():
            nz_ts = ts[nonzero]
            if nz_ts.max() <= 1.0 and torch.all(nz_ts == 1.0):
                idx_ts = torch.arange(1, self.seq_len + 1, device=x.device, dtype=x.dtype)
                ts = idx_ts.unsqueeze(0).expand_as(x) * nonzero.to(x.dtype)
                x_eff = d * ts
            else:
                x_eff = x
        else:
            x_eff = x

        # Causal left-padding of length (tau - 1) so each step t covers [t - tau + 1 : t + 1]
        pad = tau - 1
        x_win = F.pad(x_eff, (pad, 0), value=0.0).unfold(-1, tau, 1)  # (B, seq_len, tau)
        up_win = x_win > 0
        down_win = x_win < 0
        ts_win = x_win.abs()

        # Volumetric Dynamics: v0, v1
        v0 = up_win.sum(dim=-1).to(x.dtype)
        v1 = down_win.sum(dim=-1).to(x.dtype)

        # Temporal Structure: v2, v3
        # For sorted timestamps T_up, sum(ts_{i+1} - ts_i) == max(T_up) - min(T_up) when |T_up| >= 2.
        neg_sentinel = torch.tensor(-1e9, device=x.device, dtype=x.dtype)
        pos_sentinel = torch.tensor(1e9, device=x.device, dtype=x.dtype)

        up_max = torch.where(up_win, ts_win, neg_sentinel).amax(dim=-1)
        up_min = torch.where(up_win, ts_win, pos_sentinel).amin(dim=-1)
        v2 = torch.where(v0 > 1.0, (up_max - up_min).clamp_min(0.0), torch.zeros_like(v0))

        down_max = torch.where(down_win, ts_win, neg_sentinel).amax(dim=-1)
        down_min = torch.where(down_win, ts_win, pos_sentinel).amin(dim=-1)
        v3 = torch.where(v1 > 1.0, (down_max - down_min).clamp_min(0.0), torch.zeros_like(v1))

        # Interaction Patterns: v4, v5, v6, v7
        up_starts = up_win[..., 0].to(x.dtype) + (
            up_win[..., 1:] & (~up_win[..., :-1])
        ).sum(dim=-1).to(x.dtype)
        down_starts = down_win[..., 0].to(x.dtype) + (
            down_win[..., 1:] & (~down_win[..., :-1])
        ).sum(dim=-1).to(x.dtype)

        v4 = up_starts
        v5 = down_starts
        v6 = torch.where(v4 > 0.0, v0 / v4.clamp_min(1.0), torch.zeros_like(v0))
        v7 = torch.where(v5 > 0.0, v1 / v5.clamp_min(1.0), torch.zeros_like(v1))

        V = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1)  # (B, 8, seq_len)
        return V

