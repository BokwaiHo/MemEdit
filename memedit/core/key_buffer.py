"""Key buffer per memory shard.

The null-space projector P_⊥ = I − K_0^T (K_0 K_0^T)^-1 K_0 requires the
set of "preserved" keys K_0. In the paper, K_0 is the set of representative
probe keys of *existing* memories at the target layer l*.

We maintain this as a rolling FIFO buffer of probe keys (the hidden-state
input to the memory module, possibly propagated to layer l*'s input via a
forward pass). Each shard has its own buffer, and the SVD of that buffer is
recomputed lazily when the buffer changes.
"""

from __future__ import annotations

from typing import Optional

import torch

from memedit.utils.linalg import (
    compute_null_space_projector,
    null_space_rank,
)


class KeyBuffer:
    """Rolling buffer of probe keys at a single target layer.

    The buffer is a (n, d) tensor; a new key is appended with `add`, and if
    the buffer exceeds `max_size` the oldest row is evicted.

    `projector()` returns P_⊥ at the current state; it is recomputed only
    when the buffer has changed since the previous call.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_size: int = 20000,
        eps_svd: float = 1e-5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        self.eps_svd = eps_svd
        self.device = torch.device(device)
        self.dtype = dtype

        self._keys = torch.empty(0, hidden_dim, device=self.device, dtype=dtype)
        self._cached_P: Optional[torch.Tensor] = None
        self._dirty = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, key: torch.Tensor) -> None:
        """Append a single key vector. key: (d,)"""
        if key.ndim != 1 or key.shape[0] != self.hidden_dim:
            raise ValueError(f"expected shape ({self.hidden_dim},), got {tuple(key.shape)}")
        k = key.detach().to(device=self.device, dtype=self.dtype).unsqueeze(0)
        if self._keys.shape[0] >= self.max_size:
            # Evict oldest.
            self._keys = torch.cat([self._keys[1:], k], dim=0)
        else:
            self._keys = torch.cat([self._keys, k], dim=0)
        self._dirty = True

    def extend(self, keys: torch.Tensor) -> None:
        """Append a batch of keys. keys: (n, d)"""
        if keys.ndim != 2 or keys.shape[1] != self.hidden_dim:
            raise ValueError(f"expected (n, {self.hidden_dim}); got {tuple(keys.shape)}")
        k = keys.detach().to(device=self.device, dtype=self.dtype)
        cat = torch.cat([self._keys, k], dim=0)
        if cat.shape[0] > self.max_size:
            cat = cat[-self.max_size:]
        self._keys = cat
        self._dirty = True

    def clear(self) -> None:
        self._keys = torch.empty(0, self.hidden_dim, device=self.device, dtype=self.dtype)
        self._dirty = True

    def remove_key(self, key: torch.Tensor, tol: float = 1e-5) -> bool:
        """Attempt to remove a key vector matching `key` from the buffer.

        Used by DELETE, where we need to exclude the deleted memory's own
        key from the preserved set (see K^{(-i)}_0 in Sec. 3.3 Delete).
        Returns True if a match was found and removed.
        """
        if self._keys.shape[0] == 0:
            return False
        k = key.to(device=self.device, dtype=self.dtype)
        diffs = (self._keys - k.unsqueeze(0)).norm(dim=1)
        idx = int(diffs.argmin().item())
        if diffs[idx].item() < tol:
            mask = torch.ones(self._keys.shape[0], dtype=torch.bool, device=self.device)
            mask[idx] = False
            self._keys = self._keys[mask]
            self._dirty = True
            return True
        return False

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    @property
    def keys(self) -> torch.Tensor:
        return self._keys

    @property
    def size(self) -> int:
        return int(self._keys.shape[0])

    def projector(self) -> torch.Tensor:
        """Return P_⊥. Cached; recomputed lazily when the buffer changes."""
        if self._cached_P is None or self._dirty:
            self._cached_P = compute_null_space_projector(self._keys, self.eps_svd)
            self._dirty = False
        return self._cached_P

    def projector_excluding(self, key: torch.Tensor, tol: float = 1e-5) -> torch.Tensor:
        """Return P_⊥ computed with `key` removed from the preserved set.

        Does NOT mutate the buffer. Used by DELETE to form P_⊥^{(-i)}.
        """
        if self._keys.shape[0] == 0:
            d = self.hidden_dim
            return torch.eye(d, device=self.device, dtype=self.dtype)
        k = key.to(device=self.device, dtype=self.dtype)
        diffs = (self._keys - k.unsqueeze(0)).norm(dim=1)
        idx = int(diffs.argmin().item())
        if diffs[idx].item() >= tol:
            # key not in buffer; just use current projector
            return self.projector()
        mask = torch.ones(self._keys.shape[0], dtype=torch.bool, device=self.device)
        mask[idx] = False
        sub_keys = self._keys[mask]
        return compute_null_space_projector(sub_keys, self.eps_svd)

    def null_rank(self) -> int:
        """Return the null-space dimension of K_0."""
        null_dim, _ = null_space_rank(self._keys, self.eps_svd)
        return null_dim

    def null_fraction(self) -> float:
        """Null-space dim / d; monitored against MoME's δ threshold."""
        if self.hidden_dim == 0:
            return 0.0
        return self.null_rank() / self.hidden_dim
