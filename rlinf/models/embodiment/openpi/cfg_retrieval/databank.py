"""DataBank for π-VLA Method 1 (training-free retrieval CFG).

Adapted from dc_cfg_reference/databank.py for the π0/π05 LIBERO setting.

What changed vs the reference:
  - Key is the 8-d proprio state (no separate goal — the language prompt acts as
    "goal" but is handled via task_id hard-filter rather than a continuous key).
  - Action is a 5-step chunk in the model's normalized 32-d slot
    (i.e. exactly what `sample_actions` operates on as `x_t`/`x_0`), so v_R
    arithmetic stays in the same space and no de-/re-normalization is needed
    inside the denoise loop.
  - KNN supports per-query `task_id_filter` so retrieval never crosses tasks
    (LIBERO Spatial = 10 distinct task descriptions; mixing actions across
    tasks is meaningless).

P1 invariant is enforced by the caller (sample_actions): when w=0 the bank is
never queried, so an empty / corrupt bank cannot break the baseline policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class DataBank:
    state_dim: int = 8           # raw proprio dim (post-norm)
    action_chunk: int = 5        # # of action steps stored per entry
    action_dim: int = 32         # padded action dim (model x_t space)
    device: torch.device = torch.device("cuda")

    _states: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    _task_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    _actions: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self):
        for name in ("_states", "_actions"):
            t = getattr(self, name)
            setattr(self, name, t.to(self.device))
        self._task_ids = self._task_ids.to(self.device)

    @property
    def size(self) -> int:
        return int(self._states.shape[0])

    def add_batch(
        self,
        states: torch.Tensor,           # (N, state_dim)
        task_ids: torch.Tensor,         # (N,) long
        actions: torch.Tensor,          # (N, action_chunk, action_dim)
    ) -> None:
        states = states.to(self.device).contiguous()
        task_ids = task_ids.to(self.device, dtype=torch.long).contiguous()
        actions = actions.to(self.device).contiguous()
        n = states.shape[0]
        assert task_ids.shape[0] == n and actions.shape[0] == n
        assert states.shape[1] == self.state_dim, (
            f"state_dim mismatch: bank={self.state_dim} vs new={states.shape[1]}"
        )
        assert actions.shape[1:] == (self.action_chunk, self.action_dim), (
            f"action shape mismatch: bank={(self.action_chunk, self.action_dim)} "
            f"vs new={tuple(actions.shape[1:])}"
        )
        if self._states.numel() == 0:
            self._states = states.clone()
            self._task_ids = task_ids.clone()
            self._actions = actions.clone()
        else:
            self._states = torch.cat([self._states, states], dim=0)
            self._task_ids = torch.cat([self._task_ids, task_ids], dim=0)
            self._actions = torch.cat([self._actions, actions], dim=0)

    @torch.no_grad()
    def knn(
        self,
        query_states: torch.Tensor,         # (B, state_dim) post-norm proprio
        k: int,
        query_task_ids: torch.Tensor | None = None,  # (B,) long; None = no filter
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve top-K bank entries per query, optionally filtered by task_id.

        Returns:
            R_states : (B, K, state_dim)   — raw stored proprio of neighbors
            R_actions: (B, K, action_chunk, action_dim)
            R_dists  : (B, K)              — L2 distance on `query_states`-`R_states`
        """
        if self.size == 0:
            B = query_states.shape[0]
            return (
                torch.zeros(B, k, self.state_dim, device=self.device),
                torch.zeros(B, k, self.action_chunk, self.action_dim, device=self.device),
                torch.full((B, k), float("inf"), device=self.device),
            )

        query_states = query_states.to(self.device)
        # (B, N) L2 distance
        dists = torch.cdist(query_states, self._states, p=2)
        if query_task_ids is not None:
            query_task_ids = query_task_ids.to(self.device, dtype=torch.long)
            # mask: True where bank task_id != query task_id → set dist to +inf
            # bank_tids: (N,), query: (B,) → broadcast (B, N)
            bank_tids = self._task_ids.unsqueeze(0).expand(query_states.shape[0], -1)
            qry_tids = query_task_ids.unsqueeze(1).expand_as(bank_tids)
            mask = bank_tids != qry_tids
            dists = dists.masked_fill(mask, float("inf"))

        # If a query has no matching task entries, topk will still work but
        # all distances will be inf — caller's softmax produces uniform weights
        # over arbitrary actions, which is bad. We could fall back to no-filter
        # in that case but it's cleaner to surface the issue and let the caller
        # log a warning. For now, just return whatever topk gives.
        topk_dists, topk_idx = torch.topk(
            dists, min(k, self._states.shape[0]), dim=-1, largest=False
        )
        # If k > N, pad
        if topk_idx.shape[-1] < k:
            pad = k - topk_idx.shape[-1]
            B = query_states.shape[0]
            topk_dists = torch.cat(
                [topk_dists, torch.full((B, pad), float("inf"), device=self.device)], dim=-1
            )
            topk_idx = torch.cat(
                [topk_idx, torch.zeros((B, pad), dtype=torch.long, device=self.device)],
                dim=-1,
            )
        R_states = self._states[topk_idx]                         # (B, K, state_dim)
        R_actions = self._actions[topk_idx]                       # (B, K, chunk, action_dim)
        return R_states, R_actions, topk_dists

    def save(self, path: str | Path, task_id_lookup: dict[str, int] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "states": self._states.cpu(),
            "task_ids": self._task_ids.cpu(),
            "actions": self._actions.cpu(),
            "state_dim": self.state_dim,
            "action_chunk": self.action_chunk,
            "action_dim": self.action_dim,
        }
        if task_id_lookup is not None:
            payload["task_id_lookup"] = dict(task_id_lookup)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cuda") -> "DataBank":
        # weights_only=False because we may store the {task_str: int} lookup dict.
        # Trusted source: only files we wrote ourselves under outputs/.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        states = ck["states"]
        task_ids = ck.get("task_ids", torch.zeros(states.shape[0], dtype=torch.long))
        actions = ck["actions"]
        state_dim = int(ck.get("state_dim", states.shape[1]))
        action_chunk = int(ck.get("action_chunk", actions.shape[1]))
        action_dim = int(ck.get("action_dim", actions.shape[2]))
        bank = cls(
            state_dim=state_dim,
            action_chunk=action_chunk,
            action_dim=action_dim,
            device=torch.device(device),
        )
        bank.add_batch(states, task_ids, actions)
        return bank


def merge_shards_to_bank(bank_path: str | Path) -> str:
    """Combine all shard files in <bank_path>.shards/ into a single bank file at bank_path.

    The shards are produced incrementally by sample_actions during a collection eval
    (each shard ~64 calls × batch_size entries). This function consolidates them.
    """
    import glob
    bank_path = Path(bank_path)
    shards_dir = Path(str(bank_path) + ".shards")
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"No shards dir at {shards_dir}")
    shard_files = sorted(glob.glob(str(shards_dir / "shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files in {shards_dir}")
    states_all, task_ids_all, actions_all = [], [], []
    task_id_lookup: dict[str, int] = {}
    state_dim = action_chunk = action_dim = None
    for sp in shard_files:
        ck = torch.load(sp, map_location="cpu", weights_only=False)
        states_all.append(ck["states"])
        task_ids_all.append(ck.get("task_ids", torch.zeros(ck["states"].shape[0], dtype=torch.long)))
        actions_all.append(ck["actions"])
        state_dim = state_dim or int(ck["state_dim"])
        action_chunk = action_chunk or int(ck["action_chunk"])
        action_dim = action_dim or int(ck["action_dim"])
        task_id_lookup.update(ck.get("task_id_lookup", {}))
    states = torch.cat(states_all, dim=0)
    task_ids = torch.cat(task_ids_all, dim=0)
    actions = torch.cat(actions_all, dim=0)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "states": states,
            "task_ids": task_ids,
            "actions": actions,
            "state_dim": state_dim,
            "action_chunk": action_chunk,
            "action_dim": action_dim,
            "task_id_lookup": task_id_lookup,
        },
        bank_path,
    )
    return f"Merged {len(shard_files)} shards into {bank_path} (N={states.shape[0]} entries)"


def build_bank_from_rollout_log(
    rollout_pkl_path: str | Path,
    success_only: bool = True,
    device: torch.device | str = "cuda",
) -> DataBank:
    """Convenience: build a bank from a rollout log produced by `collect_bank.py`.

    The log is expected to be a `torch.save`d dict with keys:
        states     : (N, 8)               post-norm proprio at each model-call timestep
        task_ids   : (N,) long
        actions    : (N, 5, 32)           normalized action chunk that was sampled
        successes  : (N,) bool            whether the parent episode succeeded

    `success_only=True` keeps only entries from successful episodes — these are
    the ones whose actions are demonstrably "good" exemplars.
    """
    ck = torch.load(rollout_pkl_path, map_location="cpu", weights_only=True)
    states = ck["states"]
    task_ids = ck["task_ids"]
    actions = ck["actions"]
    if success_only:
        mask = ck["successes"].bool()
        states, task_ids, actions = states[mask], task_ids[mask], actions[mask]
    bank = DataBank(
        state_dim=states.shape[1],
        action_chunk=actions.shape[1],
        action_dim=actions.shape[2],
        device=torch.device(device),
    )
    bank.add_batch(states, task_ids, actions)
    return bank
