# Copyright (c) 2026
# ADPP (Adaptive Densification/Pruning & Perceptual sharpening) controller
#
# This file is designed to be *robust* across iterative development:
# - ADPPConfig accepts both "old" and "new" keyword names.
# - ADPPConfig tolerates unknown kwargs (stored in .extra) to avoid breaking runs.
# - ADPPController.step supports both call styles:
#     step(iteration, signals_dict)
#     step(iteration, edge_score=..., fog_score=..., health_bad=...)
#
# NOTE: The controller only *proposes* actions; the training loop / gaussian model
# decides how to apply them.

from __future__ import annotations

from dataclasses import dataclass, field


def _safe_float(v: Any, default: float) -> float:
    """Convert v to float safely; return default if v is None/non-numeric."""
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)
from typing import Any, Dict, Optional, Union

ADPPAction = Dict[str, Any]


@dataclass(init=False)
class ADPPConfig:
    # Triggering / cadence
    trigger: str = "cycle"  # {"cycle","continuous"}
    decision_interval: int = 50

    # Quality gates (edge: higher is better; fog: higher is worse)
    q_edge_init: float = 0.25
    q_edge_final: float = 0.60
    q_fog_init: float = 0.15
    q_fog_final: float = 0.35

    # Health
    bad_streak_kill: int = 0
    loss_spike_factor: float = 1.30

    # Edge sharpening loss weights (max is a cap; controller ramps up/down)
    max_edge_loss_weight: float = 0.10
    edge_grad_weight: float = 1.00
    edge_lap_weight: float = 0.00

    # Densification/pruning adaptation (multipliers relative to the "base" schedule)
    densify_interval_mult_max: float = 3.0
    densify_grad_thr_mult_max: float = 3.0

    # Anti-fog controls
    anti_fog_strength_max: float = 1.0
    fog_prune_frac_max: float = 0.15

    # Keep any unknown parameters for forward compatibility
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        # Backward-compatible aliases (historical names)
        aliases = {
            # cadence
            "log_interval": "decision_interval",
            "tex_interval": "decision_interval",
            "adpp_log_interval": "decision_interval",
            "adpp_tex_interval": "decision_interval",
            # edge gate naming
            "q_tex_gate_init": "q_edge_init",
            "q_tex_gate_final": "q_edge_final",
            # fog gate naming
            "adpp_q_fog_init": "q_fog_init",
            "adpp_q_fog_final": "q_fog_final",
        }
        remapped: Dict[str, Any] = {}
        for k, v in kwargs.items():
            remapped[aliases.get(k, k)] = v

        # Initialize defaults from dataclass fields
        for fname, fdef in self.__class__.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if fname == "extra":
                continue
            setattr(self, fname, fdef.default)

        # Apply provided values
        self.extra = {}
        for k, v in remapped.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.extra[k] = v

        # Normalize trigger
        if self.trigger not in ("cycle", "continuous"):
            self.extra["trigger_raw"] = self.trigger
            self.trigger = "cycle"

        # Safety clamps
        self.decision_interval = int(max(1, self.decision_interval))
        self.loss_spike_factor = float(max(1.0, self.loss_spike_factor))
        self.max_edge_loss_weight = float(max(0.0, self.max_edge_loss_weight))
        self.fog_prune_frac_max = float(min(max(0.0, self.fog_prune_frac_max), 1.0))


class ADPPController:
    """Stateful controller that outputs an action dict each iteration."""

    def __init__(self, cfg: ADPPConfig):
        self.cfg = cfg
        self._iter: int = 0
        self._edge_score: float = 1.0
        self._fog_score: float = 0.0
        self._health_bad: bool = False

        self._loss_ema: Optional[float] = None
        self._bad_streak: int = 0

        self._edge_w: float = 0.0
        self._anti_fog: float = 0.0
        self._densify_interval_mult: float = 1.0
        self._densify_grad_thr_mult: float = 1.0
        self._fog_prune_frac: float = 0.0

    def observe_loss(self, loss_value: float, momentum: float = 0.98) -> bool:
        """Track loss EMA and detect spikes; returns health_bad."""
        if self._loss_ema is None:
            self._loss_ema = float(loss_value)
            return False
        self._loss_ema = momentum * self._loss_ema + (1.0 - momentum) * float(loss_value)
        spike = float(loss_value) > float(self._loss_ema) * float(self.cfg.loss_spike_factor)
        return bool(spike)

    def _gate_linear(self, t: float, a: float, b: float) -> float:
        t = float(min(max(t, 0.0), 1.0))
        return a + (b - a) * t

    def _should_decide(self, iteration: int) -> bool:
        if self.cfg.trigger == "continuous":
            return True
        return (iteration % self.cfg.decision_interval) == 0

    def step(
        self,
        iteration: int,
        signals: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ADPPAction:
        """Produce an action dict for this iteration.

        Compatibility:
          - Old call style: step(iteration, signals_dict)
          - New call style: step(iteration, edge_score=..., fog_score=..., health_bad=...)
        """
        self._iter = int(iteration)

        if signals is None:
            signals = {}
        # New style kwargs override signals dict.
        if "edge_score" in kwargs:
            signals["edge_score"] = kwargs["edge_score"]
        if "fog_score" in kwargs:
            signals["fog_score"] = kwargs["fog_score"]
        if "health_bad" in kwargs:
            signals["health_bad"] = kwargs["health_bad"]
        # Update cached scores safely (signals may contain None)
        self._edge_score = _safe_float(signals.get("edge_score"), self._edge_score)
        self._fog_score = _safe_float(signals.get("fog_score"), self._fog_score)
        self._health_bad = bool(signals.get("health_bad", self._health_bad))
        # Progress proxy: assume 30k iters typical; keep it generic
        prog = float(min(max(iteration / 30000.0, 0.0), 1.0))
        q_edge = self._gate_linear(prog, self.cfg.q_edge_init, self.cfg.q_edge_final)
        q_fog = self._gate_linear(prog, self.cfg.q_fog_init, self.cfg.q_fog_final)

        # Default: keep previous action unless a decision point
        action: ADPPAction = {
            "q_edge": q_edge,
            "q_fog": q_fog,
            "edge_score": self._edge_score,
            "fog_score": self._fog_score,
            "health_bad": self._health_bad,
            "edge_loss_weight": float(self._edge_w),
            "anti_fog_strength": float(self._anti_fog),
            "densify_interval_mult": float(self._densify_interval_mult),
            "densify_grad_thr_mult": float(self._densify_grad_thr_mult),
            "fog_prune_frac": float(self._fog_prune_frac),
            "bad_streak": int(self._bad_streak),
            "action_name": "hold",
        }

        if not self._should_decide(iteration):
            return action

        # Update bad streak
        if self._health_bad:
            self._bad_streak += 1
        else:
            self._bad_streak = max(0, self._bad_streak - 1)

        # Edge sharpening policy:
        # - If edge_score is below gate, ramp edge loss up (cap at max_edge_loss_weight)
        # - Else decay gently toward 0
        if self._edge_score < q_edge:
            # proportional ramp
            gap = (q_edge - self._edge_score) / max(q_edge, 1e-6)
            target = self.cfg.max_edge_loss_weight * min(1.0, 0.3 + 0.7 * gap)
            self._edge_w = float(min(self.cfg.max_edge_loss_weight, max(self._edge_w, target)))
            action["action_name"] = "sharpen"
        else:
            self._edge_w = float(max(0.0, self._edge_w * 0.9))

        # Fog suppression policy:
        # - If fog_score is above gate, ramp anti_fog + prune fraction
        # - Else decay
        if self._fog_score > q_fog:
            excess = (self._fog_score - q_fog) / max(1.0 - q_fog, 1e-6)
            self._anti_fog = float(min(self.cfg.anti_fog_strength_max, max(self._anti_fog, excess)))
            self._fog_prune_frac = float(min(self.cfg.fog_prune_frac_max, max(self._fog_prune_frac, 0.25 * excess)))
            action["action_name"] = "defog" if action["action_name"] == "hold" else action["action_name"] + "+defog"
        else:
            self._anti_fog = float(max(0.0, self._anti_fog * 0.9))
            self._fog_prune_frac = float(max(0.0, self._fog_prune_frac * 0.85))

        # Densify control:
        # - If health_bad, slow densification and increase grad threshold
        # - Else move back toward 1
        if self._health_bad:
            self._densify_interval_mult = float(min(self.cfg.densify_interval_mult_max, max(self._densify_interval_mult, 1.5)))
            self._densify_grad_thr_mult = float(min(self.cfg.densify_grad_thr_mult_max, max(self._densify_grad_thr_mult, 1.5)))
            action["action_name"] = "stabilize" if action["action_name"] == "hold" else action["action_name"] + "+stabilize"
        else:
            self._densify_interval_mult = float(1.0 + (self._densify_interval_mult - 1.0) * 0.9)
            self._densify_grad_thr_mult = float(1.0 + (self._densify_grad_thr_mult - 1.0) * 0.9)

        # Kill-switch (optional)
        if self.cfg.bad_streak_kill and self._bad_streak >= int(self.cfg.bad_streak_kill):
            action["action_name"] = "kill"
            action["kill"] = True

        # Emit
        action.update({
            "edge_loss_weight": float(self._edge_w),
            "anti_fog_strength": float(self._anti_fog),
            "densify_interval_mult": float(self._densify_interval_mult),
            "densify_grad_thr_mult": float(self._densify_grad_thr_mult),
            "fog_prune_frac": float(self._fog_prune_frac),
            "bad_streak": int(self._bad_streak),
        })
        return action

    def get_log_dict(self, prefix: str = "adpp_") -> Dict[str, float]:
        # Keep logs numeric for CSV friendliness
        return {
            f"{prefix}edge_w": float(self._edge_w),
            f"{prefix}anti_fog": float(self._anti_fog),
            f"{prefix}densify_interval_mult": float(self._densify_interval_mult),
            f"{prefix}densify_grad_thr_mult": float(self._densify_grad_thr_mult),
            f"{prefix}fog_prune_frac": float(self._fog_prune_frac),
            f"{prefix}edge_score": float(self._edge_score),
            f"{prefix}fog_score": float(self._fog_score),
            f"{prefix}bad_streak": float(self._bad_streak),
        }
