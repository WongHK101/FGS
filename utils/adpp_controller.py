# -*- coding: utf-8 -*-
"""
utils/adpp_controller.py

ADPP (Adaptive Dual-Phase Policy) controller: turns *signals* into per-iteration "actions"
that modulate training (loss weights + optimizer LR multipliers + anti-fog pruning knobs).

Design goals (for GeoTGS/FGS two-stage RGB->Thermal training):
- One unified policy for both stages (no hand-tuned stage-specific hyperparameters).
- Stable decisions: hysteresis + patience + optional "cycle" trigger.
- Paper-friendly: explicit signals -> interpretable actions; actions are logged.

This file is used by:
- train_ADPP_COMPAT_v2.py (controller.step(), controller.get_log_dict())
- scene/gaussian_model_ADPP_UPDATED.py (reads action keys set via gaussians.adpp_set_action()).

Action dict keys expected by train/gaussian_model:
  enabled: bool
  edge_active: bool
  fog_active: bool
  edge_strength: float in [0,1]
  fog_strength: float in [0,1]

  edge_loss_weight: float (overall edge-loss weight multiplier)
  edge_grad_weight: float (weight for grad-alignment term inside edge loss)
  edge_lap_weight: float  (weight for laplacian term inside edge loss)

  scaling_lr_mult: float  (multiplier applied to optimizer param group "scaling")
  allow_scaling: bool     (if False, scaling LR is forced to 0)

  densify_interval_mult: float (multiplier for densification interval; >1 = densify less often)
  densify_grad_thr_mult: float (multiplier for densify grad threshold; >1 = densify harder)

  anti_fog_strength: float (0..anti_fog_strength_max) used by gaussian_model.adpp_* hooks
  fog_prune_frac: float    (0..fog_prune_frac_max) used by gaussian_model.adpp_post_densify_hook
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class ADPPConfig:
    # Master switches
    enabled: bool = True
    mode: str = "full"  # "full" | "edge_only" | "fog_only"

    # Decision trigger
    trigger: str = "cycle"  # "cycle" | "continuous"
    decision_interval: int = 50  # only used when trigger=="cycle"

    # Warmup
    warmup_iters: int = 300

    # EMA smoothing for signals (lower -> smoother)
    ema_beta: float = 0.98

    # Hysteresis thresholds (fallbacks if q_* not used)
    edge_on: float = 0.45   # edge_score < edge_on  => consider "edge bad"
    edge_off: float = 0.60  # edge_score > edge_off => consider "edge recovered"
    fog_on: float = 0.30    # fog_score > fog_on    => consider "fog bad"
    fog_off: float = 0.20   # fog_score < fog_off   => consider "fog recovered"

    # "q-" gates (preferred): continuous strength mapping + hysteresis thresholds.
    # We interpret edge_score higher=sharper, fog_score higher=worse.
    q_edge_init: float = 0.45
    q_edge_final: float = 0.60
    q_fog_init: float = 0.20
    q_fog_final: float = 0.30

    # Patience (how many *decision points* to confirm badness before activating)
    patience_edge: int = 2
    patience_fog: int = 2

    # Optional safety: if controller stays "bad" for too long, relax to avoid collapse
    bad_streak_kill: int = 4  # measured in decision points

    # ----- Action strength ranges -----

    # Edge loss weights
    max_edge_loss_weight: float = 0.25
    edge_grad_weight: float = 1.0
    edge_lap_weight: float = 0.5

    # Scaling LR (can help gaussians shrink to fit sharp edges, but can also create haze if too aggressive)
    scaling_lr_mult_max: float = 2.0
    allow_scaling: bool = True

    # Densify control (we use these mainly for anti-fog: >1 => densify less often / harder to densify)
    densify_interval_mult_max: float = 2.0
    densify_grad_thr_mult_max: float = 2.0

    # Anti-fog controls (used by gaussian_model hooks)
    anti_fog_strength_max: float = 0.65
    fog_prune_frac_max: float = 0.15


@dataclass
class ADPPState:
    # Smoothed signals
    edge_ema: Optional[float] = None
    fog_ema: Optional[float] = None

    # Hysteresis
    edge_active: bool = False
    fog_active: bool = False
    edge_bad_count: int = 0
    fog_bad_count: int = 0

    # Safety
    bad_streak: int = 0

    # Last action
    last_action: Dict[str, Any] = field(default_factory=dict)


class ADPPController:
    """
    Convert signals -> actions.

    Signals dict (from utils/adpp_signals.py) should contain:
      - "edge_score": float (higher = sharper / better)
      - "fog_score": float  (higher = more fog / worse)
    Missing signals are treated as "neutral" (no action).
    """

    def __init__(self, cfg: ADPPConfig):
        self.cfg = cfg
        self.state = ADPPState()

    # ---- Public API ----

    def step(self, iteration: int, signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Produce an action dict for this iteration.
        """
        if not self.cfg.enabled:
            action = self._action_disabled()
            self.state.last_action = action
            return action

        if iteration < int(self.cfg.warmup_iters):
            action = self._action_warmup()
            # still update EMA so we start stable right after warmup
            self._update_ema(signals)
            self.state.last_action = action
            return action

        # always update EMAs
        edge_ema, fog_ema = self._update_ema(signals)

        # Decide whether we are allowed to update mode/strengths at this iter
        if not self._should_decide(iteration):
            # Keep previous action, but update the exposed strength values using latest EMA
            action = dict(self.state.last_action) if self.state.last_action else self._action_neutral()
            action.update(self._strength_fields(edge_ema, fog_ema))
            self.state.last_action = action
            return action

        # Determine thresholds (prefer q_*, fallback to edge_on/off/fog_on/off)
        edge_on, edge_off, fog_off, fog_on = self._effective_hysteresis()

        edge_score = edge_ema if edge_ema is not None else 1.0
        fog_score = fog_ema if fog_ema is not None else 0.0

        # Hysteresis + patience
        self._update_hysteresis(edge_score, fog_score, edge_on, edge_off, fog_on, fog_off)

        # Strength in [0,1]
        edge_strength, fog_strength = self._compute_strength(edge_score, fog_score)

        # Optional safety: if both get worse for too long, relax (turn off actions for one cycle)
        self._update_bad_streak(edge_score, fog_score, edge_on, fog_on)

        if self.state.bad_streak >= int(self.cfg.bad_streak_kill):
            action = self._action_neutral()
            action.update(self._strength_fields(edge_ema, fog_ema))
            # keep active flags but relax for next interval
            self.state.bad_streak = 0
            self.state.last_action = action
            return action

        # Apply mode gating
        edge_active = bool(self.state.edge_active) and (self.cfg.mode in ("full", "edge_only"))
        fog_active = bool(self.state.fog_active) and (self.cfg.mode in ("full", "fog_only"))

        # Build actions
        action = self._compose_action(edge_strength, fog_strength, edge_active=edge_active, fog_active=fog_active)
        self.state.last_action = action
        return action

    def get_log_dict(self, prefix: str = "adpp_") -> Dict[str, float]:
        """
        Lightweight scalar stats to push into TensorBoard or CSV.
        """
        s = self.state
        out: Dict[str, float] = {}
        out[prefix + "edge_ema"] = float(s.edge_ema) if s.edge_ema is not None else -1.0
        out[prefix + "fog_ema"] = float(s.fog_ema) if s.fog_ema is not None else -1.0
        out[prefix + "edge_active"] = 1.0 if s.edge_active else 0.0
        out[prefix + "fog_active"] = 1.0 if s.fog_active else 0.0
        out[prefix + "bad_streak"] = float(s.bad_streak)
        # also expose last strengths if present
        la = s.last_action or {}
        out[prefix + "edge_strength"] = float(la.get("edge_strength", 0.0))
        out[prefix + "fog_strength"] = float(la.get("fog_strength", 0.0))
        out[prefix + "edge_loss_weight"] = float(la.get("edge_loss_weight", 0.0))
        out[prefix + "anti_fog_strength"] = float(la.get("anti_fog_strength", 0.0))
        out[prefix + "fog_prune_frac"] = float(la.get("fog_prune_frac", 0.0))
        out[prefix + "densify_interval_mult"] = float(la.get("densify_interval_mult", 1.0))
        out[prefix + "densify_grad_thr_mult"] = float(la.get("densify_grad_thr_mult", 1.0))
        return out

    # ---- Internals ----

    def _should_decide(self, iteration: int) -> bool:
        trig = (self.cfg.trigger or "cycle").lower()
        if trig == "continuous":
            return True
        # default: cycle
        k = max(int(self.cfg.decision_interval), 1)
        return (iteration % k) == 0

    def _effective_hysteresis(self) -> tuple[float, float, float, float]:
        """
        Return (edge_on, edge_off, fog_off, fog_on).
        Prefer q_* gates if they look valid.
        """
        # Edge: on < off
        edge_on = float(getattr(self.cfg, "q_edge_init", self.cfg.edge_on))
        edge_off = float(getattr(self.cfg, "q_edge_final", self.cfg.edge_off))
        if edge_off <= edge_on:
            edge_on, edge_off = self.cfg.edge_on, self.cfg.edge_off

        # Fog: off < on
        fog_off = float(getattr(self.cfg, "q_fog_init", self.cfg.fog_off))
        fog_on = float(getattr(self.cfg, "q_fog_final", self.cfg.fog_on))
        if fog_on <= fog_off:
            fog_off, fog_on = self.cfg.fog_off, self.cfg.fog_on

        return edge_on, edge_off, fog_off, fog_on

    def _update_ema(self, signals: Dict[str, float]) -> tuple[Optional[float], Optional[float]]:
        beta = _clamp(float(self.cfg.ema_beta), 0.0, 0.9999)
        e = signals.get("edge_score", None)
        f = signals.get("fog_score", None)

        if e is not None:
            e = float(e)
            if self.state.edge_ema is None:
                self.state.edge_ema = e
            else:
                self.state.edge_ema = beta * self.state.edge_ema + (1.0 - beta) * e

        if f is not None:
            f = float(f)
            if self.state.fog_ema is None:
                self.state.fog_ema = f
            else:
                self.state.fog_ema = beta * self.state.fog_ema + (1.0 - beta) * f

        return self.state.edge_ema, self.state.fog_ema

    def _update_hysteresis(self, edge_score: float, fog_score: float,
                           edge_on: float, edge_off: float,
                           fog_on: float, fog_off: float) -> None:
        # Edge
        if not self.state.edge_active:
            if edge_score < edge_on:
                self.state.edge_bad_count += 1
                if self.state.edge_bad_count >= int(self.cfg.patience_edge):
                    self.state.edge_active = True
                    self.state.edge_bad_count = 0
            else:
                self.state.edge_bad_count = 0
        else:
            # active: wait for recovery
            if edge_score > edge_off:
                self.state.edge_active = False
                self.state.edge_bad_count = 0

        # Fog
        if not self.state.fog_active:
            if fog_score > fog_on:
                self.state.fog_bad_count += 1
                if self.state.fog_bad_count >= int(self.cfg.patience_fog):
                    self.state.fog_active = True
                    self.state.fog_bad_count = 0
            else:
                self.state.fog_bad_count = 0
        else:
            if fog_score < fog_off:
                self.state.fog_active = False
                self.state.fog_bad_count = 0

    def _compute_strength(self, edge_score: float, fog_score: float) -> tuple[float, float]:
        edge_on, edge_off, fog_off, fog_on = self._effective_hysteresis()
        eps = 1e-6

        # edge_strength: 1 when very blurry (<= edge_on), 0 when recovered (>= edge_off)
        edge_strength = (edge_off - edge_score) / (edge_off - edge_on + eps)
        edge_strength = _clamp(edge_strength, 0.0, 1.0)

        # fog_strength: 0 when clean (<= fog_off), 1 when foggy (>= fog_on)
        fog_strength = (fog_score - fog_off) / (fog_on - fog_off + eps)
        fog_strength = _clamp(fog_strength, 0.0, 1.0)

        return edge_strength, fog_strength

    def _update_bad_streak(self, edge_score: float, fog_score: float, edge_on: float, fog_on: float) -> None:
        """
        Heuristic safety: count how many *decision points* we are simultaneously:
        - edge still bad (below edge_on) AND fog still bad (above fog_on)
        If the streak is too long, relax for one cycle.
        """
        if (edge_score < edge_on) and (fog_score > fog_on) and (self.cfg.mode == "full"):
            self.state.bad_streak += 1
        else:
            self.state.bad_streak = 0

    def _strength_fields(self, edge_ema: Optional[float], fog_ema: Optional[float]) -> Dict[str, Any]:
        edge_score = float(edge_ema) if edge_ema is not None else 1.0
        fog_score = float(fog_ema) if fog_ema is not None else 0.0
        edge_strength, fog_strength = self._compute_strength(edge_score, fog_score)
        return {
            "edge_strength": edge_strength,
            "fog_strength": fog_strength,
            "edge_active": bool(self.state.edge_active),
            "fog_active": bool(self.state.fog_active),
        }

    # ---- Action templates ----

    def _action_disabled(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "edge_active": False,
            "fog_active": False,
            "edge_strength": 0.0,
            "fog_strength": 0.0,
            "edge_loss_weight": 0.0,
            "edge_grad_weight": float(self.cfg.edge_grad_weight),
            "edge_lap_weight": float(self.cfg.edge_lap_weight),
            "scaling_lr_mult": 1.0,
            "allow_scaling": bool(self.cfg.allow_scaling),
            "densify_interval_mult": 1.0,
            "densify_grad_thr_mult": 1.0,
            "anti_fog_strength": 0.0,
            "fog_prune_frac": 0.0,
        }

    def _action_warmup(self) -> Dict[str, Any]:
        a = self._action_disabled()
        a["enabled"] = True
        return a

    def _action_neutral(self) -> Dict[str, Any]:
        a = self._action_warmup()
        a.update({
            "edge_loss_weight": 0.0,
            "densify_interval_mult": 1.0,
            "densify_grad_thr_mult": 1.0,
            "anti_fog_strength": 0.0,
            "fog_prune_frac": 0.0,
            "scaling_lr_mult": 1.0,
            "allow_scaling": bool(self.cfg.allow_scaling),
        })
        return a

    def _compose_action(self, edge_strength: float, fog_strength: float,
                        *, edge_active: bool, fog_active: bool) -> Dict[str, Any]:
        # Edge losses (only when edge_active)
        if edge_active:
            edge_loss_weight = float(self.cfg.max_edge_loss_weight) * float(edge_strength)
            scaling_lr_mult = _lerp(1.0, float(self.cfg.scaling_lr_mult_max), float(edge_strength))
        else:
            edge_loss_weight = 0.0
            scaling_lr_mult = 1.0

        # Anti-fog knobs (only when fog_active)
        if fog_active:
            densify_interval_mult = _lerp(1.0, float(self.cfg.densify_interval_mult_max), float(fog_strength))
            densify_grad_thr_mult = _lerp(1.0, float(self.cfg.densify_grad_thr_mult_max), float(fog_strength))
            anti_fog_strength = float(self.cfg.anti_fog_strength_max) * float(fog_strength)
            fog_prune_frac = float(self.cfg.fog_prune_frac_max) * float(fog_strength)
        else:
            densify_interval_mult = 1.0
            densify_grad_thr_mult = 1.0
            anti_fog_strength = 0.0
            fog_prune_frac = 0.0

        return {
            "enabled": True,
            "edge_active": bool(edge_active),
            "fog_active": bool(fog_active),
            "edge_strength": float(edge_strength),
            "fog_strength": float(fog_strength),

            "edge_loss_weight": float(edge_loss_weight),
            "edge_grad_weight": float(self.cfg.edge_grad_weight),
            "edge_lap_weight": float(self.cfg.edge_lap_weight),

            "scaling_lr_mult": float(scaling_lr_mult),
            "allow_scaling": bool(self.cfg.allow_scaling),

            "densify_interval_mult": float(densify_interval_mult),
            "densify_grad_thr_mult": float(densify_grad_thr_mult),

            "anti_fog_strength": float(anti_fog_strength),
            "fog_prune_frac": float(fog_prune_frac),
        }
