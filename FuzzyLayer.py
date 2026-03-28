"""
Fuzzy Decision Layer
=====================
Sits on top of the CNN model's raw probability output and
produces a structured verdict using probability zones +
feature-based override rules.

Architecture:
    P(malicious)  ──►  base zone
    features      ──►  override rules   ──►  final zone → action
                                        ──►  reason string
                                        ──►  confidence label

Why a fuzzy layer instead of a fixed threshold?
    A single threshold (e.g. 0.5) treats all emails in the same
    way regardless of what features they contain. An email at
    p=0.35 that asks for your password is more dangerous than one
    at p=0.65 that just has the word "urgent". The fuzzy layer
    lets domain knowledge override the raw probability in a
    transparent, auditable way.

Zones:
    SAFE        P < 0.30   → ALLOW
    SUSPICIOUS  0.30–0.69  → WARN
    HIGH_RISK   P ≥ 0.70   → BLOCK

Override rules (can only escalate, never de-escalate):
    SAFE → SUSPICIOUS  if contains_sensitive (even low-p sensitive = warn)
    SUSPICIOUS → HIGH_RISK  if any strong multi-signal combination
"""

from dataclasses import dataclass, field
from typing import Literal

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
# Based on Phase 3 threshold sweep:
#   ~0.25 → optimal F1 (best_f1_threshold)
#   ~0.30 → start of uncertain zone
#   ~0.70 → very high precision, few false positives
P_SAFE_MAX      = 0.30   # below this → SAFE (unless overridden)
P_HIGH_RISK_MIN = 0.70   # at or above → HIGH_RISK directly

# Override trigger thresholds
# Overrides only fire when P is above these minimums
# (prevents noise at very low probabilities from triggering rules)
P_SENSITIVE_OVERRIDE  = 0.20   # contains_sensitive fires from this low
                                # because credential requests are high-risk
                                # even when the model is uncertain
P_COMBO_OVERRIDE      = 0.30   # multi-signal combos need at least this


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────
Action     = Literal["ALLOW", "WARN", "BLOCK"]
Zone       = Literal["SAFE", "SUSPICIOUS", "HIGH_RISK"]
Confidence = Literal["low", "medium", "high"]

@dataclass
class Verdict:
    """
    Structured output from the fuzzy decision layer.

    Fields
    ------
    action      : what to do with this email
                  ALLOW  → deliver normally
                  WARN   → deliver with a warning banner
                  BLOCK  → quarantine / move to spam

    zone        : which probability zone the email fell into
                  (before any overrides)

    final_zone  : zone after applying override rules
                  if final_zone != zone, an override fired

    probability : raw model output  P(malicious) ∈ [0, 1]

    confidence  : how certain we are about the verdict
                  low    → model uncertain, human review recommended
                  medium → reasonable signal, override may have fired
                  high   → strong signal, high probability

    reasons     : list of human-readable strings explaining the verdict
                  always contains at least one entry
                  e.g. ["High risk probability: 0.83",
                        "Override: sensitive keywords + urgency detected"]

    overridden  : True if a feature rule changed the base zone
    """
    action      : Action
    zone        : Zone
    final_zone  : Zone
    probability : float
    confidence  : Confidence
    reasons     : list = field(default_factory=list)
    overridden  : bool = False

    def __str__(self):
        override_tag = " [OVERRIDDEN]" if self.overridden else ""
        lines = [
            f"  Action      : {self.action}",
            f"  Zone        : {self.zone} → {self.final_zone}{override_tag}",
            f"  Probability : {self.probability:.4f}",
            f"  Confidence  : {self.confidence}",
            f"  Reasons     :",
        ]
        for r in self.reasons:
            lines.append(f"    • {r}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _base_zone(p: float) -> Zone:
    """Map raw probability to base zone — no feature logic here."""
    if p >= P_HIGH_RISK_MIN:
        return "HIGH_RISK"
    elif p >= P_SAFE_MAX:
        return "SUSPICIOUS"
    else:
        return "SAFE"


def _confidence(p: float, overridden: bool) -> Confidence:
    """
    Derive confidence label from probability and override status.

    Overridden verdicts are capped at 'medium' because the model
    was uncertain — a feature rule stepped in, not strong probability.
    """
    if overridden:
        return "medium"
    if p >= 0.85 or p <= 0.10:
        return "high"
    if p >= 0.60 or p <= 0.20:
        return "medium"
    return "low"


def _zone_to_action(zone: Zone) -> Action:
    return {"SAFE": "ALLOW", "SUSPICIOUS": "WARN", "HIGH_RISK": "BLOCK"}[zone]


# ─────────────────────────────────────────────
# OVERRIDE RULES
# ─────────────────────────────────────────────
def _apply_overrides(
    p          : float,
    base_zone  : Zone,
    features   : dict,
) -> tuple[Zone, list[str]]:
    """
    Apply feature-based override rules.

    Design principles:
        1. Overrides can only ESCALATE, never de-escalate.
           A HIGH_RISK email stays HIGH_RISK regardless of features.

        2. Rules require COMBINATIONS of signals, not single keywords.
           Exception: contains_sensitive alone is enough — asking
           for credentials is high-risk even without other signals.

        3. Each override has a minimum probability floor so very-low-p
           emails don't get escalated by noise in the feature extractor.

        4. Every override logs a human-readable reason.

    Returns
    -------
    (final_zone, reasons)
    """
    reasons    = []
    final_zone = base_zone

    # Pull features (safe default to 0 if key missing)
    sensitive = int(features.get("contains_sensitive", 0))
    urgent    = int(features.get("contains_urgent",    0))
    verify    = int(features.get("contains_verify",    0))
    click     = int(features.get("contains_click",     0))

    # Weighted signal score
    # contains_sensitive is weighted 2× — credential requests are a
    # much stronger phishing signal than urgency or click language
    weighted_signals = (sensitive * 2) + urgent + verify + click

    # ── Rule 1: Sensitive keywords (SAFE → SUSPICIOUS) ────
    # Even at very low probability, an email asking for passwords/
    # bank details / SSN warrants a warning.
    if base_zone == "SAFE" and sensitive and p >= P_SENSITIVE_OVERRIDE:
        final_zone = "SUSPICIOUS"
        reasons.append(
            f"Override: sensitive keywords detected at P={p:.3f} "
            f"(credential-related language warrants caution)"
        )

    # ── Rule 2: Sensitive + any other signal (→ HIGH_RISK) ──
    # Sensitive keywords combined with urgency, verification requests,
    # or click-action language is a classic phishing pattern.
    if final_zone != "HIGH_RISK" and p >= P_COMBO_OVERRIDE:
        if sensitive and (urgent or verify or click):
            final_zone = "HIGH_RISK"
            triggers = [k for k, v in {
                "urgent language"       : urgent,
                "verification request"  : verify,
                "action/click language" : click,
            }.items() if v]
            reasons.append(
                f"Override: sensitive keywords + {', '.join(triggers)} "
                f"— classic credential-phishing pattern"
            )

    # ── Rule 3: Urgency + verify together (→ HIGH_RISK) ───
    # "Your account will be suspended, verify now" is a
    # textbook phishing combination even without sensitive keywords.
    if final_zone != "HIGH_RISK" and p >= P_COMBO_OVERRIDE:
        if urgent and verify:
            final_zone = "HIGH_RISK"
            reasons.append(
                "Override: urgency + verification request together "
                "— high-confidence phishing pattern"
            )

    # ── Rule 4: High weighted signal score (→ HIGH_RISK) ──
    # 3+ weighted signals (e.g. urgent + verify + click, or
    # sensitive + click) = strong multi-signal phishing pattern.
    if final_zone != "HIGH_RISK" and p >= P_COMBO_OVERRIDE:
        if weighted_signals >= 3:
            final_zone = "HIGH_RISK"
            reasons.append(
                f"Override: weighted signal score={weighted_signals} "
                f"(≥3 combined phishing signals detected)"
            )

    return final_zone, reasons


# ─────────────────────────────────────────────
# MAIN FUZZY DECISION FUNCTION
# ─────────────────────────────────────────────
def fuzzy_decision(p: float, features: dict) -> Verdict:
    """
    Convert a model probability + feature dict into a structured Verdict.

    Parameters
    ----------
    p        : float
               Raw P(malicious) from the CNN model. Must be in [0, 1].

    features : dict
               Handcrafted feature dict as produced by Phase 2.
               Expected keys:
                 contains_sensitive, contains_urgent,
                 contains_verify, contains_click,
                 num_words, num_exclamations, repeated_words_ratio
               Missing keys default to 0 — function is tolerant of
               partial feature dicts.

    Returns
    -------
    Verdict dataclass — see class definition above.

    Examples
    --------
    >>> v = fuzzy_decision(0.82, {"contains_sensitive": 0})
    >>> v.action
    'BLOCK'
    >>> v.overridden
    False

    >>> v = fuzzy_decision(0.35, {"contains_sensitive": 1, "contains_urgent": 1})
    >>> v.action
    'BLOCK'
    >>> v.overridden
    True
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Probability must be in [0, 1], got {p}")

    base_zone          = _base_zone(p)
    final_zone, o_reasons = _apply_overrides(p, base_zone, features)
    overridden         = final_zone != base_zone

    # Build reason list — always starts with the probability reason
    reasons = []

    if base_zone == "SAFE":
        reasons.append(f"Low malicious probability: {p:.4f} (below {P_SAFE_MAX})")
    elif base_zone == "SUSPICIOUS":
        reasons.append(
            f"Uncertain probability: {p:.4f} "
            f"(between {P_SAFE_MAX} and {P_HIGH_RISK_MIN})"
        )
    else:
        reasons.append(f"High malicious probability: {p:.4f} (≥ {P_HIGH_RISK_MIN})")

    reasons.extend(o_reasons)

    # If no override fired, add a brief feature summary
    if not overridden:
        active = [k for k in
                  ["contains_sensitive","contains_urgent",
                   "contains_verify","contains_click"]
                  if features.get(k, 0)]
        if active:
            reasons.append(f"Active signals: {', '.join(active)} "
                           f"(insufficient to override at P={p:.3f})")
        else:
            reasons.append("No strong feature signals detected")

    return Verdict(
        action      = _zone_to_action(final_zone),
        zone        = base_zone,
        final_zone  = final_zone,
        probability = round(p, 6),
        confidence  = _confidence(p, overridden),
        reasons     = reasons,
        overridden  = overridden,
    )


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":

    test_cases = [
        # (description, p, features)
        (
            "Clean legit email",
            0.04,
            {"contains_sensitive": 0, "contains_urgent": 0,
             "contains_verify": 0, "contains_click": 0}
        ),
        (
            "Low-p but asks for password",
            0.22,
            {"contains_sensitive": 1, "contains_urgent": 0,
             "contains_verify": 0, "contains_click": 0}
        ),
        (
            "Mid-p, sensitive + urgent (classic phishing)",
            0.45,
            {"contains_sensitive": 1, "contains_urgent": 1,
             "contains_verify": 0, "contains_click": 0}
        ),
        (
            "Mid-p, urgent + verify only (no sensitive)",
            0.38,
            {"contains_sensitive": 0, "contains_urgent": 1,
             "contains_verify": 1, "contains_click": 0}
        ),
        (
            "Mid-p, 3 signals but no sensitive",
            0.33,
            {"contains_sensitive": 0, "contains_urgent": 1,
             "contains_verify": 1, "contains_click": 1}
        ),
        (
            "High-p, no features needed",
            0.91,
            {"contains_sensitive": 0, "contains_urgent": 0,
             "contains_verify": 0, "contains_click": 0}
        ),
        (
            "Borderline suspicious, no strong signals",
            0.55,
            {"contains_sensitive": 0, "contains_urgent": 1,
             "contains_verify": 0, "contains_click": 0}
        ),
    ]

    print("=" * 60)
    print("  Fuzzy Decision Layer — Self Test")
    print("=" * 60)

    for desc, p, feats in test_cases:
        verdict = fuzzy_decision(p, feats)
        print(f"\n▸ {desc}")
        print(verdict)

    print("\n" + "=" * 60)