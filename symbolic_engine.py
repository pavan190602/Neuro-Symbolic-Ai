"""
Symbolic State Engine — Declarative Rule Engine with Provenance Tracing
=======================================================================

Architecture
------------
  Rule          : First-class declarative constraint object (data, not code).
  ModeRule      : Like Rule but determines the symbolic decision mode.
  FiredRule     : A Rule that evaluated True, with full provenance (which exact
                  fact values triggered it).
  FiredModeRule : A ModeRule that determined the mode, with provenance.
  RuleEngine    : Forward-chaining evaluator. Purely functional — it evaluates
                  rules against a fact store and returns FiredRule objects.
                  It does NOT mutate state.
  DecisionState : The fact store. The LLM layer (llm_interface.py) may only
                  write to the fact store via update(). It cannot modify rules
                  or override conclusions.

Neuro-Symbolic boundary
-----------------------
  LLM side  →  populates DecisionState via update()
  Symbolic side  →  RuleEngine evaluates rules deterministically over that state

  The symbolic layer is structurally independent: swapping the LLM for a
  different model (or for a human typing facts) would produce identical rule
  evaluations for identical fact values.

Provenance
----------
  Every FiredRule records:
    - Which rule fired (rule_id, name, severity, agent_lens)
    - The preconditions that were checked (human-readable)
    - The exact fact values that triggered it (triggering_facts dict)
    - A rendered conclusion with those values substituted in

  get_reasoning_trace() returns the complete audit trail:
    context / derived_facts / fired_rules / unfired_rules / decision_mode
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import copy
import json
import re


# ─────────────────────────────────────────────────────────────────────────────
# Core declarative types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    """
    A declarative constraint rule.

    A Rule is a data object. Its condition is a pure function over the fact
    store (state_dict).  The RuleEngine evaluates it; nothing else does.
    Adding a new constraint requires only adding a new Rule here — no other
    code needs to change.
    """
    rule_id: str
    name: str
    severity: str                             # "critical" | "warning" | "info"
    agent_lens: str                           # "financial"|"growth"|"wellbeing"|"values"
    provenance_fields: List[Tuple[str, str]]  # [(category, field), ...]  facts this rule reads
    condition: Callable[[Dict], bool]         # (state_dict) -> bool
    explanation_template: str                 # text with {category.field} placeholders
    precondition_descriptions: List[str]      # human-readable list of what is checked
    applies_when: Optional[Callable[[Dict], bool]] = None  # context guard (decision type/subtype)


@dataclass
class ModeRule:
    """
    A declarative rule that determines the symbolic decision mode.
    Evaluated in priority order; first match wins.
    """
    rule_id: str
    name: str
    mode: str       # "SURVIVAL_MODE" | "CAUTIOUS_MODE" | "GROWTH_MODE" | "INSUFFICIENT_DATA"
    priority: int   # lower = checked first
    provenance_fields: List[Tuple[str, str]]
    condition: Callable[[Dict], bool]
    explanation_template: str
    precondition_descriptions: List[str]
    applies_when: Optional[Callable[[Dict], bool]] = None


@dataclass
class FiredRule:
    """
    A Rule that evaluated True against the current fact store.

    Stores exactly which fact values triggered it.  This is the provenance
    record consumed by the decision-tree visualizer and the what-if analyser.
    """
    rule_id: str
    rule_name: str
    severity: str
    agent_lens: str
    precondition_descriptions: List[str]
    triggering_facts: Dict[str, Any]    # {"category.field": value}
    conclusion: str                     # rendered explanation with actual values
    fired_at: str                       # ISO timestamp

    # ── Backward-compat with old ConstraintViolation attribute names ──────
    @property
    def violation_type(self) -> str:
        return self.rule_id

    @property
    def description(self) -> str:
        return self.conclusion

    def __repr__(self) -> str:
        icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        return f"{icons.get(self.severity, '⚠️')} [{self.rule_id}] {self.conclusion}"

    def to_dict(self) -> Dict:
        return {
            # New provenance fields
            "rule_id":                   self.rule_id,
            "rule_name":                 self.rule_name,
            "severity":                  self.severity,
            "agent_lens":                self.agent_lens,
            "precondition_descriptions": self.precondition_descriptions,
            "triggering_facts":          self.triggering_facts,
            "conclusion":                self.conclusion,
            "fired_at":                  self.fired_at,
            # Legacy keys (app.py + llm_interface.py reference these)
            "type":                      self.rule_id,
            "description":               self.conclusion,
        }


# Backward-compat alias  —  app.py does `from symbolic_engine import ConstraintViolation`
ConstraintViolation = FiredRule


@dataclass
class FiredModeRule:
    """A ModeRule that determined the current decision mode — with provenance."""
    rule_id: str
    rule_name: str
    mode: str
    triggering_facts: Dict[str, Any]
    explanation: str
    precondition_descriptions: List[str]
    fired_at: str

    def to_dict(self) -> Dict:
        return {
            "rule_id":                   self.rule_id,
            "rule_name":                 self.rule_name,
            "mode":                      self.mode,
            "triggering_facts":          self.triggering_facts,
            "explanation":               self.explanation,
            "precondition_descriptions": self.precondition_descriptions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_template(template: str, state_dict: Dict) -> str:
    """Fill {category.field} placeholders with actual values from the fact store."""
    def _replace(m: re.Match) -> str:
        parts = m.group(1).split(".")
        if len(parts) == 2:
            val = state_dict.get(parts[0], {}).get(parts[1])
            if val is not None:
                # Round floats to 1 decimal for readability
                if isinstance(val, float):
                    return str(round(val, 1))
                return str(val)
        return m.group(0)
    return re.sub(r"\{([^}]+)\}", _replace, template)


def _get(state_dict: Dict, category: str, field: str, default=None):
    """Safe, concise accessor for nested state_dict."""
    return state_dict.get(category, {}).get(field, default)


def _safe_salary_diff(state_dict: Dict) -> float:
    """Return absolute salary difference; inf on any parse error."""
    try:
        sal_a = float(_get(state_dict, "offer_a", "salary") or 0)
        sal_b = float(_get(state_dict, "offer_b", "salary") or 0)
        return abs(sal_a - sal_b)
    except (TypeError, ValueError):
        return float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Rule registry  (the only place constraints are defined)
# ─────────────────────────────────────────────────────────────────────────────

def _build_rule_registry() -> List[Rule]:
    """
    Build and return all constraint rules.

    Every rule is a data object with declared provenance.
    To add a new constraint, add a new Rule here — nothing else changes.
    Rules are evaluated in the order they appear; order does not affect
    which rules fire, only the order of the results list.
    """
    R: List[Rule] = []

    # ── R001  Extreme Debt Load ───────────────────────────────────────────────
    R.append(Rule(
        rule_id="R001",
        name="Extreme Debt Load",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("financial", "debt_monthly_payment"),
            ("financial", "current_income"),
        ],
        condition=lambda s: (
            bool(_get(s, "financial", "debt_monthly_payment")) and
            bool(_get(s, "financial", "current_income")) and
            _get(s, "financial", "debt_monthly_payment") >
            0.5 * _get(s, "financial", "current_income")
        ),
        explanation_template=(
            "Debt payments ({financial.debt_monthly_payment}/mo) exceed 50% of income "
            "({financial.current_income}/mo) — extreme financial stress"
        ),
        precondition_descriptions=[
            "financial.debt_monthly_payment IS NOT NULL",
            "financial.current_income IS NOT NULL",
            "financial.debt_monthly_payment > 0.5 × financial.current_income",
        ],
    ))

    # ── R002  Values–Runway Mismatch ─────────────────────────────────────────
    R.append(Rule(
        rule_id="R002",
        name="Values–Runway Mismatch",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("values", "financial_security"),
            ("financial", "financial_runway_months"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "financial_security")) and
            _get(s, "values", "financial_security") >= 8 and
            bool(_get(s, "financial", "financial_runway_months")) and
            _get(s, "financial", "financial_runway_months") < 6
        ),
        explanation_template=(
            "Financial security priority = {values.financial_security}/10, but runway = "
            "{financial.financial_runway_months} months — stated priority conflicts with "
            "actual financial position"
        ),
        precondition_descriptions=[
            "values.financial_security ≥ 8",
            "financial.financial_runway_months < 6",
        ],
    ))

    # ── R003  WLB Unverified at New Opportunity ───────────────────────────────
    R.append(Rule(
        rule_id="R003",
        name="Work-Life Balance Unverified at New Opportunity",
        severity="critical",
        agent_lens="wellbeing",
        provenance_fields=[
            ("values", "work_life_balance"),
            ("current", "current_wlb"),
            ("opportunity", "work_life_balance_known"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "work_life_balance")) and
            _get(s, "values", "work_life_balance") >= 8 and
            _get(s, "current", "current_wlb") == "great" and
            not _get(s, "opportunity", "work_life_balance_known")
        ),
        explanation_template=(
            "WLB priority = {values.work_life_balance}/10 and current WLB = "
            "{current.current_wlb}, but new opportunity WLB is unverified — "
            "a critical unknown given this priority"
        ),
        precondition_descriptions=[
            "values.work_life_balance ≥ 8",
            "current.current_wlb = 'great'",
            "opportunity.work_life_balance_known = False",
        ],
        applies_when=lambda s: bool(_get(s, "opportunity", "company")),
    ))

    # ── R004  Thin Runway with Dependents ────────────────────────────────────
    R.append(Rule(
        rule_id="R004",
        name="Thin Runway with Dependents",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("personal", "has_dependents"),
            ("financial", "financial_runway_months"),
        ],
        condition=lambda s: (
            _get(s, "personal", "has_dependents") is True and
            bool(_get(s, "financial", "financial_runway_months")) and
            _get(s, "financial", "financial_runway_months") < 3
        ),
        explanation_template=(
            "Has dependents + only {financial.financial_runway_months} months runway — "
            "any income disruption is extremely high-risk for the household"
        ),
        precondition_descriptions=[
            "personal.has_dependents = True",
            "financial.financial_runway_months < 3",
        ],
    ))

    # ── R005  Leadership Ambition vs. WLB Priority ───────────────────────────
    R.append(Rule(
        rule_id="R005",
        name="Leadership Ambition vs. WLB Priority",
        severity="warning",
        agent_lens="wellbeing",
        provenance_fields=[
            ("values", "work_life_balance"),
            ("career_vision", "desired_role_5yr"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "work_life_balance")) and
            _get(s, "values", "work_life_balance") >= 8 and
            bool(_get(s, "career_vision", "desired_role_5yr")) and
            any(
                kw in str(_get(s, "career_vision", "desired_role_5yr", "")).lower()
                for kw in ["lead", "manager", "director", "executive", "cto", "vp"]
            )
        ),
        explanation_template=(
            "WLB priority = {values.work_life_balance}/10, but target role "
            "({career_vision.desired_role_5yr}) typically demands long hours — "
            "structural tension between priority and ambition"
        ),
        precondition_descriptions=[
            "values.work_life_balance ≥ 8",
            "career_vision.desired_role_5yr contains leadership keyword",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) in ("career_choice", "education"),
    ))

    # ── R006  Research Aversion vs. PhD Path ─────────────────────────────────
    R.append(Rule(
        rule_id="R006",
        name="Research Aversion vs. PhD Path",
        severity="critical",
        agent_lens="values",
        provenance_fields=[
            ("interests", "research"),
            ("decision_metadata", "options_being_compared"),
        ],
        condition=lambda s: (
            _get(s, "interests", "research") is False and
            "PhD" in str(_get(s, "decision_metadata", "options_being_compared", []))
        ),
        explanation_template=(
            "interests.research = False, yet PhD remains an option — "
            "a PhD is 4–5 years of sustained research; this is a structural "
            "incompatibility, not a preference conflict"
        ),
        precondition_descriptions=[
            "interests.research = False",
            "'PhD' ∈ decision_metadata.options_being_compared",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) in ("career_choice", "education"),
    ))

    # ── R007  Financial Priority vs. PhD Stipend ─────────────────────────────
    R.append(Rule(
        rule_id="R007",
        name="Financial Priority vs. PhD Stipend",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("values", "financial_security"),
            ("career_vision", "post_graduation_goal"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "financial_security")) and
            _get(s, "values", "financial_security") >= 8 and
            _get(s, "career_vision", "post_graduation_goal") == "phd"
        ),
        explanation_template=(
            "Financial security priority = {values.financial_security}/10, yet "
            "post_graduation_goal = 'phd' — PhD stipends are well below industry "
            "salaries for equivalent experience"
        ),
        precondition_descriptions=[
            "values.financial_security ≥ 8",
            "career_vision.post_graduation_goal = 'phd'",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) in ("career_choice", "education"),
    ))

    # ── R008  Hands-On Preference vs. Research Path ──────────────────────────
    R.append(Rule(
        rule_id="R008",
        name="Hands-On Preference vs. Research Path",
        severity="warning",
        agent_lens="values",
        provenance_fields=[
            ("interests", "hands_on_work"),
            ("career_vision", "research_vs_applied"),
        ],
        condition=lambda s: (
            _get(s, "interests", "hands_on_work") is True and
            _get(s, "career_vision", "research_vs_applied") == "research"
        ),
        explanation_template=(
            "interests.hands_on_work = True, but research_vs_applied = 'research' — "
            "academic research is predominantly theoretical, not hands-on"
        ),
        precondition_descriptions=[
            "interests.hands_on_work = True",
            "career_vision.research_vs_applied = 'research'",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) in ("career_choice", "education"),
    ))

    # ── R009  Salary Parity Neutralises Financial Factor ─────────────────────
    R.append(Rule(
        rule_id="R009",
        name="Salary Parity — Financial Factor Neutralised",
        severity="info",
        agent_lens="financial",
        provenance_fields=[
            ("offer_a", "salary"),
            ("offer_b", "salary"),
            ("values", "financial_security"),
        ],
        condition=lambda s: (
            bool(_get(s, "offer_a", "salary")) and
            bool(_get(s, "offer_b", "salary")) and
            bool(_get(s, "values", "financial_security")) and
            _get(s, "values", "financial_security") >= 8 and
            _safe_salary_diff(s) < 5000
        ),
        explanation_template=(
            "Offers A ({offer_a.salary}) and B ({offer_b.salary}) differ by less than $5k — "
            "despite financial_security = {values.financial_security}/10, salary cannot "
            "differentiate these options"
        ),
        precondition_descriptions=[
            "offer_a.salary IS NOT NULL",
            "offer_b.salary IS NOT NULL",
            "|offer_a.salary − offer_b.salary| < 5,000",
            "values.financial_security ≥ 8",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R010  Career Growth Data Missing Despite High Priority ───────────────
    R.append(Rule(
        rule_id="R010",
        name="Career Growth Data Missing",
        severity="warning",
        agent_lens="growth",
        provenance_fields=[
            ("values", "career_growth"),
            ("offer_a", "growth_potential"),
            ("offer_b", "growth_potential"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "career_growth")) and
            _get(s, "values", "career_growth") >= 8 and
            not _get(s, "offer_a", "growth_potential") and
            not _get(s, "offer_b", "growth_potential")
        ),
        explanation_template=(
            "Career growth priority = {values.career_growth}/10, but growth potential "
            "for neither offer has been collected — top-priority factor is unanalysable"
        ),
        precondition_descriptions=[
            "values.career_growth ≥ 8",
            "offer_a.growth_potential IS NULL",
            "offer_b.growth_potential IS NULL",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R011  WLB Data Missing Despite High Priority ──────────────────────────
    R.append(Rule(
        rule_id="R011",
        name="Work-Life Balance Data Missing for Both Offers",
        severity="warning",
        agent_lens="wellbeing",
        provenance_fields=[
            ("values", "work_life_balance"),
            ("offer_a", "work_life_balance"),
            ("offer_b", "work_life_balance"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "work_life_balance")) and
            _get(s, "values", "work_life_balance") >= 8 and
            not _get(s, "offer_a", "work_life_balance") and
            not _get(s, "offer_b", "work_life_balance")
        ),
        explanation_template=(
            "WLB priority = {values.work_life_balance}/10, but WLB has not been "
            "compared across either offer — a key differentiating factor is unknown"
        ),
        precondition_descriptions=[
            "values.work_life_balance ≥ 8",
            "offer_a.work_life_balance IS NULL",
            "offer_b.work_life_balance IS NULL",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R012  Relocation Hard Constraint ─────────────────────────────────────
    R.append(Rule(
        rule_id="R012",
        name="Relocation Hard Constraint",
        severity="critical",
        agent_lens="values",
        provenance_fields=[
            ("offer_a", "requires_relocation"),
            ("offer_b", "requires_relocation"),
            ("personal", "can_relocate"),
        ],
        condition=lambda s: (
            (
                _get(s, "offer_a", "requires_relocation") is True or
                _get(s, "offer_b", "requires_relocation") is True
            ) and
            _get(s, "personal", "can_relocate") is False
        ),
        explanation_template=(
            "One or more offers require relocation but personal.can_relocate = False — "
            "this is a hard eliminator, not a preference"
        ),
        precondition_descriptions=[
            "offer_a.requires_relocation = True OR offer_b.requires_relocation = True",
            "personal.can_relocate = False",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R013  Unvalidated Business + Thin Safety Net ──────────────────────────
    R.append(Rule(
        rule_id="R013",
        name="Unvalidated Business with Thin Safety Net",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("current", "business_validated"),
            ("current", "financial_runway"),
        ],
        condition=lambda s: (
            _get(s, "current", "business_validated") is False and
            (
                # Prefer numeric runway months if available
                (
                    _get(s, "financial", "financial_runway_months") is not None and
                    isinstance(_get(s, "financial", "financial_runway_months"), (int, float)) and
                    _get(s, "financial", "financial_runway_months") < 6
                ) or
                # Fall back to string runway label
                (
                    _get(s, "financial", "financial_runway_months") is None and
                    str(_get(s, "current", "financial_runway") or "").lower()
                    in ("none", "minimal", "no savings", "null", "0")
                )
            )
        ),
        explanation_template=(
            "business_validated = False and financial runway is thin "
            "(runway months: {financial.financial_runway_months}, "
            "label: {current.financial_runway}) — "
            "unvalidated concept with no safety net compounds execution risk"
        ),
        precondition_descriptions=[
            "current.business_validated = False",
            "financial_runway_months < 6 months OR financial_runway is minimal/unknown",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "job_vs_business",
    ))

    # ── R014  Frustration-Driven Exit Risk ────────────────────────────────────
    R.append(Rule(
        rule_id="R014",
        name="Frustration-Driven Exit Risk",
        severity="warning",
        agent_lens="wellbeing",
        provenance_fields=[
            ("current", "leave_reason"),
            ("current", "current_satisfaction"),
        ],
        condition=lambda s: (
            bool(_get(s, "current", "leave_reason")) and
            "frustrat" in str(_get(s, "current", "leave_reason", "")).lower() and
            bool(_get(s, "current", "current_satisfaction")) and
            isinstance(_get(s, "current", "current_satisfaction"), (int, float)) and
            _get(s, "current", "current_satisfaction") >= 5
        ),
        explanation_template=(
            "Primary exit driver: '{current.leave_reason}', but current_satisfaction "
            "= {current.current_satisfaction}/10 — frustration may be temporary; "
            "a permanent decision deserves more than a temporary feeling"
        ),
        precondition_descriptions=[
            "current.leave_reason contains 'frustrat'",
            "current.current_satisfaction ≥ 5  (not actually miserable)",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "job_vs_business",
    ))

    # ── R015  Dependents + No Partner Income + No Runway ─────────────────────
    R.append(Rule(
        rule_id="R015",
        name="Dependents Without Safety Net in Business Exit",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("personal", "has_dependents"),
            ("personal", "partner_employed"),
            ("current", "financial_runway"),
        ],
        condition=lambda s: (
            _get(s, "personal", "has_dependents") is True and
            _get(s, "personal", "partner_employed") is not True and
            (
                # Prefer numeric runway months if available
                (
                    _get(s, "financial", "financial_runway_months") is not None and
                    isinstance(_get(s, "financial", "financial_runway_months"), (int, float)) and
                    _get(s, "financial", "financial_runway_months") < 6
                ) or
                # Fall back to string runway label
                (
                    _get(s, "financial", "financial_runway_months") is None and
                    str(_get(s, "current", "financial_runway") or "").lower()
                    in ("none", "minimal", "no savings", "null", "0")
                )
            )
        ),
        explanation_template=(
            "Has dependents + partner_employed != True + "
            "financial runway is thin — leaving employment has very high household risk"
        ),
        precondition_descriptions=[
            "personal.has_dependents = True",
            "personal.partner_employed != True",
            "financial_runway_months < 6 OR financial_runway is minimal/unknown",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "job_vs_business",
    ))

    # ── R016  Cost-of-Living Salary Illusion ─────────────────────────────────
    R.append(Rule(
        rule_id="R016",
        name="Cost-of-Living Salary Illusion",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("offer_a", "salary"),
            ("offer_a", "city"),
            ("offer_b", "salary"),
            ("offer_b", "city"),
        ],
        condition=lambda s: (
            bool(_get(s, "offer_a", "salary")) and
            bool(_get(s, "offer_b", "salary")) and
            bool(_get(s, "offer_a", "city")) and
            bool(_get(s, "offer_b", "city")) and
            _get(s, "offer_a", "city", "").lower() != _get(s, "offer_b", "city", "").lower() and
            abs(float(_get(s, "offer_a", "salary") or 0) - float(_get(s, "offer_b", "salary") or 0)) < 20000
        ),
        explanation_template=(
            "Offers ({offer_a.salary} in {offer_a.city}) vs ({offer_b.salary} in {offer_b.city}) "
            "are in different cities — nominal salary comparison ignores cost-of-living differences "
            "that can swing real purchasing power by 20–40%"
        ),
        precondition_descriptions=[
            "offer_a.salary IS NOT NULL",
            "offer_b.salary IS NOT NULL",
            "offer_a.city ≠ offer_b.city",
            "|offer_a.salary - offer_b.salary| < 20,000",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R017  Non-Compete Risk ────────────────────────────────────────────────
    R.append(Rule(
        rule_id="R017",
        name="Non-Compete Agreement Risk",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("legal", "non_compete"),
        ],
        condition=lambda s: (
            _get(s, "legal", "non_compete") is True
        ),
        explanation_template=(
            "legal.non_compete = True — a non-compete clause may legally bar this career "
            "move or expose the person to litigation; this is a hard legal constraint, "
            "not a preference"
        ),
        precondition_descriptions=[
            "legal.non_compete = True",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) in ("offer_comparison", "job_vs_business"),
    ))

    # ── R018  Health Insurance Gap with Dependents ────────────────────────────
    R.append(Rule(
        rule_id="R018",
        name="Health Insurance Gap with Dependents",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("legal", "health_insurance_needed"),
            ("personal", "has_dependents"),
        ],
        condition=lambda s: (
            _get(s, "legal", "health_insurance_needed") is True and
            _get(s, "personal", "has_dependents") is True
        ),
        explanation_template=(
            "health_insurance_needed = True with dependents — any gap in employer "
            "health coverage directly affects the household; this is a non-negotiable "
            "cost that must be factored into the financial comparison"
        ),
        precondition_descriptions=[
            "legal.health_insurance_needed = True",
            "personal.has_dependents = True",
        ],
    ))

    # ── R019  Student Debt + Low-Income Career Path ───────────────────────────
    R.append(Rule(
        rule_id="R019",
        name="Student Debt vs. Low-Income Career Path",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("financial", "debt_total"),
            ("financial", "expected_salary"),
        ],
        condition=lambda s: (
            bool(_get(s, "financial", "debt_total")) and
            _get(s, "financial", "debt_total") > 0 and
            bool(_get(s, "financial", "expected_salary")) and
            _get(s, "financial", "expected_salary") < 50000 and
            _get(s, "financial", "debt_total") > _get(s, "financial", "expected_salary")
        ),
        explanation_template=(
            "Student debt ({financial.debt_total}) exceeds expected annual salary "
            "({financial.expected_salary}) — debt-to-income ratio above 1.0 at career start "
            "is a significant financial stress indicator"
        ),
        precondition_descriptions=[
            "financial.debt_total > 0",
            "financial.expected_salary < 50,000",
            "financial.debt_total > financial.expected_salary",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) in ("career_choice", "education"),
    ))

    # ── R020  Remote Work Preference Mismatch ─────────────────────────────────
    R.append(Rule(
        rule_id="R020",
        name="Remote Work Preference Mismatch",
        severity="warning",
        agent_lens="wellbeing",
        provenance_fields=[
            ("values", "work_life_balance"),
            ("offer_a", "work_location"),
            ("offer_b", "work_location"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "work_life_balance")) and
            _get(s, "values", "work_life_balance") >= 8 and
            (
                (
                    bool(_get(s, "offer_a", "work_location")) and
                    _get(s, "offer_a", "work_location", "").lower() == "onsite"
                ) or
                (
                    bool(_get(s, "offer_b", "work_location")) and
                    _get(s, "offer_b", "work_location", "").lower() == "onsite"
                )
            )
        ),
        explanation_template=(
            "WLB priority = {values.work_life_balance}/10, yet one or more offers are "
            "fully onsite — commute and schedule rigidity are documented WLB stressors; "
            "this tradeoff is not captured in salary numbers alone"
        ),
        precondition_descriptions=[
            "values.work_life_balance ≥ 8",
            "offer_a.work_location = 'onsite' OR offer_b.work_location = 'onsite'",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R021  High Financial Priority + Startup + No Security Data ───────────
    R.append(Rule(
        rule_id="R021",
        name="Financial Security Priority vs. Startup Risk",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("values", "financial_security"),
            ("offer_a", "job_security"),
            ("offer_b", "job_security"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "financial_security")) and
            _get(s, "values", "financial_security") >= 8 and
            (
                _get(s, "offer_a", "job_security", "").lower() == "low" or
                _get(s, "offer_b", "job_security", "").lower() == "low"
            )
        ),
        explanation_template=(
            "Financial security priority = {values.financial_security}/10, yet at least "
            "one offer has low job_security — startups have a median 5-year survival rate "
            "under 50%; this directly conflicts with the stated financial priority"
        ),
        precondition_descriptions=[
            "values.financial_security ≥ 8",
            "offer_a.job_security = 'low' OR offer_b.job_security = 'low'",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R022  Salary Regression Warning ──────────────────────────────────────
    R.append(Rule(
        rule_id="R022",
        name="Salary Regression at New Opportunity",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("financial", "current_income"),
            ("financial", "new_opportunity_income"),
        ],
        condition=lambda s: (
            bool(_get(s, "financial", "current_income")) and
            bool(_get(s, "financial", "new_opportunity_income")) and
            _get(s, "financial", "new_opportunity_income") <
            0.9 * _get(s, "financial", "current_income")
        ),
        explanation_template=(
            "New opportunity income ({financial.new_opportunity_income}) is more than 10% "
            "below current income ({financial.current_income}) — "
            "salary regression compounds debt and runway risk"
        ),
        precondition_descriptions=[
            "financial.current_income IS NOT NULL",
            "financial.new_opportunity_income IS NOT NULL",
            "new_opportunity_income < 0.9 × current_income",
        ],
    ))

    # ── R023  Sole Earner + High-Risk Transition ──────────────────────────────
    R.append(Rule(
        rule_id="R023",
        name="Sole Earner with High-Risk Transition",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("personal", "has_dependents"),
            ("personal", "partner_employed"),
            ("financial", "financial_runway_months"),
        ],
        condition=lambda s: (
            _get(s, "personal", "has_dependents") is True and
            _get(s, "personal", "partner_employed") is not True and
            bool(_get(s, "financial", "financial_runway_months")) and
            _get(s, "financial", "financial_runway_months") < 6 and
            _get(s, "decision_metadata", "decision_subtype") == "offer_comparison"
        ),
        explanation_template=(
            "Sole earner (partner_employed != True) with dependents and only "
            "{financial.financial_runway_months} months runway — "
            "any gap between jobs represents immediate household income risk"
        ),
        precondition_descriptions=[
            "personal.has_dependents = True",
            "personal.partner_employed != True",
            "financial.financial_runway_months < 6",
            "decision_subtype = offer_comparison",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R024  PhD Funding Unknown ─────────────────────────────────────────────
    R.append(Rule(
        rule_id="R024",
        name="PhD Path Without Confirmed Funding",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("career_vision", "post_graduation_goal"),
            ("financial", "taking_student_debt"),
            ("uni_a", "scholarship"),
            ("uni_b", "scholarship"),
        ],
        condition=lambda s: (
            _get(s, "career_vision", "post_graduation_goal") == "phd" and
            _get(s, "financial", "taking_student_debt") is True and
            _get(s, "uni_a", "scholarship") in (None, "none", "") and
            _get(s, "uni_b", "scholarship") in (None, "none", "")
        ),
        explanation_template=(
            "PhD path selected + taking_student_debt = True + no scholarship confirmed — "
            "unfunded PhD programs leave graduates with high debt and stipend-level income "
            "for 4–6 years; funded programs should be a hard filter, not a preference"
        ),
        precondition_descriptions=[
            "career_vision.post_graduation_goal = 'phd'",
            "financial.taking_student_debt = True",
            "uni_a.scholarship IS NULL/none AND uni_b.scholarship IS NULL/none",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) == "education",
    ))

    # ── R025  Early Career + Pure Salary Optimisation ────────────────────────
    R.append(Rule(
        rule_id="R025",
        name="Early Career Over-Indexing on Salary",
        severity="info",
        agent_lens="growth",
        provenance_fields=[
            ("values", "financial_security"),
            ("values", "career_growth"),
            ("current", "current_year"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "financial_security")) and
            bool(_get(s, "values", "career_growth")) and
            _get(s, "values", "financial_security") >= 9 and
            _get(s, "values", "career_growth") <= 5 and
            str(_get(s, "current", "current_year", "")).lower()
            in ("freshman", "sophomore", "pre-college", "junior", "recent graduate")
        ),
        explanation_template=(
            "Early career stage ({current.current_year}) + financial_security = "
            "{values.financial_security}/10 >> career_growth = {values.career_growth}/10 — "
            "salary optimisation at career start often trades long-term compounding for "
            "short-term comfort; skill premium diverges significantly after 5–10 years"
        ),
        precondition_descriptions=[
            "values.financial_security ≥ 9",
            "values.career_growth ≤ 5",
            "current.current_year ∈ {freshman, sophomore, pre-college, junior, recent graduate}",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_type"
        ) in ("career_choice", "education"),
    ))

    # ── R026  Culture Unverified for High-Culture-Priority Person ─────────────
    R.append(Rule(
        rule_id="R026",
        name="Culture Fit Unverified for Preferred Offer",
        severity="warning",
        agent_lens="wellbeing",
        provenance_fields=[
            ("values", "work_life_balance"),
            ("offer_a", "culture"),
            ("offer_b", "culture"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "work_life_balance")) and
            _get(s, "values", "work_life_balance") >= 8 and
            not _get(s, "offer_a", "culture") and
            not _get(s, "offer_b", "culture")
        ),
        explanation_template=(
            "WLB priority = {values.work_life_balance}/10, but team culture is unknown "
            "for both offers — culture is the single largest predictor of sustainable "
            "work-life balance; this gap cannot be resolved from salary data"
        ),
        precondition_descriptions=[
            "values.work_life_balance ≥ 8",
            "offer_a.culture IS NULL",
            "offer_b.culture IS NULL",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R027  Contractual Obligation Risk ─────────────────────────────────────
    R.append(Rule(
        rule_id="R027",
        name="Unresolved Contractual Obligation",
        severity="critical",
        agent_lens="financial",
        provenance_fields=[
            ("legal", "contractual_obligations"),
        ],
        condition=lambda s: (
            bool(_get(s, "legal", "contractual_obligations"))
        ),
        explanation_template=(
            "legal.contractual_obligations = {legal.contractual_obligations} — "
            "unresolved contractual obligations (clawbacks, vesting, garden leave) "
            "can make the switch financially negative regardless of new offer terms"
        ),
        precondition_descriptions=[
            "legal.contractual_obligations IS NOT NULL",
        ],
    ))

    # ── R028  Impact Priority with Neither Option Scaled ──────────────────────
    R.append(Rule(
        rule_id="R028",
        name="High Impact Priority with Unverified Impact Paths",
        severity="info",
        agent_lens="wellbeing",
        provenance_fields=[
            ("values", "impact"),
            ("offer_a", "growth_potential"),
            ("offer_b", "growth_potential"),
        ],
        condition=lambda s: (
            bool(_get(s, "values", "impact")) and
            _get(s, "values", "impact") >= 8 and
            not _get(s, "offer_a", "growth_potential") and
            not _get(s, "offer_b", "growth_potential")
        ),
        explanation_template=(
            "values.impact = {values.impact}/10, but growth_potential is unknown for "
            "both offers — impact compounds through career leverage; choosing blindly "
            "on this priority risks a high-paying but low-leverage role"
        ),
        precondition_descriptions=[
            "values.impact ≥ 8",
            "offer_a.growth_potential IS NULL",
            "offer_b.growth_potential IS NULL",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "offer_comparison",
    ))

    # ── R029  Passion Exit Without Side Income Validation ────────────────────
    R.append(Rule(
        rule_id="R029",
        name="Passion Exit Without Any Revenue Validation",
        severity="warning",
        agent_lens="financial",
        provenance_fields=[
            ("current", "business_validated"),
            ("current", "business_idea"),
        ],
        condition=lambda s: (
            bool(_get(s, "current", "business_idea")) and
            _get(s, "current", "business_validated") is not True
        ),
        explanation_template=(
            "business_idea present but business_validated != True — "
            "transitioning to an unvalidated concept without revenue evidence "
            "is hypothesis-driven, not market-driven; validation before full exit "
            "dramatically reduces failure risk"
        ),
        precondition_descriptions=[
            "current.business_idea IS NOT NULL",
            "current.business_validated ≠ True",
        ],
        applies_when=lambda s: _get(
            s, "decision_metadata", "decision_subtype"
        ) == "job_vs_business",
    ))

    # ── R030  High Dissatisfaction — Cost of Staying ─────────────────────────
    R.append(Rule(
        rule_id="R030",
        name="High Dissatisfaction — Status Quo Has a Cost",
        severity="info",
        agent_lens="wellbeing",
        provenance_fields=[
            ("current", "current_satisfaction"),
            ("current", "leave_reason"),
        ],
        condition=lambda s: (
            bool(_get(s, "current", "current_satisfaction")) and
            isinstance(_get(s, "current", "current_satisfaction"), (int, float)) and
            _get(s, "current", "current_satisfaction") <= 3 and
            bool(_get(s, "current", "leave_reason"))
        ),
        explanation_template=(
            "current_satisfaction = {current.current_satisfaction}/10 — "
            "sustained low satisfaction has documented health and productivity costs; "
            "staying is not a neutral choice; the cost of inaction is real"
        ),
        precondition_descriptions=[
            "current.current_satisfaction ≤ 3",
            "current.leave_reason IS NOT NULL",
        ],
    ))

    return R


# ─────────────────────────────────────────────────────────────────────────────
# Mode rule registry
# ─────────────────────────────────────────────────────────────────────────────

def _build_mode_rules() -> List[ModeRule]:
    """
    Build the decision-mode rule registry.
    Evaluated in priority order; first match wins.
    """
    M: List[ModeRule] = []

    # ── Career / education decisions: mode based on data completeness ─────────

    M.append(ModeRule(
        rule_id="M001", name="Career: No Data Yet",
        mode="INSUFFICIENT_DATA", priority=1,
        provenance_fields=[("interests", "*"), ("career_vision", "*")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") in ("career_choice", "education") and
            _get(s, "decision_metadata", "decision_subtype") not in ("offer_comparison", "job_vs_business") and
            sum(1 for v in s.get("interests", {}).values() if v is not None) +
            sum(1 for v in s.get("career_vision", {}).values() if v is not None) == 0
        ),
        explanation_template="No interests or career vision data collected yet",
        precondition_descriptions=[
            "decision_type ∈ {career_choice, education}",
            "decision_subtype ≠ offer_comparison",
            "COUNT(interests.*) + COUNT(career_vision.*) = 0",
        ],
        applies_when=lambda s: (
            _get(s, "decision_metadata", "decision_type") in ("career_choice", "education") and
            _get(s, "decision_metadata", "decision_subtype") != "offer_comparison"
        ),
    ))

    M.append(ModeRule(
        rule_id="M002", name="Career: Partial Data",
        mode="CAUTIOUS_MODE", priority=2,
        provenance_fields=[("interests", "*"), ("career_vision", "*")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") in ("career_choice", "education") and
            _get(s, "decision_metadata", "decision_subtype") not in ("offer_comparison", "job_vs_business") and
            0 < (
                sum(1 for v in s.get("interests", {}).values() if v is not None) +
                sum(1 for v in s.get("career_vision", {}).values() if v is not None)
            ) < 3
        ),
        explanation_template=(
            "Some career data collected but fewer than 3 fields known — analysis is partial"
        ),
        precondition_descriptions=[
            "decision_type ∈ {career_choice, education}",
            "decision_subtype ≠ offer_comparison",
            "0 < COUNT(interests.*) + COUNT(career_vision.*) < 3",
        ],
        applies_when=lambda s: (
            _get(s, "decision_metadata", "decision_type") in ("career_choice", "education") and
            _get(s, "decision_metadata", "decision_subtype") != "offer_comparison"
        ),
    ))

    M.append(ModeRule(
        rule_id="M003", name="Career: Sufficient Data",
        mode="GROWTH_MODE", priority=3,
        provenance_fields=[("interests", "*"), ("career_vision", "*")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") in ("career_choice", "education") and
            _get(s, "decision_metadata", "decision_subtype") not in ("offer_comparison", "job_vs_business") and
            (
                sum(1 for v in s.get("interests", {}).values() if v is not None) +
                sum(1 for v in s.get("career_vision", {}).values() if v is not None)
            ) >= 3
        ),
        explanation_template="Sufficient interests and career vision data for full analysis",
        precondition_descriptions=[
            "decision_type ∈ {career_choice, education}",
            "decision_subtype ≠ offer_comparison",
            "COUNT(interests.*) + COUNT(career_vision.*) ≥ 3",
        ],
        applies_when=lambda s: (
            _get(s, "decision_metadata", "decision_type") in ("career_choice", "education") and
            _get(s, "decision_metadata", "decision_subtype") != "offer_comparison"
        ),
    ))

    # ── Job vs Business mode rules ───────────────────────────────────────────────

    M.append(ModeRule(
        rule_id="M020", name="Job vs Business: No Financial Data",
        mode="INSUFFICIENT_DATA", priority=0,
        provenance_fields=[("financial", "financial_runway_months"), ("current", "financial_runway")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "job_vs_business" and
            _get(s, "financial", "financial_runway_months") is None and
            not str(_get(s, "current", "financial_runway") or "").strip()
        ),
        explanation_template="Job vs business decision: no financial runway data collected yet",
        precondition_descriptions=[
            "decision_subtype = job_vs_business",
            "financial_runway_months IS NULL",
            "current.financial_runway is empty",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "job_vs_business",
    ))

    M.append(ModeRule(
        rule_id="M021", name="Job vs Business: Critical Risk",
        mode="SURVIVAL_MODE", priority=0,
        provenance_fields=[
            ("financial", "financial_runway_months"),
            ("personal", "has_dependents"),
        ],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "job_vs_business" and
            _get(s, "personal", "has_dependents") is True and
            (
                (
                    _get(s, "financial", "financial_runway_months") is not None and
                    _get(s, "financial", "financial_runway_months") < 3
                ) or
                str(_get(s, "current", "financial_runway") or "").lower()
                in ("none", "minimal", "no savings", "null", "0")
            )
        ),
        explanation_template=(
            "Dependents present + runway < 3 months — leaving employment is extremely high risk"
        ),
        precondition_descriptions=[
            "decision_subtype = job_vs_business",
            "personal.has_dependents = True",
            "financial_runway_months < 3 OR no savings",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "job_vs_business",
    ))

    M.append(ModeRule(
        rule_id="M022", name="Job vs Business: Cautious",
        mode="CAUTIOUS_MODE", priority=0,
        provenance_fields=[("financial", "financial_runway_months"), ("current", "financial_runway")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "job_vs_business" and
            (
                (
                    _get(s, "financial", "financial_runway_months") is not None and
                    3 <= _get(s, "financial", "financial_runway_months") < 12
                ) or
                str(_get(s, "current", "financial_runway") or "").lower()
                in ("no savings", "minimal", "1", "2", "3", "few months")
            )
        ),
        explanation_template=(
            "Job vs business: runway {financial.financial_runway_months} months "
            "({current.financial_runway}) — limited safety net → CAUTIOUS_MODE"
        ),
        precondition_descriptions=[
            "decision_subtype = job_vs_business",
            "3 ≤ financial_runway_months < 12",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "job_vs_business",
    ))

    M.append(ModeRule(
        rule_id="M023", name="Job vs Business: Strong Runway",
        mode="GROWTH_MODE", priority=0,
        provenance_fields=[("financial", "financial_runway_months"), ("current", "financial_runway")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "job_vs_business" and
            (
                (
                    _get(s, "financial", "financial_runway_months") is not None and
                    _get(s, "financial", "financial_runway_months") >= 12
                )
            )
        ),
        explanation_template=(
            "Job vs business: runway {financial.financial_runway_months} months "
            "— sufficient safety net → GROWTH_MODE"
        ),
        precondition_descriptions=[
            "decision_subtype = job_vs_business",
            "financial_runway_months ≥ 12",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "job_vs_business",
    ))

    # ── Financial / general decisions: mode based on runway ───────────────────

    # ── Offer comparison mode rules (checked before generic financial rules) ──────

    M.append(ModeRule(
        rule_id="M010", name="Offer Comparison: Minimal Data",
        mode="INSUFFICIENT_DATA", priority=2,
        provenance_fields=[("offer_a", "company"), ("offer_b", "company"),
                           ("values", "financial_security")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "offer_comparison" and
            (
                _get(s, "offer_a", "company") is None or
                _get(s, "offer_b", "company") is None
            )
        ),
        explanation_template="Offer comparison: one or both offers not yet identified",
        precondition_descriptions=[
            "decision_subtype = offer_comparison",
            "offer_a.company IS NULL OR offer_b.company IS NULL",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "offer_comparison",
    ))

    M.append(ModeRule(
        rule_id="M011", name="Offer Comparison: Hard Constraint Present",
        mode="CAUTIOUS_MODE", priority=3,
        provenance_fields=[("personal", "can_relocate"), ("personal", "has_dependents"),
                           ("legal", "visa_constrained")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "offer_comparison" and
            _get(s, "offer_a", "company") is not None and
            _get(s, "offer_b", "company") is not None and
            (
                _get(s, "personal", "can_relocate") is False or
                _get(s, "personal", "has_dependents") is True or
                _get(s, "legal", "visa_constrained") is True
            )
        ),
        explanation_template=(
            "Offer comparison with personal constraint (relocation/dependents/visa) → CAUTIOUS_MODE"
        ),
        precondition_descriptions=[
            "decision_subtype = offer_comparison",
            "both offers identified",
            "can_relocate=False OR has_dependents=True OR visa_constrained=True",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "offer_comparison",
    ))

    M.append(ModeRule(
        rule_id="M012", name="Offer Comparison: Full Data",
        mode="GROWTH_MODE", priority=3,
        provenance_fields=[("offer_a", "salary"), ("offer_b", "salary"),
                           ("values", "financial_security")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") == "offer_comparison" and
            _get(s, "offer_a", "company") is not None and
            _get(s, "offer_b", "company") is not None and
            _get(s, "personal", "can_relocate") is not False and
            _get(s, "personal", "has_dependents") is not True and
            _get(s, "legal", "visa_constrained") is not True
        ),
        explanation_template="Offer comparison with sufficient data, no hard constraints → GROWTH_MODE",
        precondition_descriptions=[
            "decision_subtype = offer_comparison",
            "both offers identified",
            "no hard personal constraints",
        ],
        applies_when=lambda s: _get(s, "decision_metadata", "decision_subtype") == "offer_comparison",
    ))

    M.append(ModeRule(
        rule_id="M004", name="Financial: No Runway Data",
        mode="INSUFFICIENT_DATA", priority=4,
        provenance_fields=[("financial", "financial_runway_months")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_subtype") not in ("offer_comparison",) and
            _get(s, "decision_metadata", "decision_type") not in ("career_choice", "education") and
            _get(s, "financial", "financial_runway_months") is None
        ),
        explanation_template=(
            "financial_runway_months not yet derived — "
            "current_savings and/or monthly_expenses not collected"
        ),
        precondition_descriptions=[
            "decision_subtype ≠ offer_comparison",
            "decision_type ∉ {career_choice, education}",
            "financial.financial_runway_months IS NULL",
        ],
    ))

    M.append(ModeRule(
        rule_id="M005", name="Survival Mode — Critical Debt",
        mode="SURVIVAL_MODE", priority=5,
        provenance_fields=[
            ("financial", "debt_monthly_payment"),
            ("financial", "current_income"),
        ],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") not in ("career_choice", "education") and
            bool(_get(s, "financial", "debt_monthly_payment")) and
            bool(_get(s, "financial", "current_income")) and
            _get(s, "financial", "debt_monthly_payment") >
            0.3 * _get(s, "financial", "current_income")
        ),
        explanation_template=(
            "Debt ({financial.debt_monthly_payment}/mo) > 30% of income "
            "({financial.current_income}/mo) → SURVIVAL_MODE"
        ),
        precondition_descriptions=[
            "financial.debt_monthly_payment > 0.3 × financial.current_income",
        ],
    ))

    M.append(ModeRule(
        rule_id="M006", name="Cautious Mode — Low Runway",
        mode="CAUTIOUS_MODE", priority=6,
        provenance_fields=[("financial", "financial_runway_months")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") not in ("career_choice", "education") and
            bool(_get(s, "financial", "financial_runway_months")) and
            _get(s, "financial", "financial_runway_months") < 6
        ),
        explanation_template=(
            "Runway = {financial.financial_runway_months} months  (<6 threshold) → CAUTIOUS_MODE"
        ),
        precondition_descriptions=["financial.financial_runway_months < 6"],
    ))

    M.append(ModeRule(
        rule_id="M007", name="Cautious Mode — Visa or Dependents",
        mode="CAUTIOUS_MODE", priority=7,
        provenance_fields=[
            ("legal", "visa_constrained"),
            ("relationships", "has_dependents"),
            ("values", "financial_security"),
        ],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") not in ("career_choice", "education") and
            (
                _get(s, "legal", "visa_constrained") is True or
                (
                    _get(s, "relationships", "has_dependents") is True and
                    (_get(s, "values", "financial_security") or 0) >= 7
                )
            )
        ),
        explanation_template="Visa or dependent constraint with high financial priority → CAUTIOUS_MODE",
        precondition_descriptions=[
            "legal.visa_constrained = True  OR  "
            "(relationships.has_dependents = True AND values.financial_security ≥ 7)",
        ],
    ))

    M.append(ModeRule(
        rule_id="M008", name="Growth Mode — Strong Runway",
        mode="GROWTH_MODE", priority=8,
        provenance_fields=[("financial", "financial_runway_months")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") not in ("career_choice", "education") and
            bool(_get(s, "financial", "financial_runway_months")) and
            _get(s, "financial", "financial_runway_months") >= 12
        ),
        explanation_template=(
            "Runway = {financial.financial_runway_months} months  (≥12 threshold) → GROWTH_MODE"
        ),
        precondition_descriptions=["financial.financial_runway_months ≥ 12"],
    ))

    M.append(ModeRule(
        rule_id="M009", name="Cautious Mode — Conservative Default",
        mode="CAUTIOUS_MODE", priority=9,
        provenance_fields=[("financial", "financial_runway_months")],
        condition=lambda s: (
            _get(s, "decision_metadata", "decision_type") not in ("career_choice", "education") and
            bool(_get(s, "financial", "financial_runway_months"))
        ),
        explanation_template=(
            "Runway = {financial.financial_runway_months} months (6–11 range) → "
            "CAUTIOUS_MODE (conservative default)"
        ),
        precondition_descriptions=["6 ≤ financial.financial_runway_months < 12"],
    ))

    return M


# ─────────────────────────────────────────────────────────────────────────────
# Rule Engine
# ─────────────────────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Forward-chaining rule evaluator.

    Purely functional — evaluates rules against a fact store dict and returns
    FiredRule objects.  Does not mutate state.

    Key methods
    -----------
    evaluate(state_dict)
        Evaluate all constraint rules; return fired rules.

    evaluate_with_override(state_dict, overrides)
        What-if variant: evaluate with specified fact overrides.

    determine_mode(state_dict)
        Evaluate mode rules; return (mode_string, FiredModeRule|None).

    get_reasoning_trace(state_dict, fired_rules)
        Build complete audit-trail dict for visualisation and export.

    compare_scenarios(state_dict, overrides)
        Return a diff dict: newly_fired, resolved, mode_changed.
        Used directly by the What-If analysis panel.
    """

    def __init__(self, rules: List[Rule], mode_rules: List[ModeRule]):
        self.rules      = rules
        self.mode_rules = sorted(mode_rules, key=lambda r: r.priority)

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(self, state_dict: Dict) -> List[FiredRule]:
        fired: List[FiredRule] = []
        for rule in self.rules:
            if not self._context_passes(rule, state_dict):
                continue
            try:
                if not rule.condition(state_dict):
                    continue
            except Exception:
                continue
            triggering = {
                f"{cat}.{fld}": state_dict.get(cat, {}).get(fld)
                for cat, fld in rule.provenance_fields
                if state_dict.get(cat, {}).get(fld) is not None
            }
            fired.append(FiredRule(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                agent_lens=rule.agent_lens,
                precondition_descriptions=rule.precondition_descriptions,
                triggering_facts=triggering,
                conclusion=_render_template(rule.explanation_template, state_dict),
                fired_at=datetime.now().isoformat(),
            ))
        return fired

    def evaluate_with_override(
        self,
        state_dict: Dict,
        overrides: Dict[Tuple[str, str], Any],
    ) -> List[FiredRule]:
        """
        Evaluate rules with fact overrides for what-if analysis.
        Operates on a deep copy — the original state is never mutated.
        """
        modified = copy.deepcopy(state_dict)
        for (cat, fld), val in overrides.items():
            if isinstance(modified.get(cat), dict):
                modified[cat][fld] = val
        # Re-derive runway if savings or expenses were overridden
        savings  = modified.get("financial", {}).get("current_savings")
        expenses = modified.get("financial", {}).get("monthly_expenses")
        if savings is not None and expenses is not None:
            try:
                if float(expenses) > 0:
                    modified["financial"]["financial_runway_months"] = (
                        float(savings) / float(expenses)
                    )
            except (TypeError, ValueError):
                pass
        return self.evaluate(modified)

    # ── Mode determination ────────────────────────────────────────────────────

    def determine_mode(
        self, state_dict: Dict
    ) -> Tuple[str, Optional[FiredModeRule]]:
        for mode_rule in self.mode_rules:
            if not self._context_passes(mode_rule, state_dict):
                continue
            try:
                if not mode_rule.condition(state_dict):
                    continue
            except Exception:
                continue
            triggering = {}
            for cat, fld in mode_rule.provenance_fields:
                if fld == "*":
                    count = sum(
                        1 for v in state_dict.get(cat, {}).values() if v is not None
                    )
                    if count:
                        triggering[f"{cat}.known_fields"] = count
                else:
                    val = state_dict.get(cat, {}).get(fld)
                    if val is not None:
                        triggering[f"{cat}.{fld}"] = val
            return mode_rule.mode, FiredModeRule(
                rule_id=mode_rule.rule_id,
                rule_name=mode_rule.name,
                mode=mode_rule.mode,
                triggering_facts=triggering,
                explanation=_render_template(mode_rule.explanation_template, state_dict),
                precondition_descriptions=mode_rule.precondition_descriptions,
                fired_at=datetime.now().isoformat(),
            )
        return "CAUTIOUS_MODE", None

    # ── Unfired rule introspection ────────────────────────────────────────────

    def get_unfired_rules(
        self, state_dict: Dict, fired_ids: List[str]
    ) -> List[Dict]:
        fired_set = set(fired_ids)
        result = []
        for rule in self.rules:
            if rule.rule_id in fired_set:
                continue
            if not self._context_passes(rule, state_dict):
                result.append({
                    "rule_id": rule.rule_id,
                    "name":    rule.name,
                    "reason":  "context guard not met (different decision type/subtype)",
                })
                continue
            missing = [
                f"{cat}.{fld}"
                for cat, fld in rule.provenance_fields
                if state_dict.get(cat, {}).get(fld) is None
            ]
            if missing:
                result.append({
                    "rule_id": rule.rule_id,
                    "name":    rule.name,
                    "reason":  f"required facts not yet collected: {', '.join(missing)}",
                })
            else:
                result.append({
                    "rule_id": rule.rule_id,
                    "name":    rule.name,
                    "reason":  "all preconditions evaluated — condition not satisfied (no conflict)",
                })
        return result

    # ── Reasoning trace ───────────────────────────────────────────────────────

    def get_reasoning_trace(
        self, state_dict: Dict, fired_rules: List[FiredRule]
    ) -> Dict:
        """
        Build a complete, structured reasoning trace.

        This dict drives both the decision-tree visualisation and the
        what-if analysis panel.  It is also suitable for export to PDF/JSON.
        """
        mode_str, mode_fired = self.determine_mode(state_dict)
        fired_ids = [r.rule_id for r in fired_rules]
        unfired   = self.get_unfired_rules(state_dict, fired_ids)

        # Derived fact: financial runway
        derived = []
        savings  = state_dict.get("financial", {}).get("current_savings")
        expenses = state_dict.get("financial", {}).get("monthly_expenses")
        runway   = state_dict.get("financial", {}).get("financial_runway_months")
        if savings is not None and expenses is not None and runway is not None:
            derived.append({
                "fact_path":   "financial.financial_runway_months",
                "value":       round(float(runway), 1),
                "derivation":  (
                    f"current_savings ({savings}) ÷ "
                    f"monthly_expenses ({expenses}) = "
                    f"{round(float(runway), 1)} months"
                ),
                "input_facts": {
                    "financial.current_savings":  savings,
                    "financial.monthly_expenses": expenses,
                },
            })

        return {
            "context": {
                "decision_type":    state_dict.get("decision_metadata", {}).get("decision_type"),
                "decision_subtype": state_dict.get("decision_metadata", {}).get("decision_subtype"),
                "options":          state_dict.get("decision_metadata", {}).get("options_being_compared", []),
            },
            "derived_facts":  derived,
            "fired_rules":    [r.to_dict() for r in fired_rules],
            "unfired_rules":  unfired,
            "decision_mode":  mode_fired.to_dict() if mode_fired else {"mode": mode_str},
            "rule_counts": {
                "total_registered": len(self.rules),
                "fired":            len(fired_rules),
                "critical":         sum(1 for r in fired_rules if r.severity == "critical"),
                "warning":          sum(1 for r in fired_rules if r.severity == "warning"),
                "info":             sum(1 for r in fired_rules if r.severity == "info"),
            },
        }

    # ── What-if comparison ────────────────────────────────────────────────────

    def compare_scenarios(
        self,
        state_dict: Dict,
        overrides: Dict[Tuple[str, str], Any],
    ) -> Dict:
        """
        Compare current state against a what-if override scenario.

        Returns
        -------
        {
            "original_fired":  list of rule_id strings
            "modified_fired":  list of rule_id strings
            "newly_fired":     list of FiredRule (appear in modified, not original)
            "resolved":        list of FiredRule (appear in original, not modified)
            "unchanged_fired": list of rule_id strings (fire in both)
            "mode_original":   str
            "mode_modified":   str
            "mode_changed":    bool
            "modified_facts":  {category.field: new_value}
        }
        """
        original_rules = self.evaluate(state_dict)
        modified_rules = self.evaluate_with_override(state_dict, overrides)

        original_ids = {r.rule_id for r in original_rules}
        modified_ids = {r.rule_id for r in modified_rules}

        mode_orig, _ = self.determine_mode(state_dict)
        mode_mod,  _ = self.determine_mode(
            self._apply_overrides(state_dict, overrides)
        )

        return {
            "original_fired":  list(original_ids),
            "modified_fired":  list(modified_ids),
            "newly_fired":     [r for r in modified_rules if r.rule_id not in original_ids],
            "resolved":        [r for r in original_rules if r.rule_id not in modified_ids],
            "unchanged_fired": list(original_ids & modified_ids),
            "mode_original":   mode_orig,
            "mode_modified":   mode_mod,
            "mode_changed":    mode_orig != mode_mod,
            "modified_facts":  {
                f"{cat}.{fld}": val for (cat, fld), val in overrides.items()
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _context_passes(rule, state_dict: Dict) -> bool:
        if rule.applies_when is None:
            return True
        try:
            return bool(rule.applies_when(state_dict))
        except Exception:
            return False

    @staticmethod
    def _apply_overrides(state_dict: Dict, overrides: Dict) -> Dict:
        modified = copy.deepcopy(state_dict)
        for (cat, fld), val in overrides.items():
            if isinstance(modified.get(cat), dict):
                modified[cat][fld] = val
        # Re-derive runway
        savings  = modified.get("financial", {}).get("current_savings")
        expenses = modified.get("financial", {}).get("monthly_expenses")
        if savings is not None and expenses is not None:
            try:
                if float(expenses) > 0:
                    modified["financial"]["financial_runway_months"] = (
                        float(savings) / float(expenses)
                    )
            except (TypeError, ValueError):
                pass
        return modified


# ─────────────────────────────────────────────────────────────────────────────
# Module-level engine instance (built once at import time)
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE = RuleEngine(
    rules=_build_rule_registry(),
    mode_rules=_build_mode_rules(),
)


# ─────────────────────────────────────────────────────────────────────────────
# DecisionState  (fact store — public API unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class DecisionState:
    """
    Fact store for all known constraints about the decision.

    Public API is identical to the previous implementation.
    Internally, constraint evaluation is delegated to the RuleEngine.

    The only change visible externally:
      - to_dict() now includes a 'reasoning_trace' key
      - get_reasoning_trace() is a new method
      - whatif_evaluate() is a new method (for the what-if panel)
    """

    def __init__(self):
        # ── Financial ────────────────────────────────────────────────────────
        self.financial = {
            "current_savings":         None,
            "monthly_expenses":        None,
            "current_income":          None,
            "debt_total":              None,
            "debt_monthly_payment":    None,
            "new_opportunity_income":  None,
            "expected_salary":         None,
            "salary_importance":       None,
            "financial_runway_months": None,
        }
        # ── Legal ────────────────────────────────────────────────────────────
        self.legal = {
            "visa_constrained":        None,
            "non_compete":             None,
            "health_insurance_needed": None,
            "contractual_obligations": None,
        }
        # ── Values ───────────────────────────────────────────────────────────
        self.values = {
            "career_growth":     None,
            "work_life_balance": None,
            "financial_security":None,
            "learning":          None,
            "status":            None,
            "impact":            None,
        }
        # ── Relationships ────────────────────────────────────────────────────
        self.relationships = {
            "has_dependents":       None,
            "partner_income_stable":None,
            "family_support":       None,
            "geographic_constraints":None,
        }
        # ── Interests ────────────────────────────────────────────────────────
        self.interests = {
            "enjoys_coding":            None,
            "enjoys_analysis":          None,
            "enjoys_theory":            None,
            "enjoys_building_systems":  None,
            "enjoys_working_with_data": None,
            "enjoys_visualization":     None,
            "enjoys_algorithms":        None,
        }
        # ── Career Vision ────────────────────────────────────────────────────
        self.career_vision = {
            "desired_role_5yr":     None,
            "wants_specialization": None,
            "research_vs_applied":  None,
            "industry_preference":  None,
            "post_graduation_goal": None,
        }
        # ── Strengths ────────────────────────────────────────────────────────
        self.strengths = {
            "math_stats_comfort":      None,
            "programming_experience":  None,
            "communication_skills":    None,
            "problem_solving_approach":None,
        }
        # ── Decision metadata ─────────────────────────────────────────────────
        self.decision_metadata = {
            "decision_type":          None,
            "options_being_compared": [],
            "decision_summary":       None,
            "decision_subtype":       None,
        }
        # ── Offer A / B ───────────────────────────────────────────────────────
        self.offer_a = {
            "company": None, "role": None, "salary": None, "salary_raw": None,
            "growth_potential": None, "work_life_balance": None,
            "work_location": None, "city": None, "requires_relocation": None,
            "culture": None, "concern": None, "job_security": None,
        }
        self.offer_b = {
            "company": None, "role": None, "salary": None, "salary_raw": None,
            "growth_potential": None, "work_life_balance": None,
            "work_location": None, "city": None, "requires_relocation": None,
            "culture": None, "concern": None, "job_security": None,
        }
        # ── Personal ─────────────────────────────────────────────────────────
        self.personal = {
            "has_family": None, "has_dependents": None,
            "partner_employed": None, "can_relocate": None,
            "relocation_concern": None, "current_city": None,
        }
        # ── University A / B ─────────────────────────────────────────────────
        self.uni_a = {
            "name": None, "program": None, "tuition": None, "tuition_raw": None,
            "scholarship": None, "location": None, "requires_relocation": None,
            "ranking": None, "job_placement": None, "living_cost": None,
            "research_interest": None, "concern": None,
        }
        self.uni_b = {
            "name": None, "program": None, "tuition": None, "tuition_raw": None,
            "scholarship": None, "location": None, "requires_relocation": None,
            "ranking": None, "job_placement": None, "living_cost": None,
            "research_interest": None, "concern": None,
        }
        # ── Opportunity ──────────────────────────────────────────────────────
        self.opportunity = {
            "role_description":       None,
            "company":                None,
            "work_life_balance_known":False,
            "team_culture_known":     False,
            "reversibility":          None,
        }
        # ── Current situation ─────────────────────────────────────────────────
        self.current = {
            "current_employer":   None, "current_role":     None,
            "current_satisfaction":None,"current_wlb":      None,
            "current_year":       None, "leaning":          None,
            "financial_concern":  None, "concern":          None,
            "job_market_concern": None, "business_idea":    None,
            "business_validated": None, "financial_runway": None,
            "leave_reason":       None,
        }
        # ── Internal tracking ─────────────────────────────────────────────────
        self.history: List[Dict]              = []
        self.violations: List[FiredRule]      = []   # populated by _check_violations()
        self.missing_critical_info: List[str] = []
        self._last_reasoning_trace: Optional[Dict] = None

    # ── Public write interface ────────────────────────────────────────────────

    def update(self, category: str, key: str, value: Any) -> None:
        """Update a fact and run the rule engine."""
        cat_obj = getattr(self, category, None)
        if cat_obj is None:
            print(f"[STATE] WARNING: unknown category '{category}', skipping")
            return

        old_value = cat_obj.get(key) if isinstance(cat_obj, dict) else None

        # ── Type coercions (unchanged from original) ─────────────────────
        if category == "financial" and value is not None:
            if isinstance(value, str):
                cleaned = (
                    value.lower()
                    .replace("usd", "").replace("inr", "")
                    .replace("$", "").replace(",", "").strip()
                )
                if "lakh" in cleaned or "lac" in cleaned:
                    num_part = cleaned.replace("lakh", "").replace("lac", "").strip()
                    try:
                        value = float(num_part) * 100_000
                    except Exception:
                        pass
                else:
                    try:
                        value = float(cleaned) if cleaned else None
                    except Exception:
                        pass

        if category == "values" and value is not None:
            if isinstance(value, str):
                try:
                    value = int(value)
                except Exception:
                    pass

        if category == "decision_metadata" and key == "options_being_compared":
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]

        cat_obj[key] = value

        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "category":  category,
            "key":       key,
            "old_value": old_value,
            "new_value": value,
        })

        self._recalculate_derived()
        self._check_violations()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _recalculate_derived(self) -> None:
        # ── Financial runway ─────────────────────────────────────────────────
        savings  = self.financial.get("current_savings")
        expenses = self.financial.get("monthly_expenses")
        if savings is not None and expenses is not None:
            try:
                s, e = float(savings), float(expenses)
                if e > 0:
                    self.financial["financial_runway_months"] = s / e
            except (TypeError, ValueError):
                pass

        # ── can_relocate inference (deterministic, not left to LLM) ──────────
        # If user has dependents/family AND at least one offer requires relocation
        # AND can_relocate has not been explicitly set, infer it as False.
        # This ensures R012 (Relocation Hard Constraint) fires correctly.
        has_family_constraint = (
            self.personal.get("has_dependents") is True or
            self.personal.get("has_family") is True
        )
        any_offer_relocates = (
            self.offer_a.get("requires_relocation") is True or
            self.offer_b.get("requires_relocation") is True or
            self.uni_a.get("requires_relocation") is True or
            self.uni_b.get("requires_relocation") is True
        )
        if (has_family_constraint and
                any_offer_relocates and
                self.personal.get("can_relocate") is None):
            self.personal["can_relocate"] = False
            print("[DERIVED] can_relocate inferred as False (family + relocation required)")

    def _check_violations(self) -> None:
        """Delegate to RuleEngine — deterministic, no side effects on engine."""
        state_dict    = self._build_fact_store()
        fired         = _ENGINE.evaluate(state_dict)
        self.violations = fired
        # Cache the full reasoning trace
        self._last_reasoning_trace = _ENGINE.get_reasoning_trace(state_dict, fired)

    # ── Public read interface ─────────────────────────────────────────────────

    def _build_fact_store(self) -> Dict:
        """
        Build a minimal fact-store dict WITHOUT calling get_decision_mode().
        Used internally by get_decision_mode() and _check_violations() to
        avoid infinite recursion (to_dict calls get_decision_mode which
        would call to_dict again).
        """
        return {
            "decision_metadata": self.decision_metadata,
            "offer_a":           self.offer_a,
            "offer_b":           self.offer_b,
            "personal":          self.personal,
            "uni_a":             self.uni_a,
            "uni_b":             self.uni_b,
            "interests":         self.interests,
            "career_vision":     self.career_vision,
            "strengths":         self.strengths,
            "financial":         self.financial,
            "legal":             self.legal,
            "values":            self.values,
            "relationships":     self.relationships,
            "opportunity":       self.opportunity,
            "current":           self.current,
        }

    def get_decision_mode(self) -> str:
        state_dict = self._build_fact_store()
        mode, _    = _ENGINE.determine_mode(state_dict)
        return mode

    def get_missing_critical_info(self) -> List[str]:
        """
        Return a list of critical facts still missing for a meaningful analysis.

        Each decision subtype requires a different set of facts — using a single
        binary split (career_choice vs else) was incorrect and produced misleading
        missing-info indicators for university_comparison and job_vs_business.
        """
        missing          = []
        decision_type    = self.decision_metadata.get("decision_type")
        decision_subtype = self.decision_metadata.get("decision_subtype", "general")

        if decision_subtype in ("major_choice", "education_path"):
            # Need to know interests and career direction
            interests_known = sum(
                1 for v in self.interests.values() if v is not None
            )
            if interests_known == 0:
                missing.append("What kind of work or study they enjoy")
            if not self.career_vision.get("desired_role_5yr"):
                missing.append("Career vision / desired role in 5 years")

        elif decision_subtype == "offer_comparison":
            # Need at least one salary data point to compare
            sal_a = self.offer_a.get("salary")
            sal_b = self.offer_b.get("salary")
            if sal_a is None and sal_b is None:
                missing.append("Salary or compensation details for at least one offer")
            if not self.values.get("financial_security") and not self.values.get("career_growth"):
                missing.append("Personal priorities (financial security vs. career growth)")

        elif decision_subtype == "university_comparison":
            # Need cost awareness and career goal — NOT work-life balance of a job
            if not self.financial.get("taking_student_debt") and \
               not self.uni_a.get("tuition") and not self.uni_b.get("tuition"):
                missing.append("Tuition cost or student debt situation")
            if not self.career_vision.get("post_graduation_goal") and \
               not self.career_vision.get("desired_role_5yr"):
                missing.append("Post-graduation goal (job, PhD, etc.)")

        elif decision_subtype == "job_vs_business":
            # Need financial runway — that is the critical constraint
            runway = self.financial.get("financial_runway_months")
            runway_label = self.current.get("financial_runway")
            if runway is None and not runway_label:
                missing.append("Financial runway (how long they can sustain without income)")
            if not self.current.get("business_idea"):
                missing.append("Description of the business idea or passion pursuit")

        elif decision_type == "career_choice":
            # Generic career choice — interests and vision are the minimum
            interests_known = sum(
                1 for v in self.interests.values() if v is not None
            )
            if interests_known == 0:
                missing.append("What kind of work they enjoy")
            if not self.career_vision.get("desired_role_5yr"):
                missing.append("Career vision / desired role in 5 years")

        else:
            # General / unknown decision type — check the classic job-change facts
            # only if there is an explicit opportunity being evaluated
            if self.opportunity.get("company"):
                if not self.opportunity.get("work_life_balance_known"):
                    missing.append("Work-life balance at new opportunity")
                if not self.opportunity.get("team_culture_known"):
                    missing.append("Team culture and expectations")

        self.missing_critical_info = missing
        return missing

    def can_analyze(self) -> bool:
        return len(self.get_missing_critical_info()) == 0

    def get_reasoning_trace(self) -> Dict:
        """
        Return the full reasoning trace from the last evaluation.
        If not yet computed, compute it now.
        """
        if self._last_reasoning_trace is None:
            self._check_violations()
        return self._last_reasoning_trace or {}

    def whatif_evaluate(
        self, overrides: Dict[Tuple[str, str], Any]
    ) -> Dict:
        """
        Run a what-if scenario: return a diff dict showing which rules
        newly fire, which are resolved, and whether the mode changes.

        Parameters
        ----------
        overrides : {(category, field): new_value}

        Returns
        -------
        See RuleEngine.compare_scenarios() docstring.
        """
        state_dict = self._build_fact_store()
        return _ENGINE.compare_scenarios(state_dict, overrides)

    def to_dict(self, include_trace: bool = True) -> Dict:
        """
        Export current state.  include_trace=False avoids recursion during
        internal _check_violations() calls.
        """
        base = {
            "decision_metadata": self.decision_metadata,
            "offer_a":           self.offer_a,
            "offer_b":           self.offer_b,
            "personal":          self.personal,
            "uni_a":             self.uni_a,
            "uni_b":             self.uni_b,
            "interests":         self.interests,
            "career_vision":     self.career_vision,
            "strengths":         self.strengths,
            "financial":         self.financial,
            "legal":             self.legal,
            "values":            self.values,
            "relationships":     self.relationships,
            "opportunity":       self.opportunity,
            "current":           self.current,
            "violations": [v.to_dict() for v in self.violations],
            "missing_info":      self.missing_critical_info,
            "decision_mode":     self.get_decision_mode(),
        }
        if include_trace and self._last_reasoning_trace:
            base["reasoning_trace"] = self._last_reasoning_trace
        return base

    def get_state_summary(self) -> str:
        lines = ["═" * 60, "CURRENT STATE", "═" * 60]
        known_count = sum(
            1 for cat in [
                self.financial, self.legal, self.values, self.relationships,
                self.opportunity, self.current, self.interests,
                self.career_vision, self.strengths, self.decision_metadata,
            ]
            for v in cat.values()
            if v is not None and v != [] and v is not False and v != ""
        )
        lines.append(f"Known facts:    {known_count}")
        lines.append(f"Decision mode:  {self.get_decision_mode()}")
        lines.append(
            f"Rules fired:    {len(self.violations)}  "
            f"({sum(1 for v in self.violations if v.severity=='critical')} critical, "
            f"{sum(1 for v in self.violations if v.severity=='warning')} warning)"
        )
        if self.violations:
            lines.append("\n⚠️  CONSTRAINT VIOLATIONS (with provenance):")
            for v in self.violations:
                lines.append(f"  [{v.rule_id}] {v.severity.upper()}: {v.conclusion}")
                for fact_path, val in v.triggering_facts.items():
                    lines.append(f"    ← {fact_path} = {val}")
        missing = self.get_missing_critical_info()
        if missing:
            lines.append("\n❌ MISSING CRITICAL INFO:")
            for m in missing:
                lines.append(f"  • {m}")
        lines.append("═" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    state = DecisionState()
    state.update("decision_metadata", "decision_type", "career_choice")
    state.update("decision_metadata", "decision_subtype", "education_path")
    state.update("decision_metadata", "options_being_compared", "PhD,Job")
    state.update("values", "financial_security", 9)
    state.update("career_vision", "post_graduation_goal", "phd")
    state.update("interests", "research", False)
    state.update("interests", "hands_on_work", True)
    state.update("career_vision", "research_vs_applied", "research")
    state.update("financial", "current_savings", 10000)
    state.update("financial", "monthly_expenses", 3000)

    print(state.get_state_summary())

    print("\n── Reasoning Trace ──")
    trace = state.get_reasoning_trace()
    print(f"Fired:   {trace['rule_counts']['fired']} rules")
    print(f"Mode:    {trace['decision_mode'].get('mode')} "
          f"← {trace['decision_mode'].get('rule_id')}")
    for r in trace["fired_rules"]:
        print(f"  [{r['rule_id']}] {r['severity'].upper()}: {r['conclusion']}")
        for k, v in r["triggering_facts"].items():
            print(f"    ← {k} = {v}")

    print("\n── What-If: financial_security = 5 ──")
    diff = state.whatif_evaluate({("values", "financial_security"): 5})
    print(f"Resolved violations: {[r.rule_id for r in diff['resolved']]}")
    print(f"Newly fired:         {[r.rule_id for r in diff['newly_fired']]}")
    print(f"Mode change:         {diff['mode_original']} → {diff['mode_modified']}")
