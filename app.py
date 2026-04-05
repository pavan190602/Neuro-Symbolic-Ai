"""
Decision Support System - Main Application
Streamlit interface for the hybrid neuro-symbolic decision support system
"""

import logging
import streamlit as st
import streamlit.components.v1 as components
import os
from symbolic_engine import DecisionState, ConstraintViolation
from llm_interface import LLMInterface
from decision_memory import DecisionMemory
import json
import re as _re

# ── API key ────────────────────────────────────────────────────────────────────
# Try Groq key from secrets; fall back to env var for local dev
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Support System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = DecisionState()
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_council" not in st.session_state:
    st.session_state.show_council = False
if "chat_locked" not in st.session_state:
    st.session_state.chat_locked = False
if "council_cache" not in st.session_state:
    # Stores the last generated council result so navigating back/forth
    # doesn't re-call the API and produce inconsistent outputs.
    st.session_state.council_cache = None
if "show_whatif" not in st.session_state:
    st.session_state.show_whatif = False
if "decision_saved" not in st.session_state:
    st.session_state.decision_saved = False
if "talk_mode" not in st.session_state:
    st.session_state.talk_mode = False
if "last_spoken_idx" not in st.session_state:
    st.session_state.last_spoken_idx = -1
if "audio_input_counter" not in st.session_state:
    st.session_state.audio_input_counter = 0

# ── Decision Memory singleton ──────────────────────────────────────────────────
@st.cache_resource
def _load_memory():
    return DecisionMemory()

_MEMORY = _load_memory()


# ── LLM init ───────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_llm(api_key: str, bls_path: str):
    """Cached at module level — BLS file (5.8MB) loads once, not per session."""
    return LLMInterface(api_key=api_key, bls_path=bls_path)


def initialize_llm():
    try:
        _app_dir  = os.path.dirname(os.path.abspath(__file__))
        _bls_path = os.path.join(_app_dir, "bls_ooh_chunks.jsonl")
        st.session_state.llm = _load_llm(GROQ_API_KEY, _bls_path)
        return True
    except Exception as e:
        st.session_state.llm_error = str(e)
        return False


# ── Conversation-complete: only when chat_locked ───────────────────────────────
def is_conversation_complete() -> bool:
    """Council button shows only when chat_locked=True."""
    return st.session_state.get("chat_locked", False)


# ── Header ─────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style='text-align:center;padding:20px;
                background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                border-radius:10px;margin-bottom:30px;'>
        <h1 style='color:white;margin:0;'>🧠 Decision Support System</h1>
        <p style='color:#f0f0f0;margin:10px 0 0 0;'>
            Neuro-Symbolic Reasoning for Complex Decisions
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar state panel ────────────────────────────────────────────────────────
def render_sidebar_state():
    state_dict = st.session_state.state.to_dict()

    st.markdown("### 📊 Live State Tracking")

    # Decision mode
    mode = state_dict["decision_mode"]
    mode_icon = {
        "SURVIVAL_MODE": "🔴",
        "CAUTIOUS_MODE": "🟡",
        "GROWTH_MODE": "🟢",
        "INSUFFICIENT_DATA": "⚪",
    }
    st.markdown(f"**Mode:** {mode_icon.get(mode, 'circle')} {mode}")

    # Decision type + options
    meta = state_dict.get("decision_metadata", {})
    decision_type = meta.get("decision_type")
    options = meta.get("options_being_compared", [])
    if decision_type:
        label = decision_type
        if options:
            label += f": {' vs '.join(options)}"
        st.markdown(f"**Decision:** {label}")

    # Violations
    if state_dict["violations"]:
        st.markdown("#### Conflicts Detected")
        for v in state_dict["violations"]:
            st.error(f"**{v['type']}:** {v['description']}")

    # Recent updates — shows category.key so it's clear what changed
    if st.session_state.state.history:
        st.markdown("#### Recent Updates")
        for change in reversed(st.session_state.state.history[-3:]):
            st.caption(
                f"+ {change['category']}.{change['key']}: {change['new_value']}"
            )

    # Known facts — now counts ALL categories (was the bug)
    all_cats = [
        state_dict.get("financial", {}),
        state_dict.get("values", {}),
        state_dict.get("current", {}),
        state_dict.get("opportunity", {}),
        state_dict.get("interests", {}),
        state_dict.get("career_vision", {}),
        state_dict.get("strengths", {}),
        state_dict.get("decision_metadata", {}),
        state_dict.get("offer_a", {}),
        state_dict.get("offer_b", {}),
    ]
    known_count = sum(
        1 for cat in all_cats
        for v in cat.values()
        if v is not None and v is not False and v != "" and v != []
    )
    st.markdown(f"**Known facts:** {known_count}")

    # Progress toward Council (soft target: ~12 meaningful facts)
    TARGET_FACTS = 12
    progress = min(known_count / TARGET_FACTS, 1.0)
    st.progress(progress, text=f"Context: {min(int(progress*100), 100)}% — council unlocks when complete")

    # Per-category breakdown
    with st.expander("Fact breakdown", expanded=True):
        subtype  = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
        opt_list = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Offer 1", "Offer 2"])

        if subtype == "offer_comparison":
            categories = [
                (f"{opt_list[0] if opt_list else 'Offer 1'} details", "offer_a"),
                (f"{opt_list[1] if len(opt_list)>1 else 'Offer 2'} details", "offer_b"),
                ("Values", "values"),
                ("Decision metadata", "decision_metadata"),
            ]
        else:
            categories = [
                ("Decision metadata", "decision_metadata"),
                ("Interests", "interests"),
                ("Career vision", "career_vision"),
                ("Strengths", "strengths"),
                ("Values", "values"),
                ("Financial", "financial"),
                ("Current situation", "current"),
            ]

        for label, key in categories:
            cat = state_dict.get(key, {})
            count = sum(
                1 for v in cat.values()
                if v is not None and v is not False and v != "" and v != []
            )
            if count > 0:
                st.caption(f"✅ {label}: {count} facts")
            else:
                st.caption(f"⬜ {label}: none yet")

    # Missing info
    if state_dict.get("missing_info"):
        with st.expander("Missing info", expanded=False):
            for m in state_dict["missing_info"]:
                st.caption(f"- {m}")


# ── Decision Tree ──────────────────────────────────────────────────────────────
# Master registry: (category, field) → (agent_id, display_label, note, impact_fn)
# impact_fn takes the raw value and returns an impact string
def _tree_impact_score(v, high_is_bad=True):
    """Generic 1-10 score → impact. high_is_bad=True means high score = more risk."""
    try:
        s = int(float(v))
        if high_is_bad:
            return "high-risk" if s >= 8 else ("moderate" if s >= 5 else "positive")
        else:
            return "positive" if s >= 7 else ("moderate" if s >= 4 else "risk")
    except:
        return "neutral"

def _tree_bool_impact(v, true_is_risk=False):
    if v is True:  return "high-risk" if true_is_risk else "positive"
    if v is False: return "positive"  if true_is_risk else "neutral"
    return "neutral"

# Registry: maps (category, field_key) → (agent_id, display_label, tooltip_note, impact_fn)
# impact_fn is a lambda that takes the raw value
_FACT_REGISTRY = {
    # ── Values ──────────────────────────────────────────────────────────────────
    ("values", "financial_security"):  ("financial", "Financial Security Priority",
        "How much salary and stability factor into the decision — higher = stronger need for income",
        lambda v: _tree_impact_score(v, high_is_bad=True)),
    ("values", "career_growth"):       ("growth", "Career Growth Priority",
        "How much fast career progression matters to the user",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "work_life_balance"):   ("wellbeing", "Work-Life Balance Priority",
        "How much sustainable hours and personal time factor in",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "impact"):              ("values_agent", "Impact / Purpose Priority",
        "How much the user wants their work to matter beyond just income",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "learning"):            ("growth", "Learning Priority",
        "How important continuous skill development is to the user",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "salary_importance"):   ("financial", "Salary Importance",
        "Explicit 1-10 score the user gave for how much salary matters",
        lambda v: _tree_impact_score(v, high_is_bad=True)),

    # ── Current situation ────────────────────────────────────────────────────────
    ("current", "current_satisfaction"): ("wellbeing", "Current Job Satisfaction",
        "Low satisfaction is a push factor away from current job — but not sufficient reason alone",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("current", "business_idea"):      ("growth", "Business Idea Clarity",
        "Vague ideas carry higher execution risk than a specific validated concept",
        lambda v: "positive" if str(v).lower() not in ("vague","none","") else "risk"),
    ("current", "business_validated"): ("growth", "Business Idea Validated?",
        "Has the user tested with real customers or earned side income — reduces startup risk significantly",
        lambda v: _tree_bool_impact(v, true_is_risk=False)),
    ("current", "financial_runway"):   ("financial", "Financial Runway",
        "How long the user can survive without income — thin runway with dependents is high risk",
        lambda v: "positive" if v and str(v).lower() not in ("minimal","none","no savings") else "risk"),
    ("current", "leave_reason"):       ("wellbeing", "Reason for Leaving",
        "Frustration-driven exits are riskier than opportunity-driven ones",
        lambda v: "caution" if "frustrat" in str(v).lower() else "positive"),
    ("current", "current_role"):       ("growth", "Current Role & Field",
        "Whether the business or new path is in the same domain — affects execution risk",
        lambda v: "neutral"),
    ("current", "concern"):            ("values_agent", "Biggest Concern",
        "The user's self-identified fear — often reveals the real deciding factor",
        lambda v: "caution"),
    ("current", "job_market_concern"): ("growth", "Industry Opportunity Cost",
        "Whether staying has real upside — promotions, growth, interesting work",
        lambda v: "neutral"),
    ("current", "financial_concern"):  ("financial", "Financial / Family Constraints",
        "Scholarships, family expectations, or cost differences affecting the decision",
        lambda v: "moderate"),
    ("current", "leaning"):            ("values_agent", "Current Leaning & Reason",
        "What the user is already leaning toward — gut instinct is often values-driven",
        lambda v: "neutral"),
    ("current", "current_year"):       ("values_agent", "Year in School",
        "Switching majors or paths later carries higher cost — affects decision urgency",
        lambda v: "neutral"),
    ("current", "business_experience"): ("growth", "Business Experience (1-10)",
        "Sales, marketing, accounting skills needed to run a business",
        lambda v: _tree_impact_score(v, high_is_bad=False)),

    # ── Personal ────────────────────────────────────────────────────────────────
    ("personal", "has_family"):        ("financial", "Has Family",
        "Partner, kids, or household obligations that are affected by income changes",
        lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("personal", "has_dependents"):    ("financial", "Has Dependents",
        "Financial dependents amplify the risk of any income reduction",
        lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("personal", "partner_employed"):  ("financial", "Partner Employed?",
        "A second income reduces personal financial risk significantly",
        lambda v: "positive" if v is True else "risk"),
    ("personal", "can_relocate"):      ("values_agent", "Can Relocate?",
        "Relocation feasibility — if required and not possible, this is a hard blocker",
        lambda v: "positive" if v is True else "high-risk"),
    ("personal", "relocation_concern"):("wellbeing", "Relocation Concern",
        "Specific worry about moving — affects wellbeing and family",
        lambda v: "caution"),
    ("personal", "current_city"):      ("values_agent", "Current City",
        "Location context for relocation assessment", lambda v: "neutral"),

    # ── Career vision ────────────────────────────────────────────────────────────
    ("career_vision", "post_graduation_goal"): ("growth", "Post-Graduation Goal",
        "Job, grad school, or startup — shapes which path makes more sense",
        lambda v: "neutral"),
    ("career_vision", "desired_role_5yr"):     ("growth", "5-Year Vision",
        "If the target role requires a specific credential or path, this is a strong signal",
        lambda v: "positive" if v and str(v).lower() != "undecided" else "neutral"),
    ("career_vision", "research_vs_applied"):  ("values_agent", "Research vs Applied Lean",
        "Research lean → PhD/academia aligns better; Applied → industry job likely sufficient",
        lambda v: "positive" if str(v).lower() == "research" else "caution" if str(v).lower() == "applied" else "neutral"),
    ("career_vision", "industry_preference"):  ("growth", "Industry Preference",
        "Target industry — affects which path provides better entry", lambda v: "neutral"),

    # ── Interests ────────────────────────────────────────────────────────────────
    ("interests", "hands_on_work"):     ("wellbeing", "Prefers Hands-On Work",
        "Hands-on preference aligns with industry/applied roles over research",
        lambda v: "positive" if v is True else "neutral"),
    ("interests", "enjoys_theory"):     ("wellbeing", "Enjoys Theoretical Work",
        "Theory enjoyment is a strong predictor of PhD or research path fit",
        lambda v: "positive" if v is True else "neutral"),
    ("interests", "enjoys_coding"):     ("growth", "Enjoys Coding",
        "Coding affinity — relevant when comparing technical paths", lambda v: "neutral"),
    ("interests", "enjoys_building_systems"): ("growth", "Enjoys Building Systems",
        "Systems-builder mindset — tends to favor engineering/applied over research",
        lambda v: "positive" if v is True else "neutral"),
    ("interests", "enjoys_working_with_data"): ("growth", "Enjoys Working with Data",
        "Data affinity — relevant when comparing data-related paths", lambda v: "neutral"),
    ("interests", "research"):          ("values_agent", "Research Interest",
        "Genuine research interest is a prerequisite for PhD success",
        lambda v: "positive" if v is True else "risk"),

    # ── Financial ────────────────────────────────────────────────────────────────
    ("financial", "expected_salary"):   ("financial", "Expected Salary",
        "Target salary range — gap between this and a stipend is the real PhD cost",
        lambda v: "neutral"),
    ("financial", "salary_importance"): ("financial", "Salary Importance Score",
        "Explicit 1-10 score for how much salary matters in this decision",
        lambda v: _tree_impact_score(v, high_is_bad=True)),

    # ── Offer A ──────────────────────────────────────────────────────────────────
    ("offer_a", "role"):           ("growth",    "Offer A — Role",        "Job title for the first offer", lambda v: "neutral"),
    ("offer_a", "salary_raw"):     ("financial", "Offer A — Salary",      "Compensation for the first offer", lambda v: "neutral"),
    ("offer_a", "work_location"):  ("wellbeing", "Offer A — Location",    "Remote/onsite/hybrid for first offer", lambda v: "neutral"),
    ("offer_a", "requires_relocation"): ("values_agent", "Offer A — Relocation?", "Whether first offer requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("offer_a", "growth_potential"):    ("growth",    "Offer A — Growth Potential", "Career advancement path at first company", lambda v: "positive" if str(v).lower() in ("high","great") else "neutral"),
    ("offer_a", "work_life_balance"):   ("wellbeing", "Offer A — Work-Life Balance", "Balance expectations at first company", lambda v: "neutral"),
    ("offer_a", "concern"):        ("values_agent", "Offer A — Your Concern", "User's biggest hesitation about first offer", lambda v: "caution"),
    ("offer_a", "job_security"):   ("financial", "Offer A — Job Security", "Stability of first offer", lambda v: "positive" if str(v).lower() == "high" else "risk" if str(v).lower() == "low" else "moderate"),

    # ── Offer B ──────────────────────────────────────────────────────────────────
    ("offer_b", "role"):           ("growth",    "Offer B — Role",        "Job title for the second offer", lambda v: "neutral"),
    ("offer_b", "salary_raw"):     ("financial", "Offer B — Salary",      "Compensation for the second offer", lambda v: "neutral"),
    ("offer_b", "work_location"):  ("wellbeing", "Offer B — Location",    "Remote/onsite/hybrid for second offer", lambda v: "neutral"),
    ("offer_b", "requires_relocation"): ("values_agent", "Offer B — Relocation?", "Whether second offer requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("offer_b", "growth_potential"):    ("growth",    "Offer B — Growth Potential", "Career advancement path at second company", lambda v: "positive" if str(v).lower() in ("high","great") else "neutral"),
    ("offer_b", "work_life_balance"):   ("wellbeing", "Offer B — Work-Life Balance", "Balance expectations at second company", lambda v: "neutral"),
    ("offer_b", "concern"):        ("values_agent", "Offer B — Your Concern", "User's biggest hesitation about second offer", lambda v: "caution"),
    ("offer_b", "job_security"):   ("financial", "Offer B — Job Security", "Stability of second offer", lambda v: "positive" if str(v).lower() == "high" else "risk" if str(v).lower() == "low" else "moderate"),

    # ── University A ─────────────────────────────────────────────────────────────
    ("uni_a", "name"):             ("growth",    "University A — Name",      "First university name", lambda v: "neutral"),
    ("uni_a", "tuition_raw"):      ("financial", "University A — Tuition",   "Cost for first university", lambda v: "neutral"),
    ("uni_a", "scholarship"):      ("financial", "University A — Scholarship","Financial aid reducing cost burden", lambda v: "positive"),
    ("uni_a", "ranking"):          ("growth",    "University A — Ranking",   "Reputation and standing in field", lambda v: "neutral"),
    ("uni_a", "job_placement"):    ("growth",    "University A — Placement", "Alumni network and placement rates", lambda v: "neutral"),
    ("uni_a", "requires_relocation"): ("values_agent", "University A — Relocation?", "Whether first university requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("uni_a", "living_cost"):      ("financial", "University A — Living Cost","Cost of living in first university's city", lambda v: "neutral"),
    ("uni_a", "concern"):          ("values_agent", "University A — Concern","User's hesitation about first university", lambda v: "caution"),

    # ── University B ─────────────────────────────────────────────────────────────
    ("uni_b", "name"):             ("growth",    "University B — Name",      "Second university name", lambda v: "neutral"),
    ("uni_b", "tuition_raw"):      ("financial", "University B — Tuition",   "Cost for second university", lambda v: "neutral"),
    ("uni_b", "scholarship"):      ("financial", "University B — Scholarship","Financial aid reducing cost burden", lambda v: "positive"),
    ("uni_b", "ranking"):          ("growth",    "University B — Ranking",   "Reputation and standing in field", lambda v: "neutral"),
    ("uni_b", "job_placement"):    ("growth",    "University B — Placement", "Alumni network and placement rates", lambda v: "neutral"),
    ("uni_b", "requires_relocation"): ("values_agent", "University B — Relocation?", "Whether second university requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("uni_b", "living_cost"):      ("financial", "University B — Living Cost","Cost of living in second university's city", lambda v: "neutral"),
    ("uni_b", "concern"):          ("values_agent", "University B — Concern","User's hesitation about second university", lambda v: "caution"),
}

# Values that should never appear in the tree — placeholders, not real answers
_PLACEHOLDER_VALUES = {"neutral", "unknown", "undecided", "n/a", "none", "", "null"}

def _paired_university_factors(state_dict, opt_a, opt_b, college_retriever=None):
    """
    For university comparisons: compare uni_a vs uni_b field-by-field.
    Returns list of factor dicts with real comparative directions.
    Also fetches College Scorecard data if retriever available.
    """
    uni_a = state_dict.get("uni_a", {})
    uni_b = state_dict.get("uni_b", {})
    personal = state_dict.get("personal", {})
    values   = state_dict.get("values", {})
    interests = state_dict.get("interests", {})
    career_vis = state_dict.get("career_vision", {})

    factors = []

    def add(name, val_a, val_b, interpret_fn, category="Comparison", source=None):
        """Add a comparative factor. interpret_fn(val_a, val_b) -> (direction, label)."""
        if val_a is None and val_b is None:
            return
        direction, label = interpret_fn(val_a, val_b)
        factors.append({
            "category":  category,
            "name":      name,
            "value":     f"{opt_a}: {val_a or '?'}  |  {opt_b}: {val_b or '?'}",
            "impact":    label,
            "direction": direction,
            "source":    source,   # "api", "bls", or None (user-provided)
        })

    def add_single(name, val, direction, label, category="User Input", source=None):
        if val in (None, False, "", []):
            return
        if isinstance(val, bool) and not val:
            return
        factors.append({
            "category":  category,
            "name":      name,
            "value":     str(val),
            "impact":    label,
            "direction": direction,
            "source":    source,
        })

    # ── Paired comparisons from user-provided data ────────────────────────────

    # Tuition comparison
    ta = uni_a.get("tuition")
    tb = uni_b.get("tuition")
    if ta and tb:
        def cmp_tuition(a, b):
            try:
                a, b = float(a), float(b)
                diff = abs(a - b)
                cheaper = opt_a if a < b else opt_b
                direction = "a" if a < b else "b"
                return direction, f"{cheaper} is ${diff:,.0f}/yr cheaper"
            except:
                return "neutral", f"Tuition: {opt_a} ${a} | {opt_b} ${b}"
        add("Tuition (annual)", ta, tb, cmp_tuition, "Cost", source=None)
    elif ta:
        add_single(f"{opt_a} Tuition", f"${ta:,}/yr" if isinstance(ta, (int,float)) else ta,
                   "neutral", "Only one tuition known", "Cost")
    elif tb:
        add_single(f"{opt_b} Tuition", f"${tb:,}/yr" if isinstance(tb, (int,float)) else tb,
                   "neutral", "Only one tuition known", "Cost")

    # Scholarship
    sa = uni_a.get("scholarship")
    sb = uni_b.get("scholarship")
    if sa and sa.lower() not in ("none", "no", "unknown"):
        add_single(f"{opt_a} — Scholarship", sa, "a", f"Financial support at {opt_a}", "Cost")
    if sb and sb.lower() not in ("none", "no", "unknown"):
        add_single(f"{opt_b} — Scholarship", sb, "b", f"Financial support at {opt_b}", "Cost")

    # Ranking / reputation
    ra = uni_a.get("ranking")
    rb = uni_b.get("ranking")
    rep_imp = values.get("reputation_importance", 0)
    try: rep_imp = float(rep_imp)
    except: rep_imp = 0
    if ra and rb:
        def cmp_rank(a, b):
            strong = ["top", "well known", "strong", "reputable", "better", "higher"]
            a_str = any(w in str(a).lower() for w in strong)
            b_str = any(w in str(b).lower() for w in strong)
            if a_str and not b_str: return "a", f"{opt_a} has stronger reputation"
            if b_str and not a_str: return "b", f"{opt_b} has stronger reputation"
            return "neutral", "Reputation roughly comparable"
        add("Program Reputation", ra, rb, cmp_rank, "Academic")
    elif ra:
        label = "Strong reputation" if any(w in str(ra).lower() for w in ["top","well known","strong"]) else f"Reputation: {ra}"
        direction = "a" if rep_imp >= 7 else "neutral"
        add_single(f"{opt_a} — Reputation", ra, direction, label, "Academic")
    elif rb:
        label = "Strong reputation" if any(w in str(rb).lower() for w in ["top","well known","strong"]) else f"Reputation: {rb}"
        direction = "b" if rep_imp >= 7 else "neutral"
        add_single(f"{opt_b} — Reputation", rb, direction, label, "Academic")

    # Location preference
    city_pref = personal.get("city_preference", "")
    la = uni_a.get("location", "")
    lb = uni_b.get("location", "")
    if city_pref and (la or lb):
        big_cities = ["corpus christi", "houston", "dallas", "austin", "san antonio",
                      "los angeles", "new york", "chicago", "boston", "seattle", "miami"]
        a_big = any(c in str(la).lower() for c in big_cities)
        b_big = any(c in str(lb).lower() for c in big_cities)
        if "big" in city_pref.lower():
            if a_big and not b_big: add_single("City Preference", f"Prefers big city ({la})", "a", f"{opt_a} is in a larger city — matches preference", "Personal")
            elif b_big and not a_big: add_single("City Preference", f"Prefers big city ({lb})", "b", f"{opt_b} is in a larger city — matches preference", "Personal")
            else: add_single("City Preference", city_pref, "neutral", "Both in comparable city sizes", "Personal")
        elif "small" in city_pref.lower() or "town" in city_pref.lower():
            if b_big and not a_big: add_single("City Preference", f"Prefers smaller town ({lb})", "b", f"{opt_b} is in a smaller town — matches preference", "Personal")
            else: add_single("City Preference", city_pref, "neutral", "Campus environment preference noted", "Personal")
        else:
            add_single("City Preference", city_pref, "neutral", "Campus environment preference noted", "Personal")

    # Social connection
    social = personal.get("social_connection", "")
    if social:
        direction = "b" if opt_b.lower() in social.lower() else ("a" if opt_a.lower() in social.lower() else "neutral")
        add_single("Social Connection", social, direction, f"Social support near {opt_b if direction=='b' else opt_a}", "Personal")

    # Field of interest
    field = interests.get("field_of_interest", "")
    if field:
        add_single("Field of Interest", field, "neutral", f"Both programs may offer {field} tracks", "Academic")

    # Career goal
    goal = career_vis.get("post_graduation_goal", "")
    if goal == "job":
        add_single("Career Goal", "Industry job", "neutral", "Both paths lead to industry — placement data matters", "Career")
    elif goal:
        add_single("Career Goal", goal, "neutral", f"Targeting: {goal}", "Career")

    desired = career_vis.get("desired_role_5yr", "")
    if desired:
        add_single("Target Role", desired, "neutral", f"Aiming for: {desired}", "Career")

    # Reputation importance
    if rep_imp >= 7:
        add_single("Reputation Priority", f"{rep_imp}/10", "neutral",
                   f"High priority ({rep_imp}/10) — program name matters for hiring", "Values")

    # Taking student debt
    if state_dict.get("financial", {}).get("taking_student_debt"):
        add_single("Financing", "Student loan", "neutral",
                   "Taking debt — net cost and post-grad earnings matter most", "Cost")

    return factors


def _api_university_factors(card_a, card_b, opt_a, opt_b):
    """
    Generate comparative factors from College Scorecard data.
    These are clearly marked as external data (source='api').
    """
    factors = []
    if card_a is None and card_b is None:
        return factors

    def add_cmp(name, val_a, val_b, fmt_fn, direction_fn):
        if val_a is None and val_b is None:
            return
        va = fmt_fn(val_a) if val_a else "no data"
        vb = fmt_fn(val_b) if val_b else "no data"
        direction = direction_fn(val_a, val_b)
        if val_a and val_b:
            diff = abs(val_a - val_b)
            higher = opt_a if val_a > val_b else opt_b
            lower  = opt_b if val_a > val_b else opt_a
        else:
            higher = lower = None
        factors.append({
            "category":  "College Scorecard Data",
            "name":      name,
            "value":     f"{opt_a}: {va}  |  {opt_b}: {vb}",
            "impact":    direction_fn(val_a, val_b, as_label=True),
            "direction": direction_fn(val_a, val_b),
            "source":    "api",
        })

    ta = card_a.tuition_in_state  if card_a else None
    tb = card_b.tuition_in_state  if card_b else None
    if ta or tb:
        va = f"${ta:,}/yr" if ta else "no data"
        vb = f"${tb:,}/yr" if tb else "no data"
        if ta and tb:
            cheaper = opt_a if ta < tb else opt_b
            direction = "a" if ta < tb else "b"
            label = f"{cheaper} is ${abs(ta-tb):,}/yr cheaper in-state"
        elif ta:
            direction, label = "neutral", f"{opt_a}: ${ta:,}/yr"
        else:
            direction, label = "neutral", f"{opt_b}: ${tb:,}/yr"
        factors.append({"category": "College Scorecard Data", "name": "In-State Tuition",
                        "value": f"{opt_a}: {va}  |  {opt_b}: {vb}",
                        "impact": label, "direction": direction, "source": "api"})

    ea = card_a.median_earnings_10yr if card_a else None
    eb = card_b.median_earnings_10yr if card_b else None
    if ea or eb:
        va = f"${ea:,}" if ea else "no data"
        vb = f"${eb:,}" if eb else "no data"
        if ea and eb:
            higher = opt_a if ea > eb else opt_b
            direction = "a" if ea > eb else "b"
            label = f"{higher} grads earn ${abs(ea-eb):,} more at 10 years"
        elif ea:
            direction, label = "neutral", f"{opt_a} median: ${ea:,}"
        else:
            direction, label = "neutral", f"{opt_b} median: ${eb:,}"
        factors.append({"category": "College Scorecard Data", "name": "Median Earnings (10yr)",
                        "value": f"{opt_a}: {va}  |  {opt_b}: {vb}",
                        "impact": label, "direction": direction, "source": "api"})

    ga = card_a.grad_rate if card_a else None
    gb = card_b.grad_rate if card_b else None
    if ga or gb:
        va = f"{ga*100:.0f}%" if ga else "no data"
        vb = f"{gb*100:.0f}%" if gb else "no data"
        if ga and gb:
            higher = opt_a if ga > gb else opt_b
            direction = "a" if ga > gb else "b"
            label = f"{higher} has higher graduation rate ({max(ga,gb)*100:.0f}%)"
        else:
            direction, label = "neutral", "Grad rate data partial"
        factors.append({"category": "College Scorecard Data", "name": "Graduation Rate (6yr)",
                        "value": f"{opt_a}: {va}  |  {opt_b}: {vb}",
                        "impact": label, "direction": direction, "source": "api"})

    return factors


def _compute_dynamic_votes(factors: list, agents: list, agent_votes_llm: dict) -> dict:
    """
    Derive per-agent vote percentages from the real weighted factors in the
    state — not from raw LLM votes which collapse to 100-0.
    Each agent's domain factors are scored by direction+impact, normalised to
    a percentage clamped [15, 85] so no agent ever shows 0% or 100%.

    AGENT_CATS: each agent claims certain factor category strings.
    Financial agent also claims 'Values & Priorities' because salary-priority
    scores (financial_security) live there and are the primary financial signal
    for career/major decisions where no dollar figures are collected.
    """
    IMPACT_W = {
        "positive": 3, "high-risk": 3,
        "moderate": 2, "caution":   2,
        "neutral":  1,
    }
    AGENT_CATS = {
        "financial":    {"Financial", "Cost", "College Scorecard Data", "Values & Priorities"},
        "growth":       {"Career Vision", "Interests & Work Style", "Academic", "BLS Data"},
        "wellbeing":    {"Personal Context", "Current Situation", "Personal"},
        "values_agent": {"Comparison", "User Input"},
    }
    dynamic = {}
    for agent in agents:
        aid      = agent["id"]
        my_cats  = AGENT_CATS.get(aid, set())
        relevant = [f for f in factors if f.get("category", "") in my_cats]
        if not relevant:
            # No domain factors — use normalized LLM vote clamped to [30,70]
            llm_v = agent_votes_llm.get(aid, {})
            raw_a = llm_v.get("option_a", 50)
            pct_a = min(70, max(30, raw_a))
            dynamic[aid] = {"option_a": pct_a, "option_b": 100 - pct_a}
            continue
        score_a = score_b = 0.0
        for f in relevant:
            w = IMPACT_W.get(f.get("impact", "neutral"), 1)
            d = f.get("direction", "neutral")
            if   d == "a": score_a += w
            elif d == "b": score_b += w
            else:          score_a += w * 0.5; score_b += w * 0.5
        total = score_a + score_b
        pct_a = round((score_a / total) * 100) if total >= 0.5 else 50
        pct_a = min(85, max(15, pct_a))
        dynamic[aid] = {"option_a": pct_a, "option_b": 100 - pct_a}
    return dynamic


def render_decision_tree(state_dict: dict, council_results: dict):
    """
    Decision factor tree — comparative analysis.
    For university comparisons: does paired field-by-field comparison.
    External data (College Scorecard / BLS) highlighted distinctly.
    """
    options  = state_dict.get("decision_metadata", {}).get("options_being_compared", ["A", "B"])
    subtype  = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
    opt_a    = options[0] if len(options) > 0 else "Option A"
    opt_b    = options[1] if len(options) > 1 else "Option B"
    avg_vote = council_results.get("avg_vote", {})
    avg_a    = avg_vote.get("option_a", 50)
    avg_b    = avg_vote.get("option_b", 50)
    winner   = opt_a if avg_a >= avg_b else opt_b
    win_pct  = max(avg_a, avg_b)
    win_color = "#16a34a" if avg_a >= avg_b else "#dc2626"
    agents   = council_results.get("agents", [])
    agent_votes = council_results.get("agent_votes", {})
    # dynamic_votes computed after factors list is built below

    # ── Build factor list ─────────────────────────────────────────────────────
    bls_factors = []  # populated in major_choice branch; must exist for all branches
    if subtype == "university_comparison":
        # Paired comparative analysis
        factors = _paired_university_factors(state_dict, opt_a, opt_b)

        # Try to get College Scorecard data for the tree
        try:
            college = st.session_state.llm._college if st.session_state.llm else None
            if college:
                cards = college.get_cards_for_decision(opt_a, opt_b)
                ca, cb = cards.get("option_a"), cards.get("option_b")
                if ca or cb:
                    api_factors = _api_university_factors(ca, cb, opt_a, opt_b)
                    if api_factors:
                        factors = api_factors + factors
                        logging.info(f"[TREE] Injected {len(api_factors)} scorecard factors")
                    else:
                        logging.info("[TREE] Scorecard cards found but no comparable fields")
                else:
                    logging.info(f"[TREE] Scorecard: no match for {opt_a!r} or {opt_b!r}")
            else:
                logging.warning("[TREE] College retriever not initialised")
        except Exception as e:
            logging.error(f"[TREE] College Scorecard error: {type(e).__name__}: {e}")

    else:
        # Generic factor list for non-university decisions
        SKIP_VALS   = (None, False, "", [], "none", "null", "unknown", "n/a")
        SKIP_FIELDS = {"financial_runway_months","salary_raw","tuition_raw",
                       "work_life_balance_known","team_culture_known","job_market_concern"}
        READABLE = {
            "financial_security":    "Financial Security Priority",
            "career_growth":         "Career Growth Priority",
            "work_life_balance":     "Work-Life Balance Priority",
            "has_dependents":        "Has Dependents",
            "has_family":            "Has Family",
            "can_relocate":          "Can Relocate",
            "partner_employed":      "Partner Employed",
            "requires_relocation":   "Requires Relocation",
            "current_satisfaction":  "Current Satisfaction",
            "post_graduation_goal":  "Post-Graduation Goal",
            "desired_role_5yr":      "Desired Role (5yr)",
            "research_vs_applied":   "Research vs Applied",
            "hands_on_work":         "Prefers Hands-On Work",
            "concern":               "Main Concern",
            "financial_concern":     "Financial Concern",
            "leaning":               "Stated Lean",
            "financial_runway":      "Financial Runway",
            "current_salary":        "Current Salary",
            "current_income":        "Current Income",
            "current_savings":       "Savings / Runway",
            "debt_total":            "Total Debt",
            "leave_reason":          "Reason for Leaving",
            "business_validated":    "Business Tested",
            "business_idea":         "Business Idea",
            "current_satisfaction":  "Job Satisfaction",
            "city_preference":       "City Preference",
            "social_connection":     "Social Connection",
            "field_of_interest":     "Field of Interest",
            "reputation_importance": "Reputation Priority",
            "taking_student_debt":   "Taking Student Debt",
            "work_anywhere":         "Open to Work Anywhere",
        }
        CAT_LABELS = {
            "values":"Values & Priorities","interests":"Interests & Work Style",
            "career_vision":"Career Vision","current":"Current Situation",
            "personal":"Personal Context","financial":"Financial",
            "offer_a": opt_a,"offer_b": opt_b,
        }

        def _impact(field, val, subtype_):
            val_str = str(val).lower()
            if field == "enjoys_coding" and val in (True, "True", "true", "yes"):
                return ("a","Enjoys coding -- favors CS path")
            if field == "enjoys_building_systems" and val in (True, "True", "true", "yes"):
                return ("a","Enjoys building systems -- favors CS")
            if field == "enjoys_analysis" and val in (True, "True", "true", "yes"):
                return ("b","Enjoys data analysis -- favors DS")
            if field == "financial_security" and isinstance(val,(int,float)):
                if int(val) >= 8:
                    return ("a", f"{val}/10 salary priority -- favors higher-paying path")
                elif int(val) >= 5:
                    return ("neutral", f"Financial security: {val}/10")
                else:
                    return ("b", f"{val}/10 -- low priority, more flexibility for risk")
            if field == "career_growth" and isinstance(val,(int,float)):
                return ("a", f"Career growth: {val}/10") if int(val)>=7 else ("neutral",f"Career growth: {val}/10")
            if field == "current_satisfaction" and isinstance(val,(int,float)):
                if int(val) >= 7: return ("a",f"Satisfied ({val}/10)")
                if int(val) <= 4: return ("b",f"Dissatisfied ({val}/10)")
                return ("neutral",f"Satisfaction: {val}/10")
            if field == "concern":
                if any(w in val_str for w in ["jobless","income","money","debt","stability"]): return ("a",f"Fear: {val}")
                if any(w in val_str for w in ["regret","miss","stuck","hate"]): return ("b",f"Fear: {val}")
                return ("neutral",f"Concern: {val}")
            if field in ("has_dependents","has_family") and val is True: return ("a","Has dependents -- stability matters")
            if field == "partner_employed" and val is True: return ("b","Partner employed -- shared safety net")
            if field in ("current_income","current_salary") and isinstance(val,(int,float)):
                return ("a",f"${val:,.0f}/yr -- high opportunity cost") if val>=80000 else ("neutral",f"${val:,.0f}/yr")
            if field in ("financial_runway","current_savings") and isinstance(val,(int,float)):
                return ("b",f"${val:,.0f} runway") if val>=100000 else ("neutral",f"${val:,.0f} runway")
            if field == "leaning":
                _lean_lower = str(val).lower().strip()
                if _lean_lower in ("none", "unknown", "uncertain", "undecided", "n/a", "not sure", ""):
                    return ("neutral", "No stated lean")
                return ("b", f"Leans: {val}")
            if field == "desired_role_5yr":
                # For CS vs DS: engineering/dev/lead roles favor CS; analyst/scientist favor DS
                cs_roles = ["software","engineer","developer","lead","manager","architect","devops","sre","backend","frontend","fullstack"]
                ds_roles  = ["data scientist","analyst","ml engineer","machine learning","research scientist","statistician"]
                if any(r in val_str for r in cs_roles): return ("a", f"Target role '{val}' aligns with CS path")
                if any(r in val_str for r in ds_roles):  return ("b", f"Target role '{val}' aligns with DS path")
                return ("neutral", f"Target role: {val}")
            if field == "post_graduation_goal":
                return ("a","Targeting industry job -- CS has broader openings") if "job" in val_str else ("neutral",str(val))
            if field == "hands_on_work" and val is True: return ("a","Prefers hands-on building -- favors CS")
            if field == "research" and val is True: return ("b","Research-oriented -- favors DS/ML path")
            return ("neutral",str(val)[:45])

        seen = set()
        factors = []
        for cat, cat_label in CAT_LABELS.items():
            cat_data = state_dict.get(cat, {})
            if not isinstance(cat_data, dict): continue
            for field, val in cat_data.items():
                if field in SKIP_FIELDS or val in (None, False, "", []): continue
                if isinstance(val, bool) and not val: continue
                dk = (field, str(val).lower().strip())
                if dk in seen: continue
                seen.add(dk)
                display_name = READABLE.get(field, field.replace("_"," ").title())
                if isinstance(val, float): val = round(val, 1)
                elif isinstance(val, list): val = ", ".join(str(x) for x in val)
                direction, impact = _impact(field, str(val), subtype)
                factors.append({"category":cat_label,"name":display_name,
                                 "value":str(val),"impact":impact,
                                 "direction":direction,"source":None})

    factors = bls_factors + factors  # BLS data shown first

    # Re-derive winner/percentages from real state factors (not raw LLM votes)
    dynamic_votes = _compute_dynamic_votes(factors, agents, agent_votes)
    if agents:
        dyn_avg_a = round(sum(v["option_a"] for v in dynamic_votes.values()) / len(agents))
        dyn_avg_b = 100 - dyn_avg_a
        winner    = opt_a if dyn_avg_a >= dyn_avg_b else opt_b
        win_pct   = max(dyn_avg_a, dyn_avg_b)
        win_color = "#16a34a" if dyn_avg_a >= dyn_avg_b else "#dc2626"

    if not factors:
        st.markdown("""
        <div style='background:#fef9c3;border:1px solid #fde047;border-radius:8px;
                    padding:14px 16px;color:#713f12;'>
            <b>🌳 Not enough data to build a decision tree.</b><br>
            Complete a full conversation first — the tree shows factors that actually shaped the decision.
        </div>""", unsafe_allow_html=True)
        return

    # ── Color scheme ──────────────────────────────────────────────────────────
    dir_color = {"a": "#16a34a", "b": "#2563eb", "neutral": "#64748b"}
    dir_bg    = {"a": "#f0fdf4", "b": "#eff6ff",  "neutral": "#f8fafc"}
    dir_arrow = {"a": f"→ {opt_a[:16]}", "b": f"→ {opt_b[:16]}", "neutral": "↔ Both"}

    # ── Build factor nodes HTML ───────────────────────────────────────────────
    factor_nodes_html = ""
    for i, f in enumerate(factors):
        dc = dir_color[f["direction"]]
        db = dir_bg[f["direction"]]
        arrow = dir_arrow[f["direction"]]
        is_api = f.get("source") == "api"
        is_bls = f.get("source") == "bls"

        # External data gets a distinct badge + highlight
        source_badge = ""
        border_style = f"1px solid {dc}30"
        bg_style     = db
        if is_api:
            source_badge = ("<span style='background:#0ea5e9;color:white;font-size:9px;"
                           "font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px;"
                           "vertical-align:middle;'>SCORECARD</span>")
            border_style = f"2px solid {dc}60"
            bg_style     = f"linear-gradient(135deg, {db}, #f0f9ff)"
        elif is_bls:
            source_badge = ("<span style='background:#8b5cf6;color:white;font-size:9px;"
                           "font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px;"
                           "vertical-align:middle;'>BLS DATA</span>")
            border_style = f"2px solid {dc}60"
            bg_style     = f"linear-gradient(135deg, {db}, #faf5ff)"

        factor_nodes_html += f"""
        <div style="display:flex;align-items:stretch;margin-bottom:6px;">
          <div style="width:24px;display:flex;flex-direction:column;align-items:center;flex-shrink:0;">
            <div style="width:2px;background:#cbd5e1;flex:1;"></div>
            <div style="width:12px;height:2px;background:#cbd5e1;"></div>
            <div style="width:2px;background:#cbd5e1;flex:1;{"display:none" if i==len(factors)-1 else ""}"></div>
          </div>
          <div style="flex:1;background:{bg_style};border:{border_style};border-radius:8px;
              padding:7px 12px;margin-left:4px;display:flex;align-items:center;gap:10px;">
            <div style="flex:1;">
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">
                {f["category"]}{source_badge}
              </div>
              <div style="font-size:12px;font-weight:600;color:#1e293b;">
                {f["name"]}: <span style="color:{dc};">{f["value"][:60]}</span>
              </div>
              <div style="font-size:11px;color:#64748b;margin-top:1px;">{f["impact"]}</div>
            </div>
            <div style="background:{dc}18;color:{dc};border:1px solid {dc}40;border-radius:4px;
                padding:2px 8px;font-size:10px;font-weight:700;white-space:nowrap;">{arrow}</div>
          </div>
        </div>"""

    # ── Agent votes panel ─────────────────────────────────────────────────────
    agent_rows_html = ""
    for ag in agents:
        v   = dynamic_votes.get(ag["id"], {"option_a": 50, "option_b": 50})
        va  = v.get("option_a", 50)
        vb  = v.get("option_b", 50)
        lean = opt_a if va >= vb else opt_b
        lc   = "#16a34a" if va >= vb else "#dc2626"
        agent_rows_html += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #f1f5f9;">
          <span style="font-size:16px;">{ag["emoji"]}</span>
          <span style="font-size:11px;font-weight:600;color:{ag["color"]};flex:1;">{ag["name"]}</span>
          <span style="font-size:11px;color:#64748b;">{opt_a[:10]} {va}% · {opt_b[:10]} {vb}%</span>
          <span style="background:{lc}18;color:{lc};border:1px solid {lc}40;
              border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;">→ {lean}</span>
        </div>"""

    # ── Legend note for external data ─────────────────────────────────────────
    has_api = any(f.get("source") == "api" for f in factors)
    has_bls = any(f.get("source") == "bls" for f in factors)
    ext_legend = ""
    if has_api:
        ext_legend += "<div style='font-size:10px;margin-top:4px;'><span style='background:#0ea5e9;color:white;font-size:9px;font-weight:700;padding:1px 4px;border-radius:3px;'>SCORECARD</span> = U.S. Dept. of Education verified data</div>"
    if has_bls:
        ext_legend += "<div style='font-size:10px;margin-top:4px;'><span style='background:#8b5cf6;color:white;font-size:9px;font-weight:700;padding:1px 4px;border-radius:3px;'>BLS DATA</span> = Bureau of Labor Statistics verified data</div>"

    # ── Final HTML ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Inter",sans-serif;background:#f8fafc;padding:16px;}}</style>
</head><body>
<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:10px;padding:14px 18px;
    margin-bottom:4px;color:white;text-align:center;">
  <div style="font-size:13px;font-weight:700;">{opt_a} vs {opt_b}</div>
  <div style="font-size:11px;opacity:0.8;margin-top:2px;">{len(factors)} factors — {"paired comparison" if subtype=="university_comparison" else "factor analysis"}</div>
</div>
<div style="display:flex;justify-content:center;"><div style="width:2px;height:14px;background:#cbd5e1;"></div></div>
<div style="display:grid;grid-template-columns:1fr 270px;gap:12px;align-items:start;">
  <div style="background:white;border-radius:10px;padding:14px;border:1px solid #e2e8f0;">
    <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;">
      Factors That Shaped This Decision
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
      <div style="width:24px;text-align:center;">
        <div style="width:12px;height:12px;border-radius:50%;background:#2563eb;margin:auto;"></div>
      </div>
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:4px 10px;
          font-size:11px;font-weight:600;color:#2563eb;">Decision Root</div>
    </div>
    {factor_nodes_html}
    {ext_legend}
  </div>
  <div>
    <div style="background:{win_color}18;border:2px solid {win_color}50;border-radius:10px;
        padding:12px 14px;margin-bottom:10px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:4px;">Council Outcome</div>
      <div style="font-size:16px;font-weight:700;color:{win_color};">{winner}</div>
      <div style="font-size:12px;color:{win_color};opacity:0.8;">{win_pct}% aggregate lean</div>
    </div>
    <div style="background:white;border-radius:10px;padding:12px 14px;border:1px solid #e2e8f0;">
      <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">Agent Votes</div>
      {agent_rows_html}
    </div>
    <div style="background:#f8fafc;border-radius:8px;padding:10px 12px;margin-top:10px;border:1px solid #e2e8f0;">
      <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Legend</div>
      <div style="font-size:10px;color:#16a34a;margin-bottom:3px;">🟢 → {opt_a[:18]}: factor favors this option</div>
      <div style="font-size:10px;color:#2563eb;margin-bottom:3px;">🔵 → {opt_b[:18]}: factor favors this option</div>
      <div style="font-size:10px;color:#64748b;">⚫ ↔ Both: neutral or applies to both</div>
    </div>
  </div>
</div>
</body></html>"""

    height = max(520, len(factors) * 68 + 300)
    components.html(html, height=height, scrolling=True)


# ── Debate Round renderer ──────────────────────────────────────────────────────
def _render_debate_round(p: dict, opt_a: str, opt_b: str):
    """Render the two-agent rebuttal exchange."""
    debating = p.get("debating_agents", {})
    ag_a = debating.get("a", {})
    ag_b = debating.get("b", {})
    r2a  = p.get("round2_a", "")
    r2b  = p.get("round2_b", "")
    gap  = p.get("debate_gap", 0)

    if not (r2a or r2b):
        return

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1e1b4b,#312e81);
        border-radius:10px;padding:14px 18px;margin-bottom:12px;color:white;'>
        <h3 style='margin:0;font-size:1em;'>⚔️ Debate Round</h3>
        <p style='margin:4px 0 0 0;font-size:0.78em;opacity:0.85;'>
            {ag_a.get("name","Agent A")} vs {ag_b.get("name","Agent B")} —
            vote gap was {gap}%, triggering a rebuttal exchange
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, agent, rebuttal in [(col1, ag_a, r2a), (col2, ag_b, r2b)]:
        with col:
            if not agent or not rebuttal:
                continue
            import re as _r
            reb_match = _r.search(r"REBUTTAL:\s*(.+?)(?=\nSTAND:|$)", rebuttal, _r.DOTALL | _r.IGNORECASE)
            stand_match = _r.search(r"STAND:\s*(.+)", rebuttal, _r.IGNORECASE)
            reb_text   = reb_match.group(1).strip()  if reb_match   else rebuttal
            stand_text = stand_match.group(1).strip() if stand_match else ""

            st.markdown(f"""
            <div style='background:{agent.get("bg","#f8fafc")};border-left:4px solid {agent.get("border","#94a3b8")};
                border-radius:8px;padding:12px;margin-bottom:8px;'>
                <div style='font-size:1.3em;'>{agent.get("emoji","🤖")}</div>
                <strong style='color:{agent.get("color","#1e293b")};font-size:0.82em;'>
                    {agent.get("name","Agent")} Rebuts:
                </strong>
                <div style='font-size:0.82em;color:#374151;margin-top:6px;line-height:1.5;'>
                    {reb_text}
                </div>
                {f'<div style="margin-top:8px;font-size:0.75em;color:{agent.get("color","#1e293b")};font-weight:700;">{stand_text}</div>' if stand_text else ""}
            </div>
            """, unsafe_allow_html=True)


# ── What-If renderer ───────────────────────────────────────────────────────────
def _render_whatif(state):
    """
    Interactive what-if panel.
    Sliders target the exact fields the symbolic mode rules evaluate so the
    mode actually shifts. A live Simulated Mode badge updates on every drag
    without needing to press a button.
    """
    st.caption(
        "Drag the sliders — the Simulated Mode badge updates instantly. "
        "Press **Show Rule Changes** to see which rules fire or resolve."
    )

    state_dict = state.to_dict()
    subtype    = state_dict.get("decision_metadata", {}).get("decision_subtype", "")

    # Slider fields chosen to match exactly what the mode rules evaluate
    if subtype == "job_vs_business":
        slider_defs = [
            ("financial", "financial_runway_months",
             "💰 Financial Runway (months)  [<3 Survival · 3-11 Cautious · ≥12 Growth]",
             0, 24, 6),
            ("personal",  "has_dependents",
             "👨‍👩‍👧 Has Dependents  (0 = No · 1 = Yes)",
             0, 1, 0),
            ("current",   "current_satisfaction",
             "😊 Job Satisfaction (1-10)",
             1, 10, 5),
        ]
    elif subtype == "offer_comparison":
        slider_defs = [
            ("personal", "can_relocate",
             "📍 Can Relocate  (0 = No · 1 = Yes)  [No→Cautious · Yes→Growth]",
             0, 1, 1),
            ("personal", "has_dependents",
             "👨‍👩‍👧 Has Dependents  (0 = No · 1 = Yes)  [Yes→Cautious]",
             0, 1, 0),
            ("values",   "financial_security",
             "💵 Salary Priority (1-10)",
             1, 10, 5),
        ]
    elif subtype in ("major_choice", "education_path") or state_dict.get("decision_metadata", {}).get("decision_type") in ("career_choice", "education"):
        # Mode rules M001-M003 count interests.* + career_vision.* fields.
        # Show sliders that directly change what the person VALUES, so the
        # factor tree direction and simulated winner visibly shift.
        slider_defs = [
            ("values",       "financial_security",
             "💵 Salary Priority (1-10)  [high→favors higher-paying path]",
             1, 10, 5),
            ("values",       "career_growth",
             "📈 Career Growth Priority (1-10)",
             1, 10, 5),
            ("values",       "work_life_balance",
             "⚖️ Work-Life Balance Priority (1-10)",
             1, 10, 5),
        ]
    else:
        slider_defs = [
            ("financial", "financial_runway_months",
             "💰 Financial Runway (months)  [<6 Cautious · ≥12 Growth]",
             0, 24, 6),
            ("values",    "financial_security",
             "💵 Financial Security Priority (1-10)",
             1, 10, 5),
            ("values",    "career_growth",
             "📈 Career Growth Priority (1-10)",
             1, 10, 5),
        ]

    _BOOL_FIELDS = {"has_dependents", "can_relocate"}

    def _read_current(cat, field, default):
        raw = state_dict.get(cat, {}).get(field)
        if field in _BOOL_FIELDS:
            if raw is True:  return 1
            if raw is False: return 0
            return int(default)
        try:
            return max(0, int(float(raw))) if raw is not None else int(default)
        except (TypeError, ValueError):
            return int(default)

    def _coerce(field, val):
        return bool(val) if field in _BOOL_FIELDS else val

    cols     = st.columns(len(slider_defs))
    overrides = {}
    for i, (cat, field, label, lo, hi, default) in enumerate(slider_defs):
        cur = max(lo, min(hi, _read_current(cat, field, default)))
        with cols[i]:
            val = st.slider(label, lo, hi, cur, key=f"whatif_{cat}_{field}")
            overrides[(cat, field)] = _coerce(field, val)

    # Live mode badge — evaluates on every slider drag
    diff      = state.whatif_evaluate(overrides)
    mode_orig = diff.get("mode_original", "?")
    mode_new  = diff.get("mode_modified", "?")

    # Simulated winner: rebuild dynamic votes with overridden state
    _sim_state = state_dict.copy()
    for (cat, fld), val in overrides.items():
        if isinstance(_sim_state.get(cat), dict):
            _sim_state[cat] = dict(_sim_state[cat])
            _sim_state[cat][fld] = val

    options   = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Option A", "Option B"])
    opt_a_lbl = options[0] if options else "Option A"
    opt_b_lbl = options[1] if len(options) > 1 else "Option B"

    from llm_interface import LLMInterface as _LLMi
    _sim_agents = _LLMi.AGENTS if st.session_state.llm is None else st.session_state.llm.AGENTS
    _council_cache = st.session_state.get("council_cache")
    _llm_votes = _council_cache.get("agent_votes", {}) if _council_cache else {}

    # Build sim factors from overridden state
    _sim_factors = [
        {"category": clabel, "name": fld,
         "direction": _dv_dir(fld, val, _sim_state),
         "impact": "moderate"}
        for clabel, ckey in [
            ("Values & Priorities", "values"),
            ("Interests & Work Style", "interests"),
            ("Career Vision", "career_vision"),
            ("Current Situation", "current"),
            ("Personal Context", "personal"),
            ("Financial", "financial"),
        ]
        for fld, val in _sim_state.get(ckey, {}).items()
        if val not in (None, False, "", [])
    ]
    _sim_dyn  = _compute_dynamic_votes(_sim_factors, _sim_agents, _llm_votes)
    _sim_a    = round(sum(v["option_a"] for v in _sim_dyn.values()) / max(len(_sim_agents), 1)) if _sim_agents else 50
    _sim_b    = 100 - _sim_a
    _sim_win  = opt_a_lbl if _sim_a >= _sim_b else opt_b_lbl
    _sim_wc   = "#16a34a" if _sim_a >= _sim_b else "#dc2626"

    MODE_STYLE = {
        "SURVIVAL_MODE":     ("#dc2626", "🔴"),
        "CAUTIOUS_MODE":     ("#d97706", "🟡"),
        "GROWTH_MODE":       ("#16a34a", "🟢"),
        "INSUFFICIENT_DATA": ("#64748b", "⚪"),
    }
    oc, oi = MODE_STYLE.get(mode_orig, ("#64748b", "⚪"))
    nc, ni = MODE_STYLE.get(mode_new,  ("#64748b", "⚪"))

    col_mode, col_win = st.columns(2)
    with col_mode:
        if mode_orig != mode_new:
            st.markdown(
                f"<div style='background:#fff7ed;border:2px solid #f97316;border-radius:8px;"
                f"padding:10px;text-align:center;'>"
                f"<div style='font-size:0.78rem;color:#78350f;font-weight:600;'>⚡ Mode shift</div>"
                f"<div style='color:{oc};font-weight:700;font-size:0.85rem;'>{oi} {mode_orig}</div>"
                f"<div style='color:#64748b;font-size:1rem;'>↓</div>"
                f"<div style='color:{nc};font-weight:700;font-size:0.85rem;'>{ni} {mode_new}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                f"padding:10px;text-align:center;'>"
                f"<div style='font-size:0.78rem;color:#64748b;'>Decision Mode</div>"
                f"<div style='color:{nc};font-weight:700;font-size:0.9rem;'>{ni} {mode_new}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with col_win:
        st.markdown(
            f"<div style='background:{_sim_wc}10;border:2px solid {_sim_wc}50;border-radius:8px;"
            f"padding:10px;text-align:center;'>"
            f"<div style='font-size:0.78rem;color:#64748b;'>Simulated Lean</div>"
            f"<div style='color:{_sim_wc};font-weight:700;font-size:0.9rem;'>{_sim_win}</div>"
            f"<div style='font-size:0.75rem;color:{_sim_wc};'>{max(_sim_a,_sim_b)}% vs {min(_sim_a,_sim_b)}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if st.button("🔄 Show Detailed Rule Changes", key="whatif_eval_btn"):
        newly_fired = diff.get("newly_fired", [])
        resolved    = diff.get("resolved", [])
        c1, c2 = st.columns(2)
        with c1:
            if newly_fired:
                st.error(f"**⚠️ {len(newly_fired)} new rule(s) would fire:**")
                for r in newly_fired:
                    sev   = r.rule_id   if hasattr(r, "rule_id")   else str(r)
                    concl = r.conclusion if hasattr(r, "conclusion") else str(r)
                    st.markdown(f"- `[{sev}]` {concl}")
            else:
                st.success("No new constraint violations in this scenario.")
        with c2:
            if resolved:
                st.success(f"**✅ {len(resolved)} rule(s) would resolve:**")
                for r in resolved:
                    sev = r.rule_id if hasattr(r, "rule_id") else str(r)
                    st.markdown(f"- `[{sev}]` resolved")
            else:
                st.info("No existing violations resolved in this scenario.")


# ── Save to Memory ─────────────────────────────────────────────────────────────
def _render_save_to_memory(p: dict):
    """Save decision to persistent memory with optional notes."""
    if st.session_state.get("decision_saved"):
        st.success("✅ Decision saved to memory.")
        return

    with st.expander("💾 Save this decision to memory", expanded=False):
        st.caption(
            "Saving stores the ruling and key facts so future decisions "
            "can reference patterns from this one."
        )
        notes = st.text_input(
            "Optional notes (e.g. outcome, how you felt after deciding)",
            key="memory_notes_input",
            placeholder="e.g. Chose Option A, felt good about it after 3 months"
        )
        if st.button("Save to Memory", key="save_memory_btn"):
            _MEMORY.save(
                st.session_state.state.to_dict(),
                p,
                notes=notes,
            )
            st.session_state.decision_saved = True
            st.rerun()


# ── Memory sidebar panel ────────────────────────────────────────────────────────
def _render_memory_sidebar():
    """Show past decisions in sidebar with delete capability."""
    records = _MEMORY.get_all()
    if not records:
        return

    with st.expander(f"🧠 Memory ({len(records)} past decisions)", expanded=False):
        for r in records[:5]:  # show newest 5
            opts  = " vs ".join(r.get("options", ["?", "?"]))
            date_ = r.get("timestamp", "")[:10]
            ruling = r.get("ruling", "No ruling")[:80] + ("..." if len(r.get("ruling","")) > 80 else "")
            va    = r.get("avg_vote", {}).get("option_a", "?")
            vb    = r.get("avg_vote", {}).get("option_b", "?")

            st.markdown(f"""
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                padding:8px 10px;margin-bottom:6px;font-size:11px;'>
                <div style='font-weight:700;color:#1e293b;'>{opts}</div>
                <div style='color:#64748b;margin-top:2px;'>{date_} · {va}% vs {vb}%</div>
                <div style='color:#374151;margin-top:4px;font-style:italic;'>{ruling}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑", key=f"del_mem_{r['id']}", help="Delete this record"):
                _MEMORY.delete(r["id"])
                st.rerun()

        if len(records) > 5:
            st.caption(f"…and {len(records)-5} older decisions")

        if st.button("Clear all memory", key="clear_all_mem"):
            _MEMORY.clear()
            st.rerun()


# ── Council view ───────────────────────────────────────────────────────────────

def _dv_dir(field: str, val, state_dict: dict) -> str:
    """Lightweight direction mapper used by the council vote seed."""
    val_str = str(val).lower()
    # Interest signals → CS/option-a favored
    if field in ("enjoys_coding", "enjoys_building_systems") and val is True: return "a"
    if field in ("enjoys_analysis", "enjoys_working_with_data") and val is True: return "b"
    if field == "research" and val is True: return "b"
    # Financial security: high → favors safer/higher-paying path (a)
    if field == "financial_security":
        try:
            return "a" if int(float(val)) >= 7 else ("b" if int(float(val)) <= 3 else "neutral")
        except: return "neutral"
    # Career vision
    if field == "desired_role_5yr":
        cs_kw = ["software","engineer","developer","lead","manager","architect","devops"]
        ds_kw = ["data scientist","analyst","ml engineer","machine learning","statistician"]
        if any(k in val_str for k in cs_kw): return "a"
        if any(k in val_str for k in ds_kw): return "b"
    # Concern / leaning
    if field == "leaning":
        if val_str in ("none","unknown","uncertain","undecided",""): return "neutral"
        return "b"
    return "neutral"


def render_council_perspectives():
    if not st.session_state.llm:
        st.warning("System not ready")
        return

    try:
        meta          = st.session_state.state.decision_metadata
        decision_type = meta.get("decision_type", "unknown")
        options       = meta.get("options_being_compared", [])
        logging.info(f"[COUNCIL] Type: {decision_type}, Options: {options}")

        if len(options) >= 2:
            option_a, option_b = options[0], options[1]

            # Header
            st.markdown(f"""
            <div style='text-align:center;padding:20px;
                        background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                        border-radius:10px;margin-bottom:20px;'>
                <h2 style='color:white;margin:0;'>Council of Experts</h2>
                <p style='color:#f0f0f0;margin:8px 0 0 0;'>
                    {option_a} vs {option_b} — 3 independent analyses
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("The council is deliberating..."):
                # Use cached result if available — prevents re-generation on back/forward
                # navigation, which wastes API calls and produces inconsistent outputs.
                if st.session_state.get("council_cache") is None:
                    _mem_ctx = _MEMORY.get_context_block(st.session_state.state.to_dict())
                    p = st.session_state.llm.generate_council_perspectives(
                        st.session_state.state.to_dict(),
                        memory_context=_mem_ctx,
                    )
                    st.session_state.council_cache = p
                else:
                    p = st.session_state.council_cache

            agents   = p.get("agents", [])
            votes    = p.get("agent_votes", {})
            avg_vote = p.get("avg_vote", {})

            # ── Single source of truth for all percentages in this view ──────
            # Build factor list now so dynamic_votes can be computed before
            # rendering agent cards, the vote bar, AND the tree — all consistent.
            _state_dict_snap = st.session_state.state.to_dict()
            _dyn_votes = _compute_dynamic_votes(
                # Build a quick flat factor list from the state to seed scores
                # (same logic the tree uses — avoids calling the full tree builder)
                [
                    {"category": cat_label, "name": fld,
                     "direction": _dv_dir(fld, val, _state_dict_snap),
                     "impact": "moderate"}
                    for cat_label, cat_key in [
                        ("Values & Priorities", "values"),
                        ("Interests & Work Style", "interests"),
                        ("Career Vision", "career_vision"),
                        ("Current Situation", "current"),
                        ("Personal Context", "personal"),
                        ("Financial", "financial"),
                    ]
                    for fld, val in _state_dict_snap.get(cat_key, {}).items()
                    if val not in (None, False, "", [])
                ],
                agents, votes,
            )
            # Aggregate from dynamic votes — always sums to 100
            _dyn_a = round(sum(v["option_a"] for v in _dyn_votes.values()) / max(len(agents), 1)) if agents else avg_vote.get("option_a", 50)
            _dyn_b = 100 - _dyn_a

            # ── 3 Agent analysis cards ────────────────────────────────────────
            st.markdown("### 🔍 Expert Analyses")
            cols = st.columns(len(agents))
            for i, agent in enumerate(agents):
                with cols[i]:
                    v      = _dyn_votes.get(agent["id"], {"option_a": 50, "option_b": 50})
                    vote_a = v.get("option_a", 50)
                    vote_b = v.get("option_b", 50)
                    lean   = option_a if vote_a >= vote_b else option_b

                    st.markdown(f"""
                    <div style='background:{agent["bg"]};padding:12px;border-radius:10px;
                                border-left:5px solid {agent["border"]};margin-bottom:8px;'>
                        <div style='font-size:1.5em;'>{agent["emoji"]}</div>
                        <strong style='color:{agent["color"]};font-size:0.85em;'>{agent["name"]}</strong><br>
                        <span style='font-size:1em;font-weight:bold;'>Leans: {lean}</span><br>
                        <span style='color:#555;font-size:0.78em;'>{option_a}: {vote_a}% | {option_b}: {vote_b}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Extract ANALYSIS and KEY INSIGHT sections robustly
                    # NOTE: raw LLM text lives in `votes` (original agent_votes),
                    # NOT in `_dyn_votes` which only holds {"option_a":X,"option_b":Y}
                    raw = votes.get(agent["id"], {}).get("raw", "")
                    analysis    = ""
                    key_insight = ""

                    import re as _re
                    # Extract ANALYSIS: block (everything up to next label or end)
                    m_analysis = _re.search(
                        r"ANALYSIS:\s*(.+?)(?=\nKEY INSIGHT:|\nLEAN:|$)",
                        raw, _re.DOTALL | _re.IGNORECASE
                    )
                    if m_analysis:
                        analysis = m_analysis.group(1).strip()

                    # Extract KEY INSIGHT: line
                    m_insight = _re.search(
                        r"KEY INSIGHT:\s*(.+?)(?=\n[A-Z]+:|$)",
                        raw, _re.DOTALL | _re.IGNORECASE
                    )
                    if m_insight:
                        key_insight = m_insight.group(1).strip()

                    # Fallback: if regex failed, try line-by-line
                    if not analysis:
                        in_analysis = False
                        lines_buf   = []
                        for line in raw.split("\n"):
                            if line.strip().upper().startswith("ANALYSIS:"):
                                in_analysis = True
                                rest = line.split(":", 1)[-1].strip()
                                if rest:
                                    lines_buf.append(rest)
                            elif in_analysis and line.strip().upper().startswith("KEY INSIGHT:"):
                                break
                            elif in_analysis and line.strip():
                                lines_buf.append(line.strip())
                        analysis = " ".join(lines_buf)

                    if analysis:
                        # Show full analysis text, not truncated
                        st.markdown(f"""
                        <div style='font-size:0.82em;color:#374151;line-height:1.5;
                                    padding:8px 0;border-top:1px solid #e5e7eb;margin-top:6px;'>
                            {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                    if key_insight:
                        st.markdown(f"""
                        <div style='background:white;padding:6px 10px;border-radius:6px;
                                    border-left:3px solid {agent["border"]};margin-top:6px;
                                    font-size:0.78em;font-weight:600;color:{agent["color"]};'>
                            💡 {key_insight}
                        </div>
                        """, unsafe_allow_html=True)

     # ── Aggregate vote bar ──────────────────────────────────────────────
            st.markdown("---")
            winner    = option_a if _dyn_a >= _dyn_b else option_b
            win_color = "#16a34a" if _dyn_a >= _dyn_b else "#dc2626"
            st.markdown(f"""
            <div style='margin:8px 0 4px 0;'>
              <div style='display:flex;border-radius:8px;overflow:hidden;height:24px;'>
                <div style='width:{_dyn_a}%;background:#16a34a;display:flex;align-items:center;
                    justify-content:center;font-size:11px;font-weight:700;color:white;min-width:0;'>
                  {"" if _dyn_a < 20 else f"{_dyn_a}%"}
                </div>
                <div style='width:{_dyn_b}%;background:#dc2626;display:flex;align-items:center;
                    justify-content:center;font-size:11px;font-weight:700;color:white;min-width:0;'>
                  {"" if _dyn_b < 20 else f"{_dyn_b}%"}
                </div>
              </div>
              <div style='display:flex;justify-content:space-between;margin-top:5px;font-size:12px;'>
                <span style='color:#16a34a;font-weight:600;'>🟢 {option_a} {_dyn_a}%</span>
                <span style='background:{win_color}15;color:{win_color};border:1px solid {win_color}40;
                    font-weight:700;border-radius:4px;padding:2px 10px;font-size:11px;'>→ {winner} wins</span>
                <span style='color:#dc2626;font-weight:600;'>{_dyn_b}% {option_b} 🔴</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Synthesizer ruling ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div style='background:#fff3e0;padding:15px;border-radius:10px;
                        border-left:5px solid #ff9800;margin-bottom:10px;'>
                <h3 style='color:#e65100;margin:0;'>⚖️ The Synthesizer Rules</h3>
            </div>
            """, unsafe_allow_html=True)

            synth_text = p.get("synthesizer", "No ruling available")
            # Extract and render CONFIDENCE badge separately
            import re as _re2
            conf_match = _re2.search(
                r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)\s*[—-]?\s*(.+?)(?=\n[A-Z ]+:|$)",
                synth_text, _re2.DOTALL | _re2.IGNORECASE
            )
            conf_level = conf_match.group(1).upper() if conf_match else ""
            conf_note  = conf_match.group(2).strip()  if conf_match else ""
            conf_color = {"HIGH": "#16a34a", "MEDIUM": "#d97706", "LOW": "#dc2626"}.get(conf_level, "#64748b")

            if conf_level:
                st.markdown(f"""
                <div style='display:inline-block;background:{conf_color}18;
                    border:1px solid {conf_color}50;border-radius:6px;
                    padding:4px 12px;margin-bottom:10px;font-size:12px;'>
                    <strong style='color:{conf_color};'>Confidence: {conf_level}</strong>
                    {"  — " + conf_note if conf_note else ""}
                </div>
                """, unsafe_allow_html=True)

            if "OPEN QUESTION:" in synth_text:
                # Strip CONFIDENCE line before rendering main text
                display_text = _re2.sub(
                    r"\nCONFIDENCE:[^\n]+\n?", "\n", synth_text, flags=_re2.IGNORECASE
                ).strip()
                parts = display_text.split("OPEN QUESTION:")
                st.markdown(parts[0].strip())
                st.markdown(f"""
                <div style='background:#e8eaf6;padding:12px;border-radius:8px;
                            border-left:4px solid #5c6bc0;margin-top:12px;'>
                    <strong style='color:#3949ab;'>💭 Something to reflect on:</strong><br>
                    {parts[1].strip()}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(synth_text)

            # ── Debate Round — shown only when it fired ────────────────────
            if p.get("has_round3"):
                st.markdown("---")
                _render_debate_round(p, option_a, option_b)

            # ── What-If Analysis ──────────────────────────────────────────────
            st.markdown("---")
            with st.expander("🔬 What-If Scenario Explorer", expanded=False):
                _render_whatif(st.session_state.state)

            # ── Decision reasoning tree ───────────────────────────────────────
            st.markdown("---")
            with st.expander("🌳 Decision Factor Tree", expanded=False):
                st.caption(
                    "Each branch is a fact collected from the conversation. "
                    "Green = favors the left option. Blue = favors the right option. "
                    "Grey = neutral or applies to both."
                )
                render_decision_tree(st.session_state.state.to_dict(), p)

            # ── Save to Memory ────────────────────────────────────────────────
            st.markdown("---")
            _render_save_to_memory(p)

        else:
            # No options — general 3-analyst fallback
            st.markdown("""
            <div style='text-align:center;padding:20px;
                        background:linear-gradient(90deg,#ff6b6b 0%,#4ecdc4 50%,#45b7d1 100%);
                        border-radius:10px;margin-bottom:20px;'>
                <h2 style='color:white;margin:0;'>Council of Experts</h2>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("The council is deliberating..."):
                # Same cache check — fallback path also benefits from consistency.
                if st.session_state.get("council_cache") is None:
                    _mem_ctx = _MEMORY.get_context_block(st.session_state.state.to_dict())
                    p = st.session_state.llm.generate_council_perspectives(
                        st.session_state.state.to_dict(),
                        memory_context=_mem_ctx,
                    )
                    st.session_state.council_cache = p
                else:
                    p = st.session_state.council_cache

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### ⚠️ Risk Analyst")
                st.markdown(p.get("risk", "No analysis"))
            with col2:
                st.markdown("#### 🚀 Opportunity Analyst")
                st.markdown(p.get("opportunity", "No analysis"))
            with col3:
                st.markdown("#### 🎯 Values Analyst")
                st.markdown(p.get("values", "No analysis"))

        # ── Navigation buttons ────────────────────────────────────────────────
        st.markdown("---")
        col_back, col_new = st.columns(2)
        with col_back:
            if st.button("← Back to Chat", use_container_width=True):
                st.session_state.show_council = False
                st.rerun()
        with col_new:
            if st.button("Start New Decision", use_container_width=True, type="primary"):
                st.session_state.state = DecisionState()
                st.session_state.messages = []
                st.session_state.show_council = False
                st.session_state.chat_locked = False
                st.session_state.council_cache = None
                if st.session_state.llm:
                    st.session_state.llm.reset_conversation()
                st.rerun()

    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        st.error("The council ran into a problem — try again or start a new decision.")
        col_back, col_new = st.columns(2)
        with col_back:
            if st.button("Back to Chat"):
                st.session_state.show_council = False
                st.rerun()
        with col_new:
            if st.button("Start New Decision"):
                st.session_state.state = DecisionState()
                st.session_state.messages = []
                st.session_state.show_council = False
                st.session_state.chat_locked = False
                st.session_state.council_cache = None
                if st.session_state.llm:
                    st.session_state.llm.reset_conversation()
                st.rerun()



# ── Schema validation for LLM-extracted values ─────────────────────────────────
# Defines acceptable types and ranges per (category, field).
# Any value that fails validation is logged and dropped — it never reaches the
# symbolic engine.  Adding a new field here is all that is needed to protect it.

_FIELD_SCHEMA: dict = {
    # ── Values ──────────────────────────────────────────────────────────────
    ("values", "financial_security"):   ("int",   1, 10),
    ("values", "career_growth"):        ("int",   1, 10),
    ("values", "work_life_balance"):    ("int",   1, 10),
    ("values", "learning"):             ("int",   1, 10),
    ("values", "impact"):               ("int",   1, 10),
    ("values", "reputation_importance"):("int",   1, 10),
    # ── Financial ───────────────────────────────────────────────────────────
    ("financial", "financial_runway_months"): ("int",   0, 600),
    ("financial", "current_income"):          ("float", 0, None),
    ("financial", "monthly_expenses"):        ("float", 0, None),
    ("financial", "current_savings"):         ("float", 0, None),
    ("financial", "expected_salary"):         ("float", 0, None),
    ("financial", "debt_total"):              ("float", 0, None),
    ("financial", "salary_importance"):       ("int",   1, 10),
    ("financial", "taking_student_debt"):     ("bool",  None, None),
    # ── Offer A / B salaries ────────────────────────────────────────────────
    ("offer_a", "salary"):  ("float", 0, None),
    ("offer_b", "salary"):  ("float", 0, None),
    # ── Current satisfaction ─────────────────────────────────────────────────
    ("current", "current_satisfaction"): ("int", 1, 10),
    # ── Personal booleans ────────────────────────────────────────────────────
    ("personal", "has_family"):      ("bool", None, None),
    ("personal", "has_dependents"):  ("bool", None, None),
    ("personal", "can_relocate"):    ("bool", None, None),
    ("personal", "partner_employed"):("bool", None, None),
    # ── Interests booleans ──────────────────────────────────────────────────
    ("interests", "research"):               ("bool", None, None),
    ("interests", "hands_on_work"):          ("bool", None, None),
    ("interests", "enjoys_coding"):          ("bool", None, None),
    ("interests", "enjoys_theory"):          ("bool", None, None),
    ("interests", "enjoys_building_systems"):("bool", None, None),
    # ── Career vision ────────────────────────────────────────────────────────
    ("career_vision", "work_anywhere"): ("bool", None, None),
    # ── Financial booleans ───────────────────────────────────────────────────
    ("financial", "business_validated"): ("bool", None, None),
}

_ALLOWED_BOOL_STRINGS = {"true": True, "false": False, "yes": True, "no": False}


def _validate_extracted_value(category: str, key: str, value) -> tuple:
    """
    Validate and coerce a single extracted value against the schema.

    Returns (coerced_value, True) on success.
    Returns (None, False) if the value is invalid — caller should skip the update.

    Any type that is NOT in _FIELD_SCHEMA is passed through unchanged (strings,
    lists, etc.) — we only enforce the fields where bad types cause rule misfires.
    """
    schema_key = (category, key)
    if schema_key not in _FIELD_SCHEMA:
        # Not in schema → accept as-is (string, list, etc.)
        return value, True

    expected_type, lo, hi = _FIELD_SCHEMA[schema_key]

    # ── bool ──────────────────────────────────────────────────────────────
    if expected_type == "bool":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str):
            mapped = _ALLOWED_BOOL_STRINGS.get(value.strip().lower())
            if mapped is not None:
                return mapped, True
        if isinstance(value, (int, float)):
            return bool(value), True
        logging.warning(
            f"[VALIDATE] {category}.{key}: expected bool, got {type(value).__name__}={value!r} — dropped"
        )
        return None, False

    # ── int / float ───────────────────────────────────────────────────────
    if expected_type in ("int", "float"):
        if isinstance(value, str):
            # Strip common suffixes so "24 months" → 24
            cleaned = value.strip().lower()
            cleaned = cleaned.split()[0].replace(",", "").replace("$", "")
            try:
                value = float(cleaned)
            except ValueError:
                logging.warning(
                    f"[VALIDATE] {category}.{key}: cannot parse {value!r} as {expected_type} — dropped"
                )
                return None, False
        try:
            coerced = int(round(value)) if expected_type == "int" else float(value)
        except (TypeError, ValueError):
            logging.warning(
                f"[VALIDATE] {category}.{key}: cannot coerce {value!r} to {expected_type} — dropped"
            )
            return None, False

        if lo is not None and coerced < lo:
            logging.warning(
                f"[VALIDATE] {category}.{key}: value {coerced} below minimum {lo} — clamped"
            )
            coerced = lo
        if hi is not None and coerced > hi:
            logging.warning(
                f"[VALIDATE] {category}.{key}: value {coerced} above maximum {hi} — clamped"
            )
            coerced = hi
        return coerced, True

    # Should never reach here
    return value, True


def speak_response(text: str):
    """
    Speak the assistant response using Google TTS (gTTS) generated server-side.
    This produces a natural-sounding voice via Google's TTS engine — far better
    than the browser's built-in Web Speech API which uses robotic OS voices.
    The audio is encoded as base64 and played inline via an <audio> tag.
    """
    import re as _re
    import io
    import base64

    # Strip markdown so TTS doesn't read asterisks, hashes, backticks aloud
    clean = _re.sub(r'[*#`_~]', '', text)
    clean = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
    clean = _re.sub(r'<!--.*?-->', '', clean, flags=_re.DOTALL)
    clean = _re.sub(r'\n+', ' ', clean)
    clean = _re.sub(r'\s{2,}', ' ', clean).strip()

    if not clean:
        return

    try:
        from gtts import gTTS
        tts = gTTS(text=clean, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        html = f"""
<audio autoplay style="display:none">
  <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
</audio>
"""
        components.html(html, height=0)
    except Exception as e:
        # gTTS unavailable or network error — fall back to browser TTS
        logging.warning(f"[TTS] gTTS failed ({e}), falling back to browser TTS")
        import json as _json
        js_text = _json.dumps(clean)
        html = f"""
<script>
(function() {{
  window.speechSynthesis.cancel();
  var raw = {js_text}.match(/[^.!?]+[.!?]+/g) || [{js_text}];
  var sentences = raw.map(function(s){{return s.trim();}}).filter(function(s){{return s.length>1;}});
  function getBestVoice() {{
    var v = window.speechSynthesis.getVoices();
    return v.find(function(x){{return /microsoft aria/i.test(x.name);}}) ||
           v.find(function(x){{return /microsoft jenny/i.test(x.name);}}) ||
           v.find(function(x){{return /google us english/i.test(x.name);}}) ||
           v.find(function(x){{return /samantha/i.test(x.name);}}) ||
           v.find(function(x){{return x.lang==='en-US'&&!x.localService;}}) ||
           null;
  }}
  function speak(i) {{
    if(i>=sentences.length) return;
    var u=new SpeechSynthesisUtterance(sentences[i]);
    u.rate=0.88; u.lang='en-US';
    var voice=getBestVoice(); if(voice) u.voice=voice;
    u.onend=function(){{speak(i+1);}};
    window.speechSynthesis.speak(u);
  }}
  if(window.speechSynthesis.getVoices().length>0){{speak(0);}}
  else{{window.speechSynthesis.onvoiceschanged=function(){{speak(0);}};setTimeout(function(){{speak(0);}},300);}}
}})();
</script>
"""
        components.html(html, height=0)


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe recorded audio via Groq Whisper. Returns empty string on failure."""
    if not st.session_state.llm:
        return ""
    return st.session_state.llm.transcribe_audio(audio_bytes)


def process_message(user_message: str):
    if not st.session_state.llm:
        st.error("Please ensure system is initialized")
        return

    logging.info(f"Processing message: {user_message[:60]}")

    try:
        # Step 1: extract constraints
        extracted = st.session_state.llm.extract_constraints(
            user_message,
            st.session_state.state.to_dict()
        )

        # Step 2: update symbolic state — validate every value before writing
        if extracted.get("extracted"):
            for category, updates in extracted["extracted"].items():
                if not isinstance(updates, dict):
                    logging.warning(f"[VALIDATE] {category}: expected dict, got {type(updates).__name__} — skipped")
                    continue
                if hasattr(st.session_state.state, category):
                    for key, value in updates.items():
                        coerced, ok = _validate_extracted_value(category, key, value)
                        if ok:
                            st.session_state.state.update(category, key, coerced)
                        # else: already logged inside _validate_extracted_value
                else:
                    logging.warning(f"[VALIDATE] Unknown category from extraction: '{category}' — skipped")

        # Step 3: generate response
        response = st.session_state.llm.generate_response(
            user_message,
            st.session_state.state.to_dict(),
            mode="conversational"
        )

        # Guard against rate-limit error passthrough
        if response.strip().lower().startswith("i encountered an error"):
            st.warning("The AI ran into a temporary issue — please try again in a moment.")
            return

        # Detect the clean machine-readable conclusion signal injected by
        # generate_response().  Strip it before storing so it never shows in the UI.
        COUNCIL_SIGNAL = "<!-- COUNCIL_READY -->"
        should_lock = COUNCIL_SIGNAL in response
        display_response = response.replace(COUNCIL_SIGNAL, "").rstrip()

        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.messages.append({"role": "assistant", "content": display_response})

        if should_lock:
            st.session_state.chat_locked = True

    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        st.error("Something went wrong processing your message. Please try again.")
        if st.secrets.get("DEBUG", False):
            st.exception(e)


# ── Chat view ──────────────────────────────────────────────────────────────────
def render_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Auto-speak new assistant messages in talk mode
    if st.session_state.get("talk_mode") and st.session_state.messages:
        msg_idx = len(st.session_state.messages) - 1
        last_msg = st.session_state.messages[msg_idx]
        if (last_msg["role"] == "assistant"
                and msg_idx != st.session_state.get("last_spoken_idx", -1)):
            st.session_state.last_spoken_idx = msg_idx
            speak_response(last_msg["content"])

    if st.session_state.chat_locked:
        st.info(
            "I have enough information. Click **'See Council of Experts'** above "
            "to get the full multi-perspective analysis.",
            icon="💡"
        )
        if st.button("↩ Actually, I want to add more context", use_container_width=False):
            st.session_state.chat_locked = False
            st.rerun()
        return

    # ── Input: talk mode vs text mode ─────────────────────────────────────────
    if st.session_state.get("talk_mode"):
        st.markdown(
            "<div style='text-align:center;color:#888;font-size:0.85rem;margin-bottom:4px'>"
            "🎙️ <b>Talk Mode</b> — record your answer, then submit</div>",
            unsafe_allow_html=True,
        )
        audio_value = st.audio_input(
            label="Record your answer",
            label_visibility="collapsed",
            key=f"audio_input_widget_{st.session_state.audio_input_counter}",
        )
        if audio_value is not None:
            with st.spinner("Transcribing…"):
                transcribed = transcribe_audio(audio_value.read())
            if transcribed:
                # Show what was heard so user can verify
                st.caption(f"🗣️ Heard: *\"{transcribed}\"*")
                st.session_state.audio_input_counter += 1   # rotate key → clears the widget
                process_message(transcribed)
                st.rerun()
            else:
                st.session_state.audio_input_counter += 1   # also rotate on failure so user can re-record
                st.warning("Couldn't transcribe that — please try again or switch to text mode.")
    else:
        if prompt := st.chat_input("What decision are you working through?"):
            process_message(prompt)
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    render_header()

    with st.sidebar:
        st.markdown("### System")

        if not st.session_state.llm:
            with st.spinner("Initializing AI..."):
                if initialize_llm():
                    st.rerun()   # re-render into the else branch so Talk Mode toggle appears immediately
                else:
                    st.error("System initialization failed")
                    if "llm_error" in st.session_state:
                        st.code(st.session_state.llm_error)
        else:
            st.success("System Ready")
            st.caption("Using Llama 3.3 70B · Groq")
            # Warn if College Scorecard is on demo key (rate-limited)
            college_key = os.getenv("COLLEGE_SCORECARD_API_KEY", "")
            if not college_key:
                st.caption("⚠️ College data on demo key (40 req/hr) — set COLLEGE_SCORECARD_API_KEY for full access")

            # Talk mode toggle
            st.markdown("---")
            talk_on = st.toggle(
                "🎙️ Talk Mode",
                value=st.session_state.talk_mode,
                help="Speak your answers instead of typing. Uses Groq Whisper for transcription and your browser's built-in TTS for playback.",
                key="talk_mode_toggle",
            )
            if talk_on != st.session_state.talk_mode:
                st.session_state.talk_mode = talk_on
                st.session_state.last_spoken_idx = -1
                st.rerun()

        st.markdown("---")
        render_sidebar_state()
        st.markdown("---")

        if st.button("New Decision", use_container_width=True):
            st.session_state.state = DecisionState()
            st.session_state.messages = []
            st.session_state.show_council = False
            st.session_state.chat_locked = False
            st.session_state.council_cache = None
            st.session_state.show_whatif = False
            st.session_state.decision_saved = False
            st.session_state.last_spoken_idx = -1
            if st.session_state.llm:
                st.session_state.llm.reset_conversation()
            st.rerun()

        # Memory panel in sidebar
        st.markdown("---")
        _render_memory_sidebar()

    # Council full-screen view
    if st.session_state.get("show_council", False):
        render_council_perspectives()
        return

    render_chat()

    # Council button appears BELOW the chat (fix: was showing at top)
    if is_conversation_complete():
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🎭 See Council of Experts",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.show_council = True
                st.rerun()
        st.markdown("---")

    if len(st.session_state.messages) == 0:
        st.markdown("""
        ### Welcome!

        I help you think clearly about complex decisions. I won't tell you what to do, but I will:

        - Track your constraints and priorities
        - Catch logical inconsistencies
        - Show you different perspectives on your situation

        **How it works:**
        1. Tell me about your decision
        2. I'll ask a few targeted questions
        3. Once I have enough context, a **Council of Experts** will debate your options

        **Example:** *"I'm deciding between Computer Science and Data Science for my Masters."*
        """)


if __name__ == "__main__":
    main()
