"""
LLM Interface - Natural language layer constrained by symbolic state
Uses Groq API (OpenAI-compatible) with Llama 3.3 70B.
"""

import logging
import os
import json
from openai import OpenAI
from typing import Dict, Optional, List, Tuple


class LLMInterface:
    """
    Handles natural language interaction with strict constraints.

    The LLM is used ONLY for:
    - Asking clarifying questions (conversational, one question per turn)
    - Constraint extraction (JSON structured output)
    - Running 3 independent expert analyses + synthesizer

    It CANNOT override symbolic facts or give direct recommendations.
    """

    # ── Fixed agent definitions ───────────────────────────────────────────────
    # Each agent sees ONLY facts in its facts_filter — structural isolation,
    # not just a "forbidden" soft guardrail.
    AGENTS = [
        {
            "id":    "financial",
            "name":  "Financial Security Agent",
            "emoji": "💰",
            "color": "#1b5e20",
            "bg":    "#e8f5e9",
            "border":"#4caf50",
            "lens":  "salary, compensation, debt burden, financial runway, earning trajectory, cost of living, income risk",
            "persona": (
                "You are a cold, numbers-only financial risk analyst. "
                "You do not have emotions. You do not care if the person will be happy. "
                "You do not care about career fulfillment, passion, growth, or culture. "
                "You care about exactly one thing: the financial outcome — income, expenses, debt, risk of financial ruin. "
                "If the numbers work, it is good. If the numbers do not work, it is bad. That is it. "
                "You will not soften your conclusion because the other option feels right. "
                "You are not mean — you are a calculator that speaks."
            ),
            "facts_filter": ["financial", "offer_a", "offer_b", "uni_a", "uni_b"],
            "must_address": (
                "What is the income or cost difference in actual numbers? "
                "What does debt load or financial runway say? "
                "Which option reduces financial risk? Argue from numbers only."
            ),
        },
        {
            "id":    "growth",
            "name":  "Career Growth Agent",
            "emoji": "📈",
            "color": "#0d47a1",
            "bg":    "#e3f2fd",
            "border":"#2196f3",
            "lens":  "career trajectory, skill acquisition, industry demand, role advancement, transferable skills, 5-10 year path",
            "persona": (
                "You are a ruthless career strategist. You think in decades, not years. "
                "You do not care about salary numbers, how the person feels, or their family situation. "
                "You care about one thing: which path builds the most valuable career capital over 10 years. "
                "You think about skill stacking, industry tailwinds, and ceiling of advancement. "
                "A lower-paying path that builds rare skills beats a higher-paying dead-end every time — in your world. "
                "You will not hedge. You will not say it depends. You pick a direction and argue it."
            ),
            "facts_filter": ["career_vision", "interests", "offer_a", "offer_b", "uni_a", "uni_b", "current"],
            "must_address": (
                "Where does each path lead in 5-10 years concretely? "
                "Which builds more transferable or rare skills? "
                "Which has stronger industry demand right now? Argue from career trajectory only."
            ),
        },
        {
            "id":    "wellbeing",
            "name":  "Mental Wellbeing Agent",
            "emoji": "🧠",
            "color": "#4a148c",
            "bg":    "#f3e5f5",
            "border":"#9c27b0",
            "lens":  "identity alignment, regret risk, fulfillment, stress tolerance, passion vs pragmatism tension, life fit",
            "persona": (
                "You are a blunt psychologist who specializes in career regret. "
                "You do not care about salary, job titles, or industry trends. "
                "You care about one thing: will this person resent this choice in 10 years? "
                "You look at the gap between what they say they value and what they are actually choosing. "
                "You surface cognitive dissonance without mercy. "
                "You never cite numbers. You never mention job market data. "
                "You think about identity, meaning, and the cost of ignoring what you actually want."
            ),
            "facts_filter": ["values", "current", "personal", "interests"],
            "must_address": (
                "Will they resent this choice in 10 years — and why specifically? "
                "What does the gap between their stated values and their stated lean reveal? "
                "Is the internal tension manageable or a slow burn? Argue from human psychology only."
            ),
        },
    ]



    def __init__(self, api_key: Optional[str] = None,
                 model: str = "llama-3.3-70b-versatile",
                 bls_path: str = "bls_ooh_chunks.jsonl"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY env var.")

        # Build a list of clients — primary key first, fallback key second
        self._clients: list = [
            OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        ]
        fallback_key = os.getenv("GROQ_API_KEY2")
        if fallback_key and fallback_key != self.api_key:
            self._clients.append(
                OpenAI(api_key=fallback_key, base_url="https://api.groq.com/openai/v1")
            )
            logging.info(f"[API] Loaded {len(self._clients)} Groq API keys")
        else:
            logging.info("[API] Single Groq API key loaded (set GROQ_API_KEY2 for fallback)")

        # Keep self.client pointing to primary for any legacy code
        self.client = self._clients[0]
        self.model_name = model
        self.conversation_history: List[Dict] = []
        self.turn_count: int = 0
        self._bls = None
        try:
            from bls_retriever import BLSRetriever
            self._bls = BLSRetriever(bls_path)
        except Exception as e:
            logging.info(f"[BLS] Could not load: {e}")

        self._college = None
        try:
            from college_retriever import CollegeRetriever
            self._college = CollegeRetriever()
        except Exception as e:
            logging.info(f"[COLLEGE] Could not load: {e}")

    # ── Core API call (Groq / OpenAI-compatible) ────────────────────────────────
    def _call_llm(self, messages: List[Dict], system_prompt: str = "",
                  max_tokens: int = 800) -> str:
        """
        Call Groq with automatic fallback on rate limits AND auth failures.

        Tries each (client, model) combination in order:
          - client1 + llama-3.3-70b-versatile  (primary)
          - client2 + llama-3.3-70b-versatile  (backup key, same model)
          - client1 + llama-3.1-8b-instant     (fallback model)
          - client1 + gemma2-9b-it             (last resort)

        Rate limit (429) → retry with backoff, then try next client/model.
        Auth error (401) → skip this client immediately, try next.
        Other errors     → bail immediately (bad request, model not found etc).
        """
        import time
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

        clients   = getattr(self, "_clients", [self.client])
        fallbacks = getattr(self, "_model_fallbacks",
                            [self.model_name, "llama-3.1-8b-instant", "gemma2-9b-it"])
        start_model_idx = getattr(self, "_current_model_idx", 0)
        last_err  = None

        for model_idx in range(start_model_idx, len(fallbacks)):
            model = fallbacks[model_idx]
            if model_idx > start_model_idx:
                logging.info(f"[API] Switched to fallback model: {model}")

            for client_idx, client in enumerate(clients):
                key_label = f"key{client_idx + 1}/{model}"

                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=chat_messages,
                            temperature=0.7,
                            max_tokens=max_tokens,
                        )
                        if model_idx != getattr(self, "_current_model_idx", 0):
                            self._current_model_idx = model_idx
                            logging.info(f"[API] Now using {model} for this session")
                        if client_idx > 0:
                            logging.info(f"[API] Success with key{client_idx + 1}")
                        return resp.choices[0].message.content.strip()

                    except Exception as e:
                        last_err = e
                        err_str  = str(e).lower()
                        is_rate  = any(x in err_str for x in
                                       ("rate", "429", "limit", "quota", "exceeded"))
                        is_auth  = "401" in err_str or "invalid_api_key" in err_str or "invalid api key" in err_str

                        if is_auth:
                            logging.info(f"[API] Auth error on key{client_idx+1} — trying next key")
                            break  # try next client

                        elif is_rate:
                            if attempt < 2:
                                wait = 5 * (2 ** attempt)  # 5s, 10s
                                logging.warning(f"[API] Rate limit ({key_label}), waiting {wait}s (attempt {attempt+1}/3)")
                                time.sleep(wait)
                            else:
                                logging.warning(f"[API] Rate limit exhausted ({key_label}) — trying next key/model...")
                                break  # try next client

                        else:
                            # Unrecoverable: bad request, model not found etc
                            logging.info(f"[API] Unrecoverable error on {key_label}: {e}")
                            return "I encountered an error. Could you repeat that?"

        logging.info(f"[API] All options exhausted. Last error: {last_err}")
        return "I encountered an error. Could you repeat that?"


    # Legacy alias — kept for any external callers during migration
    def _call_gemini(self, messages: List[Dict], system_prompt: str = "") -> str:
        return self._call_llm(messages, system_prompt)

    # ── Constraint extraction ─────────────────────────────────────────────────
    def extract_constraints(self, user_message: str, current_state: Dict) -> Dict:
        is_first = not current_state.get("decision_metadata", {}).get("decision_type")

        if is_first:
            system_prompt = f"""The user said: "{user_message}"

Detect the decision type, subtype, and options. Be precise.

CRITICAL RULE — check OFFER COMPARISON first:
If the message mentions "offer", "offers", "job offer", OR two company names with context
like "join", "choose", "which one", "deciding between" — it is ALWAYS offer_comparison.
Do NOT classify as major_choice or career_choice if offers/companies are mentioned.

Examples (in priority order):

1. JOB OFFER COMPARISON (highest priority):
   Triggers: "offer", "job offer", "offer from X", "offer from X and Y", "join X or Y",
             "company A vs company B", "two offers", "which one to join"
   → {{"decision_metadata": {{"decision_type": "career_choice", "decision_subtype": "offer_comparison", "options_being_compared": ["<company A>", "<company B>"]}}}}
   Examples: "offer from Apple and Google" → offer_comparison: ["Apple", "Google"]
             "join Amazon or Microsoft" → offer_comparison: ["Amazon", "Microsoft"]
             "choosing between two job offers" → offer_comparison: ["Offer 1", "Offer 2"]

2. UNIVERSITY COMPARISON:
   Triggers: "university", "college", "school", "MIT vs Stanford", "which program"
   → {{"decision_metadata": {{"decision_type": "education", "decision_subtype": "university_comparison", "options_being_compared": ["<university A>", "<university B>"]}}}}

3. EDUCATION PATH (PhD vs work):
   Triggers: "PhD vs job", "should I do a PhD", "masters or work", "academia vs industry"
   → {{"decision_metadata": {{"decision_type": "education", "decision_subtype": "education_path", "options_being_compared": ["PhD", "Job"]}}}}

4. MAJOR/FIELD CHOICE (academic fields, no company names):
   Triggers: "CS vs DS", "computer science or data science", "which major", "which field to study"
   Note: Only use this if NO company names or job offers are mentioned.
   → {{"decision_metadata": {{"decision_type": "career_choice", "decision_subtype": "major_choice", "options_being_compared": ["<field A>", "<field B>"]}}}}

5. JOB VS BUSINESS:
   Triggers: "quit my job", "start a business", "leave job", "entrepreneurship vs job",
             "passion vs job", "follow my passion", "pursue passion", "job or passion",
             "job or my dream", "continue job or pursue"
   → {{"decision_metadata": {{"decision_type": "career_choice", "decision_subtype": "job_vs_business", "options_being_compared": ["Continue Job", "Pursue Passion"]}}}}
   IMPORTANT: Any message about choosing between a stable job and a passion/dream/business
   is ALWAYS job_vs_business -- even if phrased as "continue my career or follow my passion"

6. ANYTHING ELSE:
   → {{"decision_metadata": {{"decision_type": "general", "decision_subtype": "general", "options_being_compared": []}}}}

Return ONLY valid JSON:
{{"extracted": {{"decision_metadata": {{...}}}}}}"""

        else:
            decision_type    = current_state.get("decision_metadata", {}).get("decision_type", "general")
            decision_subtype = current_state.get("decision_metadata", {}).get("decision_subtype", "general")
            options          = current_state.get("decision_metadata", {}).get("options_being_compared", [])
            opt_a = options[0] if options else "Offer 1"
            opt_b = options[1] if len(options) > 1 else "Offer 2"

            if decision_subtype == "offer_comparison":
                system_prompt = f"""Extract facts from this message about a job offer comparison ({opt_a} vs {opt_b}).

Message: "{user_message}"

OFFER A details (category: "offer_a") — facts about {opt_a}:
- company: string
- role: string (job title)
- salary: number ("80k"→80000, "6 LPA"→600000, "1.5 lakh/month"→1800000)
- salary_raw: string (original text)
- work_location: "remote" / "onsite" / "hybrid"
- city: string (where the job is located)
- requires_relocation: true / false
- growth_potential: "high" / "medium" / "low" or description
- work_life_balance: "great" / "ok" / "poor" or 1-10
- job_security: "high" / "medium" / "low"
- culture: string description
- concern: string (user's hesitation)

OFFER B details (category: "offer_b") — same fields as offer_a

PERSONAL CONTEXT (category: "personal"):
- has_family: true / false
- has_dependents: true / false
- partner_employed: true / false
- can_relocate: true / false
- relocation_concern: string description
- current_city: string

VALUES (category: "values"):
- financial_security (1-10), career_growth (1-10), work_life_balance (1-10)

MAPPING RULES:
- "first offer / offer 1 / {opt_a}" → offer_a; "second offer / offer 2 / {opt_b}" → offer_b
- "same salary / same pay / both pay the same" → offer_a: {{salary_raw: "same", salary: null}}, offer_b: {{salary_raw: "same", salary: null}}
- "same benefits / same compensation" → offer_a: {{benefits: "same"}}, offer_b: {{benefits: "same"}}
- "same work location / both onsite / both remote" → offer_a: {{work_location: <value>}}, offer_b: {{work_location: <value>}}
- "neither requires relocation / both in same city" → offer_a: {{requires_relocation: false}}, offer_b: {{requires_relocation: false}}
- CRITICAL: When user says "same X for both" — ALWAYS set that field on BOTH offer_a AND offer_b
- "work from home" / "remote" → work_location = "remote"
- "need to move" / "different city" → requires_relocation = true
- "same city" / "no relocation" → requires_relocation = false
- "out of state" / "out-of-state" → tuition tier only, NOT requires_relocation
  (out-of-state = higher tuition; requires_relocation = must physically move)
- "I have a family" / "my spouse" / "my kids" → has_family = true, has_dependents = true
- If has_family/has_dependents = true AND any offer requires_relocation = true → personal.can_relocate = false
- "I can't relocate" / "can't move" / "need to stay" / "wife/husband won't move" → personal.can_relocate = false
- "open to relocating" / "fine with moving" / "flexible on location" → personal.can_relocate = true
- "startup feels risky" → job_security = "low", concern = "startup risk"
- "lots of room to grow" → growth_potential = "high"

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"offer_a": {{}}, "offer_b": {{}}, "personal": {{}}, "values": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            elif decision_subtype == "university_comparison":
                system_prompt = f"""Extract facts from this conversation message about choosing between universities ({opt_a} vs {opt_b}).

Message to extract from: "{user_message}"

CONTEXT: The question context will be prepended separately. Use it to interpret short answers.

UNIVERSITY A — {opt_a} (category: "uni_a"):
- name: string
- program: string (degree/major — e.g. "Computer Science", "MBA")
- tuition: number (annual in USD; convert "35k/yr"→35000, "20k"→20000)
  ONLY set if user explicitly states a dollar amount. If they say "out of state" with no number, leave null.
- scholarship: string ("none", "partial", "full", or amount)
- location: string (city name)
- requires_relocation: true/false
- living_cost: "high"/"medium"/"low"
- ranking: string (reputation note — e.g. "well known", "top 50", "lesser known")
- job_placement: string (known placement quality)
- concern: string (user's hesitation about THIS university)
- preferred: true/false (user explicitly prefers this one)

UNIVERSITY B — {opt_b} (category: "uni_b"):
- same fields as uni_a

PERSONAL CONTEXT (category: "personal"):
- has_family: true/false
- has_dependents: true/false
- can_relocate: true/false  ("relocate all around" / "flexible" / "no restrictions" → true)
- current_city: string
- social_connection: string (friends/family near which campus)
- city_preference: "big city" / "small town" / "no preference"

INTERESTS (category: "interests"):
- field_of_interest: string (specific area within their major — e.g. "software development", "AI", "networking")
- hands_on_work: true/false
- research: true/false

CAREER VISION (category: "career_vision"):
- post_graduation_goal: "job" / "phd" / "startup" / "undecided"
- desired_role_5yr: string (e.g. "software engineer", "data scientist")
- target_location: string (where they want to work after graduating)
- work_anywhere: true/false (willing to work anywhere in country)

FINANCIAL (category: "financial"):
- taking_student_debt: true/false
- debt_concern: "high"/"medium"/"low"

CURRENT SITUATION (category: "current"):
- concern: string (biggest overall worry — e.g. "missing out on job opportunities", "cost of living")
- leaning: string (which university they lean toward and why)

VALUES (category: "values"):
- financial_security: 1-10
- career_growth: 1-10
- work_life_balance: 1-10
- reputation_importance: 1-10 (how much the school's name matters)

EXTRACTION RULES — apply these to interpret the user's message:
- "relocate all around the country" / "can move anywhere" / "flexible on location" → personal.can_relocate = true, career_vision.work_anywhere = true
- "taking student debt" / "student loan" / "no scholarships" → financial.taking_student_debt = true, uni_a.scholarship = "none", uni_b.scholarship = "none"
- "friend at {opt_b}" / "know people at {opt_b}" → personal.social_connection = "friend at {opt_b}"
- "prefer bigger city" / "big city" → personal.city_preference = "big city", supports {opt_a} if {opt_a} is in bigger city
- "reputation matters" / "name recognition" → values.reputation_importance = 8
- "FOMO on jobs" / "fear of missing out on opportunities" → current.concern = "fear of missing out on better job opportunities"
- "a bit important" for cost of living → values.financial_security = 6
- "development" / "software development" as area of interest → interests.field_of_interest = "software development"
- If user says same program at both universities → set program field on BOTH uni_a AND uni_b
- "want to work in industry" / "get a job" → career_vision.post_graduation_goal = "job"

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"uni_a": {{}}, "uni_b": {{}}, "personal": {{}}, "interests": {{}}, "career_vision": {{}}, "financial": {{}}, "current": {{}}, "values": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            elif decision_subtype == "major_choice":
                system_prompt = f"""Extract facts from this message about a major/field choice ({opt_a} vs {opt_b}).

Message: "{user_message}"

CURRENT SITUATION (category: "current"):
- current_year: string (e.g. "freshman", "pre-college", "2nd year", "high school senior")
- leaning: string (which option they lean toward and why)
- financial_concern: string (any financial/family constraints)
- concern: string (biggest worry about making the wrong choice)
- job_market_concern: true/false

INTERESTS (category: "interests"):
- hands_on_work: true/false ("building", "tinkering", "practical" → true)
- enjoys_theory: true/false ("abstract", "theory", "math-heavy" → true)
- enjoys_coding: true/false
- enjoys_building_systems: true/false
- research: true/false

CAREER VISION (category: "career_vision"):
- post_graduation_goal: "job" / "grad_school" / "undecided"
- desired_role_5yr: string (e.g. "software engineer", "hardware engineer", "researcher")
- research_vs_applied: "research" / "applied" / "both"
- industry_preference: string (e.g. "tech", "hardware", "defense", "any")

VALUES (category: "values"):
- financial_security (1-10)
- career_growth (1-10)
- learning (1-10)
- impact (1-10)

MAPPING RULES:
- "I like building things / circuits / hardware" → interests.enjoys_building_systems = true, interests.hands_on_work = true
- "I prefer coding / software" → interests.enjoys_coding = true
- "I like math / theory / abstract problems" → interests.enjoys_theory = true
- "want a job after college" → career_vision.post_graduation_goal = "job"
- "want to do grad school / masters / PhD" → career_vision.post_graduation_goal = "grad_school"
- "software engineer / developer" → career_vision.desired_role_5yr = "software engineer"
- "salary is important / want to earn well" → values.financial_security = 8
- "parents want me to" / "family pressure" / "parents forcing me" → current.financial_concern = "family expectation"
- "freshman / first year / just starting" → current.current_year = "freshman"
- "sophomore / second year" → current.current_year = "sophomore"
- "junior / third year" → current.current_year = "junior"
- "senior / fourth year / final year" → current.current_year = "senior"
- "still deciding / haven't started / not started yet / pre-college / deciding before starting / still in high school / about to start" → current.current_year = "pre-college"
- "just graduated / just finished / recently graduated" → current.current_year = "recent graduate"
- "leaning toward X because Y" → current.leaning = "X because Y"

CRITICAL — desired_role_5yr EXTRACTION:
The user often answers the "5-year vision" question vaguely or indirectly. You MUST extract something.
- "I want to be in arts" → desired_role_5yr = "arts professional"
- "something related to arts like graphic designing" → desired_role_5yr = "graphic designer"
- "I see myself in arts like X" → desired_role_5yr = X
- "I want to be an arts professor" → desired_role_5yr = "arts professor"
- "I don't know / still figuring out" → desired_role_5yr = "undecided" (do NOT leave null)
- "I want a good job in [field]" → desired_role_5yr = "[field] professional"
- ANY mention of a role, even vague → extract the best approximation
- If they mention a specific field (arts, engineering, tech) without a role → use "[field] professional"
- NEVER return null for desired_role_5yr if the user said anything about their future

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"interests": {{}}, "career_vision": {{}}, "values": {{}}, "current": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            elif decision_subtype == "job_vs_business":
                system_prompt = f"""Extract facts from this message about a job-vs-business decision.

Message: "{user_message}"

CURRENT SITUATION (category: "current"):
- current_satisfaction: 1-10 or "happy"/"ok"/"unhappy"/"miserable"
- business_idea: string (description of the business concept)
- business_validated: true/false (has tested with real customers or earned side income)
- financial_runway: string — keep this as a human-readable label ("no savings", "6 months", "2 years")
- leave_reason: string (frustration with job / excitement about business / both / freedom)
- concern: string (biggest fear about making the wrong choice)
- current_monthly_salary: number (monthly, convert if needed: "60k/year"→5000, "5k/month"→5000)

FINANCIAL (category: "financial"):
- financial_runway_months: number — CRITICAL: convert any runway mention to months
  Examples: "2 years" → 24, "1.5 years" → 18, "6 months" → 6, "3 months" → 3,
            "no savings" → 0, "minimal" → 1, "a year maybe" → 12,
            "2 years maybe" → 24, "about a year" → 12, "few months" → 3
- current_income: number (monthly salary/income)
- monthly_expenses: number (monthly expenses if mentioned)

PERSONAL CONTEXT (category: "personal"):
- has_family: true/false
- has_dependents: true/false
- partner_employed: true/false
- can_relocate: true/false

VALUES (category: "values"):
- financial_security (1-10)
- career_growth (1-10)
- work_life_balance (1-10)
- impact (1-10)

MAPPING RULES:
- "I hate my job / bored / no growth" → current_satisfaction = 3, leave_reason = "job frustration"
- "I have a business idea" → business_idea = description
- "I've already got some customers / side income" → business_validated = true
- "no savings / can't afford to quit" → current.financial_runway = "no savings", financial.financial_runway_months = 0
- "6 months runway / savings" → current.financial_runway = "6 months", financial.financial_runway_months = 6
- "2 years savings / runway" → current.financial_runway = "2 years", financial.financial_runway_months = 24
- "I want freedom / be my own boss" → leave_reason = "desire for autonomy"
- "I have a family / kids / mortgage" → has_family = true, has_dependents = true
- "salary / stability matters to me" → values.financial_security = 8
- Monthly salary mentions: "5k", "$5000/month", "earning 5000" → financial.current_income = 5000
- Annual salary: "60k/year", "$60,000" → financial.current_income = 5000

CRITICAL: financial_runway_months MUST be a number. Always convert time expressions to months.

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"current": {{}}, "personal": {{}}, "values": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            else:
                system_prompt = f"""Extract facts from this message for a {decision_type} decision ({' vs '.join(options)}).

Message: "{user_message}"

Extract ANY of these mentioned (don't force all):

INTERESTS (category: "interests"):
- enjoys_coding, enjoys_analysis, enjoys_theory, enjoys_building_systems,
  enjoys_working_with_data, hands_on_work, research
- "hands-on", "practical", "building things" → hands_on_work = true
- "research", "theory", "academia" → research = true/false

CAREER VISION (category: "career_vision"):
- desired_role_5yr (string), research_vs_applied ("research"/"applied"/"both")
- post_graduation_goal: "job" / "phd" / "startup" / "undecided"
- "I want a job" → post_graduation_goal = "job"
- "applied", "real-world" → research_vs_applied = "applied"

VALUES (category: "values"):
- financial_security (1-10), career_growth (1-10), work_life_balance (1-10),
  learning (1-10), impact (1-10)

FINANCIAL (category: "financial"):
- salary_importance (1-10)
- expected_salary: number (annual USD — "100k" → 100000, "six figures" → 100000)
- debt_total: number ("no debt" / "nothing" / "none" / "zero" → 0, "30k" → 30000)
- taking_student_debt: true/false

CURRENT (category: "current"):
- job_market_concern (true/false), current_satisfaction (1-10)
- leaning: string (which option they lean toward)
- concern: string (biggest fear)

MAPPING RULES for this generic extractor:
- "nothing" / "no debt" / "debt-free" as answer to debt question → financial.debt_total = 0
- "$100k" / "100k" / "six figures" as salary target → financial.expected_salary = 100000
- "single" / "no family" / "no partner" → personal.has_dependents = false
- "i can relocate" / "flexible on location" → personal.can_relocate = true
- "hands-on" / "coding" / "technical" → interests.hands_on_work = true, interests.enjoys_coding = true
- "team lead" / "lead engineer" → career_vision.desired_role_5yr = "team lead"
- "collaborat" → values.work_life_balance = 7 (team-oriented person)

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"interests": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

        # FIX: inject last assistant question so extractor understands bare answers
        last_bot_question = ""
        if self.conversation_history:
            for msg in reversed(self.conversation_history):
                if msg["role"] == "assistant":
                    last_bot_question = msg["content"]
                    break
        if last_bot_question:
            system_prompt = f"The assistant just asked the user: \"{last_bot_question}\"\nUse this to interpret bare or ambiguous answers (e.g. a lone number like '7' or 'yes').\n\n" + system_prompt

        response = self._call_gemini(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt
        )

        try:
            start = response.find("{")
            end   = response.rfind("}") + 1
            if start >= 0:
                data = json.loads(response[start:end])
                logging.debug(f"[EXTRACT] {data}")
                return data
        except Exception as e:
            logging.debug(f"[EXTRACT] Parse error: {e} | Response: {response[:200]}")

        return {"extracted": {}, "user_emotional_state": "unknown"}

    # ── Response generation ───────────────────────────────────────────────────
    def generate_response(self, user_message: str, state_dict: Dict, mode: str = "conversational") -> str:
        """
        Pure conversational approach — no question plans, no classification.
        The LLM sees the full conversation history + a system prompt that tells it:
          - What decision is being made
          - What topics to cover across the conversation
          - When it has enough info, wrap up and mention Council of Experts
        The LLM decides which question to ask next naturally.
        """
        self.turn_count += 1
        options   = state_dict.get("decision_metadata", {}).get("options_being_compared", [])
        opt_a     = options[0] if len(options) > 0 else "Option A"
        opt_b     = options[1] if len(options) > 1 else "Option B"
        violations = [v.get("description", "") for v in state_dict.get("violations", [])]

        # Build violation callout if symbolic engine found a constraint issue
        violation_note = ""
        if violations:
            trace = state_dict.get("reasoning_trace", {})
            fired = trace.get("fired_rules", [])
            priority = next((r for r in fired if r.get("severity") == "critical"),
                            next((r for r in fired if r.get("severity") == "warning"), None))
            if priority:
                violation_note = (
                    f"\n\nIMPORTANT: Before asking your next question, briefly flag this "
                    f"contradiction the system detected: \"{priority.get('conclusion', '')}\". "
                    f"One sentence, warm tone, then continue."
                )

        # ── Agent coverage check — conclude when all 3 buckets filled OR turn limit hit ──
        #
        # Each agent needs specific facts to avoid hallucinating.
        # Financial bucket:  salary/tuition/debt/runway
        # Career bucket:     role goal, work style, industry preference
        # Wellbeing bucket:  concern, passion, stress tolerance, city/life preference
        #
        # Conclude when: (all 3 buckets have >= 1 fact AND turns >= 8)
        #             OR turns >= 15 (hard cap)
        # University comparisons also require reputation + cost data.

        financial  = state_dict.get("financial", {})
        current    = state_dict.get("current", {})
        values     = state_dict.get("values", {})
        personal   = state_dict.get("personal", {})
        interests  = state_dict.get("interests", {})
        career_vis = state_dict.get("career_vision", {})
        uni_a_data = state_dict.get("uni_a", {})
        uni_b_data = state_dict.get("uni_b", {})

        def _has(*dicts_and_keys):
            for d, *keys in dicts_and_keys:
                if any(d.get(k) not in (None, False, "", []) for k in keys):
                    return True
            return False

        def _count(*dicts_and_keys):
            total = 0
            for d, *keys in dicts_and_keys:
                total += sum(1 for k in keys if d.get(k) not in (None, False, "", []))
            return total

        # Financial bucket: covered when ≥2 financial facts are known, OR user
        # explicitly said money doesn't matter (financial_security ≤ 2)
        fin_low_priority = (
            values.get("financial_security") is not None and
            isinstance(values.get("financial_security"), (int, float)) and
            values["financial_security"] <= 2
        )
        fin_covered = fin_low_priority or _count(
            (financial, "current_income", "financial_runway_months"),
            (financial, "taking_student_debt"),
            (values,    "financial_security"),
            (uni_a_data,"tuition"), (uni_b_data,"tuition"),
            (current,   "financial_runway"),
            (current,   "current_monthly_salary", "debt_amount", "financial_concern"),
        ) >= 2

        # Career bucket: ≥2 career-domain facts required
        career_covered = _count(
            (career_vis, "desired_role_5yr", "post_graduation_goal"),
            (interests,  "field_of_interest", "hands_on_work", "research"),
            (current,    "current_role", "leave_reason"),
        ) >= 2

        # Wellbeing bucket: ≥2 wellbeing-domain facts required
        wellbeing_covered = _count(
            (current,   "concern"),
            (personal,  "city_preference", "social_connection"),
            (values,    "work_life_balance", "reputation_importance"),
            (current,   "leaning", "business_idea"),
        ) >= 2

        all_covered   = fin_covered and career_covered and wellbeing_covered
        hard_cap      = self.turn_count >= 15
        soft_conclude = all_covered and self.turn_count >= 10  # raised from 8 → 10

        should_conclude = hard_cap or soft_conclude

        covered_count = sum([fin_covered, career_covered, wellbeing_covered])
        logging.debug(f"[TURN {self.turn_count}] Coverage: fin={fin_covered} career={career_covered} wellbeing={wellbeing_covered} ({covered_count}/3) conclude={should_conclude}")

        if should_conclude:
            missing = []
            if not fin_covered:     missing.append("financial situation")
            if not career_covered:  missing.append("career goals")
            if not wellbeing_covered: missing.append("personal preferences")

            known_facts = []
            if current.get("leaning"):       known_facts.append(f"leaning: {current['leaning']}")
            if current.get("concern"):       known_facts.append(f"concern: {current['concern']}")
            if values.get("financial_security"): known_facts.append(f"financial priority: {values['financial_security']}/10")
            if personal.get("has_dependents"): known_facts.append("has dependents")
            facts_summary = (", ".join(known_facts) + ".") if known_facts else ""

            missing_note = ""
            if missing:
                missing_note = (f" Note: limited data on {', '.join(missing)} — "
                                f"agents will flag where analysis is thin.")

            conclusion_prompt = f"""The conversation about "{opt_a} vs {opt_b}" has gathered enough information.

Key facts: {facts_summary}{missing_note}

Write a 3-sentence wrap-up:
1. "Great, I think I have a solid picture of your situation."
2. One sentence naming the central tension or tradeoff this person faces.
3. "The council of expert agents will now analyze this — click the Council of Experts button below."

Rules: Do NOT recommend. Do NOT score. Must end with "Council of Experts"."""

            response = self._call_gemini(
                messages=self.conversation_history + [{"role": "user", "content": user_message}],
                system_prompt=conclusion_prompt,
            )
            if "Council of Experts" not in response:
                response = (response.rstrip() +
                    " Click the **Council of Experts** button below to see the full analysis.")
            # Append a machine-readable signal so app.py can detect conclusion
            # reliably without fragile phrase-matching on natural language output.
            response = response.rstrip() + "\n\n<!-- COUNCIL_READY -->"
            self.conversation_history.append({"role": "user",      "content": user_message})
            self.conversation_history.append({"role": "assistant",  "content": response})
            return response

        # Normal conversational turn
        logging.debug(f"[TURN {self.turn_count}] Collecting info about '{opt_a}' vs '{opt_b}'")

        # Detect subtype FIRST — everything below depends on it
        subtype = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")

        # Map generic option labels to natural language.
        # 'Continue Job' / 'Keep Job' -> 'your job'
        # 'Pursue Passion' / 'Start Business' -> 'your passion' / 'your business idea'
        GENERIC_JOB_LABELS = {
            "continue job", "keep job", "stay at job", "current job",
            "job", "stay", "my job", "the job",
        }
        GENERIC_PASSION_LABELS = {
            "pursue passion", "follow passion", "passion", "my passion",
            "start business", "start a business", "entrepreneurship",
            "quit job", "leave job", "the business", "new career",
        }
        def natural(label):
            ll = label.lower().strip()
            if ll in GENERIC_JOB_LABELS:     return "your current job"
            if ll in GENERIC_PASSION_LABELS: return "your passion"
            return label

        # Detect if this is semantically a job-vs-passion decision even if subtype is wrong
        is_jvb = (
            subtype == "job_vs_business" or
            (opt_a.lower() in GENERIC_JOB_LABELS | GENERIC_PASSION_LABELS or
             opt_b.lower() in GENERIC_JOB_LABELS | GENERIC_PASSION_LABELS)
        )
        if is_jvb:
            label_a = natural(opt_a)
            label_b = natural(opt_b)
            subtype = "job_vs_business"   # normalise subtype for topic selection
        else:
            label_a = opt_a
            label_b = opt_b


        if subtype == "university_comparison":
            topics = f"""
DECISION TYPE: University comparison — {opt_a} vs {opt_b}.
The user is choosing between these two universities. Cover ALL of these topics — one per turn, in roughly this order:

FINANCIAL (ask early — but PIVOT IMMEDIATELY if user says money doesn't matter):
  1. In-state or out-of-state for each school? (one question, not two)
  2. Any scholarships or financial aid offered?
  3. Expected salary target after graduating?
  4. Rough living cost awareness — but ONLY if they seem financially aware. If they say "I don't know" or "doesn't matter", skip to ACADEMIC section immediately.

IMPORTANT: If user says "I don't care about money" or gives a very low financial priority, stop ALL financial questions and move to ACADEMIC section right away.

ACADEMIC & CAREER:
  5. Major / specialization — what specific area within their field? (ask early if not known)
  6. Program reputation — which school is known for their specific field? Does name matter?
  7. Research vs coursework — do they want to do research or just complete coursework?
  8. Job market after graduation — do they want to work in the same city or relocate?

PERSONAL:
  9. Social / support system — anyone near either campus? Friends, family?
  10. City preference — lifestyle fit (big city, small town, weather, campus environment)
  11. Timeline — do they need to finish quickly or is a longer program okay?
  12. Biggest concern — what would make this the wrong choice?

Do NOT ask all at once. One question per turn. Start with financial questions since those have the most impact on the analysis.
Do NOT ask about major if they already told you."""
        elif subtype == "offer_comparison":
            topics = f"""
DECISION TYPE: Job offer comparison — {opt_a} vs {opt_b}.
Cover these topics naturally:
  • Role and day-to-day work at each — what would they actually be doing?
  • Salary and total compensation at each
  • Relocation — does either require moving? Is that feasible?
  • Growth potential at each company
  • Team and culture fit
  • Work-life balance expectations
  • Family / personal constraints
  • Their biggest concern about each offer"""
        elif subtype == "job_vs_business":
            topics = (
                "DECISION TYPE: Job vs passion/business.\n"
                "Cover these topics naturally, one per turn:\n"
                "- What IS the passion/business idea -- what would they actually do? (ask early)\n"
                "- Have they tested it -- customers, side income, any validation?\n"
                "- Current job satisfaction -- fleeing frustration or chasing something real?\n"
                "- Current salary and monthly expenses\n"
                "- Savings/financial runway -- how many months without income?\n"
                "- Family or dependents who rely on their income?\n"
                "- Partner employed? Second income in the household?\n"
                "- Worst-case scenario they can live with?\n"
                "- Biggest fear about making the leap"
            )
        elif subtype == "education_path":
            topics = (
                f"DECISION TYPE: Education path -- {opt_a} vs {opt_b}.\n"
                "Cover these topics naturally:\n"
                "- Current situation -- what is pushing toward further study?\n"
                "- Financial situation -- can they afford reduced income during study?\n"
                "- What specific role does this path lead to?\n"
                "- Do they genuinely enjoy research, or just want the credential?\n"
                "- Family obligations -- partner, dependents, location constraints\n"
                "- What draws them to each option specifically\n"
                "- Biggest concern about the wrong choice"
            )
        elif subtype == "major_choice":
            topics = f"""
DECISION TYPE: Academic field / major choice — {opt_a} vs {opt_b}.
The user is choosing between two fields of study. These questions separate a good analysis from a generic one.

Ask these in order — one per turn:

INTERESTS (most important — ask early):
  1. What specifically excites them about {opt_a}? Push past "I like it" to concrete activities.
  2. What specifically excites them about {opt_b}? Same — concrete.
  3. Day-to-day work preference: do they prefer building/coding, analysing data/patterns, doing research, or managing systems? Be specific to {opt_a} vs {opt_b}.
  4. Math and statistics comfort — relevant for differentiating these fields.

FINANCIAL:
  5. How important is salary — ask for a 1-10 rating. This drives agent weighting.
  6. Do they have debt or financial pressure that makes earning potential critical?

CAREER VISION:
  7. What specific job title or role in 3-5 years? "Team lead" or "data scientist" or "software engineer" — get concrete.
  8. Industry preference — tech company, finance, healthcare, research, startup?

PERSONAL:
  9. Do they have a natural lean toward one field, and if so why?
  10. Biggest concern about choosing the wrong one — missing out on opportunities, ending up bored, salary risk?

CRITICAL RULES for this decision type:
- Do NOT ask about relocation or family early — those matter less for a field choice.
- DO ask about what kind of work energizes them — this is the #1 differentiator.
- Reference {opt_a} and {opt_b} by name in every question."""

        else:
            topics = f"""
Cover these topics naturally across the conversation:
  • Financial situation — salary importance (ask 1-10), debt, savings pressures
  • Career goals — role, level, kind of work in 3-5 years
  • Work style — what kind of daily work energizes them day-to-day
  • What specifically draws them toward {opt_a} vs {opt_b}
  • Personal constraints — family, location, partner
  • Risk tolerance — stability vs upside
  • Their biggest concern about the wrong choice"""

        system_prompt = f"""You are a sharp, empathetic decision coach. You think like a trusted friend who happens to be brilliant at untangling complex choices.

Character: Warm but direct. You notice what people *avoid* saying as much as what they say. You catch contradictions and name them — with curiosity, not judgment. You don't celebrate answers, but you show you genuinely processed what was said. You ask questions that make people realize something, not just extract data.

You are helping someone decide between {label_a} and {label_b}.
{topics}

Turn {self.turn_count} of max 15. Stop collecting when you have financial + career + personal data.
{"EARLY — Understand the texture of the situation first. What's really at stake for them?" if self.turn_count <= 4 else ""}
{"MID — Go deeper. Ask about fear, regret, and what they're avoiding. Surface what they haven't said." if 5 <= self.turn_count <= 9 else ""}
{"NEAR END — Focus on the one or two things that are still genuinely unclear. Quality over completeness." if self.turn_count >= 10 else ""}

HOW TO ASK QUESTIONS:
- Lead with 1-2 sentences showing you actually absorbed what they said — a real observation, not a summary recap.
- Then ONE specific, direct question. Use the options by name, not "option A".
- When they give a number, anchor to it: "You said salary is an 8/10 — what specific number would actually make you feel safe?"
- When they contradict themselves, surface it gently: "You said stability matters most, but you're leaning toward {label_b}. What's pulling you there despite that?"
- When they're vague ("it's fine", "I don't know"): try once with a more concrete angle. If they dodge again, move on without labeling the dodge.
- Occasionally ask about the downside: "What would make this the wrong choice in two years?" — fear-based questions reveal more than preference questions.
- If they seem stressed or overwhelmed, briefly acknowledge it before moving on.

HARD RULES:
1. ONE question per turn. Never two.
2. Maximum 35 words total across your entire response. Short, direct, conversational — like a text message from a smart friend, not a paragraph from a therapist.
3. No sycophancy. "Great answer!", "That's helpful!", "Interesting!" — all banned.
4. Never tell them to go research something and come back.
5. Never volunteer information about the options (no rankings, salary stats, reputation claims).
6. Never give advice — the council handles that.
7. Use "{label_a}" and "{label_b}" by name — never "option A" or "option B".
8. Don't ask two related questions in the same turn by hiding one as a clarification.{violation_note}"""

        messages = self.conversation_history + [{"role": "user", "content": user_message}]
        response = self._call_gemini(messages=messages, system_prompt=system_prompt)
        self.conversation_history.append({"role": "user",      "content": user_message})
        self.conversation_history.append({"role": "assistant",  "content": response})
        return response



    def _build_symbolic_constraints_block(self, state_dict: Dict) -> str:
        """
        Build the SYMBOLIC ENGINE CONSTRAINTS section from the reasoning trace.
        Injected into every agent and synthesizer prompt so the LLM layer
        must respect deterministic rule outputs, not reason around them.
        """
        trace      = state_dict.get("reasoning_trace", {})
        fired      = trace.get("fired_rules", [])
        mode_info  = trace.get("decision_mode", {})
        rule_counts = trace.get("rule_counts", {})

        if not fired and not mode_info:
            return ""

        mode_str  = mode_info.get("mode", "UNKNOWN")
        mode_rule = mode_info.get("rule_id", "")
        mode_expl = mode_info.get("explanation", "")
        n_total   = rule_counts.get("total_registered", "?")
        n_fired   = rule_counts.get("fired", len(fired))

        lines = [
            "",
            "=" * 60,
            "SYMBOLIC ENGINE CONSTRAINTS",
            f"(Deterministic rule evaluation: {n_fired}/{n_total} rules fired)",
            "=" * 60,
            f"Decision Mode: {mode_str}  [{mode_rule}]",
            f"Mode reason:   {mode_expl}",
            "",
        ]

        if not fired:
            lines.append("No constraint violations detected.")
        else:
            critical = [r for r in fired if r.get("severity") == "critical"]
            warnings = [r for r in fired if r.get("severity") == "warning"]
            info     = [r for r in fired if r.get("severity") == "info"]

            if critical:
                lines.append("CRITICAL VIOLATIONS — hard structural incompatibilities.")
                lines.append("  Your analysis MUST acknowledge these.")
                for r in critical:
                    lines.append(f"  [{r['rule_id']}] {r['rule_name']}")
                    lines.append(f"  -> {r['conclusion']}")
                    facts_str = ", ".join(
                        f"{k.split('.')[-1]}={v}"
                        for k, v in r.get("triggering_facts", {}).items()
                    )
                    if facts_str:
                        lines.append(f"  Triggering facts: {facts_str}")
                    lines.append("")

            if warnings:
                lines.append("WARNINGS — tensions to weigh:")
                for r in warnings:
                    lines.append(f"  [{r['rule_id']}] {r['rule_name']}")
                    lines.append(f"  -> {r['conclusion']}")
                    lines.append("")

            if info:
                lines.append("INFO:")
                for r in info:
                    lines.append(f"  [{r['rule_id']}] {r['conclusion']}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Profile builder ───────────────────────────────────────────────────────────

    def _build_profile(self, state_dict: Dict) -> str:
        """
        Build the user profile section for LLM prompts.
        Appends the symbolic constraints block so agents see both facts
        and deterministic rule outputs in every prompt.
        """
        interests  = state_dict.get("interests", {})
        career_vis = state_dict.get("career_vision", {})
        strengths  = state_dict.get("strengths", {})
        values     = state_dict.get("values", {})
        financial  = state_dict.get("financial", {})
        current    = state_dict.get("current", {})
        offer_a    = state_dict.get("offer_a", {})
        offer_b    = state_dict.get("offer_b", {})
        uni_a      = state_dict.get("uni_a", {})
        uni_b      = state_dict.get("uni_b", {})
        personal   = state_dict.get("personal", {})
        options    = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Option A", "Option B"])
        subtype    = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")

        opt_a = options[0] if options else "Option A"
        opt_b = options[1] if len(options) > 1 else "Option B"

        if subtype == "offer_comparison":
            facts_section = f"""User is comparing two job offers:

{opt_a}:
{json.dumps({k:v for k,v in offer_a.items() if v is not None and v != ""}, indent=2)}

{opt_b}:
{json.dumps({k:v for k,v in offer_b.items() if v is not None and v != ""}, indent=2)}

Personal / life context:
{json.dumps({k:v for k,v in personal.items() if v is not None and v != ""}, indent=2)}

Their values and priorities:
{json.dumps({k:v for k,v in values.items() if v is not None}, indent=2)}"""

        elif subtype == "university_comparison":
            college_note = ""
            if hasattr(self, "_college") and self._college:
                try:
                    cards = self._college.get_cards_for_decision(opt_a, opt_b)
                    ca, cb = cards.get("option_a"), cards.get("option_b")
                    if ca or cb:
                        college_note = "\n" + self._college.format_comparison_block(
                            ca, cb, opt_a, opt_b
                        )
                except Exception:
                    pass

            facts_section = f"""User is comparing two universities:

{opt_a}:
{json.dumps({k:v for k,v in uni_a.items() if v is not None and v != ""}, indent=2)}

{opt_b}:
{json.dumps({k:v for k,v in uni_b.items() if v is not None and v != ""}, indent=2)}

Personal / life context:
{json.dumps({k:v for k,v in personal.items() if v is not None and v != ""}, indent=2)}

Their values and priorities:
{json.dumps({k:v for k,v in values.items() if v is not None}, indent=2)}{college_note}"""

        elif subtype == "major_choice":
            facts_section = f"""User is choosing between {opt_a} and {opt_b} for their degree.

Current situation:
{json.dumps({k:v for k,v in current.items() if v is not None and v != ""}, indent=2)}

Interests and work style:
{json.dumps({k:v for k,v in interests.items() if v is not None and v is not False}, indent=2)}

Career vision:
{json.dumps({k:v for k,v in career_vis.items() if v is not None and v != ""}, indent=2)}

Values and priorities:
{json.dumps({k:v for k,v in values.items() if v is not None}, indent=2)}"""

        elif subtype == "job_vs_business":
            facts_section = f"""User is deciding whether to leave their job and start a business.

Current situation:
{json.dumps({k:v for k,v in current.items() if v is not None and v != ""}, indent=2)}

Personal / family context:
{json.dumps({k:v for k,v in personal.items() if v is not None and v != ""}, indent=2)}

Values and priorities:
{json.dumps({k:v for k,v in values.items() if v is not None}, indent=2)}"""

        else:
            facts_section = f"""User profile (facts from conversation):
- Interests:      {json.dumps({k:v for k,v in interests.items() if v is not None and v is not False})}
- Values:         {json.dumps({k:v for k,v in values.items() if v is not None})}
- Career vision:  {json.dumps({k:v for k,v in career_vis.items() if v is not None and v != ""})}
- Financial:      {json.dumps({k:v for k,v in financial.items() if v is not None})}
- Strengths:      {json.dumps({k:v for k,v in strengths.items() if v is not None})}
- Current:        {json.dumps({k:v for k,v in current.items() if v is not None})}"""

        symbolic_block = self._build_symbolic_constraints_block(state_dict)
        return facts_section + symbolic_block

    def _build_agent_profile(self, state_dict: Dict, allowed_categories: list, bls_block: str) -> str:
        """
        Build a fact profile restricted to the categories an agent is allowed to see.
        Prevents agents from cross-contaminating their analysis with out-of-lens data.
        Each agent gets only the facts in its facts_filter — not the full user profile.
        """
        options = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Option A", "Option B"])
        opt_a   = options[0] if options else "Option A"
        opt_b   = options[1] if len(options) > 1 else "Option B"

        lines = [f"Decision: {opt_a} vs {opt_b}"]

        for cat in allowed_categories:
            data = state_dict.get(cat, {})
            if not data:
                continue
            filtered = {
                k: v for k, v in data.items()
                if v is not None and v is not False and v != "" and v != []
            }
            if filtered:
                lines.append(f"\n{cat}:\n" + json.dumps(filtered, indent=2))

        # Inject BLS/College data only for agents whose lens covers career/offer categories
        career_cats = {"career_vision", "interests", "offer_a", "offer_b", "uni_a", "uni_b"}
        if bls_block and any(c in career_cats for c in allowed_categories):
            lines.append(f"\nExternal market data:\n{bls_block}")

        # Symbolic constraints always included — they are structural facts, not opinion
        symbolic = self._build_symbolic_constraints_block(state_dict)
        if symbolic:
            lines.append(symbolic)

        return "\n".join(lines) if len(lines) > 1 else "No facts collected yet for this lens."

    # ── Council debate (3 fixed agents + debate round + synthesizer) ──────────
    def generate_council_perspectives(
        self, state_dict: Dict, memory_context: str = ""
    ) -> Dict:
        """
        Run 3 independent expert analyses + optional debate round + synthesizer.

        The two most-disagreeing agents (gap >= 15%) exchange rebuttals.
        The synthesizer reads all outputs and produces a ruling with
        confidence score + open question.

        memory_context : optional block from DecisionMemory.get_context_block()
        """
        options  = state_dict.get("decision_metadata", {}).get("options_being_compared", [])
        opt_a    = options[0] if len(options) > 0 else "Option A"
        opt_b    = options[1] if len(options) > 1 else "Option B"

        # Count meaningful facts — warn agents if data is thin
        total_facts = sum(
            1 for cat in ["values","interests","career_vision","current","personal","financial",
                          "offer_a","offer_b","uni_a","uni_b"]
            for v in state_dict.get(cat, {}).values()
            if v is not None and v is not False and v != "" and v != []
        )
        thin_data_warning = ""
        if total_facts < 5:
            thin_data_warning = (
                f"\n\nDATA WARNING: Only {total_facts} meaningful facts were collected "
                f"from the conversation. The analysis below is based on limited information. "
                f"Do NOT invent facts, assume demographics, or hallucinate specifics about "
                f"{opt_a} or {opt_b}. If you don't have the data to evaluate something, say so."
            )
        logging.info(f"[COUNCIL] Facts collected: {total_facts}")

        profile  = self._build_profile(state_dict)

        # Data lookup — BLS for career/job, College Scorecard for university comparisons
        bls_block    = ""
        options_list = state_dict.get("decision_metadata", {}).get("options_being_compared", [])
        subtype_str  = state_dict.get("decision_metadata", {}).get("decision_subtype", "")

        if subtype_str == "university_comparison" and hasattr(self, "_college") and self._college and len(options_list) >= 2:
            try:
                qa, qb = options_list[0], options_list[1]
                cards  = self._college.get_cards_for_decision(qa, qb)
                ca, cb = cards.get("option_a"), cards.get("option_b")
                if ca or cb:
                    bls_block = self._college.format_comparison_block(ca, cb, qa, qb)
                    logging.info(f"[COLLEGE] Injected scorecard data for {qa} vs {qb}")
            except Exception as e:
                logging.info(f"[COLLEGE] Lookup failed: {e}")

        elif self._bls and len(options_list) >= 2:
            try:
                qa, qb = options_list[0], options_list[1]
                if subtype_str == "offer_comparison":
                    qa = (state_dict.get("offer_a", {}).get("role") or qa).strip()
                    qb = (state_dict.get("offer_b", {}).get("role") or qb).strip()
                cards = self._bls.get_cards_for_decision(qa, qb)
                ca, cb = cards.get("option_a"), cards.get("option_b")
                if ca or cb:
                    bls_block = self._bls.format_comparison_block(ca, cb, qa, qb)
            except Exception as e:
                logging.info(f"[BLS] Lookup failed: {e}")

        results  = {}

        # ── Run 3 independent agent analyses — each sees only its own facts ───────
        agent_votes = {}
        user_lean   = state_dict.get("current", {}).get("leaning") or ""

        # Build compact critical-violation note for agents
        trace = state_dict.get("reasoning_trace", {})
        crit  = [r for r in trace.get("fired_rules", []) if r.get("severity") == "critical"]
        viol_note = (" CRITICAL CONSTRAINT: " + "; ".join(r["conclusion"] for r in crit)) if crit else ""

        for agent in self.AGENTS:
            # Each agent gets only the facts in its facts_filter — structural isolation
            agent_profile = self._build_agent_profile(
                state_dict, agent["facts_filter"], bls_block
            )

            prompt = (
                f"{agent['persona']}\n\n"
                f"=== FACTS YOU ARE ALLOWED TO USE ===\n"
                f"{agent_profile}\n"
                f"=====================================\n\n"
                f"Decision: {opt_a} vs {opt_b}\n"
                f"{'User stated lean: ' + str(user_lean) + chr(10) if user_lean else ''}"
                f"{viol_note}\n"
                f"{thin_data_warning}\n\n"
                f"You MUST address: {agent['must_address']}\n\n"
                f"STRICT RULES:\n"
                f"- You only reason from the facts in the block above. Nothing else exists.\n"
                f"- Do NOT acknowledge facts outside your domain even if you think you know them.\n"
                f"- Do NOT hedge. Pick a direction. Argue it hard.\n"
                f"- 3 sentences max in ANALYSIS. Every sentence must cite a specific fact from above.\n"
                f"- If you have no facts for a point, say 'insufficient data' — do not invent.\n\n"
                f"Respond EXACTLY in this format:\n"
                f"LEAN: {opt_a}: [X]% | {opt_b}: [Y]%\n"
                f"ANALYSIS: [3 sharp sentences — your domain only, cite specific facts or BLS numbers]\n"
                f"KEY INSIGHT: [1 sentence that is genuinely memorable and specific to this person]\n\n"
                f"X+Y must equal 100. No hedging. Argue your lens to its logical conclusion."
            )
            raw = self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=550,
            )
            results[f"agent_{agent['id']}"] = raw

            # Parse lean percentages — robust: tolerates spaces before %, normalizes to 100
            import re as _re
            vote_a, vote_b = 50, 50
            try:
                for line in raw.split("\n"):
                    if line.strip().upper().startswith("LEAN:"):
                        segs = line.split("|")
                        if len(segs) == 2:
                            ma = _re.search(r'(\d+)\s*%', segs[0])
                            mb = _re.search(r'(\d+)\s*%', segs[1])
                            if ma and mb:
                                vote_a = int(ma.group(1))
                                vote_b = int(mb.group(1))
                                # Always normalize so votes sum to exactly 100
                                total = vote_a + vote_b
                                if total > 0 and total != 100:
                                    vote_a = round(vote_a * 100 / total)
                                    vote_b = 100 - vote_a
                                break
            except Exception as e:
                logging.warning(f"[VOTE PARSE] {agent['id']}: {e}")

            agent_votes[agent["id"]] = {"option_a": vote_a, "option_b": vote_b, "raw": raw}

        results["agent_votes"] = agent_votes

        # ── Aggregate vote ────────────────────────────────────────────────────
        avg_a = sum(v["option_a"] for v in agent_votes.values()) / len(self.AGENTS)
        avg_b = sum(v["option_b"] for v in agent_votes.values()) / len(self.AGENTS)
        # Normalize aggregate to exactly 100
        _total = avg_a + avg_b
        if _total > 0 and abs(_total - 100) > 0.5:
            avg_a = round(avg_a * 100 / _total)
            avg_b = 100 - avg_a
        else:
            avg_a, avg_b = round(avg_a), round(avg_b)
        results["tally_after_r1"] = {"option_a": avg_a, "option_b": avg_b}
        logging.info(f"[TALLY] {opt_a}={avg_a}% {opt_b}={avg_b}%")

        # ── Debate Round — two most-disagreeing agents exchange rebuttals ─────
        # Find the pair of agents with the largest vote gap on option_a
        agent_id_list = list(agent_votes.keys())
        max_delta  = 0
        debater_a_id = agent_id_list[0]
        debater_b_id = agent_id_list[1] if len(agent_id_list) > 1 else agent_id_list[0]

        for i in range(len(agent_id_list)):
            for j in range(i + 1, len(agent_id_list)):
                delta = abs(
                    agent_votes[agent_id_list[i]]["option_a"]
                    - agent_votes[agent_id_list[j]]["option_a"]
                )
                if delta > max_delta:
                    max_delta  = delta
                    debater_a_id = agent_id_list[i]
                    debater_b_id = agent_id_list[j]

        debater_a = next((ag for ag in self.AGENTS if ag["id"] == debater_a_id), None)
        debater_b = next((ag for ag in self.AGENTS if ag["id"] == debater_b_id), None)

        rebuttal_a_raw = ""
        rebuttal_b_raw = ""

        if debater_a and debater_b and max_delta >= 15:
            logging.info(f"[DEBATE] {debater_a_id} vs {debater_b_id} — gap={max_delta}%")

            # Agent A reads Agent B's analysis and rebuts it from its own lens
            rebuttal_a_prompt = (
                f"{debater_a['persona']}\n\n"
                f"You just reviewed the analysis from the {debater_b['name']}:\n"
                f"---\n{agent_votes[debater_b_id]['raw']}\n---\n\n"
                f"You strongly disagree with their conclusion. You are not going to change your mind.\n"
                f"Write a sharp, specific 2-sentence rebuttal using only facts from YOUR lens "
                f"({debater_a['lens']}).\n"
                f"Do not repeat their argument. Attack the weakest point in their reasoning.\n"
                f"CRITICAL: Double-check every number you cite. Never invert a comparison (e.g. do not say X > Y if Y > X).\n\n"
                f"Format EXACTLY:\n"
                f"REBUTTAL: [your 2-sentence counter-argument, citing a specific fact or number]\n"
                f"STAND: {opt_a}: [X]% | {opt_b}: [Y]%  (X+Y=100, no hedging)"
            )
            rebuttal_a_raw = self._call_llm(
                messages=[{"role": "user", "content": rebuttal_a_prompt}],
                max_tokens=250,
            )

            # Agent B reads Agent A's analysis and rebuts it
            rebuttal_b_prompt = (
                f"{debater_b['persona']}\n\n"
                f"You just reviewed the analysis from the {debater_a['name']}:\n"
                f"---\n{agent_votes[debater_a_id]['raw']}\n---\n\n"
                f"You strongly disagree with their conclusion. You are not going to change your mind.\n"
                f"Write a sharp, specific 2-sentence rebuttal using only facts from YOUR lens "
                f"({debater_b['lens']}).\n"
                f"Do not repeat their argument. Attack the weakest point in their reasoning.\n"
                f"CRITICAL: Double-check every number you cite. Never invert a comparison (e.g. do not say X > Y if Y > X).\n\n"
                f"Format EXACTLY:\n"
                f"REBUTTAL: [your 2-sentence counter-argument, citing a specific fact or number]\n"
                f"STAND: {opt_a}: [X]% | {opt_b}: [Y]%  (X+Y=100, no hedging)"
            )
            rebuttal_b_raw = self._call_llm(
                messages=[{"role": "user", "content": rebuttal_b_prompt}],
                max_tokens=250,
            )
        else:
            logging.info(f"[DEBATE] Skipped — max gap={max_delta}% (threshold=15%)")

        results["round2_a"]        = rebuttal_a_raw
        results["round2_b"]        = rebuttal_b_raw
        results["debating_agents"] = {
            "a": debater_a or {},
            "b": debater_b or {},
        }
        results["has_round3"]      = bool(rebuttal_a_raw or rebuttal_b_raw)
        results["debate_gap"]      = max_delta

        # ── Synthesizer ───────────────────────────────────────────────────────
        # Build block of all outputs: 3 initial analyses + up to 2 rebuttals
        analyses_parts = []
        for ag in self.AGENTS:
            key = f"agent_{ag['id']}"
            analyses_parts.append(f"ROUND 1 — {ag['name']} ({ag['emoji']}):\n{results.get(key, 'N/A')}")

        if rebuttal_a_raw and debater_a:
            analyses_parts.append(
                f"DEBATE REBUTTAL — {debater_a['name']} ({debater_a['emoji']}) "
                f"responding to {debater_b['name']}:\n{rebuttal_a_raw}"
            )
        if rebuttal_b_raw and debater_b:
            analyses_parts.append(
                f"DEBATE REBUTTAL — {debater_b['name']} ({debater_b['emoji']}) "
                f"responding to {debater_a['name']}:\n{rebuttal_b_raw}"
            )

        analyses_block = "\n\n".join(analyses_parts)

        trace        = state_dict.get("reasoning_trace", {})
        fired_rules  = trace.get("fired_rules", [])
        mode_info    = trace.get("decision_mode", {})
        mode_str     = mode_info.get("mode", "UNKNOWN")
        critical_txt = ""
        if fired_rules:
            critical = [r for r in fired_rules if r.get("severity") == "critical"]
            if critical:
                critical_txt = "CRITICAL CONSTRAINTS (MUST acknowledge in ruling):\n"
                for r in critical:
                    critical_txt += f"  [{r['rule_id']}] {r['conclusion']}\n"

        lean_instruction = ""
        if user_lean:
            lean_instruction = (
                f"\nIMPORTANT: Person explicitly leans toward: \"{user_lean}\"\n"
                "Mental Wellbeing Agent gives this significant weight. Ruling must address "
                "whether data supports or conflicts with this stated preference."
            )

        debate_note = (
            f"\nNote: A debate round occurred between {debater_a['name']} and {debater_b['name']} "
            f"(gap={max_delta}%). Their rebuttals are included above. Your confidence score "
            f"should reflect whether the debate resolved or deepened the disagreement.\n"
        ) if results["has_round3"] else ""

        memory_block = (f"\n{memory_context}\n" if memory_context else "")

        synth_prompt = (
            f"{profile}\n\n"
            f"{bls_block}\n\n"
            f"{memory_block}"
            f"Expert agents analyzed {opt_a} vs {opt_b}:{thin_data_warning}\n\n"
            f"{analyses_block}\n\n"
            f"Aggregate lean after Round 1: {opt_a}: {round(avg_a)}% | {opt_b}: {round(avg_b)}%\n"
            f"Decision Mode: {mode_str}\n"
            f"{critical_txt}"
            f"{lean_instruction}"
            f"{debate_note}\n\n"
            "You are the SYNTHESIZER. Your job is to be the clearest voice in the room — not the loudest, but the most honest.\n\n"
            "Format EXACTLY:\n"
            "STRONGEST ANALYSIS: [which agent gave the most grounded, specific argument — and why in one sentence]\n"
            "WEAKEST ANALYSIS: [which agent relied on generics or made assumptions — and why in one sentence]\n"
            f"VOTE TALLY: {opt_a} {round(avg_a)}% | {opt_b} {round(avg_b)}%\n"
            "RULING: [2-3 sentences — what the full picture leans toward; must cite at least one specific fact or number from the profile]\n"
            "BOTTOM LINE: [1 plain-English sentence starting with 'If this person were a close friend, I'd tell them...' — direct, no hedging]\n"
            "WATCH OUT FOR: [1 sentence — the single most likely reason this choice could still go wrong or be regretted]\n"
            "CONFIDENCE: [HIGH / MEDIUM / LOW] — [1 sentence: why, based on data completeness + degree of agent agreement]\n"
            "OPEN QUESTION: [the one unresolved question whose answer could genuinely flip the recommendation]\n\n"
            "HARD RULES:\n"
            "- RULING uses 'the data suggests...' or 'the profile points toward...'\n"
            "- BOTTOM LINE starts with 'If this person were a close friend, I'd tell them...'\n"
            "- RULING must cite at least one number or named constraint from the facts\n"
            "- CONFIDENCE is HIGH only if agents broadly agree AND key facts are present\n"
            "- OPEN QUESTION must be specific to this person's situation — not a generic 'have you considered?'\n"
            "- If external data and agent votes point opposite directions, flag this clearly in RULING\n"
            "- WATCH OUT FOR must be the *most likely* failure mode for the recommended path, not a disclaimer"
        )

        results["synthesizer"]  = self._call_llm(
            messages=[{"role": "user", "content": synth_prompt}],
            max_tokens=750,
        )
        results["options"]  = [opt_a, opt_b]
        results["avg_vote"] = {"option_a": round(avg_a), "option_b": round(avg_b)}
        results["agents"]   = self.AGENTS
        return results


    def reset_conversation(self):
        self.conversation_history = []
        self.turn_count = 0
        self._current_model_idx = 0   # reset to primary model for new conversation

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribe audio bytes using Groq's Whisper endpoint.
        Returns the transcribed text, or an empty string on failure.
        """
        import io
        try:
            # Groq supports OpenAI-compatible audio transcription
            client = self._clients[0]
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename
            result = client.audio.transcriptions.create(
                file=(filename, audio_bytes, "audio/wav"),
                model="whisper-large-v3",
                response_format="text",
            )
            # result is a plain string in text mode
            text = result.strip() if isinstance(result, str) else (result.text or "").strip()
            logging.info(f"[WHISPER] Transcribed: {text[:80]}")
            return text
        except Exception as e:
            logging.warning(f"[WHISPER] Transcription failed: {e}")
            return ""
