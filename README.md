# 🧠 Neuro-Symbolic Agentic Decision Support System

<p align="center">
  <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LLM-Llama%203.3%2070B-F54E00?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/Architecture-Neuro--Symbolic-764BA2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Data-BLS%20%7C%20College%20Scorecard-0057A8?style=for-the-badge&logo=gov&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
</p>

<p align="center">
  <strong>🚀 <a href="https://neurosymbolic-ai.streamlit.app/">Try it live → neurosymbolic-ai.streamlit.app</a></strong>
</p>

---

An AI decision support system that combines a **deterministic symbolic constraint engine** with a **neural LLM backend** — grounded in real-world labor market and university data. This is not a chatbot that generates generic advice. It tracks your constraints, flags logical contradictions in your reasoning, retrieves actual salary and career outlook figures from the U.S. Bureau of Labor Statistics, fetches real tuition and earnings data from the U.S. Department of Education, and runs a structured multi-agent debate to surface the genuine tension in your decision.

Built as a Master's capstone at **Texas A&M University – Corpus Christi** (Dept. of Computer Science), accompanied by an ACM-format research paper.

---

## What Makes This Different From a Chatbot

| Layer | Technology | What It Does |
|---|---|---|
| **Symbolic Engine** | Pure Python, deterministic | Stores every fact the user states, detects logical contradictions, computes decision mode (`SURVIVAL` / `CAUTIOUS` / `GROWTH`) |
| **Neural Layer** | Llama 3.3 70B via Groq | Natural language understanding, structured fact extraction (JSON), conversational questioning, agent debate |
| **BLS Retriever** | 342-occupation JSONL corpus (bls.gov) | Retrieves median salary, 10-year job growth %, and education requirements for any career query — injected into agent prompts as hard facts |
| **College Scorecard** | U.S. Dept. of Education live API | Fetches real tuition (in/out-of-state), net price after aid, graduation rate, median earnings 10 years post-enrollment, and acceptance rate for any university |

**The LLM cannot override symbolic facts.** Every agent prompt is grounded in the current symbolic state. Agents are explicitly instructed to cite exact BLS numbers — `$131,450 vs $112,590`, not `"higher salary"`. If the symbolic engine detects a contradiction (e.g., financial stability rated 9/10 while planning to quit with no savings), it is surfaced — the LLM does not get to silently ignore it.

---

## How It Works

1. User describes their decision in plain English — no form to fill out
2. System auto-detects the decision type and collects facts through a focused conversational interview, only asking what it doesn't already know
3. A live **Symbolic State** builds in the sidebar — every fact, constraint, and value tracked and auditable in real time
4. **BLS or College Scorecard data is retrieved automatically** based on the decision type and injected into agent prompts before the debate begins
5. The **Council of Experts** runs: three specialized AI agents debate from distinct lenses (Financial Security, Career Growth, Mental Wellbeing), each forbidden from straying outside their domain
6. A **Synthesizer** reads the full debate, produces a weighted vote tally, and issues a ruling — required to cite at least one specific number or named constraint. If external data contradicts agent votes, it must flag the conflict explicitly
7. The Synthesizer closes with one **OPEN QUESTION** — a genuine unresolved dilemma the user still has to answer themselves
8. A **dynamic decision tree** renders every collected fact, color-coded by impact, mapped to the agent that owns it

---

## Decisions It Handles

| Type | Real Data Source | Example |
|---|---|---|
| Career / field comparison | BLS OOH — salary, outlook, entry education | "Computer Science or Data Science?" |
| Job offer comparison | BLS OOH — matched to job role | "Google vs a startup offer" |
| PhD vs. industry | BLS OOH — research scientist vs engineer | "PhD or stay in my current role?" |
| University comparison | College Scorecard — tuition, earnings, grad rate | "MIT vs CMU for my Master's" |
| Job vs. starting a business | Symbolic engine — financial runway, risk | "Should I quit and go all-in?" |

---

## Real Data Integration

### BLS Occupational Outlook Handbook (`bls_retriever.py`)
- **342 occupations** loaded from a structured JSONL corpus sourced from bls.gov/ooh
- Fuzzy name matching with a synonym layer — `"cs"` resolves to `software developers`, `"ml engineer"` resolves to `computer and information research scientists`
- Returns median pay, 10-year projected job growth (%), total jobs, entry education, and a role summary
- Comparison blocks are formatted and injected directly into every agent's prompt before analysis begins

### College Scorecard (`college_retriever.py`)
- Live API calls to the **U.S. Department of Education College Scorecard** — no stale local data
- Session-level cache so each school is only fetched once
- Returns in-state tuition, out-of-state tuition, average net price after financial aid, graduation rate, median earnings 10 years after enrollment, acceptance rate, and enrollment size
- 200+ university name aliases built in (`"mit"`, `"gatech"`, `"tamucc"`, `"ucsd"`, etc.)
- Automatically computes and surfaces the earnings gap between the two schools when both have data

---

## Running Locally

```bash
git clone https://github.com/SahithReddyVellenki/NeuroSymbolic-AI-Chatbot.git
cd NeuroSymbolic-AI-Chatbot
pip install streamlit openai requests
```

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY         = "your-groq-key"
GROQ_API_KEY2        = "optional-fallback-key"
COLLEGE_SCORECARD_API_KEY = "your-scorecard-key"  # free at api.data.gov
```

```bash
streamlit run app.py
```

> The system runs with a single Groq key. The second key and College Scorecard key are optional — the system degrades gracefully if either is missing.

---

## Project Structure

```
app.py                  # Streamlit UI, council rendering, decision tree visualization
llm_interface.py        # Fact extraction, topic rotation, BLS/college injection, agent debate
symbolic_engine.py      # Deterministic constraint engine, violation detection, decision mode
bls_retriever.py        # BLS OOH fuzzy retriever — 342 occupations, synonym resolution
college_retriever.py    # College Scorecard live API client — tuition, earnings, grad rate
bls_ooh_chunks.jsonl    # BLS Occupational Outlook Handbook structured corpus
```

---

## Research

**"Neuro-Symbolic Agentic Decision Support for Constraint-Aware Career Analysis"**  
Texas A&M University – Corpus Christi · Dept. of Computer Science · 2025  
Author: Sahith Reddy Vellenki

The core academic contribution is the architecture pattern: a deterministic symbolic state as the ground truth layer that constrains and audits neural LLM reasoning, combined with authoritative external data retrieval to replace hallucinated figures with real numbers — making the system's conclusions traceable, grounded, and challengeable.

---
