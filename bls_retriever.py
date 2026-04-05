"""
BLS OOH Retriever
=================
Loads the BLS Occupational Outlook Handbook JSONL file at startup and
provides fast fuzzy-match retrieval of career facts by occupation name.

No external vector DB or embeddings required — 342 occupations is small
enough for difflib similarity scoring with a synonym layer on top.

Public API
----------
    retriever = BLSRetriever("bls_ooh_chunks.jsonl")
    card = retriever.get_career_card("Data Science")
    # returns a BLSCard or None

    cards = retriever.get_cards_for_decision("Computer Science", "Data Science")
    # returns {"option_a": BLSCard|None, "option_b": BLSCard|None}

    block = retriever.format_for_agent(card)
    # returns a compact string for LLM prompt injection
"""

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ── Synonym map ───────────────────────────────────────────────────────────────
# Maps common user-input terms → BLS occupation title keywords.
# Extend freely — keys are lowercased user terms, values are BLS title substrings.

SYNONYMS: Dict[str, str] = {
    # Tech
    "cs":                          "computer science",
    "computer science":            "software developers",
    "software engineering":        "software developers",
    "software engineer":           "software developers",
    "swe":                         "software developers",
    "software dev":                "software developers",
    "data science":                "data scientists",
    "data scientist":              "data scientists",
    "ds":                          "data scientists",
    "ml engineer":                 "computer and information research scientists",
    "machine learning":            "computer and information research scientists",
    "ai engineer":                 "computer and information research scientists",
    "cybersecurity":               "information security analysts",
    "cyber security":              "information security analysts",
    "infosec":                     "information security analysts",
    "network admin":               "network and computer systems administrators",
    "sysadmin":                    "network and computer systems administrators",
    "database admin":              "database administrators",
    "dba":                         "database administrators",
    "computer hardware":           "computer hardware engineers",
    "hardware engineering":        "computer hardware engineers",
    "it manager":                  "computer and information systems managers",
    "it management":               "computer and information systems managers",
    "programmer":                  "computer programmers",
    "coding":                      "computer programmers",
    "systems analyst":             "computer systems analysts",
    # Business / Finance
    "business":                    "management analysts",
    "mba":                         "management analysts",
    "consulting":                  "management analysts",
    "finance":                     "financial analysts",
    "financial analyst":           "financial analysts",
    "accounting":                  "accountants and auditors",
    "accountant":                  "accountants and auditors",
    "cpa":                         "accountants and auditors",
    "financial advisor":           "personal financial advisors",
    "financial manager":           "financial managers",
    "marketing":                   "advertising, promotions, and marketing managers",
    "sales":                       "sales managers",
    "hr":                          "human resources managers",
    "human resources":             "human resources managers",
    "project manager":             "project management specialists",
    "pm":                          "project management specialists",
    # Engineering
    "electrical engineering":      "electrical and electronics engineers",
    "ee":                          "electrical and electronics engineers",
    "mechanical engineering":      "mechanical engineers",
    "me":                          "mechanical engineers",
    "civil engineering":           "civil engineers",
    "chemical engineering":        "chemical engineers",
    "aerospace engineering":       "aerospace engineers",
    "biomedical engineering":      "bioengineers and biomedical engineers",
    "environmental engineering":   "environmental engineers",
    "industrial engineering":      "industrial engineers",
    "nuclear engineering":         "nuclear engineers",
    "petroleum engineering":       "petroleum engineers",
    "materials engineering":       "materials engineers",
    # Healthcare
    "medicine":                    "physicians and surgeons",
    "doctor":                      "physicians and surgeons",
    "md":                          "physicians and surgeons",
    "physician":                   "physicians and surgeons",
    "nursing":                     "registered nurses",
    "nurse":                       "registered nurses",
    "rn":                          "registered nurses",
    "pharmacy":                    "pharmacists",
    "pharmacist":                  "pharmacists",
    "dentistry":                   "dentists",
    "dentist":                     "dentists",
    "physical therapy":            "physical therapists",
    "pt":                          "physical therapists",
    "occupational therapy":        "occupational therapists",
    "speech therapy":              "speech-language pathologists",
    "psychology":                  "psychologists",
    "psychiatry":                  "physicians and surgeons",
    "veterinary":                  "veterinarians",
    "vet":                         "veterinarians",
    "physician assistant":         "physician assistants",
    "pa":                          "physician assistants",
    "np":                          "nurse anesthetists, nurse midwives",
    # Sciences
    "biology":                     "biological technicians",
    "chemistry":                   "chemists and materials scientists",
    "physics":                     "physicists and astronomers",
    "statistics":                  "mathematicians and statisticians",
    "math":                        "mathematicians and statisticians",
    "mathematics":                 "mathematicians and statisticians",
    "environmental science":       "environmental scientists and specialists",
    "geology":                     "geoscientists",
    "epidemiology":                "epidemiologists",
    # Education
    "teacher":                     "high school teachers",
    "teaching":                    "high school teachers",
    "education":                   "postsecondary teachers",
    "professor":                   "postsecondary teachers",
    "counselor":                   "school and career counselors",
    # Law / Policy
    "law":                         "lawyers",
    "lawyer":                      "lawyers",
    "attorney":                    "lawyers",
    "paralegal":                   "paralegals and legal assistants",
    "urban planning":              "urban and regional planners",
    "public policy":               "political scientists",
    "political science":           "political scientists",
    "economics":                   "economists",
    "economist":                   "economists",
    # Arts / Media
    "graphic design":              "graphic designers",
    "ux":                          "web developers and digital designers",
    "ui":                          "web developers and digital designers",
    "web design":                  "web developers and digital designers",
    "web development":             "web developers and digital designers",
    "animation":                   "special effects artists and animators",
    "game design":                 "special effects artists and animators",
    "film":                        "film and video editors and camera operators",
    "journalism":                  "news analysts, reporters, and journalists",
    "writing":                     "writers and authors",
    "architecture":                "architects",
    "interior design":             "interior designers",
    "fashion":                     "fashion designers",
    "music":                       "musicians and singers",
    "arts":                        "craft and fine artists",
    "social work":                 "social workers",
    "public relations":            "public relations specialists",
    # Trades
    "electrician":                 "electricians",
    "plumber":                     "plumbers, pipefitters",
    "carpenter":                   "carpenters",
    "construction":                "construction managers",
    "welding":                     "welders, cutters",
    # Other common
    "pilot":                       "airline and commercial pilots",
    "flight attendant":            "flight attendants",
    "actuary":                     "actuaries",
    "librarian":                   "librarians",
    "chef":                        "chefs and head cooks",
    "culinary":                    "chefs and head cooks",
}


@dataclass
class BLSCard:
    """Structured career facts for a single BLS occupation."""
    title:           str
    median_pay:      str
    outlook:         str          # e.g. "34% (Much faster than average)"
    outlook_pct:     Optional[float]  # numeric outlook, None if unparseable
    num_jobs:        str
    entry_education: str
    url:             str
    what_they_do:    str          # first 300 chars of "What X Do" section
    job_outlook_text:str          # first 300 chars of "Job Outlook" section
    similar_to:      List[str] = field(default_factory=list)
    match_score:     float = 1.0  # similarity score from fuzzy match
    matched_query:   str = ""

    def pay_annual(self) -> Optional[float]:
        """Parse median_pay string → float USD annual."""
        try:
            cleaned = re.sub(r"[^\d,.]", "", self.median_pay)
            return float(cleaned.replace(",", ""))
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"BLSCard({self.title}: {self.median_pay} | {self.outlook})"


class BLSRetriever:
    """
    Loads the BLS JSONL file once and provides fuzzy occupation lookup.

    The retrieval pipeline:
    1. Normalize the query (lowercase, strip punctuation)
    2. Check synonym map — if match, rewrite query to canonical BLS keyword
    3. Score all 342 occupation titles with SequenceMatcher
    4. Return the best match above the threshold (default 0.35)
    """

    SECTIONS_FOR_CARD = {
        "what_they_do":   None,   # filled by "What X Do" section
        "job_outlook":    None,   # filled by "Job Outlook" section
    }

    def __init__(self, jsonl_path: str, similarity_threshold: float = 0.35):
        self.threshold = similarity_threshold
        self._occupations: Dict[str, Dict] = {}   # title → {meta, sections}
        self._titles_lower: List[Tuple[str, str]] = []  # (lower_title, original_title)
        self._embedder = None          # lazy-loaded sentence-transformer
        self._title_embeddings = None  # np.ndarray once embedder is ready
        self._load(jsonl_path)
        self._try_load_embedder()

    def _try_load_embedder(self) -> None:
        """
        Attempt to load a sentence-transformer model for embedding-based retrieval.
        Falls back silently to difflib if the package is not installed.
        The model (all-MiniLM-L6-v2) is ~80 MB and loads once at startup.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer("all-MiniLM-L6-v2")
            titles = [t for _, t in self._titles_lower]
            self._title_embeddings = model.encode(titles, show_progress_bar=False)
            self._embedder = model
            self._embed_titles = titles
            logging.info(f"[BLS] Sentence-transformer loaded — {len(titles)} title embeddings ready")
        except ImportError:
            logging.info("[BLS] sentence-transformers not installed — using difflib fallback")
        except Exception as e:
            logging.warning(f"[BLS] Embedder load failed ({e}) — using difflib fallback")

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d    = json.loads(line)
                meta = d["metadata"]
                title = meta["title"]

                if title not in self._occupations:
                    self._occupations[title] = {
                        "meta":     meta,
                        "sections": {},
                    }
                    self._titles_lower.append((title.lower(), title))

                self._occupations[title]["sections"][meta["section"]] = d["text"]

        print(f"[BLS] Loaded {len(self._occupations)} occupations")

    # ── Query normalization ───────────────────────────────────────────────────

    @staticmethod
    def _normalize(query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r"[^a-z0-9 ]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _apply_synonyms(self, query: str) -> str:
        norm = self._normalize(query)
        # Exact synonym match first
        if norm in SYNONYMS:
            return SYNONYMS[norm]
        # Partial match — check if any synonym key appears as a substring
        for key, val in SYNONYMS.items():
            if key in norm:
                return val
        return norm

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, query_norm: str, title_lower: str) -> float:
        # SequenceMatcher base score
        base = SequenceMatcher(None, query_norm, title_lower).ratio()
        # Bonus if query is a substring of title or vice versa
        if query_norm in title_lower:
            base = max(base, 0.75)
        if title_lower in query_norm:
            base = max(base, 0.70)
        # Word overlap bonus
        q_words = set(query_norm.split())
        t_words = set(title_lower.split())
        overlap = len(q_words & t_words)
        if overlap:
            base = max(base, 0.4 + 0.1 * overlap)
        return base

    def _embedding_search(self, query: str) -> Optional[str]:
        """
        Use sentence-transformer cosine similarity to find the best occupation
        title for a free-form query.  Returns the matched title or None.
        Only invoked when the difflib path returns no match above threshold.
        """
        if self._embedder is None or self._title_embeddings is None:
            return None
        try:
            import numpy as np
            q_emb = self._embedder.encode([query], show_progress_bar=False)
            # Cosine similarity
            norms_t = self._title_embeddings / (
                np.linalg.norm(self._title_embeddings, axis=1, keepdims=True) + 1e-10
            )
            norms_q = q_emb / (np.linalg.norm(q_emb) + 1e-10)
            sims    = norms_t @ norms_q.T
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= 0.35:
                matched = self._embed_titles[best_idx]
                logging.info(f"[BLS][EMBED] query={query!r} → {matched!r} (sim={best_sim:.2f})")
                return matched
        except Exception as e:
            logging.warning(f"[BLS][EMBED] Error: {e}")
        return None

    # ── Card construction ─────────────────────────────────────────────────────

    def _build_card(self, title: str, score: float, query: str) -> BLSCard:
        occ      = self._occupations[title]
        meta     = occ["meta"]
        sections = occ["sections"]

        # Find "What X Do" section
        what_text = ""
        for sec_name, sec_text in sections.items():
            if sec_name.startswith("What ") and "Do" in sec_name:
                # Strip the header line (first line), take next 3 sentences
                lines  = sec_text.split("\n")
                body   = " ".join(l for l in lines if l.strip() and "Median Pay" not in l)
                what_text = body[:350].strip()
                break

        # Find "Job Outlook" section
        outlook_text = ""
        if "Job Outlook" in sections:
            lines = sections["Job Outlook"].split("\n")
            body  = " ".join(l for l in lines if l.strip() and "Median Pay" not in l)
            outlook_text = body[:350].strip()

        # Parse outlook percentage
        outlook_pct = None
        m = re.search(r"(-?\d+)%", meta["outlook"])
        if m:
            try:
                outlook_pct = float(m.group(1))
            except Exception:
                pass

        # Similar occupations
        similar = []
        if "Similar Occupations" in sections:
            txt = sections["Similar Occupations"]
            # Extract occupation names — they appear as title-cased lines
            for line in txt.split("\n"):
                line = line.strip()
                if line and len(line) < 60 and line[0].isupper():
                    similar.append(line)
            similar = similar[:5]

        return BLSCard(
            title           = title,
            median_pay      = meta["median_pay"],
            outlook         = meta["outlook"],
            outlook_pct     = outlook_pct,
            num_jobs        = meta["num_jobs"],
            entry_education = meta["entry_education"],
            url             = meta["url"],
            what_they_do    = what_text,
            job_outlook_text= outlook_text,
            similar_to      = similar,
            match_score     = score,
            matched_query   = query,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_career_card(self, query: str) -> Optional[BLSCard]:
        """
        Return the best-matching BLSCard for a career/occupation query.
        Returns None if no match exceeds the similarity threshold.
        """
        if not query or not query.strip():
            return None

        expanded = self._apply_synonyms(query)
        norm     = self._normalize(expanded)

        best_score = 0.0
        best_title = None

        for title_lower, original_title in self._titles_lower:
            score = self._score(norm, title_lower)
            if score > best_score:
                best_score = score
                best_title = original_title

        if best_score < self.threshold or best_title is None:
            # Difflib failed — try embedding-based fallback
            embed_title = self._embedding_search(query)
            if embed_title:
                return self._build_card(embed_title, 0.5, query)
            return None

        return self._build_card(best_title, best_score, query)

    def get_cards_for_decision(
        self, option_a: str, option_b: str
    ) -> Dict[str, Optional[BLSCard]]:
        """Return BLS cards for both options in a decision."""
        return {
            "option_a": self.get_career_card(option_a),
            "option_b": self.get_career_card(option_b),
        }

    def format_for_agent(self, card: Optional[BLSCard], label: str = "") -> str:
        """
        Format a BLSCard as a compact string for LLM prompt injection.
        Returns empty string if card is None.
        """
        if card is None:
            return f"[No BLS data found for {label}]" if label else "[No BLS data found]"

        outlook_label = ""
        if card.outlook_pct is not None:
            if card.outlook_pct >= 20:
                outlook_label = "🚀 Much faster than average"
            elif card.outlook_pct >= 10:
                outlook_label = "📈 Faster than average"
            elif card.outlook_pct >= 3:
                outlook_label = "➡️ Average growth"
            elif card.outlook_pct >= 0:
                outlook_label = "⚠️ Slower than average"
            else:
                outlook_label = "📉 Declining"

        pay = card.pay_annual()
        pay_str = f"${pay:,.0f}/yr" if pay else card.median_pay

        lines = [
            f"[BLS: {card.title}]",
            f"  Median pay:       {pay_str}",
            f"  10-yr job growth: {card.outlook}  {outlook_label}",
            f"  Total jobs:       {card.num_jobs}",
            f"  Entry education:  {card.entry_education}",
        ]
        if card.what_they_do:
            lines.append(f"  Role summary:     {card.what_they_do[:200]}")
        if card.job_outlook_text:
            lines.append(f"  Outlook detail:   {card.job_outlook_text[:200]}")

        return "\n".join(lines)

    def format_comparison_block(
        self,
        card_a: Optional[BLSCard],
        card_b: Optional[BLSCard],
        label_a: str,
        label_b: str,
    ) -> str:
        """
        Format both cards as a side-by-side comparison block for prompt injection.
        """
        block_a = self.format_for_agent(card_a, label_a)
        block_b = self.format_for_agent(card_b, label_b)

        lines = [
            "═" * 55,
            "BLS OCCUPATIONAL OUTLOOK DATA (source: bls.gov/ooh)",
            "═" * 55,
            "",
            f"── {label_a} ──",
            block_a,
            "",
            f"── {label_b} ──",
            block_b,
            "",
            "Use these figures in your analysis. Cite the specific numbers.",
            "═" * 55,
        ]
        return "\n".join(lines)


# ── Module-level singleton ────────────────────────────────────────────────────

_RETRIEVER: Optional[BLSRetriever] = None


def get_retriever(jsonl_path: str = "bls_ooh_chunks.jsonl") -> BLSRetriever:
    """Return the module-level BLSRetriever singleton, creating it if needed."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = BLSRetriever(jsonl_path)
    return _RETRIEVER


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "bls_ooh_chunks.jsonl"
    r    = BLSRetriever(path)

    test_queries = [
        ("Computer Science", "Data Science"),
        ("nursing", "medicine"),
        ("software engineer", "data scientist"),
        ("MBA", "law"),
        ("electrical engineering", "computer science"),
        ("arts", "cs"),
        ("PhD", "job at Google"),
    ]

    for qa, qb in test_queries:
        cards = r.get_cards_for_decision(qa, qb)
        ca, cb = cards["option_a"], cards["option_b"]
        print(f"\n{'='*55}")
        print(f"Query: '{qa}' vs '{qb}'")
        print(f"  A → {ca.title if ca else 'NO MATCH'} (score={ca.match_score:.2f})" if ca else f"  A → NO MATCH")
        print(f"  B → {cb.title if cb else 'NO MATCH'} (score={cb.match_score:.2f})" if cb else f"  B → NO MATCH")
        if ca and cb:
            print(r.format_comparison_block(ca, cb, qa, qb))
