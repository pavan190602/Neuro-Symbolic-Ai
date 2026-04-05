"""
College Scorecard Retriever
===========================
Fetches real university data from the U.S. Department of Education
College Scorecard API and provides fuzzy name matching for decision support.

Data provided per university:
  - In-state and out-of-state tuition (annual)
  - Average net price after financial aid
  - Graduation rate (150% of normal time)
  - Median earnings 10 years after enrollment
  - Acceptance rate
  - Enrollment size
  - City and state

No local file needed — data is fetched live from the API and cached
for the session so each university is only looked up once.

Public API
----------
    retriever = CollegeRetriever(api_key="your_key")
    card = retriever.get_college_card("TAMUCC")
    # returns a CollegeCard or None

    cards = retriever.get_cards_for_decision("TAMUCC", "UMBC")
    # returns {"option_a": CollegeCard|None, "option_b": CollegeCard|None}

    block = retriever.format_comparison_block(card_a, card_b, "TAMUCC", "UMBC")
    # returns a compact string for LLM prompt injection

API key
-------
Free key at https://api.data.gov/signup (takes 2 minutes).
Set as COLLEGE_SCORECARD_API_KEY env var or pass to constructor.
Falls back to DEMO_KEY (40 requests/hour) if not set.
"""

import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    import urllib.request
    import json as _json


# ── University name aliases ───────────────────────────────────────────────────
# Maps common abbreviations / nicknames → fragments of the official name.
# Used to expand user input before the API search.

NAME_ALIASES: Dict[str, str] = {
    # Texas A&M system
    "tamucc":            "Texas A&M University-Corpus Christi",
    "tamu corpus christi": "Texas A&M University-Corpus Christi",
    "tamu cc":           "Texas A&M University-Corpus Christi",
    "tamuk":             "Texas A&M University-Kingsville",
    "tamu kingsville":   "Texas A&M University-Kingsville",
    "tamu":              "Texas A&M University",
    "tamu college station": "Texas A&M University",
    "texas a&m":         "Texas A&M University",
    "texas a&m main":    "Texas A&M University",

    # UCs
    "ucla":              "University of California-Los Angeles",
    "ucsd":              "University of California-San Diego",
    "ucsb":              "University of California-Santa Barbara",
    "ucb":               "University of California-Berkeley",
    "uc berkeley":       "University of California-Berkeley",
    "uci":               "University of California-Irvine",
    "ucd":               "University of California-Davis",
    "ucr":               "University of California-Riverside",

    # UTs
    "ut austin":         "University of Texas at Austin",
    "utsa":              "University of Texas at San Antonio",
    "utd":               "University of Texas at Dallas",
    "ut dallas":         "University of Texas at Dallas",
    "uta":               "University of Texas at Arlington",

    # CUNY/SUNY
    "cuny":              "City University of New York",
    "suny":              "State University of New York",

    # Common abbreviations
    "mit":               "Massachusetts Institute of Technology",
    "cmu":               "Carnegie Mellon University",
    "gatech":            "Georgia Institute of Technology",
    "georgia tech":      "Georgia Institute of Technology",
    "umbc":              "University of Maryland-Baltimore County",
    "umd":               "University of Maryland-College Park",
    "vt":                "Virginia Polytechnic Institute",
    "virginia tech":     "Virginia Polytechnic Institute",
    "uva":               "University of Virginia",
    "unc":               "University of North Carolina",
    "unc chapel hill":   "University of North Carolina at Chapel Hill",
    "ncsu":              "North Carolina State University",
    "nc state":          "North Carolina State University",
    "psu":               "Pennsylvania State University",
    "penn state":        "Pennsylvania State University",
    "ohio state":        "Ohio State University",
    "osu":               "Ohio State University",
    "asu":               "Arizona State University",
    "usc":               "University of Southern California",
    "nyu":               "New York University",
    "bu":                "Boston University",
    "northeastern":      "Northeastern University",
    "drexel":            "Drexel University",
    "rutgers":           "Rutgers University",
    "purdue":            "Purdue University",
    "uiuc":              "University of Illinois Urbana-Champaign",
    "illinois":          "University of Illinois Urbana-Champaign",
    "umich":             "University of Michigan",
    "u michigan":        "University of Michigan",
    "michigan":          "University of Michigan-Ann Arbor",
    "uw":                "University of Washington",
    "uw madison":        "University of Wisconsin-Madison",
    "wisc":              "University of Wisconsin-Madison",
    "colorado":          "University of Colorado Boulder",
    "cu boulder":        "University of Colorado Boulder",
    "csu":               "Colorado State University",
    "sfsu":              "San Francisco State University",
    "sjsu":              "San Jose State University",
    "csulb":             "California State University-Long Beach",
    "cal state long beach": "California State University-Long Beach",
    "csuf":              "California State University-Fullerton",
    "sdsu":              "San Diego State University",

    # Missouri / Midwest
    "ucm":                         "University of Central Missouri",
    "university of central missouri": "University of Central Missouri",
    "mst":               "Missouri University of Science and Technology",
    "missouri s&t":      "Missouri University of Science and Technology",
    "umsl":              "University of Missouri-St. Louis",
    "umkc":              "University of Missouri-Kansas City",
    "mizzou":            "University of Missouri-Columbia",

    # Additional common ones students compare
    "fiu":               "Florida International University",
    "fau":               "Florida Atlantic University",
    "usf":               "University of South Florida",
    "ucf":               "University of Central Florida",
    "fsu":               "Florida State University",
    "uf":                "University of Florida",
    "lsu":               "Louisiana State University",
    "tulane":            "Tulane University",
    "rice":              "Rice University",
    "ttu":               "Texas Tech University",
    "texas tech":        "Texas Tech University",
    "unt":               "University of North Texas",
    "twu":               "Texas Woman's University",
    "utpb":              "University of Texas of the Permian Basin",
    "utrgv":             "University of Texas Rio Grande Valley",
    "lamar":             "Lamar University",
    "sfasu":             "Stephen F. Austin State University",
    "shsu":              "Sam Houston State University",
    "txstate":           "Texas State University",
    "texas state":       "Texas State University",

    # Ivy / elite
    "harvard":           "Harvard University",
    "yale":              "Yale University",
    "princeton":         "Princeton University",
    "columbia":          "Columbia University",
    "penn":              "University of Pennsylvania",
    "upenn":             "University of Pennsylvania",
    "dartmouth":         "Dartmouth College",
    "brown":             "Brown University",
    "cornell":           "Cornell University",
    "stanford":          "Stanford University",
    "caltech":           "California Institute of Technology",

    # Others frequently compared
    "wsu":               "Washington State University",
    "oregonstate":       "Oregon State University",
    "osu oregon":        "Oregon State University",
    "colorado state":    "Colorado State University",
    "ku":                "University of Kansas",
    "kstate":            "Kansas State University",
    "iowa state":        "Iowa State University",
    "iastate":           "Iowa State University",
    "uiowa":             "University of Iowa",
    "unl":               "University of Nebraska-Lincoln",
    "huskers":           "University of Nebraska-Lincoln",
    "auburn":            "Auburn University",
    "ua":                "University of Alabama",
    "bama":              "University of Alabama",
    "clemson":           "Clemson University",
    "sc":                "University of South Carolina",
    "usc columbia":      "University of South Carolina",
    "virginia":          "University of Virginia",
    "wm":                "College of William and Mary",
    "jmu":               "James Madison University",
    "vcu":               "Virginia Commonwealth University",
    "pitt":              "University of Pittsburgh",
    "penn state":        "Pennsylvania State University",
    "temple":            "Temple University",
    "fordham":           "Fordham University",
    "sbu":               "Stony Brook University",
    "stony brook":       "Stony Brook University",
    "binghamton":        "Binghamton University",
    "buffalo":           "University at Buffalo",
    "ub":                "University at Buffalo",
}


@dataclass
class CollegeCard:
    """Structured university facts from College Scorecard."""
    name:               str
    city:               str
    state:              str
    tuition_in_state:   Optional[int]    # annual, USD
    tuition_out_state:  Optional[int]    # annual, USD
    net_price:          Optional[int]    # avg net price after aid
    grad_rate:          Optional[float]  # 0.0–1.0
    median_earnings_10yr: Optional[int]  # USD, 10 years after enrollment
    acceptance_rate:    Optional[float]  # 0.0–1.0
    enrollment:         Optional[int]
    unit_id:            Optional[str]    # IPEDS unit ID
    matched_query:      str = ""
    match_score:        float = 1.0

    def tuition_display(self, is_instate: bool = True) -> str:
        """Human-readable tuition string."""
        val = self.tuition_in_state if is_instate else self.tuition_out_state
        if val is None:
            return "unknown"
        return f"${val:,}/yr"

    def grad_rate_display(self) -> str:
        if self.grad_rate is None:
            return "unknown"
        return f"{self.grad_rate * 100:.0f}%"

    def earnings_display(self) -> str:
        if self.median_earnings_10yr is None:
            return "unknown"
        return f"${self.median_earnings_10yr:,}"

    def acceptance_display(self) -> str:
        if self.acceptance_rate is None:
            return "unknown"
        return f"{self.acceptance_rate * 100:.0f}%"

    def __repr__(self) -> str:
        return (f"CollegeCard({self.name}, {self.city} {self.state} | "
                f"in-state: {self.tuition_display(True)} | "
                f"earnings 10yr: {self.earnings_display()} | "
                f"grad rate: {self.grad_rate_display()})")


class CollegeRetriever:
    """
    Looks up university data from the College Scorecard API.

    Session-level cache prevents duplicate API calls for the same school.
    If the API is unavailable, returns None gracefully — the system
    continues without university data rather than crashing.
    """

    BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"

    FIELDS = ",".join([
        "school.name",
        "school.city",
        "school.state",
        "school.carnegie_size_setting",
        "id",
        "latest.cost.tuition.in_state",
        "latest.cost.tuition.out_of_state",
        "latest.cost.avg_net_price.public",
        "latest.cost.avg_net_price.private",
        "latest.earnings.10_yrs_after_entry.median",
        "latest.completion.completion_rate_4yr_150nt",
        "latest.admissions.admission_rate.overall",
        "latest.student.size",
    ])

    def __init__(self, api_key: Optional[str] = None):
        self.api_key   = (api_key
                         or os.getenv("COLLEGE_SCORECARD_API_KEY")
                         or "DEMO_KEY")
        self._cache: Dict[str, Optional[CollegeCard]] = {}
        if self.api_key == "DEMO_KEY":
            print("[COLLEGE] Using DEMO_KEY (40 req/hr). "
                  "Get a free key at api.data.gov/signup for 1000 req/day.")
        else:
            print(f"[COLLEGE] Retriever initialized with custom API key")

    # ── Query helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"\s+", " ", name.lower().strip())

    def _resolve_alias(self, name: str) -> str:
        """Expand abbreviation to full name if known."""
        n = self._normalize(name)
        return NAME_ALIASES.get(n, name)

    def _fetch(self, school_name: str) -> Optional[Dict]:
        """
        Call College Scorecard API with the given school name.
        Returns the raw JSON result dict or None on failure.
        """
        params = {
            "school.name": school_name,
            "fields":      self.FIELDS,
            "api_key":     self.api_key,
            "_per_page":   5,    # get top 5 matches to pick best
        }

        try:
            if _HAS_REQUESTS:
                r = _requests.get(self.BASE_URL, params=params, timeout=8)
                if r.status_code == 200:
                    return r.json()
                print(f"[COLLEGE] API returned {r.status_code} for '{school_name}'")
                return None
            else:
                # Fallback: urllib
                import urllib.request
                import urllib.parse
                import json
                query = urllib.parse.urlencode(params)
                url   = f"{self.BASE_URL}?{query}"
                with urllib.request.urlopen(url, timeout=8) as resp:
                    return json.loads(resp.read())
        except Exception as e:
            print(f"[COLLEGE] API error for '{school_name}': {e}")
            return None

    def _parse_result(self, result: Dict, query: str) -> Optional[CollegeCard]:
        """Parse one API result dict into a CollegeCard."""
        try:
            school   = result.get("school", {})
            latest   = result.get("latest", {})
            cost     = latest.get("cost", {})
            earnings = latest.get("earnings", {})
            compl    = latest.get("completion", {})
            admiss   = latest.get("admissions", {})
            student  = latest.get("student", {})

            name     = school.get("name", "Unknown")
            city     = school.get("city", "")
            state    = school.get("state", "")

            tuition_in  = cost.get("tuition", {}).get("in_state")
            tuition_out = cost.get("tuition", {}).get("out_of_state")
            net_pub  = cost.get("avg_net_price", {}).get("public")
            net_priv = cost.get("avg_net_price", {}).get("private")
            net_price = net_pub or net_priv

            earnings_10yr = (earnings
                             .get("10_yrs_after_entry", {})
                             .get("median"))

            grad_rate = compl.get("completion_rate_4yr_150nt")
            accept    = admiss.get("admission_rate", {}).get("overall")
            enroll    = student.get("size")
            unit_id   = str(result.get("id", ""))

            return CollegeCard(
                name               = name,
                city               = city,
                state              = state,
                tuition_in_state   = int(tuition_in)   if tuition_in   else None,
                tuition_out_state  = int(tuition_out)  if tuition_out  else None,
                net_price          = int(net_price)    if net_price    else None,
                grad_rate          = float(grad_rate)  if grad_rate    else None,
                median_earnings_10yr = int(earnings_10yr) if earnings_10yr else None,
                acceptance_rate    = float(accept)     if accept       else None,
                enrollment         = int(enroll)       if enroll       else None,
                unit_id            = unit_id,
                matched_query      = query,
                match_score        = 1.0,
            )
        except Exception as e:
            print(f"[COLLEGE] Parse error: {e}")
            return None

    def _best_match(self, results: List[Dict], query: str) -> Optional[CollegeCard]:
        """
        Pick the best matching school from API results using
        name similarity scoring.
        """
        if not results:
            return None

        query_norm = self._normalize(query)
        best_score = 0.0
        best_card  = None

        for r in results:
            name = r.get("school", {}).get("name", "")
            score = SequenceMatcher(
                None, query_norm, self._normalize(name)
            ).ratio()
            if score > best_score:
                best_score = score
                best_card  = self._parse_result(r, query)

        if best_card:
            best_card.match_score = best_score
        return best_card

    # ── Public API ────────────────────────────────────────────────────────────

    def get_college_card(self, name: str) -> Optional[CollegeCard]:
        """
        Look up a university by name or abbreviation.
        Tries multiple search terms if the first attempt returns nothing.
        Results are cached for the session.
        """
        cache_key = self._normalize(name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build a list of search terms to try in order
        resolved  = self._resolve_alias(name)
        norm_name = self._normalize(name)

        search_attempts = [resolved]
        # Do NOT add the raw abbreviation — short codes return 0 results from API
        # Add smart variants of the full name
        words = resolved.split()
        if len(words) >= 3:
            search_attempts.append(" ".join(words[:3]))  # first 3 words
        if len(words) >= 2:
            search_attempts.append(" ".join(words[:2]))  # first 2 words
        # "University of X" → try "X" alone
        if resolved.lower().startswith("university of "):
            search_attempts.append(resolved[14:])
        # "Texas A&M University-City" → try "Texas A&M City"
        if "university-" in resolved.lower():
            parts = resolved.lower().split("university-")
            city = parts[-1].title()
            search_attempts.append(f"Texas A&M {city}")
            search_attempts.append(city)
        # "University-City" format → try city name
        if "-" in resolved:
            city_part = resolved.split("-")[-1].strip()
            if len(city_part) > 3:
                search_attempts.append(city_part)

        # Deduplicate while preserving order
        seen = set()
        unique_attempts = []
        for s in search_attempts:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique_attempts.append(s)

        print(f"[COLLEGE] Looking up: '{name}' — trying: {unique_attempts}")

        all_results = []
        for attempt in unique_attempts:
            data = self._fetch(attempt)
            results = (data or {}).get("results", [])
            if results:
                all_results.extend(results)
                break  # found something, stop trying

        card = self._best_match(all_results, resolved) if all_results else None
        if card:
            print(f"[COLLEGE] Found: {card.name} ({card.city}, {card.state}) "
                  f"score={card.match_score:.2f}")
        else:
            print(f"[COLLEGE] No match for '{name}' after {len(unique_attempts)} attempts")

        self._cache[cache_key] = card
        return card

    def get_cards_for_decision(
        self, name_a: str, name_b: str
    ) -> Dict[str, Optional[CollegeCard]]:
        """Look up both universities for a comparison decision."""
        # Small delay between calls to avoid rate limiting on DEMO_KEY
        card_a = self.get_college_card(name_a)
        time.sleep(0.5)   # avoid DEMO_KEY rate limit (40 req/hour)
        card_b = self.get_college_card(name_b)
        return {"option_a": card_a, "option_b": card_b}

    def format_comparison_block(
        self,
        card_a: Optional["CollegeCard"],
        card_b: Optional["CollegeCard"],
        label_a: str,
        label_b: str,
    ) -> str:
        """
        Format a compact comparison block for LLM prompt injection.
        Only includes fields that have real data — no nulls passed to agents.
        """
        def fmt_card(card: Optional[CollegeCard], label: str) -> str:
            if card is None:
                return f"{label}: No data available from College Scorecard."

            lines = [f"{label} ({card.name}, {card.city} {card.state}):"]

            if card.tuition_in_state:
                lines.append(f"  In-state tuition:   ${card.tuition_in_state:,}/yr")
            if card.tuition_out_state:
                lines.append(f"  Out-of-state tuition: ${card.tuition_out_state:,}/yr")
            if card.net_price:
                lines.append(f"  Avg net price (after aid): ${card.net_price:,}/yr")
            if card.median_earnings_10yr:
                lines.append(
                    f"  Median earnings 10yr after enrollment: "
                    f"${card.median_earnings_10yr:,}"
                )
            if card.grad_rate:
                lines.append(f"  Graduation rate (6yr): {card.grad_rate_display()}")
            if card.acceptance_rate:
                lines.append(f"  Acceptance rate: {card.acceptance_display()}")
            if card.enrollment:
                lines.append(f"  Enrollment: {card.enrollment:,} students")

            return "\n".join(lines)

        if card_a is None and card_b is None:
            return ""

        lines = [
            "=" * 55,
            "COLLEGE SCORECARD DATA (U.S. Dept. of Education)",
            "=" * 55,
            fmt_card(card_a, label_a),
            "",
            fmt_card(card_b, label_b),
        ]

        # Add a derived comparison note if both have earnings data
        if card_a and card_b and card_a.median_earnings_10yr and card_b.median_earnings_10yr:
            diff = card_a.median_earnings_10yr - card_b.median_earnings_10yr
            if abs(diff) > 1000:
                higher = label_a if diff > 0 else label_b
                lines.append("")
                lines.append(
                    f"Note: {higher} graduates earn ~${abs(diff):,} more "
                    f"at 10 years (median). This is an all-majors median — "
                    f"CS/engineering typically runs higher."
                )

        lines.append("=" * 55)
        return "\n".join(lines)
