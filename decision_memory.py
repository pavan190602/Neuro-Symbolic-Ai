"""
Decision Memory — Persistent cross-session decision store
==========================================================

Stores resolved decisions to disk so the system can surface patterns
from past conversations.

Public API
----------
    mem = DecisionMemory()
    mem.save(state_dict, council_result, notes)
    mem.find_similar(state_dict)
    mem.get_context_block(state_dict)
    mem.get_all()
    mem.delete(record_id)
    mem.clear()
"""

from __future__ import annotations
import json, logging, os, uuid, re
from datetime import datetime
from typing import Any, Dict, List

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decisions.json")


class DecisionMemory:
    """Persistent store for past decisions."""

    def __init__(self, path: str = _DEFAULT_PATH):
        self._path = path
        self._records: List[Dict] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            self._records = []
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._records = data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"[MEMORY] Could not load: {e} — starting fresh")
            self._records = []

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._records, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logging.error(f"[MEMORY] Write failed: {e}")

    # ── Write API ────────────────────────────────────────────────────────────

    def save(self, state_dict: Dict, council_result: Dict, notes: str = "") -> str:
        """Persist a completed decision. Returns the record_id."""
        record_id = str(uuid.uuid4())[:8]
        meta      = state_dict.get("decision_metadata", {})
        options   = meta.get("options_being_compared", [])
        synth_raw = council_result.get("synthesizer", "")
        avg       = council_result.get("avg_vote", {})
        votes     = council_result.get("agent_votes", {})

        record = {
            "id":               record_id,
            "timestamp":        datetime.now().isoformat(),
            "decision_type":    meta.get("decision_type", "general"),
            "decision_subtype": meta.get("decision_subtype", ""),
            "options":          options,
            "decision_mode":    state_dict.get("decision_mode", ""),
            "notes":            notes,
            "ruling":           _extract_section(synth_raw, "RULING"),
            "open_question":    _extract_section(synth_raw, "OPEN QUESTION"),
            "avg_vote":         avg,
            "agent_votes":      {
                ag_id: {"option_a": v.get("option_a", 50), "option_b": v.get("option_b", 50)}
                for ag_id, v in votes.items()
            },
            "facts_snapshot":   _compact_snapshot(state_dict),
        }
        self._records.insert(0, record)
        self._save()
        logging.info(f"[MEMORY] Saved decision {record_id}: {options}")
        return record_id

    def delete(self, record_id: str) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r.get("id") != record_id]
        changed = len(self._records) < before
        if changed:
            self._save()
        return changed

    def clear(self) -> None:
        self._records = []
        self._save()

    # ── Read API ─────────────────────────────────────────────────────────────

    def get_all(self) -> List[Dict]:
        return list(self._records)

    def find_similar(self, state_dict: Dict, max_results: int = 3) -> List[Dict]:
        """Return past records that match the current decision type/subtype."""
        meta    = state_dict.get("decision_metadata", {})
        d_type  = meta.get("decision_type", "")
        d_sub   = meta.get("decision_subtype", "")
        options = set(meta.get("options_being_compared", []))

        scored = []
        for r in self._records:
            score = 0
            if r.get("decision_type") == d_type:
                score += 2
            if r.get("decision_subtype") == d_sub:
                score += 1
            past_opts = set(r.get("options", []))
            if options and options == past_opts:
                score += 3
            if score > 0:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:max_results]]

    def get_context_block(self, state_dict: Dict) -> str:
        """Formatted text block for LLM injection. Empty if no matches."""
        similar = self.find_similar(state_dict, max_results=2)
        if not similar:
            return ""
        lines = ["=== MEMORY: Relevant past decisions ==="]
        for r in similar:
            opts   = " vs ".join(r.get("options", ["?"]))
            date_  = r.get("timestamp", "")[:10]
            ruling = r.get("ruling") or "No ruling recorded"
            oq     = r.get("open_question", "")
            va     = r.get("avg_vote", {}).get("option_a", "?")
            vb     = r.get("avg_vote", {}).get("option_b", "?")
            lines.append(
                f"• [{date_}] {opts}  |  Outcome: {va}% vs {vb}%\n"
                f"  Ruling: {ruling}"
                + (f"\n  Unresolved: {oq}" if oq else "")
            )
        lines.append("=" * 40)
        return "\n".join(lines)

    def count(self) -> int:
        return len(self._records)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_section(text: str, label: str) -> str:
    m = re.search(
        rf"{re.escape(label)}:\s*(.+?)(?=\n[A-Z ]+:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _compact_snapshot(state_dict: Dict) -> Dict[str, Any]:
    CATS = ["values","interests","career_vision","current","personal",
            "financial","offer_a","offer_b","uni_a","uni_b"]
    snap: Dict[str, Any] = {}
    for cat in CATS:
        filled = {
            k: v for k, v in state_dict.get(cat, {}).items()
            if v is not None and v is not False and v != "" and v != []
        }
        if filled:
            snap[cat] = filled
    return snap
