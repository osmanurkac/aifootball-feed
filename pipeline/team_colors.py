"""Our own team marks (colours + 3-letter code).

HARD RULE: we never use official club crests or logos. We only use team *names*
(data) and our own colour palette. This table mirrors the design's badge palette;
teams not listed get a deterministic colour derived from their name.
"""
from __future__ import annotations

import hashlib

# name-substring -> (shortCode, primary, secondary, text)
KNOWN: dict[str, tuple[str, str, str, str]] = {
    "arsenal": ("ARS", "#D7263D", "#F2F5FA", "#F2F5FA"),
    "chelsea": ("CHE", "#1F44A8", "#F2F5FA", "#F2F5FA"),
    "liverpool": ("LIV", "#B5121B", "#F2C14E", "#F2C14E"),
    "manchester city": ("MCI", "#7DB9E8", "#1C2C5B", "#1C2C5B"),
    "manchester united": ("MUN", "#C70101", "#111111", "#F2F5FA"),
    "tottenham": ("TOT", "#F2F5FA", "#132257", "#132257"),
    "newcastle": ("NEW", "#141414", "#F2F5FA", "#F2F5FA"),
    "aston villa": ("AVL", "#6C1D45", "#95BFE5", "#95BFE5"),
    "real madrid": ("RMA", "#F2F5FA", "#C9A43A", "#1A2A5E"),
    "barcelona": ("BAR", "#9B1B4A", "#2C4FA8", "#F2F5FA"),
    "atletico": ("ATM", "#CB3524", "#F2F5FA", "#F2F5FA"),
    "bayern": ("BAY", "#DC052D", "#F2F5FA", "#F2F5FA"),
    "dortmund": ("BVB", "#FDE100", "#111111", "#111111"),
    "gladbach": ("BMG", "#F2F5FA", "#111111", "#111111"),
    "inter": ("INT", "#0B3D91", "#111111", "#F2F5FA"),
    "milan": ("MIL", "#C8102E", "#111111", "#F2F5FA"),
    "juventus": ("JUV", "#F2F5FA", "#111111", "#111111"),
    "marseille": ("OM", "#F2F5FA", "#2FAEE0", "#0B3D91"),
    "lyon": ("OL", "#F2F5FA", "#E30613", "#1B3A8F"),
    "paris": ("PSG", "#0B1F4D", "#DA291C", "#F2F5FA"),
    "fenerbah": ("FEN", "#13224F", "#F7D417", "#F7D417"),
    "lakers": ("LAL", "#4B2A85", "#F5B82E", "#F5B82E"),
    "celtics": ("BOS", "#0F7A3E", "#F2F5FA", "#F2F5FA"),
    "warriors": ("GSW", "#1D428A", "#FFC72C", "#FFC72C"),
    "heat": ("MIA", "#98002E", "#F9A01B", "#F2F5FA"),
    "knicks": ("NYK", "#006BB6", "#F58426", "#F2F5FA"),
    "nuggets": ("DEN", "#0E2240", "#FEC524", "#FEC524"),
}

_PALETTE = ["#2C4FA8", "#9B1B4A", "#0F7A3E", "#C8102E", "#4B2A85", "#1D428A", "#CB3524", "#0E2240"]


def _slug_code(name: str) -> str:
    letters = [c for c in name.upper() if c.isalpha()]
    return "".join(letters[:3]) if len(letters) >= 3 else (name.upper() + "XXX")[:3]


def team_mark(name: str) -> tuple[str, str, str, str]:
    """Return (shortCode, primary, secondary, text) for a team name."""
    low = name.lower()
    for key, mark in KNOWN.items():
        if key in low:
            return mark
    # Deterministic fallback from the name hash.
    h = int(hashlib.sha1(low.encode()).hexdigest(), 16)
    primary = _PALETTE[h % len(_PALETTE)]
    return (_slug_code(name), primary, "#F2F5FA", "#F2F5FA")
