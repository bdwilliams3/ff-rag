"""
Manual test suite for PlayerResolver.
Run: .venv/bin/python3 scripts/test_resolver.py
"""

import sys
sys.path.insert(0, ".")
from utils.player_resolver import PlayerResolver

resolver = PlayerResolver()

cases = [
    # (description, record, expect_player_id_contains, expect_min_confidence)

    # Tier 1 — exact draft slot
    ("Tier1: Ja'Marr Chase (pick 5, 2021)",
     {"draft_year": 2021, "draft_pick": 5},
     "chase", 1.0),

    ("Tier1: Justin Jefferson (pick 22, 2020)",
     {"draft_year": 2020, "draft_pick": 22},
     "jefferson", 1.0),

    # Tier 2 — draft year/round/team + fuzzy last name (nickname variation)
    ("Tier2: 'Dre Miller' → should find via team+round+last name",
     {"name": "Dre Miller", "draft_year": 2022, "draft_round": 4, "drafting_team": "CIN"},
     None, 0.0),   # will show whatever it finds

    # Tier 3 — fuzzy name + college + position family
    ("Tier3: name+college+position family",
     {"name": "Jaylen Waddle", "college": "Alabama", "position": "WR"},
     None, 0.75),

    # Tier 3 — position family mismatch (WR vs TE) should still match
    ("Tier3: position mismatch WR vs TE same college",
     {"name": "Kyle Pitts", "college": "Florida", "position": "WR"},
     None, 0.75),

    # Tier 4 — fuzzy name + draft year only
    ("Tier4: first+last name + draft year",
     {"name": "Bijan Robinson", "draft_year": 2023},
     None, 0.50),

    # Should not match — gibberish
    ("No match: nonsense input",
     {"name": "Zzzz Xxxxxx", "draft_year": 2099},
     None, 0.0),
]

print(f"\n{'='*70}")
print(f"{'DESCRIPTION':<45} {'ID':<14} {'CONF':<6} {'NAME'}")
print(f"{'='*70}")

ref = resolver._ref  # for display only

for desc, record, _, min_conf in cases:
    pid, conf = resolver.resolve(record)
    if pid:
        row = ref[ref["player_id"] == pid]
        matched_name = row.iloc[0]["player_name"] if not row.empty else "?"
    else:
        matched_name = "— no match —"
    status = "✓" if conf >= min_conf else "✗"
    print(f"{status} {desc:<43} {str(pid or ''):<14} {conf:<6.2f} {matched_name}")

unmatched = resolver.flush_unmatched()
print(f"\n{unmatched} unmatched record(s) written to data/unmatched/")
