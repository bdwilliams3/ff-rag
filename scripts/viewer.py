"""
Local data viewer — browse any collected Parquet dataset at http://localhost:8080
Usage: .venv/bin/python3 scripts/viewer.py
"""

from pathlib import Path
from typing import Optional
from flask import Flask, request, render_template_string
import pandas as pd

app = Flask(__name__)
DATA_ROOT = Path("data")

DATASETS = {
    "stats/weekly":    "Weekly Stats",
    "stats/seasonal":  "Seasonal Stats",
    "athletic/combine_draft":  "Athletic Profiles",
    "athletic/college_stats":  "College Stats",
    "injuries":        "Injuries",
    "coaching/coaching_weekly":                  "Coaching Weekly",
    "coaching/offensive_coordinator_profiles":   "OC Profiles",
    "coaching/defensive_coordinator_profiles":   "DC Profiles",
    "adp/adp_historical":              "Historical ADP",
    "winners/league_winner_frequency": "League Winners",
    "salaries/player_salaries":        "Player Salaries",
}

PAGE_SIZE = 100

BASE = """
<!doctype html>
<html>
<head>
  <title>FF RAG — Data Viewer</title>
  <style>
    body { font-family: monospace; margin: 20px; background: #111; color: #eee; }
    h1 { color: #f0a500; }
    h2 { color: #aaa; }
    nav a { color: #f0a500; margin-right: 20px; text-decoration: none; font-size: 14px; }
    nav a:hover { text-decoration: underline; }
    form { margin: 12px 0; }
    input[type=text] { background: #222; color: #eee; border: 1px solid #444;
                       padding: 6px 10px; width: 320px; font-family: monospace; }
    input[type=text]::placeholder { color: #666; }
    input[type=submit], a.btn {
      background: #f0a500; color: #111; border: none; padding: 6px 14px;
      cursor: pointer; font-family: monospace; font-weight: bold;
      text-decoration: none; margin-left: 6px; }
    .meta { color: #888; font-size: 12px; margin-bottom: 10px; }
    .pager { margin: 10px 0; }
    .pager a { color: #f0a500; margin-right: 12px; }
    table { border-collapse: collapse; font-size: 12px; width: 100%; }
    th { background: #222; color: #f0a500; padding: 6px 10px;
         text-align: left; border-bottom: 1px solid #444; white-space: nowrap; }
    td { padding: 4px 10px; border-bottom: 1px solid #1e1e1e; white-space: nowrap; }
    tr:hover td { background: #1a1a1a; }
    .null { color: #555; font-style: italic; }
  </style>
</head>
<body>
<h1>FF RAG — Data Viewer</h1>
<nav>
  {% for key, label in datasets.items() %}
    <a href="/view/{{ key }}">{{ label }}</a>
  {% endfor %}
</nav>
<hr style="border-color:#333; margin: 16px 0;">
{% block content %}{% endblock %}
</body>
</html>
"""

TABLE = BASE.replace("{% block content %}{% endblock %}", """
<h2>{{ label }} {% if year %}— {{ year }}{% endif %}</h2>
<div class="meta">
  {{ total }} rows &nbsp;|&nbsp; {{ cols }} columns
  {% if available_years %}
  &nbsp;|&nbsp; Year:
  <a href="/view/{{ key }}">All</a>
  {% for y in available_years %}<a href="/view/{{ key }}?year={{ y }}">{{ y }}</a> {% endfor %}
  {% endif %}
</div>
<form method="get">
  {% if year %}<input type="hidden" name="year" value="{{ year }}">{% endif %}
  <input type="text" name="q" placeholder="Search player name..." value="{{ q }}">
  <input type="submit" value="Search">
  {% if q %}<a class="btn" href="/view/{{ key }}{% if year %}?year={{ year }}{% endif %}">Clear</a>{% endif %}
</form>
<div class="pager">
  {% if page > 0 %}<a href="?{{ prev_qs }}">&larr; Prev</a>{% endif %}
  Page {{ page + 1 }} of {{ total_pages }}
  {% if page + 1 < total_pages %}<a href="?{{ next_qs }}">Next &rarr;</a>{% endif %}
</div>
<table>
  <tr>{% for col in columns %}<th>{{ col }}</th>{% endfor %}</tr>
  {% for row in rows %}
  <tr>{% for cell in row %}
    <td>{% if cell == '' or cell == 'nan' or cell == 'None' %}<span class="null">—</span>
    {% else %}{{ cell }}{% endif %}</td>
  {% endfor %}</tr>
  {% endfor %}
</table>
<div class="pager" style="margin-top:10px">
  {% if page > 0 %}<a href="?{{ prev_qs }}">&larr; Prev</a>{% endif %}
  Page {{ page + 1 }} of {{ total_pages }}
  {% if page + 1 < total_pages %}<a href="?{{ next_qs }}">Next &rarr;</a>{% endif %}
</div>
""")


def load_dataset(key: str, year: str = None) -> Optional[pd.DataFrame]:
    path = DATA_ROOT / key
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            return None
        if year:
            files = [f for f in files if f.stem == year]
            if not files:
                return None
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    single = Path(str(path) + ".parquet")
    if single.exists():
        return pd.read_parquet(single)
    return None


def available_years(key: str) -> list:
    path = DATA_ROOT / key
    if path.is_dir():
        return [f.stem for f in sorted(path.glob("*.parquet"))]
    return []


def name_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if "name" in c.lower()]


@app.route("/")
def index():
    return render_template_string(BASE.replace(
        "{% block content %}{% endblock %}",
        "<p style='color:#888'>Select a dataset above to begin.</p>"
    ), datasets=DATASETS)


@app.route("/view/<path:key>")
def view(key):
    if key not in DATASETS:
        return "Dataset not found", 404

    year  = request.args.get("year", "")
    q     = request.args.get("q", "").strip()
    page  = int(request.args.get("page", 0))

    df = load_dataset(key, year or None)
    if df is None:
        return render_template_string(TABLE, datasets=DATASETS, label=DATASETS[key],
            key=key, year=year, q=q, page=0, total_pages=1, total=0, cols=0,
            columns=[], rows=[], available_years=available_years(key),
            prev_qs="", next_qs="")

    if q:
        nc = name_cols(df)
        if nc:
            mask = df[nc[0]].astype(str).str.contains(q, case=False, na=False)
            for c in nc[1:]:
                mask |= df[c].astype(str).str.contains(q, case=False, na=False)
            df = df[mask]

    total = len(df)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages - 1)
    chunk = df.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]

    def qs(**kwargs):
        p = {"year": year, "q": q, **kwargs}
        return "&".join(f"{k}={v}" for k, v in p.items() if v)

    return render_template_string(TABLE,
        datasets=DATASETS, label=DATASETS[key], key=key,
        year=year, q=q, page=page, total=total, total_pages=total_pages,
        cols=len(df.columns), columns=list(chunk.columns),
        rows=[[str(v) for v in row] for row in chunk.values],
        available_years=available_years(key),
        prev_qs=qs(page=page - 1), next_qs=qs(page=page + 1))


if __name__ == "__main__":
    app.run(port=8080, debug=False)
