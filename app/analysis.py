"""Small deterministic analysis helpers for forward-looking draft questions."""

from __future__ import annotations

import re
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POSITIONS = ("QB", "RB", "WR", "TE")


class ProjectionContext:
    def __init__(self) -> None:
        self._seasonal: pd.DataFrame | None = None
        self._adp: pd.DataFrame | None = None
        self._combine: pd.DataFrame | None = None
        self._college: pd.DataFrame | None = None
        self._draft_picks: pd.DataFrame | None = None
        self._current_combine: pd.DataFrame | None = None

    def build(self, question: str) -> str:
        if self.is_rookie_question(question):
            return self._rookie_projection()
        position = self.detect_position(question)
        if not position or not self.is_projection_question(question):
            return ""
        return self._position_projection(position)

    def _seasonal_df(self) -> pd.DataFrame:
        if self._seasonal is None:
            files = sorted((ROOT / "data" / "stats" / "seasonal").glob("*.parquet"))
            self._seasonal = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        return self._seasonal

    def _adp_df(self) -> pd.DataFrame:
        if self._adp is None:
            self._adp = pd.read_parquet(ROOT / "data" / "adp" / "adp_historical.parquet")
        return self._adp

    def _combine_df(self) -> pd.DataFrame:
        if self._combine is None:
            self._combine = pd.read_parquet(ROOT / "data" / "athletic" / "combine_draft.parquet")
        return self._combine

    def _draft_picks_df(self) -> pd.DataFrame:
        if self._draft_picks is None:
            self._draft_picks = nfl.import_draft_picks([2026])
        return self._draft_picks

    def _current_combine_df(self) -> pd.DataFrame:
        if self._current_combine is None:
            self._current_combine = nfl.import_combine_data(
                years=[2026],
                positions=list(POSITIONS),
            )
        return self._current_combine

    def _college_df(self) -> pd.DataFrame:
        if self._college is None:
            self._college = pd.read_parquet(ROOT / "data" / "athletic" / "college_stats.parquet")
        return self._college

    def detect_position(self, question: str) -> str | None:
        q = question.lower()
        patterns = {
            "QB": [r"\bqbs?\b", r"\bquarterbacks?\b"],
            "RB": [r"\brbs?\b", r"\brunning backs?\b"],
            "WR": [r"\bwrs?\b", r"\bwide receivers?\b"],
            "TE": [r"\btes?\b", r"\btight ends?\b"],
        }
        for position, position_patterns in patterns.items():
            if any(re.search(pattern, q) for pattern in position_patterns):
                return position
        return None

    def is_projection_question(self, question: str) -> bool:
        q = question.lower()
        return any(token in q for token in ["2026", "upcoming", "next season", "draft", "rank", "top", "best", "tier"])

    def is_rookie_question(self, question: str) -> bool:
        q = question.lower()
        return any(token in q for token in ["rookie", "rookies", "prospect", "prospects", "athletic testing", "college stats"])

    def _position_projection(self, position: str) -> str:
        seasonal = self._seasonal_df()
        latest_season = int(seasonal["season"].max())
        lookback_start = max(latest_season - 2, int(seasonal["season"].min()))

        rows = seasonal[
            (seasonal["position"].eq(position))
            & (seasonal["season"].between(lookback_start, latest_season))
        ].copy()
        if rows.empty:
            return ""

        name_col = "player_display_name" if "player_display_name" in rows.columns else "player_name"
        rows["points"] = rows["fantasy_points_ppr"].fillna(rows["fantasy_points"])
        rows["games"] = rows["games"].fillna(0)

        latest = (
            rows[rows["season"].eq(latest_season)]
            .sort_values("points", ascending=False)
            .drop_duplicates("player_id")
        )
        recent = (
            rows.groupby("player_id", dropna=False)
            .agg(
                player_name=(name_col, "last"),
                seasons=("season", "nunique"),
                recent_points=("points", "mean"),
                recent_games=("games", "mean"),
            )
            .reset_index()
        )

        adp = self._adp_df()
        latest_adp = adp[
            (adp["position"].eq(position))
            & (adp["season"].eq(latest_season))
            & (adp["format"].isin(["ppr", "std", "half_ppr"]))
        ]
        adp_rank = (
            latest_adp.groupby("player_id", dropna=False)
            .agg(adp=("avg_adp", "mean"))
            .reset_index()
        )

        board = latest.merge(recent, on="player_id", how="left", suffixes=("", "_recent"))
        board = board.merge(adp_rank, on="player_id", how="left")
        board["adp_score"] = board["adp"].apply(lambda value: max(0, 220 - value) if pd.notna(value) else 80)
        board["projection_score"] = (
            board["points"] * 0.58
            + board["recent_points"].fillna(board["points"]) * 0.27
            + board["adp_score"] * 0.15
        )
        board = board.sort_values("projection_score", ascending=False).head(16)

        lines = [
            f"Derived forward-looking {position} board for 2026 using latest available data.",
            f"Basis: {latest_season} actual fantasy points, {lookback_start}-{latest_season} actual recent production, and {latest_season} ADP when available.",
            "This is a projection aid, not a true 2026 projection feed.",
            "Use these exact table headers if you render this board: Rank | Player | 2026 Projection Tier | 2025 Actual Points | 2023-2025 Avg Actual Points | 2025 ADP | Key 2025 Actual Stats",
        ]
        for rank, row in enumerate(board.itertuples(index=False), start=1):
            passing = f"{int(getattr(row, 'passing_yards', 0) or 0)} pass yds, {int(getattr(row, 'passing_tds', 0) or 0)} pass TD"
            rushing = f"{int(getattr(row, 'rushing_yards', 0) or 0)} rush yds, {int(getattr(row, 'rushing_tds', 0) or 0)} rush TD"
            adp_value = getattr(row, "adp", None)
            adp_text = f"{adp_value:.1f}" if pd.notna(adp_value) else "n/a"
            tier = "Elite QB1" if rank <= 3 else "QB1" if rank <= 10 else "Upside QB2"
            lines.append(
                f"{rank}. {getattr(row, name_col)} | {tier} | {getattr(row, 'points'):.1f} | "
                f"{getattr(row, 'recent_points'):.1f} | {adp_text} | {passing}; {rushing}"
            )
        return "\n".join(lines)

    def _rookie_projection(self) -> str:
        draft_picks = self._draft_picks_df()
        current_combine = self._current_combine_df()
        college = self._college_df()

        draft_year = int(draft_picks["season"].max())
        rookies = draft_picks[
            draft_picks["season"].eq(draft_year)
            & draft_picks["position"].isin(POSITIONS)
        ].copy()
        if rookies.empty:
            return self._fallback_previous_rookie_projection()

        rookies = rookies.rename(columns={
            "season": "draft_year",
            "round": "draft_round",
            "pick": "draft_pick",
            "team": "draft_team",
            "pfr_player_name": "player_name",
        })
        rookies["player_id"] = rookies["gsis_id"].fillna(rookies["pfr_player_id"]).fillna(rookies["cfb_player_id"])

        if not current_combine.empty:
            measurables = current_combine.rename(columns={"pos": "position", "school": "combine_school"})
            measurables = (
                measurables.sort_values(["cfb_id", "forty", "vertical"], na_position="last")
                .drop_duplicates("cfb_id")
            )
            rookies = rookies.merge(
                measurables[["cfb_id", "pfr_id", "ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]],
                left_on="cfb_player_id",
                right_on="cfb_id",
                how="left",
            )
            missing = rookies["forty"].isna() & rookies["vertical"].isna() & rookies["broad_jump"].isna()
            if missing.any():
                by_pfr = measurables[["pfr_id", "ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]]
                pfr_join = rookies.loc[missing, ["pfr_player_id"]].merge(
                    by_pfr,
                    left_on="pfr_player_id",
                    right_on="pfr_id",
                    how="left",
                )
                for column in ["ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]:
                    fill_values = pd.Series(pfr_join[column].values, index=rookies.loc[missing].index)
                    rookies.loc[missing, column] = rookies.loc[missing, column].fillna(fill_values)

        college_rows = college[
            college["player_id"].isin(set(rookies["player_id"].dropna()))
            | college["player_name"].isin(set(rookies["player_name"].dropna()))
        ].copy()
        for column in ["pass_yds", "pass_td", "rush_yds", "rush_td", "rec", "rec_yds", "rec_td"]:
            if column in college_rows.columns:
                college_rows[column] = pd.to_numeric(college_rows[column], errors="coerce").fillna(0)
        college_agg = (
            college_rows.groupby("player_id", dropna=False)
            .agg(
                college_seasons=("season", "nunique"),
                pass_yds=("pass_yds", "sum"),
                pass_td=("pass_td", "sum"),
                rush_yds=("rush_yds", "sum"),
                rush_td=("rush_td", "sum"),
                rec=("rec", "sum"),
                rec_yds=("rec_yds", "sum"),
                rec_td=("rec_td", "sum"),
            )
            .reset_index()
        )

        board = rookies.merge(college_agg, on="player_id", how="left")
        for column in ["pass_yds", "pass_td", "rush_yds", "rush_td", "rec", "rec_yds", "rec_td", "college_seasons"]:
            board[column] = board[column].fillna(0)

        board["draft_capital_score"] = board["draft_pick"].apply(lambda pick: max(0, 260 - float(pick or 260)))
        board["production_score"] = (
            board["pass_yds"] * 0.015
            + board["pass_td"] * 3.0
            + board["rush_yds"] * 0.035
            + board["rush_td"] * 4.0
            + board["rec"] * 1.0
            + board["rec_yds"] * 0.055
            + board["rec_td"] * 5.0
        )
        board["athletic_score"] = 0.0
        board.loc[board["forty"].notna(), "athletic_score"] += (4.75 - board["forty"].clip(upper=4.75)) * 80
        board.loc[board["vertical"].notna(), "athletic_score"] += (board["vertical"] - 30).clip(lower=0) * 2
        board.loc[board["broad_jump"].notna(), "athletic_score"] += (board["broad_jump"] - 112).clip(lower=0) * 1.2
        board["rookie_score"] = (
            board["draft_capital_score"] * 0.80
            + board["production_score"].clip(upper=180) * 0.20
            + board["athletic_score"].clip(upper=80) * 0.10
        )
        board.loc[board["draft_pick"] <= 8, "rookie_score"] += 35
        board.loc[board["draft_pick"] <= 32, "rookie_score"] += 15
        board.loc[board["position"].eq("QB") & (board["draft_pick"] <= 16), "rookie_score"] += 18
        board = board.drop_duplicates(["player_name", "draft_pick", "draft_team"])
        board = board.sort_values("rookie_score", ascending=False).head(18)

        lines = [
            f"Derived rookie fantasy value board using the {draft_year} draft class.",
            "Basis: 2026 draft capital is weighted most heavily, then capped matched college production, then 2026 combine/athletic testing.",
            "Historical fantasy trend heuristic: early draft capital plus strong college production is more predictive than athletic testing alone; athletic testing is a tie-breaker and upside flag. Cumulative college production is capped so older multi-year starters do not dominate solely by longevity.",
            "Use these exact table headers if you render this board: Rank | Player | Position | NFL Team | Draft Pick | Fantasy Signal | College Production | Athletic Testing",
        ]
        for rank, row in enumerate(board.itertuples(index=False), start=1):
            production = self._rookie_production_text(row)
            testing = self._athletic_text(row)
            signal = "Immediate fantasy priority" if rank <= 6 else "Strong watchlist" if rank <= 12 else "Upside stash"
            lines.append(
                f"{rank}. {row.player_name} | {row.position} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{signal} | {production} | {testing}"
            )
        return "\n".join(lines)

    def _fallback_previous_rookie_projection(self) -> str:
        combine = self._combine_df()
        latest_year = int(combine["draft_year"].dropna().max())
        return (
            f"True 2026 rookie draft data is not loaded. The local historical combine table only reaches {latest_year}. "
            f"Do not call the {latest_year} draft class rookies for 2026; treat them as second-year players."
        )

    def _rookie_production_text(self, row: object) -> str:
        if getattr(row, "college_seasons", 0) <= 0:
            return "college production not matched in local dataset"
        if row.position == "QB":
            return f"{int(row.pass_yds)} pass yds, {int(row.pass_td)} pass TD; {int(row.rush_yds)} rush yds, {int(row.rush_td)} rush TD"
        if row.position == "RB":
            return f"{int(row.rush_yds)} rush yds, {int(row.rush_td)} rush TD; {int(row.rec)} rec, {int(row.rec_yds)} rec yds"
        return f"{int(row.rec)} rec, {int(row.rec_yds)} rec yds, {int(row.rec_td)} rec TD"

    def _athletic_text(self, row: object) -> str:
        parts = []
        if pd.notna(row.forty):
            parts.append(f"{row.forty:.2f} forty")
        if pd.notna(row.vertical):
            parts.append(f"{row.vertical:.1f} vertical")
        if pd.notna(row.broad_jump):
            parts.append(f"{int(row.broad_jump)} broad")
        return ", ".join(parts) if parts else "limited/no combine testing"
