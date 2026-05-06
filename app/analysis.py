"""Small deterministic analysis helpers for forward-looking draft questions."""

from __future__ import annotations

import re
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POSITIONS = ("QB", "RB", "WR", "TE")
TEAM_ALIASES = {
    "ARZ": "ARI",
    "JAC": "JAX",
    "KAN": "KC",
    "LA": "LAR",
    "LAR": "LA",
    "LVR": "LV",
    "NOR": "NO",
    "NWE": "NE",
    "SFO": "SF",
    "TAM": "TB",
}


class ProjectionContext:
    def __init__(self) -> None:
        self._seasonal: pd.DataFrame | None = None
        self._adp: pd.DataFrame | None = None
        self._combine: pd.DataFrame | None = None
        self._college: pd.DataFrame | None = None
        self._draft_picks: pd.DataFrame | None = None
        self._current_combine: pd.DataFrame | None = None
        self._oc_context: pd.DataFrame | None = None

    def build(self, question: str) -> str:
        if self.is_rookie_question(question):
            return self._rookie_projection(question)
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

    def _weekly_stats_df(self) -> pd.DataFrame:
        files = sorted((ROOT / "data" / "stats" / "weekly").glob("*.parquet"))
        return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    def _oc_context_df(self) -> pd.DataFrame:
        if self._oc_context is not None:
            return self._oc_context

        coaching = pd.read_parquet(ROOT / "data" / "coaching" / "coaching_weekly.parquet")
        stats = self._weekly_stats_df()
        latest_season = int(min(coaching["season"].max(), stats["season"].max()))

        latest_coaching = (
            coaching[coaching["season"].eq(latest_season) & coaching["had_game"]]
            .sort_values(["team", "week"])
            .groupby("team", as_index=False)
            .agg(
                oc_name=("oc_name", "last"),
                hc_name=("hc_name", "last"),
                team_name=("team_name", "last"),
            )
        )

        team_stats = stats[(stats["season"].eq(latest_season)) & (stats["season_type"].eq("REG"))].copy()
        for column in ["targets", "attempts", "passing_yards"]:
            team_stats[column] = pd.to_numeric(team_stats[column], errors="coerce").fillna(0)

        wr = (
            team_stats[team_stats["position"].eq("WR")]
            .groupby(["week", "recent_team", "player_id"], as_index=False)
            .agg(targets=("targets", "sum"))
            .sort_values(["week", "recent_team", "targets"], ascending=[True, True, False])
        )
        wr["wr_rank"] = wr.groupby(["week", "recent_team"]).cumcount() + 1
        wr_ranks = (
            wr[wr["wr_rank"].isin([2, 3])]
            .pivot_table(index=["week", "recent_team"], columns="wr_rank", values="targets", fill_value=0)
            .reset_index()
            .rename(columns={"recent_team": "team", 2: "wr2_targets", 3: "wr3_targets"})
        )

        team_totals = (
            team_stats.groupby(["week", "recent_team"], as_index=False)
            .agg(
                total_targets=("targets", "sum"),
                pass_attempts=("attempts", "sum"),
                pass_yards=("passing_yards", "sum"),
            )
            .rename(columns={"recent_team": "team"})
        )
        weekly = team_totals.merge(wr_ranks, on=["week", "team"], how="left").fillna(0)
        context = (
            weekly.groupby("team", as_index=False)
            .agg(
                total_targets=("total_targets", "sum"),
                pass_attempts=("pass_attempts", "sum"),
                pass_yards=("pass_yards", "sum"),
                wr2_targets=("wr2_targets", "sum"),
                wr3_targets=("wr3_targets", "sum"),
            )
        )
        context["wr2_target_share"] = context["wr2_targets"] / context["total_targets"].where(context["total_targets"] != 0)
        context["wr3_target_share"] = context["wr3_targets"] / context["total_targets"].where(context["total_targets"] != 0)
        context["secondary_wr_target_share"] = (
            context["wr2_targets"] + context["wr3_targets"]
        ) / context["total_targets"].where(context["total_targets"] != 0)
        context["pass_volume_rank"] = context["pass_attempts"].rank(ascending=False, method="min")
        context["pass_yards_rank"] = context["pass_yards"].rank(ascending=False, method="min")
        context["team_pass_yards_per_attempt"] = context["pass_yards"] / context["pass_attempts"].where(context["pass_attempts"] != 0)
        context["season"] = latest_season
        self._oc_context = context.merge(latest_coaching, on="team", how="left")
        return self._oc_context

    def _rb_oc_context_df(self) -> pd.DataFrame:
        profiles = pd.read_parquet(ROOT / "data" / "coaching" / "offensive_coordinator_profiles.parquet").copy()
        latest_season = int(profiles["season"].max())
        profiles["rb_carries_per_game"] = profiles["rb_carries"] / profiles["games_coached"].where(profiles["games_coached"] != 0)
        profiles["rb_targets_per_game"] = (
            profiles["rb_target_share"] * profiles["total_targets"]
        ) / profiles["games_coached"].where(profiles["games_coached"] != 0)

        latest = profiles[profiles["season"].eq(latest_season)].copy()
        historical = (
            profiles.groupby("oc_name", as_index=False)
            .agg(
                oc_seasons=("season", "nunique"),
                hist_rb_target_share=("rb_target_share", "mean"),
                hist_rb_carries_per_game=("rb_carries_per_game", "mean"),
                hist_rb_targets_per_game=("rb_targets_per_game", "mean"),
                hist_run_rate=("run_rate", "mean"),
            )
        )
        latest = latest.merge(historical, on="oc_name", how="left")
        latest["rb_opportunity_score"] = (
            latest["hist_rb_target_share"].fillna(latest["rb_target_share"]) * 130
            + latest["hist_rb_carries_per_game"].fillna(latest["rb_carries_per_game"]) * 2.2
            + latest["hist_run_rate"].fillna(latest["run_rate"]) * 35
        )
        return latest[
            [
                "season",
                "team",
                "oc_name",
                "rb_target_share",
                "rb_carries_per_game",
                "rb_targets_per_game",
                "hist_rb_target_share",
                "hist_rb_carries_per_game",
                "hist_rb_targets_per_game",
                "hist_run_rate",
                "oc_seasons",
                "rb_opportunity_score",
            ]
        ]

    def _offensive_trend_context_df(self) -> pd.DataFrame:
        profiles = pd.read_parquet(ROOT / "data" / "coaching" / "offensive_coordinator_profiles.parquet").copy()
        latest_season = int(profiles["season"].max())
        profiles["pass_attempts_per_game"] = profiles["pass_attempts"] / profiles["games_coached"].where(profiles["games_coached"] != 0)
        profiles["pass_yards_per_attempt"] = profiles["pass_yards"] / profiles["pass_attempts"].where(profiles["pass_attempts"] != 0)
        profiles["te_targets_per_game"] = (
            profiles["te_target_share"] * profiles["total_targets"]
        ) / profiles["games_coached"].where(profiles["games_coached"] != 0)

        latest = profiles[profiles["season"].eq(latest_season)].copy()
        historical = (
            profiles.groupby("oc_name", as_index=False)
            .agg(
                oc_seasons=("season", "nunique"),
                hist_pass_rate=("pass_rate", "mean"),
                hist_pass_attempts_per_game=("pass_attempts_per_game", "mean"),
                hist_pass_yards_per_attempt=("pass_yards_per_attempt", "mean"),
                hist_te_target_share=("te_target_share", "mean"),
                hist_te_targets_per_game=("te_targets_per_game", "mean"),
            )
        )
        latest = latest.merge(historical, on="oc_name", how="left")
        latest["qb_environment_score"] = (
            latest["hist_pass_rate"].fillna(latest["pass_rate"]) * 45
            + latest["hist_pass_attempts_per_game"].fillna(latest["pass_attempts_per_game"]) * 1.4
            + latest["hist_pass_yards_per_attempt"].fillna(latest["pass_yards_per_attempt"]) * 6
        )
        latest["te_environment_score"] = (
            latest["hist_te_target_share"].fillna(latest["te_target_share"]) * 160
            + latest["hist_te_targets_per_game"].fillna(latest["te_targets_per_game"]) * 6
            + latest["hist_pass_rate"].fillna(latest["pass_rate"]) * 25
        )
        return latest[
            [
                "season",
                "team",
                "oc_name",
                "pass_rate",
                "pass_attempts_per_game",
                "pass_yards_per_attempt",
                "te_target_share",
                "te_targets_per_game",
                "hist_pass_rate",
                "hist_pass_attempts_per_game",
                "hist_pass_yards_per_attempt",
                "hist_te_target_share",
                "hist_te_targets_per_game",
                "oc_seasons",
                "qb_environment_score",
                "te_environment_score",
            ]
        ]

    def detect_position(self, question: str) -> str | None:
        q = question.lower()
        patterns = {
            "QB": [r"\bqbs?\b", r"\bquarterbacks?\b"],
            "RB": [r"\brbs?\b", r"\brunning\s?backs?\b", r"\bhalfbacks?\b"],
            "WR": [r"\bwrs?\b", r"\bwide receivers?\b"],
            "TE": [r"\btes?\b", r"\btight\s?ends?\b"],
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

    def _rookie_projection(self, question: str = "") -> str:
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
            "cfb_player_id": "draft_cfb_id",
        })
        rookies["player_id"] = rookies["gsis_id"].fillna(rookies["pfr_player_id"]).fillna(rookies["draft_cfb_id"])

        if not current_combine.empty:
            measurables = current_combine.rename(columns={"pos": "position", "school": "combine_school", "player_name": "combine_player_name"})
            measurables = (
                measurables.sort_values(["cfb_id", "forty", "vertical"], na_position="last")
                .drop_duplicates("cfb_id")
            )
            measure_cols = ["ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
            for column in measure_cols:
                rookies[column] = pd.NA

            valid_cfb = measurables.dropna(subset=["cfb_id"])
            cfb_join = rookies[["draft_cfb_id"]].merge(
                valid_cfb[["cfb_id", *measure_cols]],
                left_on="draft_cfb_id",
                right_on="cfb_id",
                how="left",
            )
            for column in measure_cols:
                rookies[column] = rookies[column].fillna(pd.Series(cfb_join[column].values, index=rookies.index))

            missing = rookies["forty"].isna() & rookies["vertical"].isna() & rookies["broad_jump"].isna()
            valid_pfr = measurables.dropna(subset=["pfr_id"])
            if missing.any() and not valid_pfr.empty:
                pfr_join = rookies.loc[missing, ["pfr_player_id"]].merge(
                    valid_pfr[["pfr_id", *measure_cols]],
                    left_on="pfr_player_id",
                    right_on="pfr_id",
                    how="left",
                )
                for column in measure_cols:
                    fill_values = pd.Series(pfr_join[column].values, index=rookies.loc[missing].index)
                    rookies.loc[missing, column] = rookies.loc[missing, column].fillna(fill_values)

            missing = rookies["forty"].isna() & rookies["vertical"].isna() & rookies["broad_jump"].isna()
            valid_name = measurables.dropna(subset=["combine_player_name"]).drop_duplicates("combine_player_name")
            if missing.any() and not valid_name.empty:
                name_join = rookies.loc[missing, ["player_name"]].merge(
                    valid_name[["combine_player_name", *measure_cols]],
                    left_on="player_name",
                    right_on="combine_player_name",
                    how="left",
                )
                for column in measure_cols:
                    fill_values = pd.Series(name_join[column].values, index=rookies.loc[missing].index)
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

        requested_position = self.detect_position(question)
        if requested_position == "WR":
            return self._rookie_wr_projection(board, draft_year)
        if requested_position == "RB":
            return self._rookie_rb_projection(board, draft_year)
        if requested_position == "QB":
            return self._rookie_qb_projection(board, draft_year, question)
        if requested_position == "TE":
            return self._rookie_te_projection(board, draft_year)
        if requested_position in POSITIONS:
            return self._rookie_position_projection(board, draft_year, requested_position)

        return self._rookie_overall_projection(board, draft_year, question)

    def _rookie_overall_projection(self, board: pd.DataFrame, draft_year: int, question: str) -> str:
        players = board.drop_duplicates(["player_name", "draft_pick", "draft_team"]).copy()
        players["oc_team"] = players["draft_team"].replace(TEAM_ALIASES)
        oc = self._oc_context_df()
        players = players.merge(oc, left_on="oc_team", right_on="team", how="left")

        q = question.lower()
        superflex = any(token in q for token in ["superflex", "super flex", "2qb", "two qb"])
        players["fantasy_score"] = players.apply(lambda row: self._overall_fantasy_score(row, superflex), axis=1)
        players = players.sort_values("fantasy_score", ascending=False).head(24)

        lines = [
            f"Derived overall rookie fantasy board using the {draft_year} draft class.",
            f"Scoring lens: {'superflex/2QB' if superflex else '1QB PPR-style fantasy draft by default'}. This is not an NFL draft-capital ranking.",
            "Fantasy weighting: position-adjusted draft capital, college production, PPR receiving path, athletic ceiling, and landing-spot/pass-game context. QBs and TEs are discounted in normal 1QB year-one fantasy unless the prompt asks for superflex.",
            "Use these exact table headers if you render this board: Rank | Player | Position | NFL Team | Draft Pick | Fantasy Draft Read | College Production | Athletic Testing | Why",
        ]
        for rank, row in enumerate(players.itertuples(index=False), start=1):
            production = self._rookie_production_text(row)
            testing = self._athletic_text(row)
            signal = self._overall_fantasy_read(rank, row, superflex)
            why = self._overall_reason_text(row, superflex)
            lines.append(
                f"{rank}. {row.player_name} | {row.position} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{signal} | {production} | {testing} | {why}"
            )
        return "\n".join(lines)

    def _rookie_position_projection(self, board: pd.DataFrame, draft_year: int, position: str) -> str:
        players = board[board["position"].eq(position)].copy()
        if players.empty:
            return ""

        players = players.drop_duplicates(["player_name", "draft_pick", "draft_team"])
        players = players.sort_values("rookie_score", ascending=False)
        lines = [
            f"Derived 2026 rookie {position} fantasy board using all {position}s in the {draft_year} draft class.",
            "This is position-specific, not a filtered top-18 overall rookie board. Draft capital still matters most, with matched college production and combine testing as supporting signals.",
            "Use these exact table headers if you render this board: Rank | Player | NFL Team | Draft Pick | Projection Read | College Production | Athletic Testing | Why",
        ]
        for rank, row in enumerate(players.itertuples(index=False), start=1):
            production = self._rookie_production_text(row)
            testing = self._athletic_text(row)
            read = self._position_projection_read(rank, row)
            why = self._position_reason_text(row)
            lines.append(
                f"{rank}. {row.player_name} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{read} | {production} | {testing} | {why}"
            )
        return "\n".join(lines)

    def _rookie_rb_projection(self, board: pd.DataFrame, draft_year: int) -> str:
        rbs = board[board["position"].eq("RB")].copy()
        if rbs.empty:
            return ""

        rb_context = self._rb_oc_context_df()
        rbs["oc_team"] = rbs["draft_team"].replace(TEAM_ALIASES)
        rbs = rbs.merge(rb_context, left_on="oc_team", right_on="team", how="left")
        rbs["context_is_stale"] = rbs["season"].fillna(0) < draft_year - 1
        rbs["rb_environment_score"] = rbs["rb_opportunity_score"].where(~rbs["context_is_stale"], 45).fillna(45)
        rbs["ppr_receiving_score"] = (rbs["rec"] * 0.55 + rbs["rec_yds"] * 0.025).clip(upper=70)
        rbs["rb_fantasy_score"] = (
            rbs["draft_capital_score"] * 0.42
            + rbs["production_score"].clip(upper=230) * 0.24
            + rbs["athletic_score"].clip(upper=90) * 0.10
            + rbs["ppr_receiving_score"] * 0.12
            + rbs["rb_environment_score"].clip(upper=95) * 0.12
        )
        rbs.loc[rbs["draft_pick"] <= 40, "rb_fantasy_score"] += 16
        rbs.loc[rbs["draft_pick"] > 96, "rb_fantasy_score"] -= 12
        rbs.loc[rbs["draft_pick"] > 160, "rb_fantasy_score"] -= 10
        rbs = rbs.drop_duplicates(["player_name", "draft_pick", "draft_team"])
        rbs = rbs.sort_values("rb_fantasy_score", ascending=False)

        oc_season = int(rbs["season"].dropna().max()) if "season" in rbs.columns and rbs["season"].notna().any() else None
        lines = [
            f"Derived 2026 rookie RB fantasy board using all RBs in the {draft_year} draft class.",
            "Diagnosis: the RB data is present; the previous answer was a retrieval/priming issue. The query said 'runningbacks' as one word, which was not detected as RB, and the generic rookie retrieval leaned into sparse evidence.",
            f"OC caveat: local coordinator profiles currently stop at {oc_season}. When that is stale for the {draft_year} landing spot, the row shows it as a stale franchise profile and uses neutral environment weight instead of pretending it is current staff.",
            "Fantasy weighting: draft investment, rushing production, PPR receiving profile, athletic ceiling, and OC/team RB opportunity environment.",
            "Use these exact table headers if you render this board: Rank | Player | NFL Team | Draft Pick | Fantasy Read | College Production | Athletic Testing | OC RB Usage | Why",
        ]
        for rank, row in enumerate(rbs.itertuples(index=False), start=1):
            lines.append(
                f"{rank}. {row.player_name} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{self._rb_read(rank, row)} | {self._rookie_production_text(row)} | {self._athletic_text(row)} | "
                f"{self._rb_oc_text(row)} | {self._rb_reason_text(row)}"
            )
        return "\n".join(lines)

    def _rb_read(self, rank: int, row: object) -> str:
        if row.draft_pick <= 40:
            return "Priority rookie RB target"
        if row.draft_pick <= 96:
            return "Strong rookie RB target"
        if rank <= 8:
            return "Production-driven sleeper"
        return "Deep stash"

    def _rb_oc_text(self, row: object) -> str:
        if pd.isna(getattr(row, "hist_rb_target_share", pd.NA)):
            return "OC RB trend unavailable"
        prefix = "stale local franchise profile, not current staff" if getattr(row, "context_is_stale", False) else "current local team profile"
        return (
            f"{prefix}: {getattr(row, 'oc_name', 'unknown OC')}: hist RB target share {row.hist_rb_target_share:.1%}, "
            f"hist RB carries/gm {row.hist_rb_carries_per_game:.1f}, current-team RB target share {row.rb_target_share:.1%}"
        )

    def _rb_reason_text(self, row: object) -> str:
        positives = []
        cautions = []
        if row.draft_pick <= 40:
            positives.append("strong team investment")
        elif row.draft_pick <= 96:
            positives.append("day-two investment")
        else:
            cautions.append("later draft capital")
        if row.rush_yds >= 2000 or row.rush_td >= 25:
            positives.append("college rushing profile")
        if row.rec >= 45:
            positives.append("PPR receiving path")
        if pd.notna(row.forty) and row.forty <= 4.42:
            positives.append("explosive testing")
        if (
            not getattr(row, "context_is_stale", False)
            and pd.notna(getattr(row, "hist_rb_target_share", pd.NA))
            and row.hist_rb_target_share >= 0.20
        ):
            positives.append("OC history supports RB passing-game usage")
        if (
            not getattr(row, "context_is_stale", False)
            and pd.notna(getattr(row, "hist_rb_carries_per_game", pd.NA))
            and row.hist_rb_carries_per_game < 18
        ):
            cautions.append("OC history has modest RB carry volume")
        if getattr(row, "context_is_stale", False):
            cautions.append("current 2026 RB usage context not loaded")
        if getattr(row, "college_seasons", 0) <= 0:
            cautions.append("college production not matched locally")
        if not positives:
            positives.append("profile needs depth chart confirmation")
        if cautions:
            return f"{'; '.join(positives)}; caution: {', '.join(cautions)}"
        return "; ".join(positives)

    def _rookie_qb_projection(self, board: pd.DataFrame, draft_year: int, question: str) -> str:
        qbs = board[board["position"].eq("QB")].copy()
        if qbs.empty:
            return ""

        trends = self._offensive_trend_context_df()
        qbs["oc_team"] = qbs["draft_team"].replace(TEAM_ALIASES)
        qbs = qbs.merge(trends, left_on="oc_team", right_on="team", how="left")
        superflex = any(token in question.lower() for token in ["superflex", "super flex", "2qb", "two qb"])
        qbs["rushing_ceiling_score"] = (qbs["rush_yds"] * 0.025 + qbs["rush_td"] * 3.0).clip(upper=70)
        qbs["passing_resume_score"] = (qbs["pass_yds"] * 0.01 + qbs["pass_td"] * 2.0).clip(upper=210)
        qbs["qb_fantasy_score"] = (
            qbs["draft_capital_score"] * 0.42
            + qbs["passing_resume_score"] * 0.20
            + qbs["rushing_ceiling_score"] * 0.18
            + qbs["qb_environment_score"].fillna(50).clip(upper=110) * 0.20
        )
        qbs.loc[qbs["draft_pick"] <= 16, "qb_fantasy_score"] += 18
        qbs.loc[qbs["draft_pick"] > 96, "qb_fantasy_score"] -= 12
        if not superflex:
            qbs["qb_fantasy_score"] -= 20
        qbs = qbs.drop_duplicates(["player_name", "draft_pick", "draft_team"])
        qbs = qbs.sort_values("qb_fantasy_score", ascending=False)

        oc_season = int(qbs["season"].dropna().max()) if "season" in qbs.columns and qbs["season"].notna().any() else None
        lines = [
            f"Derived 2026 rookie QB fantasy board using all QBs in the {draft_year} draft class.",
            f"Scoring lens: {'superflex/2QB' if superflex else '1QB default, so QBs are discounted versus skill-position rookies but ranked properly within QB'}.",
            f"OC caveat: local pass-environment context uses the latest available coordinator profile season ({oc_season}) plus that OC's historical pass rate, pass attempts per game, and pass efficiency.",
            "Fantasy weighting: draft investment, college passing production, rushing ceiling, and pass-environment opportunity.",
            "Use these exact table headers if you render this board: Rank | Player | NFL Team | Draft Pick | Fantasy Read | College Production | Athletic Testing | OC Pass Environment | Why",
        ]
        for rank, row in enumerate(qbs.itertuples(index=False), start=1):
            lines.append(
                f"{rank}. {row.player_name} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{self._qb_read(rank, row, superflex)} | {self._rookie_production_text(row)} | {self._athletic_text(row)} | "
                f"{self._qb_oc_text(row)} | {self._qb_reason_text(row, superflex)}"
            )
        return "\n".join(lines)

    def _rookie_te_projection(self, board: pd.DataFrame, draft_year: int) -> str:
        tes = board[board["position"].eq("TE")].copy()
        if tes.empty:
            return ""

        trends = self._offensive_trend_context_df()
        tes["oc_team"] = tes["draft_team"].replace(TEAM_ALIASES)
        tes = tes.merge(trends, left_on="oc_team", right_on="team", how="left")
        tes["te_receiving_score"] = (tes["rec"] * 0.85 + tes["rec_yds"] * 0.06 + tes["rec_td"] * 5.5).clip(upper=210)
        tes["te_fantasy_score"] = (
            tes["draft_capital_score"] * 0.38
            + tes["te_receiving_score"] * 0.28
            + tes["athletic_score"].clip(upper=95) * 0.14
            + tes["te_environment_score"].fillna(42).clip(upper=100) * 0.20
        )
        tes.loc[tes["draft_pick"] <= 32, "te_fantasy_score"] += 12
        tes.loc[tes["draft_pick"] > 96, "te_fantasy_score"] -= 12
        tes["te_fantasy_score"] -= 10
        tes = tes.drop_duplicates(["player_name", "draft_pick", "draft_team"])
        tes = tes.sort_values("te_fantasy_score", ascending=False)

        oc_season = int(tes["season"].dropna().max()) if "season" in tes.columns and tes["season"].notna().any() else None
        lines = [
            f"Derived 2026 rookie TE fantasy board using all TEs in the {draft_year} draft class.",
            f"OC caveat: local TE opportunity context uses the latest available coordinator profile season ({oc_season}) plus that OC's historical TE target share and TE targets per game.",
            "Fantasy weighting: draft investment, college receiving production, athletic mismatch profile, TE-friendly offensive environment, and a year-one rookie TE discount.",
            "Use these exact table headers if you render this board: Rank | Player | NFL Team | Draft Pick | Fantasy Read | College Production | Athletic Testing | OC TE Usage | Why",
        ]
        for rank, row in enumerate(tes.itertuples(index=False), start=1):
            lines.append(
                f"{rank}. {row.player_name} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{self._te_read(rank, row)} | {self._rookie_production_text(row)} | {self._athletic_text(row)} | "
                f"{self._te_oc_text(row)} | {self._te_reason_text(row)}"
            )
        return "\n".join(lines)

    def _qb_read(self, rank: int, row: object, superflex: bool) -> str:
        if superflex and row.draft_pick <= 32:
            return "Priority superflex rookie QB"
        if row.draft_pick <= 16:
            return "Best rookie QB bet"
        if row.draft_pick <= 96:
            return "Format-dependent QB stash"
        return "Developmental QB stash"

    def _qb_oc_text(self, row: object) -> str:
        if pd.isna(getattr(row, "hist_pass_rate", pd.NA)):
            return "OC pass trend unavailable"
        return (
            f"{getattr(row, 'oc_name', 'unknown OC')}: hist pass rate {row.hist_pass_rate:.1%}, "
            f"hist pass att/gm {row.hist_pass_attempts_per_game:.1f}, hist pass Y/A {row.hist_pass_yards_per_attempt:.1f}"
        )

    def _qb_reason_text(self, row: object, superflex: bool) -> str:
        positives = []
        cautions = []
        if row.draft_pick <= 16:
            positives.append("premium QB investment")
        elif row.draft_pick <= 96:
            positives.append("drafted with plausible development path")
        else:
            cautions.append("later QB draft capital")
        if row.pass_yds >= 7000 or row.pass_td >= 60:
            positives.append("strong college passing resume")
        if row.rush_yds >= 400 or row.rush_td >= 8:
            positives.append("rushing creates fantasy ceiling")
        if pd.notna(getattr(row, "hist_pass_attempts_per_game", pd.NA)) and row.hist_pass_attempts_per_game >= 35:
            positives.append("OC history supports pass volume")
        if not superflex:
            cautions.append("1QB format limits rookie QB draft priority")
        if getattr(row, "college_seasons", 0) <= 0:
            cautions.append("college production not matched locally")
        if not positives:
            positives.append("needs role and development confirmation")
        if cautions:
            return f"{'; '.join(positives)}; caution: {', '.join(cautions)}"
        return "; ".join(positives)

    def _te_read(self, rank: int, row: object) -> str:
        if row.draft_pick <= 32:
            return "Best rookie TE bet"
        if row.draft_pick <= 96 and rank <= 6:
            return "TE-premium target"
        if row.draft_pick <= 96:
            return "Developmental TE target"
        return "Deep TE stash"

    def _te_oc_text(self, row: object) -> str:
        if pd.isna(getattr(row, "hist_te_target_share", pd.NA)):
            return "OC TE trend unavailable"
        return (
            f"{getattr(row, 'oc_name', 'unknown OC')}: hist TE target share {row.hist_te_target_share:.1%}, "
            f"hist TE targets/gm {row.hist_te_targets_per_game:.1f}, current-team TE target share {row.te_target_share:.1%}"
        )

    def _te_reason_text(self, row: object) -> str:
        positives = []
        cautions = ["rookie TE year-one production is volatile"]
        if row.draft_pick <= 32:
            positives.append("premium TE investment")
        elif row.draft_pick <= 96:
            positives.append("day-two investment")
        else:
            cautions.append("later draft capital")
        if row.rec_yds >= 1000 or row.rec_td >= 8:
            positives.append("receiving profile translates")
        if pd.notna(row.forty) and row.forty <= 4.60:
            positives.append("athletic mismatch upside")
        if pd.notna(getattr(row, "hist_te_target_share", pd.NA)) and row.hist_te_target_share >= 0.22:
            positives.append("OC history supports TE targets")
        if getattr(row, "college_seasons", 0) <= 0:
            cautions.append("college production not matched locally")
        if not positives:
            positives.append("needs route role confirmation")
        return f"{'; '.join(positives)}; caution: {', '.join(cautions)}"

    def _overall_fantasy_score(self, row: object, superflex: bool) -> float:
        pick = float(getattr(row, "draft_pick", 260) or 260)
        draft = max(0.0, 130 - pick * 0.85)
        production = min(float(getattr(row, "production_score", 0) or 0), 240)
        athletic = min(float(getattr(row, "athletic_score", 0) or 0), 95)
        rec = float(getattr(row, "rec", 0) or 0)
        rec_yds = float(getattr(row, "rec_yds", 0) or 0)
        pass_yards_rank = getattr(row, "pass_yards_rank", pd.NA)
        secondary_wr_share = getattr(row, "secondary_wr_target_share", pd.NA)

        if row.position == "RB":
            score = draft * 0.42 + production * 0.30 + athletic * 0.10 + min(rec * 0.45, 35)
            if pick <= 40:
                score += 18
            elif pick > 160:
                score -= 22
            elif pick > 96:
                score -= 14
            if rec >= 50:
                score += 10
            if pd.notna(row.forty) and row.forty <= 4.42:
                score += 8
            return score

        if row.position == "WR":
            landing = 0.0
            if pd.notna(secondary_wr_share):
                landing += min(float(secondary_wr_share) * 125, 40)
            if pd.notna(pass_yards_rank):
                landing += max(0, 24 - float(pass_yards_rank)) * 1.1
                if float(pass_yards_rank) >= 26:
                    landing -= 12
            score = draft * 0.36 + production * 0.30 + athletic * 0.12 + landing * 0.22
            if pick <= 12:
                score += 14
            if rec_yds >= 2200 or getattr(row, "rec_td", 0) >= 20:
                score += 10
            if getattr(row, "draft_team", "") == "CLE":
                score -= 10
            return score

        if row.position == "TE":
            score = draft * 0.34 + production * 0.28 + athletic * 0.18
            if pick <= 32:
                score += 8
            score -= 16
            return score

        if row.position == "QB":
            score = draft * 0.30 + production * 0.18
            if getattr(row, "rush_yds", 0) >= 400 or getattr(row, "rush_td", 0) >= 8:
                score += 12
            if pick <= 16:
                score += 15
            score += 42 if superflex else -34
            return score

        return draft * 0.4 + production * 0.25 + athletic * 0.1

    def _overall_fantasy_read(self, rank: int, row: object, superflex: bool) -> str:
        if row.draft_pick > 96 and rank <= 12:
            return "Production-driven sleeper with draft-capital caveat"
        if rank <= 5:
            return "Priority fantasy rookie target"
        if rank <= 12:
            return "Strong rookie draft target"
        if row.position == "QB" and not superflex:
            return "Format-dependent QB stash"
        return "Upside stash"

    def _overall_reason_text(self, row: object, superflex: bool) -> str:
        positives = []
        cautions = []
        if row.draft_pick <= 32:
            positives.append("strong draft investment")
        elif row.draft_pick <= 96:
            positives.append("day-two investment")
        else:
            cautions.append("later draft capital")

        if row.position == "RB":
            if row.rush_yds >= 2000 or row.rush_td >= 25:
                positives.append("rushing production translates")
            if row.rec >= 45:
                positives.append("PPR receiving path")
            if pd.notna(row.forty) and row.forty <= 4.42:
                positives.append("explosive athletic profile")
        elif row.position == "WR":
            if row.rec_yds >= 2200 or row.rec_td >= 20:
                positives.append("strong college receiving profile")
            if pd.notna(row.forty) and row.forty <= 4.40:
                positives.append("speed ceiling")
            if pd.notna(getattr(row, "secondary_wr_target_share", pd.NA)) and row.secondary_wr_target_share >= 0.27:
                positives.append("landing spot has fed secondary WRs")
            if pd.notna(getattr(row, "pass_yards_rank", pd.NA)) and row.pass_yards_rank >= 26:
                cautions.append("weak recent passing environment")
            if row.draft_team == "CLE":
                cautions.append("Cleveland QB/pass-game risk")
        elif row.position == "TE":
            if row.rec_yds >= 900 or row.rec_td >= 8:
                positives.append("real receiving resume")
            if pd.notna(row.forty) and row.forty <= 4.60:
                positives.append("athletic mismatch upside")
            cautions.append("rookie TE year-one hit rates are lower")
        elif row.position == "QB":
            if superflex:
                positives.append("superflex format boosts QB value")
            else:
                cautions.append("QB discounted in 1QB fantasy")
            if row.rush_yds >= 400 or row.rush_td >= 8:
                positives.append("rushing adds fantasy ceiling")

        if getattr(row, "college_seasons", 0) <= 0:
            cautions.append("college production not matched locally")
        if not positives:
            positives.append("profile has usable fantasy traits")
        if cautions:
            return f"{'; '.join(positives)}; caution: {', '.join(cautions)}"
        return "; ".join(positives)

    def _rookie_wr_projection(self, board: pd.DataFrame, draft_year: int) -> str:
        wrs = board[board["position"].eq("WR")].copy()
        if wrs.empty:
            return ""

        oc = self._oc_context_df()
        wrs["oc_team"] = wrs["draft_team"].replace(TEAM_ALIASES)
        wrs = wrs.merge(oc, left_on="oc_team", right_on="team", how="left")
        wrs["secondary_wr_share"] = wrs["secondary_wr_target_share"].fillna(0)
        wrs["pass_volume_component"] = (33 - wrs["pass_volume_rank"].fillna(33)).clip(lower=0)
        wrs["pass_efficiency_component"] = (33 - wrs["pass_yards_rank"].fillna(33)).clip(lower=0)
        wrs["opportunity_score"] = (
            wrs["secondary_wr_share"] * 160
            + wrs["pass_volume_component"] * 1.8
            + wrs["pass_efficiency_component"] * 1.0
        )
        wrs["wr_projection_score"] = (
            wrs["draft_capital_score"] * 0.45
            + wrs["production_score"].clip(upper=220) * 0.25
            + wrs["athletic_score"].clip(upper=90) * 0.10
            + wrs["opportunity_score"].clip(upper=95) * 0.20
        )
        wrs.loc[wrs["draft_pick"] <= 12, "wr_projection_score"] += 15
        wrs.loc[wrs["draft_pick"] > 64, "wr_projection_score"] -= 8
        wrs = wrs.drop_duplicates(["player_name", "draft_pick", "draft_team"])
        wrs = wrs.sort_values("wr_projection_score", ascending=False)

        oc_season = int(wrs["season"].dropna().max()) if "season" in wrs.columns and wrs["season"].notna().any() else None
        lines = [
            f"Derived 2026 rookie WR fantasy board using the {draft_year} draft class.",
            "No: this is not simply the first wide receivers drafted. Draft capital still matters, but this board also includes matched college production, combine testing, and the player's NFL team's latest local OC/pass-game environment.",
            f"Coordinator caveat: OC context is from the latest local coaching/stat season ({oc_season}) and should be refreshed when 2026 staffs/roles are finalized.",
            "Opportunity heuristic: a high WR2+WR3 target share helps a rookie who may open behind an established WR1; weak pass volume or passing efficiency is a downgrade even when draft capital is strong.",
            "Use these exact table headers if you render this board: Rank | Player | NFL Team | Draft Pick | Projection Read | College Production | Athletic Testing | OC/Secondary WR Usage | Pass Environment | Why",
        ]
        for rank, row in enumerate(wrs.head(14).itertuples(index=False), start=1):
            production = self._rookie_production_text(row)
            testing = self._athletic_text(row)
            read = self._wr_projection_read(rank, row)
            oc_text = self._oc_usage_text(row)
            pass_text = self._pass_environment_text(row)
            why = self._wr_reason_text(row)
            lines.append(
                f"{rank}. {row.player_name} | {row.draft_team} | pick {int(row.draft_pick)} | "
                f"{read} | {production} | {testing} | {oc_text} | {pass_text} | {why}"
            )
        return "\n".join(lines)

    def _wr_projection_read(self, rank: int, row: object) -> str:
        if rank <= 3:
            return "Best bet for year-one fantasy relevance"
        if rank <= 7:
            return "Useful rookie draft target with usage caveat"
        return "Upside stash or landing-spot dependent"

    def _position_projection_read(self, rank: int, row: object) -> str:
        if row.draft_pick <= 32:
            return "Immediate fantasy priority"
        if row.draft_pick <= 96:
            return "Rookie draft target"
        if rank <= 8:
            return "Depth chart watchlist"
        return "Deep stash"

    def _position_reason_text(self, row: object) -> str:
        positives = []
        cautions = []
        if row.draft_pick <= 32:
            positives.append("premium draft capital")
        elif row.draft_pick <= 96:
            positives.append("day-two draft capital")
        else:
            cautions.append("later draft capital")
        if row.position == "RB":
            if row.rush_yds >= 2000 or row.rush_td >= 25:
                positives.append("strong rushing production")
            if row.rec >= 40:
                positives.append("receiving profile helps PPR path")
            if pd.notna(row.forty) and row.forty <= 4.45:
                positives.append("plus speed")
        elif row.position == "TE":
            if row.rec_yds >= 800 or row.rec_td >= 8:
                positives.append("usable receiving production")
            if pd.notna(row.forty) and row.forty <= 4.60:
                positives.append("athletic upside")
        elif row.position == "QB":
            if row.pass_yds >= 6000 or row.pass_td >= 50:
                positives.append("multi-year passing production")
            if row.rush_yds >= 400 or row.rush_td >= 8:
                positives.append("rushing adds fantasy ceiling")
        if getattr(row, "college_seasons", 0) <= 0:
            cautions.append("college production not matched locally")
        if not positives:
            positives.append("profile needs depth chart confirmation")
        if cautions:
            return f"{'; '.join(positives)}; caution: {', '.join(cautions)}"
        return "; ".join(positives)

    def _oc_usage_text(self, row: object) -> str:
        if pd.isna(getattr(row, "secondary_wr_target_share", pd.NA)):
            return "OC secondary-WR usage unavailable"
        return (
            f"{getattr(row, 'oc_name', 'unknown OC')}: WR2 {row.wr2_target_share:.1%}, "
            f"WR3 {row.wr3_target_share:.1%}, WR2+WR3 {row.secondary_wr_target_share:.1%}"
        )

    def _pass_environment_text(self, row: object) -> str:
        if pd.isna(getattr(row, "pass_volume_rank", pd.NA)):
            return "team pass environment unavailable"
        ypa = getattr(row, "team_pass_yards_per_attempt", None)
        ypa_text = f", {ypa:.1f} pass Y/A" if pd.notna(ypa) else ""
        return f"pass volume rank {int(row.pass_volume_rank)}, pass yards rank {int(row.pass_yards_rank)}{ypa_text}"

    def _wr_reason_text(self, row: object) -> str:
        positives = []
        cautions = []
        if row.draft_pick <= 12:
            positives.append("premium draft capital")
        elif row.draft_pick <= 40:
            positives.append("top-40 draft capital")
        else:
            cautions.append("less insulated draft capital")
        if row.rec_yds >= 2200 or row.rec_td >= 20:
            positives.append("strong college production")
        if pd.notna(row.forty) and row.forty <= 4.40:
            positives.append("speed separates")
        if pd.notna(getattr(row, "secondary_wr_target_share", pd.NA)) and row.secondary_wr_target_share >= 0.27:
            positives.append("OC has fed WR2/WR3 targets")
        if pd.notna(getattr(row, "pass_yards_rank", pd.NA)) and row.pass_yards_rank >= 24:
            cautions.append("poor recent passing environment")
        if row.draft_team == "CLE":
            cautions.append("CLE landing spot needs QB/pass-game improvement")
        if not positives:
            positives.append("profile has usable traits")
        if cautions:
            return f"{'; '.join(positives)}; caution: {', '.join(cautions)}"
        return "; ".join(positives)

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
