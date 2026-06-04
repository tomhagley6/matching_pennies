"""
Parse the two oTree "all_apps_wide" exports from the Duan retreat 2026 games
into tidy pandas DataFrames ready for analysis.

Two games, two source files
---------------------------
matching_pennies (a.k.a. the "cursor game", oTree app ``matching_retreat_1``)
    Two players move a cursor over a screen split into LEFT / MIDDLE / RIGHT
    zones. The trial ends at a uniformly-random time; whichever zone each
    cursor is in at that moment is the player's choice (L / R, or NONE if
    still in the middle). The *matcher* is rewarded when the two choices
    match, the *avoider* when they mismatch.

corridor (oTree app ``matching_blocks``)
    Two players are squares advancing towards each other along one of two
    lanes (0 / 1) and may switch lane (rate-limited). Control is locked at a
    uniformly-random "control cutoff" time, after which the final lanes are
    fixed. The *matcher* is rewarded on a collision (same lane = pass through
    each other -> collide), the *avoider* on a pass (different lanes).

In both files the row granularity is one row per *participant* (2 players),
and the real trial-by-trial data lives in group-level JSON columns:
    matching_retreat_1.1.group.trial_log_json   - one record per trial
    matching_retreat_1.1.group.cursor_log_json  - cursor trajectory samples
    matching_blocks.1.group.trial_log_json      - one record per trial
    matching_blocks.1.group.command_log_json    - every control message

Public API
----------
    build_all(mp_path, corridor_path) -> dict of DataFrames:
        'trials'         one row per trial, both games stacked (unified schema)
        'cursor_samples' one row per cursor sample (matching_pennies only)
        'commands'       one row per control message (corridor only)

Run as a script to parse the two CSVs in this folder and write parquet/csv
copies of each table next to them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MP_FILE = HERE / "all_apps_wide-2026-05-29-cursor-game-roshni-vs-orsi.csv"
CORRIDOR_FILE = HERE / "all_apps_wide-2026-06-01-corridor-game-bene-vs-chen-chen.csv"

MP_APP = "matching_retreat_1"
CORRIDOR_APP = "matching_blocks"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_raw(path) -> pd.DataFrame:
    """Read an all_apps_wide CSV (utf-8-sig handles the BOM on one file)."""
    return pd.read_csv(path, encoding="utf-8-sig")

def _gjson(raw: pd.DataFrame, app: str, field: str):
    """Parse a group-level JSON column (identical across the group's rows)."""
    col = f"{app}.1.group.{field}"
    val = raw[col].iloc[0]
    return json.loads(val) if pd.notna(val) else []

def _session_code(raw: pd.DataFrame) -> str:
    return raw["session.code"].iloc[0]


# --------------------------------------------------------------------------- #
# matching pennies / cursor game
# --------------------------------------------------------------------------- #
def parse_matching_pennies(path=MP_FILE):
    """Return (trials, cursor_samples) for the cursor / matching-pennies game.

    Role convention (verified against the reward structure): the *matcher* is
    rewarded on 'match', the *avoider* on 'mismatch'. p1 is the matcher and p2
    the avoider in this dataset, but we derive it from the data rather than
    assume it.
    """
    raw = _load_raw(path)
    session = _session_code(raw)
    log = _gjson(raw, MP_APP, "trial_log_json")

    # derive which player id is the matcher from a decisive trial
    matcher_pid = 1
    for t in log:
        if t["outcome_reason"] == "match":
            matcher_pid = 1 if t["p1_reward"] > 0 else 2
            break

    def role_cols(t, role_p1, role_p2):
        return (t[role_p1], t[role_p2]) if matcher_pid == 1 else (t[role_p2], t[role_p1])

    rows = []
    for t in log:
        m_choice, a_choice = role_cols(t, "p1_choice", "p2_choice")
        m_valid, a_valid = role_cols(t, "p1_valid_choice", "p2_valid_choice")
        m_rew, a_rew = role_cols(t, "p1_reward", "p2_reward")
        m_tot, a_tot = role_cols(t, "p1_total_points_after", "p2_total_points_after")
        m_rt, a_rt = role_cols(t, "p1_rt_ms", "p2_rt_ms")
        m_code, a_code = role_cols(t, "p1_code", "p2_code")
        rows.append(dict(
            game="matching_pennies",
            session_code=session,
            trial=t["overall_trial"],
            block=t["block"],
            opponent_type=t["opponent_type"],
            matcher_code=m_code,
            avoider_code=a_code,
            matcher_action=m_choice,            # 'L' / 'R' / 'NONE'
            avoider_action=a_choice,
            matcher_valid=bool(m_valid),
            avoider_valid=bool(a_valid),
            matched=(t["outcome_reason"] == "match"),
            matcher_won=m_rew > 0,
            avoider_won=a_rew > 0,
            matcher_reward=m_rew,
            avoider_reward=a_rew,
            matcher_total=m_tot,
            avoider_total=a_tot,
            outcome_reason=t["outcome_reason"],
            random_time_ms=t["timer_ms"],       # uniformly-random trial duration
            matcher_rt_ms=m_rt,
            avoider_rt_ms=a_rt,
            server_ts=t["server_ts"],
        ))
    trials = pd.DataFrame(rows)

    # --- cursor trajectory samples: one row per (trial, player, sample) ---
    role_of = {matcher_pid: "matcher", (3 - matcher_pid): "avoider"}
    samp_rows = []
    for entry in _gjson(raw, MP_APP, "cursor_log_json"):
        pid = entry["player_id"]
        for s in entry["samples"]:
            samp_rows.append(dict(
                game="matching_pennies",
                session_code=session,
                trial=entry["trial"],
                player_id=pid,
                participant_code=entry["participant_code"],
                role=role_of.get(pid),
                t_ms=s["t_ms"],          # ms since trial start
                x=s["x"],                # normalised 0..1 (0=left, 1=right)
                y=s["y"],
                zone=s["zone"],          # 'L' / 'R' / 'NONE'
            ))
    cursor_samples = pd.DataFrame(samp_rows).sort_values(
        ["trial", "player_id", "t_ms"]).reset_index(drop=True)
    return trials, cursor_samples


# --------------------------------------------------------------------------- #
# corridor game
# --------------------------------------------------------------------------- #
def parse_corridor(path=CORRIDOR_FILE):
    """Return (trials, commands) for the corridor / matching-blocks game.

    Role names are recorded per player (``role_name`` = matcher / avoider);
    p1 is the matcher here. matcher wins on collision, avoider on a pass.
    """
    raw = _load_raw(path)
    session = _session_code(raw)
    log = _gjson(raw, CORRIDOR_APP, "trial_log_json")

    # map player id -> role from the player-level columns
    pid = raw[f"{CORRIDOR_APP}.1.player.id_in_group"]
    rolename = raw[f"{CORRIDOR_APP}.1.player.role_name"]
    role_by_pid = dict(zip(pid, rolename))
    code_by_pid = dict(zip(pid, raw["participant.code"]))
    matcher_pid = next((p for p, r in role_by_pid.items() if r == "matcher"), 1)

    def role_cols(t, key_p1, key_p2):
        return (t[key_p1], t[key_p2]) if matcher_pid == 1 else (t[key_p2], t[key_p1])

    rows = []
    for t in log:
        m_final, a_final = role_cols(t, "p1_final_lane", "p2_final_lane")
        m_start, a_start = role_cols(t, "p1_start_lane", "p2_start_lane")
        m_rew, a_rew = role_cols(t, "p1_reward", "p2_reward")
        m_tot, a_tot = role_cols(t, "p1_total_reward", "p2_total_reward")
        rows.append(dict(
            game="corridor",
            session_code=session,
            trial=t["trial_index"],
            block="multi",
            opponent_type="human",
            matcher_code=code_by_pid.get(matcher_pid),
            avoider_code=code_by_pid.get(3 - matcher_pid),
            matcher_action=m_final,             # final lane 0 / 1
            avoider_action=a_final,
            matcher_start_lane=m_start,
            avoider_start_lane=a_start,
            matcher_valid=True,
            avoider_valid=True,
            matched=bool(t["collision"]),       # collision == same lane == match
            matcher_won=m_rew > 0,
            avoider_won=a_rew > 0,
            matcher_reward=m_rew,
            avoider_reward=a_rew,
            matcher_total=m_tot,
            avoider_total=a_tot,
            outcome_reason="collision" if t["collision"] else "pass",
            random_time_ms=t["p1_control_cutoff_ms"],  # uniformly-random control lock
            shared_control_lock=bool(t["shared_control_lock"]),
            total_pause_ms=t["total_pause_ms"],
            server_ts=t["server_ts_ms"] / 1000.0,
        ))
    trials = pd.DataFrame(rows)

    # --- command log: one row per control message ---
    cmd = _gjson(raw, CORRIDOR_APP, "command_log_json")
    cmd_rows = []
    for c in cmd:
        msg = c["msg"]
        cmd_rows.append(dict(
            game="corridor",
            session_code=session,
            trial=c["trial_index"],
            player_id=c["player_id"],
            role=c["role"],
            participant_code=code_by_pid.get(c["player_id"]),
            server_ts_ms=c["server_ts_ms"],
            msg_type=msg["type"],                # position / final_choice / ready / ...
            lane=msg.get("lane"),
        ))
    commands = pd.DataFrame(cmd_rows)
    commands = commands.sort_values(["trial", "server_ts_ms"]).reset_index(drop=True)

    # Within-trial clock. Anchor each trial to its first gameplay command
    # (a 'position' update, or 'final_choice' if the player never moved); the
    # 'next_trial' marker is logged at the *end* of a trial, not its start.
    # This makes t_rel_ms broadly comparable to control_cutoff_ms. Pre-game
    # setup commands ('set_game_settings'/'ready') get negative t_rel_ms.
    def _trial_start(g):
        play = g.loc[g["msg_type"].isin(["position", "final_choice"]), "server_ts_ms"]
        return play.min() if len(play) else g["server_ts_ms"].min()

    starts = commands.groupby("trial", group_keys=False).apply(_trial_start)
    commands["t_rel_ms"] = commands["server_ts_ms"] - commands["trial"].map(starts)
    return trials, commands


# --------------------------------------------------------------------------- #
# combined
# --------------------------------------------------------------------------- #
def build_all(mp_path=MP_FILE, corridor_path=CORRIDOR_FILE):
    mp_trials, cursor_samples = parse_matching_pennies(mp_path)
    cor_trials, commands = parse_corridor(corridor_path)
    # stack the per-trial tables on the shared schema; game-specific columns
    # (rt, start lane, ...) become NaN where they don't apply.
    trials = pd.concat([mp_trials, cor_trials], ignore_index=True, sort=False)
    lead = ["game", "session_code", "trial", "matcher_code", "avoider_code",
            "matcher_action", "avoider_action", "matched", "matcher_won",
            "avoider_won", "matcher_reward", "avoider_reward", "random_time_ms"]
    trials = trials[lead + [c for c in trials.columns if c not in lead]]
    return {"trials": trials, "cursor_samples": cursor_samples, "commands": commands}


if __name__ == "__main__":
    tables = build_all()
    for name, df in tables.items():
        print(f"\n=== {name}: shape {df.shape} ===")
        print(df.head(6).to_string())
        out_parquet = HERE / f"{name}.parquet"
        out_csv = HERE / f"{name}.csv"
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"  wrote {out_parquet.name}")
        except Exception as exc:  # parquet engine optional
            print(f"  (parquet skipped: {exc})")
        df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv.name}")
