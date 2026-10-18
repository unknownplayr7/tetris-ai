"""Local technical dashboard for live PPO training.

Run with: streamlit run dashboard.py
The dashboard starts a separate train.py process and reads its append-only metrics log.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def tail_text(path, lines=35):
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "No training log yet."


def load_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return pd.DataFrame(rows)


def stop_process():
    proc = st.session_state.get("trainer_process")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    handle = st.session_state.get("trainer_log_handle")
    if handle:
        handle.close()
    st.session_state.trainer_process = None
    st.session_state.trainer_log_handle = None


def start_training(config, resume_path=None):
    RUNS.mkdir(parents=True, exist_ok=True)
    run_name = "ppo-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (run_dir / "trainer.log").open("w", encoding="utf-8")
    command = [
        sys.executable, str(ROOT / "train.py"),
        "--updates", str(config["updates"]),
        "--envs", str(config["envs"]),
        "--rollout", str(config["rollout"]),
        "--epochs", str(config["epochs"]),
        "--minibatch-size", str(config["minibatch_size"]),
        "--lr", str(config["lr"]),
        "--seed", str(config["seed"]),
        "--device", config["device"],
        "--run-dir", str(run_dir),
    ]
    if resume_path:
        command.extend(["--resume", str(Path(resume_path).expanduser())])
    proc = subprocess.Popen(
        command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    st.session_state.run_dir = str(run_dir)
    st.session_state.trainer_process = proc
    st.session_state.trainer_log_handle = log_handle
    st.session_state.trainer_command = command


st.set_page_config(page_title="Tetris RL Lab", page_icon="▦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#101827; border:1px solid #273449; padding:14px 16px; border-radius:12px;}
div[data-testid="stAlert"] {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("▦ TETRIS RL LAB")
st.caption("PROXIMAL POLICY OPTIMIZATION  /  ACTOR–CRITIC  /  LIVE TRAINING TELEMETRY")
st.markdown("A fresh run starts from randomly initialized weights. Training only resumes from a checkpoint when you explicitly provide one.")

with st.sidebar:
    st.header("Experiment setup")
    updates = st.number_input("Training updates", min_value=1, max_value=100000, value=1000, step=100)
    envs = st.number_input("Parallel environments", min_value=1, max_value=256, value=16)
    rollout = st.number_input("Steps per environment / update", min_value=8, max_value=2048, value=128, step=8)
    epochs = st.number_input("PPO epochs", min_value=1, max_value=20, value=4)
    minibatch_size = st.select_slider("Minibatch size", options=[32, 64, 128, 256, 512, 1024], value=256)
    lr = st.select_slider("Learning rate", options=[0.00003, 0.0001, 0.00025, 0.0003, 0.001], value=0.00025, format_func=lambda x: f"{x:g}")
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=7)
    device = st.selectbox("Compute device", ["auto", "mps", "cpu", "cuda"], index=0)
    st.divider()
    resume_enabled = st.checkbox("Resume from checkpoint", value=False)
    resume_path = st.text_input("Checkpoint path", value="checkpoints/tetris_rl.pt", disabled=not resume_enabled)
    st.caption("Unchecked = genuinely fresh policy. Resume is opt-in.")

    running = (
        st.session_state.get("trainer_process") is not None
        and st.session_state.trainer_process.poll() is None
    )
    start_col, stop_col = st.columns(2)
    with start_col:
        start_clicked = st.button("▶ Start", type="primary", disabled=running, use_container_width=True)
    with stop_col:
        stop_clicked = st.button("■ Stop", disabled=not running, use_container_width=True)

    if start_clicked:
        config = {
            "updates": int(updates), "envs": int(envs), "rollout": int(rollout),
            "epochs": int(epochs), "minibatch_size": int(minibatch_size),
            "lr": float(lr), "seed": int(seed), "device": device,
        }
        resume = resume_path.strip() if resume_enabled else None
        if resume and not Path(resume).expanduser().is_file():
            st.error(f"Checkpoint not found: {resume}")
        else:
            if st.session_state.get("trainer_process") is not None:
                stop_process()
            start_training(config, resume)
            st.rerun()
    if stop_clicked:
        stop_process()
        st.rerun()

    if st.session_state.get("run_dir"):
        st.caption("Active run")
        st.code(st.session_state.run_dir, language=None)

run_dir = st.session_state.get("run_dir")
if not run_dir:
    candidates = sorted(RUNS.glob("ppo-*")) if RUNS.exists() else []
    if candidates:
        run_dir = str(candidates[-1])
        st.session_state.run_dir = run_dir

proc = st.session_state.get("trainer_process")
is_running = proc is not None and proc.poll() is None
if is_running:
    st.success(f"TRAINING RUNNING · PID {proc.pid}")
elif proc is not None:
    code = proc.poll()
    st.info(f"TRAINING PROCESS EXITED · exit code {code}")
elif run_dir:
    st.info("Showing the latest available run. Start a new run from the sidebar.")
else:
    st.info("No run yet. Choose settings on the left and press Start to begin training from scratch.")

@st.fragment(run_every=2)
def live_telemetry(active_run):
    if not active_run:
        st.subheader("Telemetry")
        st.write("Metrics will appear here after the first PPO update.")
        return
    df = load_metrics(active_run)
    if df.empty:
        st.subheader("Telemetry")
        st.write("Waiting for the first metrics record. The environment rollout may take a moment.")
        log = tail_text(Path(active_run) / "trainer.log")
        if log:
            st.code(log, language="text")
        return

    latest = df.iloc[-1]
    cols = st.columns(5)
    cols[0].metric("Update", f"{int(latest.get('update', 0)):,}")
    cols[1].metric("Environment steps", f"{int(latest.get('env_steps', 0)):,}")
    cols[2].metric("Steps / sec", f"{int(latest.get('sps', 0)):,}")
    value = latest.get("mean_episode_return_100")
    cols[3].metric("Mean return · last 100", "n/a" if pd.isna(value) else f"{value:.2f}")
    value = latest.get("mean_lines_100")
    cols[4].metric("Mean lines · last 100", "n/a" if pd.isna(value) else f"{value:.2f}")

    tabs = st.tabs(["Learning", "Optimization", "Episode performance", "Run details"])
    with tabs[0]:
        st.subheader("Reward and return")
        reward_cols = [c for c in ("mean_reward", "mean_episode_return_100") if c in df and df[c].notna().any()]
        if reward_cols:
            st.line_chart(df.set_index("update")[reward_cols], height=260)
        st.subheader("Value targets")
        value_cols = [c for c in ("value_loss", "return_mean", "advantage_mean") if c in df and df[c].notna().any()]
        if value_cols:
            st.line_chart(df.set_index("update")[value_cols], height=220)
    with tabs[1]:
        st.subheader("PPO optimization diagnostics")
        loss_cols = [c for c in ("policy_loss", "value_loss") if c in df and df[c].notna().any()]
        if loss_cols:
            st.line_chart(df.set_index("update")[loss_cols], height=240)
        diag_cols = [c for c in ("entropy", "approx_kl", "clip_fraction", "learning_rate") if c in df and df[c].notna().any()]
        if diag_cols:
            st.line_chart(df.set_index("update")[diag_cols], height=240)
        st.caption("KL and clip fraction help spot overly large policy updates; entropy is a rough indicator of policy exploration.")
    with tabs[2]:
        st.subheader("Game performance")
        perf_cols = [c for c in ("mean_lines_100", "mean_pieces_100", "mean_completed_lines", "mean_completed_pieces") if c in df and df[c].notna().any()]
        if perf_cols:
            st.line_chart(df.set_index("update")[perf_cols], height=260)
        st.dataframe(df.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
    with tabs[3]:
        st.json({
            "run_directory": str(active_run),
            "device": latest.get("device"),
            "last_update": int(latest.get("update", 0)),
            "environment_steps": int(latest.get("env_steps", 0)),
            "wall_seconds": latest.get("wall_seconds"),
            "checkpoint_exists": (Path(active_run) / "checkpoint.pt").exists(),
            "metric_records": len(df),
        })
        log_path = Path(active_run) / "trainer.log"
        st.subheader("Trainer output")
        st.code(tail_text(log_path), language="text")
        st.download_button(
            "Download metrics JSONL", data=(Path(active_run) / "metrics.jsonl").read_bytes(),
            file_name="metrics.jsonl", mime="application/x-ndjson",
        )

live_telemetry(run_dir)

st.divider()
st.caption("Local-only dashboard · metrics are recorded in runs/<run-name>/metrics.jsonl · checkpoints are saved inside each run directory.")
