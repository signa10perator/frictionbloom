import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from friction_bloom.environment import GridWorld
from friction_bloom.agent import GreedyAgent
from friction_bloom.experiment import run_experiment
from friction_bloom.friction import (
    block_first_third,
    block_baseline_midpoint,
    add_wall_gap_constraint,
    add_scattered_obstacles,
)

st.set_page_config(
    page_title="Friction Bloom Observatory",
    layout="wide",
)

DATA = Path("data/runs.jsonl")

# --- Friction registry ---
FRICTION_MAP = {
    "block_first_third": block_first_third,
    "block_baseline_midpoint": block_baseline_midpoint,
    "add_wall_gap_constraint": add_wall_gap_constraint,
    "add_scattered_obstacles": add_scattered_obstacles,
}


# --- Logger ---
def log_run(env, friction_name, result, baseline):
    DATA.parent.mkdir(exist_ok=True)

    record = {
        "grid_size": env.width,
        "friction": friction_name,
        "outcome": result.outcome,
        "bloom_score": result.bloom_score,
        "steps": result.altered.steps,
        "novelty": result.novelty,
        "coherence": result.coherence,
        "chaos": result.chaos,
        "baseline_path": baseline.path,
        "altered_path": result.altered.path,
        "notes": result.notes,
    }

    with open(DATA, "a") as f:
        f.write(pd.Series(record).to_json() + "\n")


# --- Drawing ---
def find_divergence_point(baseline_path, altered_path):
    for point in altered_path:
        if point not in baseline_path:
            return point
    return None


def draw_overlay(env, baseline_path, altered_path, title=""):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Grid
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.obstacles:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="black"))
            else:
                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        1,
                        1,
                        fill=False,
                        edgecolor="gray",
                        linewidth=0.8,
                    )
                )

    # Baseline path
    if baseline_path:
        xs = [p[0] + 0.5 for p in baseline_path]
        ys = [p[1] + 0.5 for p in baseline_path]
        ax.plot(xs, ys, color="limegreen", linewidth=2.5, label="baseline")

    # Altered path
    if altered_path:
        xs = [p[0] + 0.5 for p in altered_path]
        ys = [p[1] + 0.5 for p in altered_path]
        ax.plot(
            xs,
            ys,
            color="purple",
            linewidth=2.5,
            linestyle="--",
            label="altered",
        )

    # Divergence point
    div_point = find_divergence_point(baseline_path, altered_path)
    if div_point:
        ax.scatter(
            div_point[0] + 0.5,
            div_point[1] + 0.5,
            color="red",
            s=90,
            zorder=5,
            label="divergence",
        )

    ax.set_title(title)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(loc="lower left")

    return fig


# --- Header ---
st.title("Dendra: Friction Bloom Observatory")
st.caption("Controlled disruption → behavioral adaptation → emergent structure")

# --- Sidebar ---
st.sidebar.header("DENDRA: OBSERVATORY")

grid_size = st.sidebar.slider("Grid Size", 5, 15, 10)

selected_friction_names = st.sidebar.multiselect(
    "Friction Types",
    list(FRICTION_MAP.keys()),
    default=[
        "block_first_third",
        "block_baseline_midpoint",
        "add_wall_gap_constraint",
        "add_scattered_obstacles",
    ],
)

selected_frictions = [FRICTION_MAP[name] for name in selected_friction_names]

save_runs = st.sidebar.toggle("Save Runs", value=True)

run_button = st.sidebar.button("Run Experiment")

# --- History Sidebar Summary ---
if DATA.exists():
    df_history = pd.read_json(DATA, lines=True)

    if not df_history.empty:
        top_friction = (
            df_history.groupby("friction")["bloom_score"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

        st.sidebar.divider()
        st.sidebar.subheader("Memory Layer")
        st.sidebar.write(f"Total Saved Runs: {len(df_history)}")
        st.sidebar.write(f"Top Friction: `{top_friction}`")

# --- Environment ---
env = GridWorld(
    width=grid_size,
    height=grid_size,
    start=(0, 0),
    goal=(grid_size - 1, grid_size - 1),
)

agent = GreedyAgent()

# --- Main Run ---
if run_button:
    experiment = run_experiment(agent, env, selected_frictions)
    baseline = experiment.baseline
    results = experiment.rankings

    st.subheader("Behavioral Field")

    for r in results:
        if save_runs:
            log_run(env, r.friction_name, r, baseline)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.pyplot(
                draw_overlay(
                    env,
                    baseline.path,
                    r.altered.path,
                    title=r.friction_name,
                )
            )

        with col2:
            st.metric("Bloom Score", round(r.bloom_score, 2))
            st.write(f"Outcome: **{r.outcome.upper()}**")
            st.write(f"Steps: {r.altered.steps}")
            st.write(f"Novelty: {round(r.novelty, 2)}")
            st.write(f"Coherence: {round(r.coherence, 2)}")
            st.write(f"Chaos: {round(r.chaos, 2)}")
            st.write(f"Notes: {r.notes}")

        st.divider()

# --- Memory Layer ---
st.subheader("Memory Layer")

if DATA.exists():
    df = pd.read_json(DATA, lines=True)

    if not df.empty:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Total Runs", len(df))

        with col_b:
            bloom_count = (df["outcome"] == "bloom").sum()
            st.metric("Bloom Runs", bloom_count)

        with col_c:
            avg_bloom = df["bloom_score"].mean()
            st.metric("Avg Bloom Score", round(avg_bloom, 2))

        st.write("### Friction Ranking")
        ranking = (
            df.groupby("friction")["bloom_score"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(ranking)

        st.write("### Recent Runs")
        st.dataframe(
            df[
                [
                    "grid_size",
                    "friction",
                    "outcome",
                    "bloom_score",
                    "steps",
                    "novelty",
                    "coherence",
                    "chaos",
                    "notes",
                ]
            ].tail(20),
            use_container_width=True,
        )

    else:
        st.info("No saved runs yet.")
else:
    st.info("No memory file yet. Run an experiment with Save Runs enabled.")
