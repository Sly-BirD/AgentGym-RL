"""
RL-WebNav Agent — Interactive Gradio Dashboard
===============================================

This dashboard provides a visual interface for monitoring, evaluating,
and understanding the RL-trained web navigation agent.

FEATURES:
1. 📊 Training Monitor — Live reward curves, loss plots, success rate over time
2. 🔍 Trajectory Viewer — Replay agent's actions step-by-step
3. ⚖️ Comparison Table — "No RL" vs "GRPO" vs "ScalingInter-RL" side-by-side
4. 🎯 Live Demo — Input a task and watch the agent attempt it
5. 📈 Reward Analyzer — Visualize reward component breakdown

TERMINOLOGY:
    - Gradio: A Python library for building ML demos/UIs. It auto-generates
      a web interface from Python functions. Runs locally at localhost:7860.
    - Epoch: One complete pass through all training data.
    - Reward Curve: Plot of average reward per epoch — should go UP over time
      if the agent is learning.
    - Success Rate: Percentage of tasks completed successfully.
    - Trajectory: The full sequence of agent actions in one task attempt.

HOW TO RUN:
    python dashboard/app.py
    → Opens at http://localhost:7860

NOTE: This dashboard uses MOCK DATA for prototyping on Windows.
      On the cloud, it will connect to real W&B logs and trajectory files.
"""

import gradio as gr
import json
import os
import random
import math
from pathlib import Path

# ============================================================================
# MOCK DATA (for local prototyping)
# ============================================================================
# In production, this data comes from:
# - W&B API (training metrics)
# - Rollout log files (trajectories)
# - Eval results JSON (comparison metrics)
#
# We generate realistic-looking mock data so the dashboard can be developed
# and tested on Windows without a GPU.
# ============================================================================


def generate_mock_training_data(num_epochs: int = 25) -> dict:
    """
    Generate mock training curves that look like real RL training.

    Real RL training typically shows:
    - Reward increases with noise (not smooth — RL is inherently stochastic)
    - Success rate follows a sigmoid-like curve (slow start, rapid middle, plateau)
    - KL divergence stays relatively flat (controlled by the penalty)
    - Policy loss decreases then fluctuates around a minimum

    Returns dict with lists for each metric, indexed by epoch.
    """
    random.seed(42)  # Reproducible mock data
    epochs = list(range(1, num_epochs + 1))

    # Reward curve: starts low, increases with noise
    # Uses a logistic function + noise to simulate learning
    rewards = []
    for e in epochs:
        base = 0.8 / (1 + math.exp(-0.25 * (e - 12)))  # Sigmoid centered at epoch 12
        noise = random.gauss(0, 0.04)
        rewards.append(round(max(0, min(1.0, base + 0.05 + noise)), 3))

    # Success rate: similar sigmoid but scaled to percentage
    success_rates = []
    for e in epochs:
        base = 68 / (1 + math.exp(-0.3 * (e - 10)))  # Sigmoid → ~68% max
        noise = random.gauss(0, 3)
        success_rates.append(round(max(0, min(100, base + 5 + noise)), 1))

    # Average steps: starts high (agent is inefficient), decreases
    avg_steps = []
    for e in epochs:
        base = 14 - 6 / (1 + math.exp(-0.3 * (e - 10)))
        noise = random.gauss(0, 0.5)
        avg_steps.append(round(max(3, base + noise), 1))

    # KL divergence: stays low (controlled by penalty)
    kl_divs = []
    for e in epochs:
        base = 0.002 + 0.001 * e / num_epochs
        noise = random.gauss(0, 0.0005)
        kl_divs.append(round(max(0, base + noise), 4))

    # Policy loss: decreases then stabilizes
    policy_losses = []
    for e in epochs:
        base = 0.5 * math.exp(-0.1 * e) + 0.05
        noise = random.gauss(0, 0.02)
        policy_losses.append(round(max(0, base + noise), 4))

    return {
        "epochs": epochs,
        "rewards": rewards,
        "success_rates": success_rates,
        "avg_steps": avg_steps,
        "kl_divergence": kl_divs,
        "policy_loss": policy_losses,
    }


def generate_mock_comparison_data() -> list:
    """
    Generate comparison data: Baseline vs GRPO vs ScalingInter-RL.

    This is the key table from the PRD — showing that RL training
    actually improves the agent compared to no training.
    """
    return [
        {
            "Model": "Qwen2.5-7B (Zero-Shot, No RL)",
            "Success Rate (%)": 28.4,
            "Avg Steps": 14.1,
            "Avg Reward": 0.31,
            "Partial Progress": 0.12,
            "Training Time": "N/A",
            "Status": "🔴 Baseline",
        },
        {
            "Model": "Qwen2.5-7B + SFT Only",
            "Success Rate (%)": 39.2,
            "Avg Steps": 12.3,
            "Avg Reward": 0.45,
            "Partial Progress": 0.22,
            "Training Time": "~2 hours",
            "Status": "🟡 Better",
        },
        {
            "Model": "Qwen2.5-7B + GRPO (Fixed Horizon)",
            "Success Rate (%)": 58.6,
            "Avg Steps": 9.1,
            "Avg Reward": 0.72,
            "Partial Progress": 0.41,
            "Training Time": "~8 hours",
            "Status": "🟢 Good",
        },
        {
            "Model": "Qwen2.5-7B + ScalingInter-RL",
            "Success Rate (%)": 65.3,
            "Avg Steps": 7.8,
            "Avg Reward": 0.82,
            "Partial Progress": 0.53,
            "Training Time": "~10 hours",
            "Status": "🟢 Best",
        },
    ]


def generate_mock_trajectory() -> list:
    """
    Generate a mock agent trajectory for the trajectory viewer.

    Each step shows:
    - Turn number
    - What the agent observed (simplified page state)
    - What action the agent took
    - The reward received for that action
    - Cumulative status
    """
    return [
        {
            "turn": 1,
            "observation": "📄 RedBus.in Homepage — Search bar visible with 'From', 'To', 'Date' fields. Popular routes section below.",
            "action": "type[source_city][Pune]",
            "reward": 0.0,
            "status": "🟡 Entering source city",
        },
        {
            "turn": 2,
            "observation": "📄 RedBus.in — Source city 'Pune' selected. Dropdown showing suggestions. 'To' field empty.",
            "action": "type[destination_city][Mumbai]",
            "reward": 0.0,
            "status": "🟡 Entering destination",
        },
        {
            "turn": 3,
            "observation": "📄 RedBus.in — Route: Pune → Mumbai. Date field showing calendar widget. Today highlighted.",
            "action": "click[date_tomorrow]",
            "reward": 0.0,
            "status": "🟡 Selecting date",
        },
        {
            "turn": 4,
            "observation": "📄 RedBus.in — Route: Pune → Mumbai, Date: Tomorrow. Search button active.",
            "action": "click[search_buses]",
            "reward": 0.05,
            "status": "🟢 Search submitted!",
        },
        {
            "turn": 5,
            "observation": "📄 Bus Results — 47 buses found. Filters: AC/Non-AC, Seater/Sleeper, Bus Type. First result: 'Neeta Tours AC Sleeper ₹650, 4.2★'",
            "action": "click[filter_ac_sleeper]",
            "reward": 0.05,
            "status": "🟢 Filter applied",
        },
        {
            "turn": 6,
            "observation": "📄 Filtered Results — 12 AC Sleeper buses. Sort options: Price, Rating, Departure. Cheapest: 'VRL Travels ₹550, 3.8★'",
            "action": "click[sort_price_low_to_high]",
            "reward": 0.03,
            "status": "🟢 Sorted by price",
        },
        {
            "turn": 7,
            "observation": "📄 Sorted Results — First bus: 'VRL Travels AC Sleeper ₹550, 10:30 PM, 3.8★, 6h journey'. View Seats button visible.",
            "action": "click[view_seats_vrl_travels]",
            "reward": 0.10,
            "status": "🟢 Bus selected!",
        },
        {
            "turn": 8,
            "observation": "📄 Seat Layout — Lower deck: seats A1-A10 (window), B1-B10 (aisle). Upper deck: C1-C10, D1-D10. Green=available, Red=booked. A3(window)=green.",
            "action": "click[seat_A3_window]",
            "reward": 0.10,
            "status": "🟢 Seat selected!",
        },
        {
            "turn": 9,
            "observation": "📄 Passenger Details — Seat A3 selected (₹550). Name, Age, Gender fields. 'Proceed to Pay' button.",
            "action": "click[proceed_to_pay]",
            "reward": 0.25,
            "status": "✅ Reached checkout!",
        },
    ]


# ============================================================================
# DASHBOARD COMPONENTS
# ============================================================================


def create_training_plots(training_data: dict) -> str:
    """
    Create HTML/JS training metric plots using inline charts.

    We use simple HTML canvas charts to avoid heavy dependencies.
    In production, these would be interactive Plotly charts.
    """
    epochs = training_data["epochs"]
    rewards = training_data["rewards"]
    success_rates = training_data["success_rates"]
    avg_steps = training_data["avg_steps"]

    # Format data as HTML table + sparkline description
    table_rows = [
        f"""
        <tr>
            <td style="text-align:center; font-weight:600;">{e}</td>
            <td>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:{int(rewards[i] * 170)}px; height:14px; background:linear-gradient(90deg, #6366f1, #8b5cf6); border-radius:7px;"></div>
                    <span style="font-size:0.8em; color:#a5b4fc;">{rewards[i]:.3f}</span>
                </div>
            </td>
            <td>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:{int(success_rates[i] * 2.5)}px; height:14px; background:linear-gradient(90deg, #10b981, #34d399); border-radius:7px;"></div>
                    <span style="font-size:0.8em; color:#6ee7b7;">{success_rates[i]}%</span>
                </div>
            </td>
            <td style="text-align:center; color:#fbbf24; font-weight:500;">{avg_steps[i]}</td>
        </tr>
        """
        for i, e in enumerate(epochs)
    ]

    html = f"""
    <div style="max-height:500px; overflow-y:auto; padding:8px;">
        <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif; font-size:0.9em;">
            <thead>
                <tr style="border-bottom:2px solid #374151; color:#9ca3af;">
                    <th style="padding:8px 4px; text-align:center; width:60px;">Epoch</th>
                    <th style="padding:8px 4px;">Avg Reward</th>
                    <th style="padding:8px 4px;">Success Rate</th>
                    <th style="padding:8px 4px; text-align:center; width:80px;">Avg Steps</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    """
    return html


def create_summary_cards(training_data: dict) -> str:
    """Create summary metric cards showing latest values."""
    latest_reward = training_data["rewards"][-1]
    latest_success = training_data["success_rates"][-1]
    latest_steps = training_data["avg_steps"][-1]
    latest_kl = training_data["kl_divergence"][-1]

    # Trend indicators
    reward_trend = "↑" if training_data["rewards"][-1] > training_data["rewards"][-2] else "↓"
    success_trend = "↑" if training_data["success_rates"][-1] > training_data["success_rates"][-2] else "↓"
    steps_trend = "↓" if training_data["avg_steps"][-1] < training_data["avg_steps"][-2] else "↑"

    html = f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:16px; font-family:'Inter',sans-serif;">
        <div style="background:linear-gradient(135deg,#312e81,#4338ca); padding:20px; border-radius:16px; box-shadow:0 4px 15px rgba(99,102,241,0.3);">
            <div style="color:#a5b4fc; font-size:0.8em; font-weight:500; text-transform:uppercase; letter-spacing:1px;">Avg Reward</div>
            <div style="color:white; font-size:2em; font-weight:700; margin-top:4px;">{latest_reward:.3f} <span style="font-size:0.5em;">{reward_trend}</span></div>
            <div style="color:#818cf8; font-size:0.75em; margin-top:4px;">Target: ≥0.60</div>
        </div>
        <div style="background:linear-gradient(135deg,#064e3b,#065f46); padding:20px; border-radius:16px; box-shadow:0 4px 15px rgba(16,185,129,0.3);">
            <div style="color:#6ee7b7; font-size:0.8em; font-weight:500; text-transform:uppercase; letter-spacing:1px;">Success Rate</div>
            <div style="color:white; font-size:2em; font-weight:700; margin-top:4px;">{latest_success}% <span style="font-size:0.5em;">{success_trend}</span></div>
            <div style="color:#34d399; font-size:0.75em; margin-top:4px;">Target: ≥60%</div>
        </div>
        <div style="background:linear-gradient(135deg,#78350f,#92400e); padding:20px; border-radius:16px; box-shadow:0 4px 15px rgba(251,191,36,0.3);">
            <div style="color:#fcd34d; font-size:0.8em; font-weight:500; text-transform:uppercase; letter-spacing:1px;">Avg Steps</div>
            <div style="color:white; font-size:2em; font-weight:700; margin-top:4px;">{latest_steps} <span style="font-size:0.5em;">{steps_trend}</span></div>
            <div style="color:#fbbf24; font-size:0.75em; margin-top:4px;">Lower is better</div>
        </div>
        <div style="background:linear-gradient(135deg,#4c1d95,#5b21b6); padding:20px; border-radius:16px; box-shadow:0 4px 15px rgba(139,92,246,0.3);">
            <div style="color:#c4b5fd; font-size:0.8em; font-weight:500; text-transform:uppercase; letter-spacing:1px;">KL Divergence</div>
            <div style="color:white; font-size:2em; font-weight:700; margin-top:4px;">{latest_kl:.4f}</div>
            <div style="color:#a78bfa; font-size:0.75em; margin-top:4px;">Penalty: 0.001</div>
        </div>
    steps_html = []
    cumulative_reward = 0.0

    for step in trajectory:
        cumulative_reward += step["reward"]
        reward_color = "#10b981" if step["reward"] > 0 else "#6b7280"
        reward_display = f"+{step['reward']:.2f}" if step["reward"] > 0 else "0.00"

        steps_html.append(f"""
        <div style="display:flex; gap:16px; margin-bottom:16px; align-items:flex-start;">
            <div style="min-width:48px; text-align:center;">
                <div style="width:36px; height:36px; background:linear-gradient(135deg,#4338ca,#6366f1);
                     border-radius:50%; display:flex; align-items:center; justify-content:center;
                     color:white; font-weight:700; font-size:0.9em; margin:0 auto;">
                    {step['turn']}
                </div>
                <div style="width:2px; height:24px; background:#374151; margin:4px auto 0;"></div>
            </div>
            <div style="flex:1; background:#1e1b4b; padding:14px 18px; border-radius:12px;
                  border-left:3px solid #6366f1; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
                <div style="color:#94a3b8; font-size:0.75em; margin-bottom:6px;">OBSERVATION</div>
                <div style="color:#e2e8f0; font-size:0.85em; margin-bottom:10px;">{step['observation']}</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#a5b4fc; font-size:0.75em;">ACTION → </span>
                        <code style="background:#312e81; color:#c4b5fd; padding:3px 8px; border-radius:6px;
                                font-size:0.8em;">{step['action']}</code>
                    </div>
                    <div style="display:flex; gap:12px; align-items:center;">
                        <span style="color:{reward_color}; font-weight:600; font-size:0.85em;">
                            R: {reward_display}
                        </span>
                        <span style="color:#fbbf24; font-size:0.75em;">
                            Σ: {cumulative_reward:.2f}
                        </span>
                    </div>
                </div>
                <div style="color:#64748b; font-size:0.75em; margin-top:6px;">{step['status']}</div>
            </div>
        </div>
        """)

    html = f"""
    <div style="font-family:'Inter',sans-serif; padding:8px;">
        <div style="color:#a5b4fc; font-weight:600; margin-bottom:16px; font-size:0.9em;">
            🎯 Task: Book Pune → Mumbai AC Sleeper bus on RedBus
        </div>
        {''.join(steps_html)}
        <div style="text-align:center; padding:12px; background:linear-gradient(135deg,#064e3b,#065f46);
              border-radius:12px; color:#34d399; font-weight:600;">
            ✅ Task Completed — Total Reward: {cumulative_reward:.2f} in {len(trajectory)} steps
        </div>
    </div>
    """color:#34d399; font-weight:600;">
            ✅ Task Completed — Total Reward: {cumulative_reward:.2f} in {len(trajectory)} steps
        </div>
    </div>
    """
    return html


def format_comparison_table(comparison_data: list) -> str:
    """Format the comparison data as a styled HTML table."""
    def get_bg_color(status):
        if "Best" in status: return "#0f2922"
        if "Baseline" in status: return "#2a1a1a"
        return "#1a1a2e"

    rows = [
        f"""
        <tr style="background:{get_bg_color(item['Status'])}; border-bottom:1px solid #2d2d3f;">
            <td style="padding:12px 16px; font-weight:600; color:#e2e8f0;">{item['Model']}</td>
            <td style="padding:12px 16px; text-align:center; color:#10b981; font-weight:700;">{item['Success Rate (%)']}</td>
            <td style="padding:12px 16px; text-align:center; color:#fbbf24;">{item['Avg Steps']}</td>
            <td style="padding:12px 16px; text-align:center; color:#8b5cf6;">{item['Avg Reward']}</td>
            <td style="padding:12px 16px; text-align:center; color:#94a3b8;">{item['Partial Progress']}</td>
            <td style="padding:12px 16px; text-align:center; color:#64748b;">{item['Training Time']}</td>
            <td style="padding:12px 16px; text-align:center;">{item['Status']}</td>
        </tr>
        """
        for item in comparison_data
    ]

    html = f"""
    <div style="overflow-x:auto; font-family:'Inter',sans-serif;">
        <table style="width:100%; border-collapse:collapse; font-size:0.85em;">
            <thead>
                <tr style="background:#1e1b4b; border-bottom:2px solid #4338ca;">
                    <th style="padding:12px 16px; text-align:left; color:#a5b4fc;">Model</th>
                    <th style="padding:12px 16px; text-align:center; color:#a5b4fc;">Success Rate</th>
                    <th style="padding:12px 16px; text-align:center; color:#a5b4fc;">Avg Steps</th>
                    <th style="padding:12px 16px; text-align:center; color:#a5b4fc;">Avg Reward</th>
                    <th style="padding:12px 16px; text-align:center; color:#a5b4fc;">Partial Progress</th>
                    <th style="padding:12px 16px; text-align:center; color:#a5b4fc;">Training Time</th>
                    <th style="padding:12px 16px; text-align:center; color:#a5b4fc;">Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """
    return html


def create_reward_breakdown_chart() -> str:
    """Visualize how the custom reward breaks down into components."""
    scenarios = [
        {"name": "Success (6 steps)", "success": 1.0, "efficiency": 0.18, "partial": 0.0, "total": 1.18},
        {"name": "Success (12 steps)", "success": 1.0, "efficiency": 0.06, "partial": 0.0, "total": 1.06},
        {"name": "Partial (Search done)", "success": 0.0, "efficiency": 0.0, "partial": 0.13, "total": 0.13},
        {"name": "Partial (Item selected)", "success": 0.0, "efficiency": 0.0, "partial": 0.23, "total": 0.23},
        {"name": "No Progress", "success": 0.0, "efficiency": 0.0, "partial": 0.0, "total": 0.0},
    ]

    max_bar = 160  # Max bar width in pixels
    rows = [
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <div style="min-width:180px; color:#94a3b8; font-size:0.85em; text-align:right;">{s['name']}</div>
            <div style="flex:1; display:flex; height:24px; border-radius:6px; overflow:hidden; background:#1e1b4b;">
                <div style="width:{int(s['success'] * max_bar / 1.6)}px; background:linear-gradient(90deg,#10b981,#34d399);" title="Success: {s['success']}"></div>
                <div style="width:{int(s['efficiency'] * max_bar / 1.6)}px; background:linear-gradient(90deg,#3b82f6,#60a5fa);" title="Efficiency: {s['efficiency']}"></div>
                <div style="width:{int(s['partial'] * max_bar / 1.6)}px; background:linear-gradient(90deg,#f59e0b,#fbbf24);" title="Partial: {s['partial']}"></div>
            </div>
            <div style="min-width:50px; color:#e2e8f0; font-weight:600; font-size:0.9em;">{s['total']:.2f}</div>
        </div>
        """
        for s in scenarios
    ]

    html = f"""
    <div style="font-family:'Inter',sans-serif; padding:8px;">
        <div style="color:#94a3b8; font-size:0.8em; margin-bottom:16px;">
            Reward = <span style="color:#34d399;">■ Success (0-1.0)</span> +
            <span style="color:#60a5fa;">■ Efficiency (0-0.3)</span> +
            <span style="color:#fbbf24;">■ Partial Progress (0-0.3)</span>
        </div>
        {''.join(rows)}
    </div>
    """
    return html


# ============================================================================
# CALLBACK FUNCTIONS (Gradio event handlers)
# ============================================================================


def refresh_training_data():
    """Refresh training metrics (simulated)."""
    data = generate_mock_training_data()
    summary = create_summary_cards(data)
    plots = create_training_plots(data)
    return summary, plots


def load_trajectory(task_name: str):
    """Load and display a trajectory."""
    traj = generate_mock_trajectory()
    return format_trajectory_viewer(traj)


def load_comparison():
    """Load comparison table."""
    data = generate_mock_comparison_data()
    return format_comparison_table(data)


def load_reward_breakdown():
    """Load reward component breakdown."""
    return create_reward_breakdown_chart()


def simulate_agent_run(task_description: str, max_steps: int):
    """
    Simulate an agent attempting a task.
    In production, this would:
    1. Load the trained model checkpoint
    2. Initialize the environment server
    3. Run the agent for max_steps
    4. Return the trajectory
    
    For now, returns mock data.
    """
    if not task_description.strip():
        return "<div style='color:#f87171; padding:16px;'>⚠️ Please enter a task description.</div>"

    # Mock response
    traj = generate_mock_trajectory()
    return format_trajectory_viewer(traj)


# ============================================================================
# BUILD THE GRADIO APP
# ============================================================================

def build_app():
    """
    Build the complete Gradio dashboard.

    Gradio works by defining:
    1. Blocks — layout containers (rows, columns, tabs)
    2. Components — UI elements (textbox, button, HTML display)
    3. Events — callbacks triggered by user actions (click, submit)
    """

    # Custom CSS for dark theme with premium aesthetics
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .gradio-container {
        font-family: 'Inter', sans-serif !important;
        background: #0a0a1a !important;
        max-width: 1400px !important;
    }

    .dark {
        --body-background-fill: #0a0a1a !important;
        --block-background-fill: #111127 !important;
        --block-border-color: #1e1b4b !important;
        --block-label-text-color: #a5b4fc !important;
        --block-title-text-color: #c4b5fd !important;
        --button-primary-background-fill: linear-gradient(135deg, #4338ca, #6366f1) !important;
        --button-primary-text-color: white !important;
        --input-background-fill: #1a1a2e !important;
    }

    #header-title {
        text-align: center;
        font-size: 2.2em;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
        letter-spacing: -0.5px;
    }

    #header-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95em;
        margin-top: -8px;
        padding-bottom: 16px;
    }

    .tab-nav button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85em !important;
        letter-spacing: 0.5px !important;
    }

    .tab-nav button.selected {
        background: linear-gradient(135deg, #4338ca, #6366f1) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    """

    with gr.Blocks() as app:

        # --- Header ---
        gr.HTML('<div id="header-title">🤖 RL-WebNav Agent Dashboard</div>')
        gr.HTML('<div id="header-subtitle">Reinforcement Learning Enhanced Web Navigation • GRPO Training Monitor</div>')

        with gr.Tabs():
            # =================================================================
            # TAB 1: Training Monitor
            # =================================================================
            with gr.TabItem("📊 Training Monitor"):
                gr.Markdown("""
                ### Training Metrics
                *Monitor the agent's learning progress across epochs. Reward and success rate should trend upward; average steps should decrease.*
                """)

                summary_display = gr.HTML()
                refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                plots_display = gr.HTML()

                # Auto-load on page open
                app.load(fn=refresh_training_data, outputs=[summary_display, plots_display])
                refresh_btn.click(fn=refresh_training_data, outputs=[summary_display, plots_display])

            # =================================================================
            # TAB 2: Trajectory Viewer
            # =================================================================
            with gr.TabItem("🔍 Trajectory Viewer"):
                gr.Markdown("""
                ### Trajectory Replay
                *Watch the agent's step-by-step decision making. Each turn shows what the agent observed, what action it chose, and the reward received.*
                
                **Terminology:**
                - **Observation**: The web page state the agent sees (simplified text representation of the HTML)
                - **Action**: What the agent decides to do (e.g., `type[field][value]`, `click[element]`, `scroll[direction]`)
                - **Reward (R)**: Immediate reward for this step
                - **Σ (Sigma)**: Cumulative reward so far in this episode
                """)

                with gr.Row():
                    task_selector = gr.Dropdown(
                        choices=[
                            "india_redbus_001 — Pune→Mumbai Bus Booking",
                            "india_redbus_002 — Bangalore→Hyderabad Volvo",
                            "india_confirmtkt_001 — Pune→Delhi Rajdhani",
                            "india_confirmtkt_002 — PNR Status Check",
                            "webarena_shopping_001 — Buy Red Shirt",
                        ],
                        label="Select Task",
                        value="india_redbus_001 — Pune→Mumbai Bus Booking",
                    )
                    load_traj_btn = gr.Button("▶️ Load Trajectory", variant="primary")

                trajectory_display = gr.HTML()
                load_traj_btn.click(fn=load_trajectory, inputs=[task_selector], outputs=[trajectory_display])

            # =================================================================
            # TAB 3: Model Comparison
            # =================================================================
            with gr.TabItem("⚖️ Comparison"):
                gr.Markdown("""
                ### No RL vs. RL Agent — Side-by-Side Comparison
                *This is the key result: does RL training actually help? Compare the zero-shot baseline, SFT-only, GRPO, and ScalingInter-RL.*
                
                **What each column means:**
                - **Success Rate**: % of tasks fully completed correctly
                - **Avg Steps**: Mean number of actions taken per task (lower = more efficient)
                - **Avg Reward**: Mean reward using our custom reward function (higher = better)
                - **Partial Progress**: Average partial milestone score for failed tasks
                - **Training Time**: Wall-clock time on a single A100 GPU
                """)

                comparison_display = gr.HTML()
                load_comp_btn = gr.Button("📊 Load Comparison Data", variant="primary")

                app.load(fn=load_comparison, outputs=[comparison_display])
                load_comp_btn.click(fn=load_comparison, outputs=[comparison_display])

            # =================================================================
            # TAB 4: Reward Analyzer
            # =================================================================
            with gr.TabItem("🎯 Reward Analyzer"):
                gr.Markdown("""
                ### Custom Reward Function Breakdown
                *Visualize how our 3-component reward function (Success + Efficiency + Partial Progress) compares across different scenarios.*
                
                **Why this matters:**
                - **Binary reward** (just 0 or 1): Agent gets almost no learning signal early on → slow training
                - **Shaped reward** (our approach): Agent gets credit for partial progress → faster, more stable learning
                - **Efficiency bonus**: Encourages the agent to find shorter paths, not just complete the task by any means
                """)

                reward_display = gr.HTML()
                app.load(fn=load_reward_breakdown, outputs=[reward_display])

            # =================================================================
            # TAB 5: Live Demo
            # =================================================================
            with gr.TabItem("🚀 Live Demo"):
                gr.Markdown("""
                ### Try the Agent
                *Enter a web navigation task and watch the agent attempt it. (Currently uses mock data — connect to trained model on cloud for real inference.)*
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        task_input = gr.Textbox(
                            label="Task Description",
                            placeholder="e.g., Book a bus from Pune to Mumbai on RedBus for tomorrow, cheapest AC sleeper",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        max_steps_slider = gr.Slider(
                            minimum=5, maximum=30, value=15, step=1,
                            label="Max Steps (Horizon)"
                        )
                        run_btn = gr.Button("🤖 Run Agent", variant="primary", size="lg")

                demo_output = gr.HTML()
                run_btn.click(fn=simulate_agent_run, inputs=[task_input, max_steps_slider], outputs=[demo_output])

            # =================================================================
            # TAB 6: Architecture
            # =================================================================
            with gr.TabItem("🏗️ Architecture"):
                gr.Markdown("""
                ### System Architecture
                
                ```
                ┌─────────────────────────────────────────────────────────────┐
                │                    RL-WebNav Agent                          │
                ├─────────────────────────────────────────────────────────────┤
                │                                                             │
                │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
                │  │  Environment │◄──►│    Agent      │◄──►│   Trainer    │  │
                │  │   Module     │    │   Module      │    │   Module     │  │
                │  │              │    │              │    │              │  │
                │  │ • WebArena   │    │ • LLM Policy │    │ • GRPO/PPO   │  │
                │  │ • RedBus sim │    │ • Qwen2.5-7B │    │ • Verl       │  │
                │  │ • ConfirmTkt │    │ • vLLM infer │    │ • Ray        │  │
                │  │              │    │              │    │ • FSDP       │  │
                │  │  HTTP Server │    │ Rollout      │    │ Advantage    │  │
                │  │  (Docker)    │    │ Handler      │    │ Computation  │  │
                │  └──────────────┘    └──────────────┘    └──────────────┘  │
                │         │                   │                    │          │
                │         ▼                   ▼                    ▼          │
                │  ┌─────────────────────────────────────────────────────┐   │
                │  │              Training Loop (per epoch)              │   │
                │  │                                                     │   │
                │  │  1. Rollout: Agent interacts with environment       │   │
                │  │     (N=4 trajectories per task, up to 15 turns)     │   │
                │  │                                                     │   │
                │  │  2. Reward: Compute custom reward                   │   │
                │  │     (success + efficiency + partial progress)       │   │
                │  │                                                     │   │
                │  │  3. Advantage: GRPO group-relative normalization    │   │
                │  │     (which of N responses was best?)                │   │
                │  │                                                     │   │
                │  │  4. Update: Backprop + clip policy gradient         │   │
                │  │     (make good actions more likely)                 │   │
                │  └─────────────────────────────────────────────────────┘   │
                │                                                             │
                │  ┌─────────────────────────────────────────────────────┐   │
                │  │              ScalingInter-RL                        │   │
                │  │                                                     │   │
                │  │  Steps 0-99:   Horizon = 5 turns  (learn basics)   │   │
                │  │  Steps 100-199: Horizon = 10 turns (multi-step)    │   │
                │  │  Steps 200+:   Horizon = 15 turns (full tasks)     │   │
                │  └─────────────────────────────────────────────────────┘   │
                └─────────────────────────────────────────────────────────────┘
                ```
                
                ### Key Concepts Explained
                
                | Concept | Meaning |
                |---------|---------|
                | **Policy** | The LLM's decision function: observation → action |
                | **Rollout** | One complete episode of agent ↔ environment interaction |
                | **GRPO** | Generate N rollouts, rank them, reinforce the better ones |
                | **KL Penalty** | Prevents over-optimization (keeps model "sane") |
                | **vLLM** | Fast inference engine (10-24x faster than HuggingFace) |
                | **FSDP** | Splits model across GPUs for distributed training |
                | **Ray** | Orchestrates multiple GPU workers |
                | **Hydra** | Configuration management (YAML → Python config) |
                """)

    return app, custom_css


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  RL-WebNav Agent — Gradio Dashboard")
    print("  Starting at http://localhost:7860")
    print("=" * 60)

    app, custom_css = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css,
        theme=gr.themes.Base(),
    )
