"""
Custom Reward Function for RL-WebNav Agent
==========================================

This module defines the reward function used during GRPO/PPO training.
Instead of the default binary reward (1.0 for success, 0.0 for failure),
we use a composite reward with three components:

1. SUCCESS REWARD (0 or 1.0):
   - Binary: did the agent complete the task?
   - This is the "ground truth" signal from the environment.

2. STEP EFFICIENCY BONUS (0 to 0.3):
   - Rewards the agent for completing tasks in fewer steps.
   - Intuition: A human who buys a shirt in 3 clicks is better than one
     who takes 15 clicks with wrong turns.
   - Formula: efficiency = 0.3 * (max_steps - steps_used) / max_steps
   - Only awarded if the task was successful (no bonus for failing fast).

3. PARTIAL PROGRESS REWARD (0 to 0.3):
   - Rewards intermediate milestones even if the task isn't fully completed.
   - WHY THIS IS CRITICAL: In early training, the agent almost never succeeds
     at full tasks. Without partial progress, the reward is 0 for 99% of
     trajectories → the GRPO advantage is ~0 for everything → no learning.
   - With partial progress, even "got to the search results page" gives
     a signal, which breaks the "sparse reward problem."
   - Milestones are defined per-task (see custom_india_tasks.json).

TERMINOLOGY:
    - Trajectory: The full sequence of (observation, action, reward) tuples
      from the start of a task to its end (success, failure, or timeout).
    - Sparse Reward: When the agent only gets a reward at the very end
      (e.g., "did you finish the whole task?"). This is hard to learn from
      because most of the trajectory has zero signal.
    - Dense Reward: When the agent gets rewards at many points during the
      trajectory. Easier to learn from, but harder to design correctly.
    - Reward Shaping: The art of designing dense rewards that guide the
      agent toward the goal without introducing unintended shortcuts.
    - Credit Assignment: The challenge of figuring out WHICH action in a
      long trajectory was responsible for the final reward.

HOW THIS INTEGRATES WITH TRAINING:
    The `custom_reward_function` config in ppo_trainer.yaml points to this file:
        custom_reward_function:
            path: "rewards/custom_web_reward.py"
            name: "compute_custom_reward"
    
    During training, after each rollout, this function is called to compute
    the reward for each trajectory. The rewards are then used by GRPO to
    compute advantages (how much better/worse each response was compared
    to the group average).
"""

import json
import re
from typing import Dict, List, Any, Optional


# =============================================================================
# Partial Progress Detectors
# =============================================================================
# These functions analyze the trajectory to detect if certain milestones
# were reached, even if the task wasn't fully completed.
#
# Each detector takes the list of observations (web pages the agent saw)
# and actions (what the agent did) and returns True/False.
# =============================================================================

def detect_search_performed(observations: List[str], actions: List[str]) -> bool:
    """
    Detects if the agent performed a search action.
    
    Looks for patterns like:
    - "click[Search]" or "click[search_button]"
    - "type[search_box][query]" followed by "click[submit]"
    - URL contains "search" or "results"
    
    Why? Performing a search is the first meaningful step in most web tasks.
    """
    search_patterns = [
        r"click\[.*search.*\]",
        r"click\[.*submit.*\]",
        r"type\[.*search.*\]",
        r"press\[.*enter.*\]",
    ]
    for action in actions:
        action_lower = action.lower()
        for pattern in search_patterns:
            if re.search(pattern, action_lower):
                return True
    
    # Also check if any observation URL suggests a results page
    for obs in observations:
        if any(keyword in obs.lower() for keyword in ["results", "search?q=", "listing"]):
            return True
    
    return False


def detect_form_filled(observations: List[str], actions: List[str]) -> bool:
    """
    Detects if the agent filled out form fields.
    
    Looks for multiple "type" actions (indicating form filling).
    Requires at least 2 fields filled to count as meaningful form interaction.
    
    Why? Form filling (source city, destination, date, passenger details)
    is a key intermediate step in booking tasks.
    """
    type_count = sum(1 for a in actions if a.lower().startswith("type["))
    return type_count >= 2


def detect_navigation_to_target(observations: List[str], actions: List[str],
                                 target_keywords: List[str]) -> bool:
    """
    Detects if the agent navigated to a page containing target keywords.
    
    Args:
        target_keywords: List of keywords that indicate the agent reached a
                        relevant page (e.g., ["cart", "checkout", "booking"]).
    
    Why? Reaching the right page (even without completing the task) shows
    the agent is on the right track.
    """
    for obs in observations:
        obs_lower = obs.lower()
        if any(kw in obs_lower for kw in target_keywords):
            return True
    return False


def detect_item_selected(observations: List[str], actions: List[str]) -> bool:
    """
    Detects if the agent selected/clicked on a specific item (bus, train, product).
    
    Looks for "click" actions on items that look like results/listings.
    
    Why? Selecting an item from search results is a key decision point.
    """
    selection_patterns = [
        r"click\[.*view.*seat.*\]",
        r"click\[.*book.*\]",
        r"click\[.*select.*\]",
        r"click\[.*add.*cart.*\]",
        r"click\[.*details.*\]",
    ]
    for action in actions:
        action_lower = action.lower()
        for pattern in selection_patterns:
            if re.search(pattern, action_lower):
                return True
    return False


# =============================================================================
# Main Reward Function
# =============================================================================

def compute_custom_reward(
    trajectory: Dict[str, Any],
    task_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute the composite reward for a single trajectory.
    
    This is the MAIN FUNCTION called by the training pipeline after each rollout.
    
    Args:
        trajectory (dict): Contains:
            - "observations": List[str] — web page states the agent saw
            - "actions": List[str] — actions the agent took
            - "env_reward": float — the environment's binary reward (0 or 1)
            - "num_steps": int — how many steps the agent took
            - "max_steps": int — maximum allowed steps (horizon)
            - "task_id": str — the task identifier
            - "done": bool — whether the episode ended
            
        task_info (dict, optional): Additional task metadata containing
            partial progress milestone definitions.
    
    Returns:
        float: The composite reward (0.0 to 1.6 max).
    
    Reward Breakdown:
        ┌────────────────────────────────┐
        │ Component        │ Range       │
        ├────────────────────────────────┤
        │ Success          │ 0.0 – 1.0   │
        │ Step Efficiency  │ 0.0 – 0.3   │
        │ Partial Progress │ 0.0 – 0.3   │
        │ TOTAL            │ 0.0 – 1.6   │
        └────────────────────────────────┘
    """
    
    # --- Extract trajectory data ---
    observations = trajectory.get("observations", [])
    actions = trajectory.get("actions", [])
    env_reward = trajectory.get("env_reward", 0.0)
    num_steps = trajectory.get("num_steps", len(actions))
    max_steps = trajectory.get("max_steps", 15)
    task_id = trajectory.get("task_id", "unknown")
    
    # =========================================================================
    # Component 1: SUCCESS REWARD
    # =========================================================================
    # The environment returns 1.0 if the task is completed, 0.0 otherwise.
    # This is the primary signal — everything else is supplementary.
    success_reward = float(env_reward)
    
    # =========================================================================
    # Component 2: STEP EFFICIENCY BONUS
    # =========================================================================
    # Only awarded if the task was successful.
    # 
    # Formula: efficiency = 0.3 * (max_steps - steps_used) / max_steps
    #
    # Example (max_steps = 15):
    #   - Completed in 5 steps:  0.3 * (15 - 5) / 15 = 0.20
    #   - Completed in 10 steps: 0.3 * (15 - 10) / 15 = 0.10
    #   - Completed in 15 steps: 0.3 * (15 - 15) / 15 = 0.00
    #
    # Intuition: We want the agent to be efficient — find the shortest path
    # to completing the task, not wander around.
    if success_reward > 0 and max_steps > 0:
        efficiency_bonus = 0.3 * max(0, (max_steps - num_steps) / max_steps)
    else:
        efficiency_bonus = 0.0
    
    # =========================================================================
    # Component 3: PARTIAL PROGRESS REWARD
    # =========================================================================
    # Even if the task wasn't completed, reward intermediate milestones.
    # This is CRITICAL for early training when the agent can't complete
    # any tasks yet.
    #
    # We detect milestones from the trajectory and sum their rewards.
    # The total is capped at 0.3 to ensure it doesn't dominate the
    # success signal.
    partial_progress = 0.0
    
    # Only compute partial progress if the task was NOT successful.
    # (If it was successful, the success reward already covers everything.)
    if success_reward == 0:
        # Generic milestone detection (works for any task)
        milestones_detected = []
        
        # Milestone 1: Did the agent fill out form fields?
        if detect_form_filled(observations, actions):
            partial_progress += 0.05
            milestones_detected.append("form_filled")
        
        # Milestone 2: Did the agent perform a search?
        if detect_search_performed(observations, actions):
            partial_progress += 0.08
            milestones_detected.append("search_performed")
        
        # Milestone 3: Did the agent select an item?
        if detect_item_selected(observations, actions):
            partial_progress += 0.10
            milestones_detected.append("item_selected")
        
        # Milestone 4: Did the agent reach a checkout/booking page?
        checkout_keywords = ["checkout", "booking", "passenger", "payment",
                           "confirm", "seat", "cart", "summary"]
        if detect_navigation_to_target(observations, actions, checkout_keywords):
            partial_progress += 0.07
            milestones_detected.append("reached_checkout")
        
        # -------------------------------------------------------------------
        # Task-specific milestones (from custom_india_tasks.json)
        # -------------------------------------------------------------------
        # If the task has custom milestone definitions, use them for more
        # precise partial progress tracking.
        if task_info and "evaluation_criteria" in task_info:
            task_milestones = task_info["evaluation_criteria"].get(
                "partial_progress_milestones", []
            )
            # Custom milestones override generic detection
            if task_milestones:
                partial_progress = 0.0  # Reset generic score
                milestones_detected = []
                for milestone in task_milestones:
                    # In a full implementation, each milestone would have
                    # its own detector. For now, we use the generic detectors
                    # mapped to milestone names.
                    milestone_name = milestone.get("milestone", "")
                    milestone_reward = milestone.get("reward", 0.0)
                    
                    if _check_milestone(milestone_name, observations, actions):
                        partial_progress += milestone_reward
                        milestones_detected.append(milestone_name)
        
        # Cap partial progress at 0.3
        partial_progress = min(partial_progress, 0.3)
    
    # =========================================================================
    # Combine all components
    # =========================================================================
    total_reward = success_reward + efficiency_bonus + partial_progress
    
    return total_reward


def _check_milestone(milestone_name: str, observations: List[str],
                     actions: List[str]) -> bool:
    """
    Check if a specific named milestone was reached.
    
    This maps milestone names (from task JSON) to detection logic.
    
    Args:
        milestone_name: The name of the milestone to check.
        observations: List of web page observations.
        actions: List of actions taken.
    
    Returns:
        bool: True if the milestone was reached.
    """
    # Map milestone names to detection functions
    milestone_detectors = {
        # Route/location entry
        "entered_source_city": lambda o, a: _detect_typed_field(a, ["from", "source", "origin", "departure"]),
        "entered_destination_city": lambda o, a: _detect_typed_field(a, ["to", "destination", "arrival"]),
        "entered_source_station": lambda o, a: _detect_typed_field(a, ["from", "source", "origin"]),
        "entered_destination_station": lambda o, a: _detect_typed_field(a, ["to", "destination"]),
        "entered_route": lambda o, a: _detect_typed_field(a, ["from", "source"]) and _detect_typed_field(a, ["to", "destination"]),
        
        # Date selection
        "selected_date": lambda o, a: any(re.search(r"click\[.*date.*\]|type\[.*date.*\]", act.lower()) for act in a),
        
        # Search
        "clicked_search": lambda o, a: detect_search_performed(o, a),
        "clicked_check_status": lambda o, a: detect_search_performed(o, a),
        
        # Results
        "reached_results": lambda o, a: detect_navigation_to_target(o, a, ["results", "listing", "buses", "trains"]),
        "browsed_results": lambda o, a: sum(1 for act in a if "scroll" in act.lower() or "click" in act.lower()) >= 3,
        
        # Filters
        "applied_filter": lambda o, a: any(re.search(r"click\[.*filter.*\]|click\[.*ac.*\]|click\[.*sleeper.*\]", act.lower()) for act in a),
        "applied_volvo_filter": lambda o, a: any(re.search(r"click\[.*volvo.*\]|click\[.*bus.*type.*\]", act.lower()) for act in a),
        "sorted_by_price": lambda o, a: any(re.search(r"click\[.*sort.*\]|click\[.*price.*\]|click\[.*cheap.*\]", act.lower()) for act in a),
        
        # Selection
        "selected_bus": lambda o, a: detect_item_selected(o, a),
        "found_rajdhani": lambda o, a: detect_navigation_to_target(o, a, ["rajdhani"]),
        "found_high_rated_bus": lambda o, a: detect_navigation_to_target(o, a, ["rating", "stars", "4."]),
        "found_most_reviewed": lambda o, a: detect_navigation_to_target(o, a, ["review", "rating"]),
        
        # Seat / availability
        "clicked_view_seats": lambda o, a: any(re.search(r"click\[.*seat.*\]|click\[.*view.*\]", act.lower()) for act in a),
        "selected_seat": lambda o, a: any(re.search(r"click\[.*seat.*\d+.*\]|click\[.*window.*\]", act.lower()) for act in a),
        "selected_window_seat": lambda o, a: any(re.search(r"click\[.*window.*\]", act.lower()) for act in a),
        "checked_availability": lambda o, a: detect_navigation_to_target(o, a, ["availability", "available", "waitlist", "confirm"]),
        "compared_classes": lambda o, a: sum(1 for obs in o if any(c in obs.lower() for c in ["1ac", "2ac", "3ac", "sl"])) >= 2,
        "identified_cheapest": lambda o, a: detect_navigation_to_target(o, a, ["cheapest", "lowest", "price"]),
        
        # PNR specific
        "navigated_to_pnr_page": lambda o, a: detect_navigation_to_target(o, a, ["pnr"]),
        "entered_pnr_number": lambda o, a: any(re.search(r"type\[.*pnr.*\]|type\[.*\d{10}.*\]", act.lower()) for act in a),
        "read_booking_status": lambda o, a: detect_navigation_to_target(o, a, ["status", "confirmed", "waitlist", "rac"]),
        "read_prediction": lambda o, a: detect_navigation_to_target(o, a, ["prediction", "probability", "chance"]),
        
        # Amenities
        "viewed_amenities": lambda o, a: detect_navigation_to_target(o, a, ["amenities", "facilities", "wifi", "charging"]),
        
        # Checkout / booking
        "reached_checkout": lambda o, a: detect_navigation_to_target(o, a, ["checkout", "payment", "booking", "passenger"]),
        "proceeded_to_booking": lambda o, a: detect_navigation_to_target(o, a, ["booking", "confirm", "proceed"]),
    }
    
    detector = milestone_detectors.get(milestone_name)
    if detector:
        try:
            return detector(observations, actions)
        except Exception:
            return False
    
    # Unknown milestone — skip
    return False


def _detect_typed_field(actions: List[str], field_keywords: List[str]) -> bool:
    """
    Detect if the agent typed into a field matching any of the keywords.
    
    A "type" action looks like: type[field_name][value]
    We check if the field_name contains any of the keywords.
    """
    for action in actions:
        action_lower = action.lower()
        if action_lower.startswith("type["):
            for keyword in field_keywords:
                if keyword in action_lower:
                    return True
    return False


# =============================================================================
# Batch Reward Computation (for integration with training pipeline)
# =============================================================================

def compute_batch_rewards(
    trajectories: List[Dict[str, Any]],
    task_infos: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[float]:
    """
    Compute rewards for a batch of trajectories.
    
    This is the batch version called by the training loop.
    
    Args:
        trajectories: List of trajectory dicts.
        task_infos: Optional dict mapping task_id → task metadata.
    
    Returns:
        List of reward floats.
    """
    rewards = []
    for traj in trajectories:
        task_id = traj.get("task_id", "unknown")
        task_info = task_infos.get(task_id) if task_infos else None
        reward = compute_custom_reward(traj, task_info)
        rewards.append(reward)
    return rewards


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    # Demo: test the reward function with a mock trajectory
    print("=" * 60)
    print("Custom Reward Function — Demo")
    print("=" * 60)
    
    # Scenario 1: Successful task completion in few steps
    traj_success = {
        "observations": ["RedBus homepage", "Bus results: Pune to Mumbai", "Seat selection page"],
        "actions": [
            'type[source_city][Pune]',
            'type[destination_city][Mumbai]',
            'click[search_buses]',
            'click[view_seats_bus_1]',
            'click[seat_A1]',
            'click[proceed_to_booking]',
        ],
        "env_reward": 1.0,
        "num_steps": 6,
        "max_steps": 15,
        "task_id": "india_redbus_001",
    }
    
    reward1 = compute_custom_reward(traj_success)
    print(f"\n✅ Scenario 1 — Success in 6/15 steps:")
    print(f"   Success:     1.0")
    print(f"   Efficiency:  {0.3 * (15 - 6) / 15:.2f}")
    print(f"   Partial:     0.00 (not needed — task succeeded)")
    print(f"   TOTAL:       {reward1:.2f}")
    
    # Scenario 2: Partial progress (got to results but didn't complete)
    traj_partial = {
        "observations": ["RedBus homepage", "Bus results: Pune to Mumbai"],
        "actions": [
            'type[source_city][Pune]',
            'type[destination_city][Mumbai]',
            'click[search_buses]',
            'scroll[down]',
            'click[wrong_button]',
        ],
        "env_reward": 0.0,
        "num_steps": 5,
        "max_steps": 15,
        "task_id": "india_redbus_001",
    }
    
    reward2 = compute_custom_reward(traj_partial)
    print(f"\n⚠️  Scenario 2 — Partial progress (searched but didn't complete):")
    print(f"   Success:     0.0")
    print(f"   Efficiency:  0.00 (task failed)")
    print(f"   Partial:     {reward2:.2f} (detected: form fill + search)")
    print(f"   TOTAL:       {reward2:.2f}")
    
    # Scenario 3: No progress (agent did nothing useful)
    traj_nothing = {
        "observations": ["RedBus homepage"],
        "actions": [
            'scroll[down]',
            'scroll[up]',
            'click[random_link]',
        ],
        "env_reward": 0.0,
        "num_steps": 3,
        "max_steps": 15,
        "task_id": "india_redbus_001",
    }
    
    reward3 = compute_custom_reward(traj_nothing)
    print(f"\n❌ Scenario 3 — No meaningful progress:")
    print(f"   Success:     0.0")
    print(f"   Efficiency:  0.00")
    print(f"   Partial:     {reward3:.2f}")
    print(f"   TOTAL:       {reward3:.2f}")
    
    print(f"\n" + "=" * 60)
    print(f"Reward comparison:")
    print(f"  Success + Efficient: {reward1:.2f}")
    print(f"  Partial Progress:    {reward2:.2f}")
    print(f"  No Progress:         {reward3:.2f}")
    print(f"=" * 60)
