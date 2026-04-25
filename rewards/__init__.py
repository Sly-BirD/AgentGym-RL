"""
Rewards package for RL-WebNav Agent.

This package contains custom reward functions used during GRPO/PPO training.
The default environment reward is binary (0 or 1), but custom rewards provide
richer learning signals through:

1. Success reward (0 or 1.0) — did the task complete?
2. Step efficiency bonus (0 to 0.3) — fewer steps = higher bonus
3. Partial progress (0 to 0.3) — credit for intermediate milestones
"""
