"""A compact PPO implementation for discrete-action Gymnasium environments.

Modules:
    networks  actor and critic neural networks
    buffer    rollout storage and GAE (NumPy only)
    agent     PPOAgent: action selection, PPO update, checkpoints
    rollout   training-data collection and policy evaluation loops
"""
