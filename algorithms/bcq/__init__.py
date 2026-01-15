"""BCQ (Batch-Constrained Q-Learning) Algorithm Implementation

This module provides a complete implementation of BCQ for offline reinforcement learning,
specifically designed for ICU drug dosing optimization.

Key Components:
- bcq_models: Neural network architectures (Q-networks, VAE)
- bcq_losses: Loss functions for BCQ training
- bcq_train_utils: Training utilities and BCQTrainer class
- train_bcq: End-to-end training pipeline
- evaluate_bcq: Model evaluation and policy analysis

Example Usage:
    from algorithms.bcq.bcq_models import BCQAgent
    from algorithms.bcq.train_bcq import train_bcq
    
    config = {
        'data_path': 'ready_data1.csv',
        'epochs': 30,
        'batch_size': 256,
    }
    
    train_bcq(config, output_dir='./bcq_output')

References:
    Fujimoto, S., Meger, D., & Precup, D. (2019). 
    "Off-Policy Deep Reinforcement Learning without Exploration." 
    In ICML 2019.
"""

__version__ = "1.0.0"
__author__ = "BCQ Implementation Team"
__all__ = [
    "bcq_models",
    "bcq_losses",
    "bcq_train_utils",
    "train_bcq",
    "evaluate_bcq",
]
