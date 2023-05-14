import wandb

# Example sweep configuration
sweep_configuration = {
    'program': 'main.py' 
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'train_l1_loss'
        },
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="MemSeg")
