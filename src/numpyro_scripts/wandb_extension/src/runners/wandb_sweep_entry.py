import wandb
from run_experiment import main

if __name__ == "__main__":
    # Initialize W&B run from sweep agent
    run = wandb.init(config=None, allow_val_change=True)

    # Pull config dict
    cfg = dict(run.config)

    # Default values if sweep config doesn’t include them
    cfg.setdefault("project", "dict-learning")
    cfg.setdefault("entity", None)

    # Run single experiment
    main(**cfg)
