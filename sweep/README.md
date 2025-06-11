### Create Sweep ID
**Command**
```bash
bash run.sh wandb sweep sweep/tdmpc2_train.yaml
```

**Result**
```
wandb: Creating sweep from: sweep/tdmpc2_train.yaml
wandb: Creating sweep with ID: f2ffjhx1
wandb: View sweep at: https://wandb.ai/cilab-robot/humanoid-bench/sweeps/f2ffjhx1
wandb: Run sweep agent with: wandb agent cilab-robot/humanoid-bench/f2ffjhx1
```

`f2ffjhx1` is the `sweep id` in this example

### Run Sweep Agent
**Command**
```bash
gpu=0 bash run.sh wandb agent cilab-robot/humanoid-bench/f2ffjhx1
gpu=1 bash run.sh wandb agent cilab-robot/humanoid-bench/f2ffjhx1
(specify gpu id)
```
Multiple run of `wandb agent` is possible in a single gpu. (e.g., running two sweep agents in `gpu=0`)

**Result**
```
wandb: Starting wandb agent 🕵️
2025-05-09 05:36:00,872 - wandb.wandb_agent - INFO - Running runs: []
2025-05-09 05:36:01,560 - wandb.wandb_agent - INFO - Agent received command: run
2025-05-09 05:36:01,560 - wandb.wandb_agent - INFO - Agent starting run with config:
        seed: 0
        task: humanoid_h1dualarm-rub-v0
2025-05-09 05:36:01,561 - wandb.wandb_agent - INFO - About to run command: python -m tdmpc2.train seed=0 task=humanoid_h1dualarm-rub-v0
Work dir: /workspace/logs/humanoid_h1dualarm-rub-v0/0/default
small obs start: None
...
```