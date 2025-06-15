# Learning Controlled Robot Contact-Rich Manipulation via Human Demonstration
This project introduces a reinforcement learning benchmark for contact-rich robotic manipulation, particularly focusing on pressure-sensitive tasks like wiping or rubbing. The setup supports multi-modal input (vision, proprioception, tactile) and learns from human demonstrations to enable generalizable contact behavior.
This repository includes:
- Multi-task learning setup across tactile contact goals
- Pressure-conditioned instructions and semantic goal vectors
- Unified training/evaluation across sensor modalities
- Sim-to-real compatible robot configurations (e.g., Unitree H1 Dual-arm with tactile hands)




## 🛠️ Setup
### Clone the Repository 
```bash
git clone git@github.com:CILAB-ROBOTICS/humanoid-bench.git
cd humanoid-bench
cp .env.example .env
```

### Configure API Key (for logging with [Weight & Biases](https://wandb.ai/))
Replace the placeholder with your actual WandB API key:

```
WANDB_API_KEY=wandb_key => ecd68e29f838358...
```

## 🚀 Training

### 🐳 Docker Image
We provide a pre-built Docker image with CUDA 12 and all required libraries installed:

```
docker pull bic4907/humanbench:cu12
```

### 🧠 Default Training: Pressure-conditioned Vector Input


The default setup uses **scalar instruction vectors** indicating pressure levels (`0.2`, `0.6`, `1.0`). Example:
The default arguments for the pressure condition is `instruct=scn-1_se-0` and `modality=vector`.
Use `overwrite=True` to overwrite existing checkpoints and start training from scratch:
```
gpu=0 bash run.sh python -m tdmpc2.train task=humanoid_h1dualarm-rub-v0 seed=0
```

### 🧩 Multi-modal Training
To train with a combination of vision, tactile, and proprioceptive observations:

```
gpu=0 bash run.sh python -m tdmpc2.train \
    task=humanoid_h1dualarm-rub-v0 \
    seed=0 \
    sensors=proprio/image/tactile
```

**Available Modalities:**
- `proprio`: robot joint positions/velocities
- `image`: front-facing RGB camera
- `tactile`: tactile sensors on hands


## 📈 Sweep (Batch Training)
For hyperparameter tuning or large-scale experiment sweeps, refer to the following documentation: [sweep/README](./sweep/README.md)


## Evaluation
```
gpu=0 bash run.sh python -m tdmpc2.eval task=humanoid_h1dualarm-rub-v0 seed=0 
```


## References
This codebase contains some files adapted from other sources:
* jaxrl_m: https://github.com/dibyaghosh/jaxrl_m/tree/main
* TD-MPC2: https://github.com/nicklashansen/tdmpc2
* Digit models: https://github.com/adubredu/KinodynamicFabrics.jl/tree/sim
* MuJoCo Menagerie (Unitree H1, Shadow Hands, Robotiq 2F-85 models): https://github.com/google-deepmind/mujoco_menagerie
* Robosuite (some texture files): https://github.com/ARISE-Initiative/robosuite