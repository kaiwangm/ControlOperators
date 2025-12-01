# Simplified Reference Implementation

This repository contains a simplified reference implementation for our Siggraph Asia 2025 paper ["Control Operators for Interactive Character Animation"](https://theorangeduck.com/page/control-operators-interactive-character-animation). Note that it is not the exact implementation as in the paper or paper videos, as the original implementation was developed integrated within Unreal Engine. Here we provide an example of implementing some core ideas in Python:

- implementation and example usage of [control operators](https://theorangeduck.com/page/implementing-control-operators)
- training and interactive testing of a flow-matching-based autoregressive character controller on [lafan1-resolved](https://github.com/orangeduck/lafan1-resolved) dataset


## Preview


https://github.com/user-attachments/assets/2dc06ffd-0b89-440c-a216-c9edbd3d90a1



### Environment setup:
Note that the current environment is tested on Windows/Linux.
```powershell
uv sync
# or
pip install -r requirements.txt
```

### (optional) Download Pretrained Models to run the demo

- [control_operators_pretrained.zip](https://drive.google.com/file/d/1nr4B9fnDllYe01fpwx4bYc89NOO3_CIE/view?usp=sharing)
- Extract to: `data/lafan1_resolved/`
- Running the demo requires the data directory to look like this:
```
├── data/
│   ├── lafan1_resolved/
│   │   ├── X.npz
│   │   ├── Z.npz
│   │   ├── autoencoder.ptz
│   │   ├── database.npz
│   │   ├── UberControlEncoder
│   │   │   └── controller.ptz
```

### Demo (`controller.py`)

- The viewer displays the character and uses the trained networks to drive animation - you will need a gamepad (tested with a xbox controller) to test the modes other than uncontrolled:

```powershell
uv run controller.py # or python controller.py
```

### Training (`train.py`)

- To train:

```powershell
uv run train.py # or python train.py
```
