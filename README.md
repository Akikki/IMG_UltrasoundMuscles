# UMUD – Muscle Architecture in Ultrasound Data

This project is developed as part of the **Image Processing course** in the **M1 Robotics program at Sorbonne Université**.

The objective is to analyze ultrasound images of muscles to automatically estimate three anatomical measurements:
- **Pennation Angle (PA)**
- **Fascicle Length (FL)**
- **Muscle Thickness (MT)**

Dataset and competition: [UMUD Challenge – Kaggle](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data)

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/data) and extract it into the `data/` directory.

The `data/` and `.venv/` directories should be excluded from Git using `.gitignore`.
