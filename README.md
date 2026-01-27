# SP CUP Phase 2 - Audio Source Separation with DCCRN-Conformer

A deep learning solution for angle-conditioned audio source separation using a DCCRN-Conformer architecture.

---

## 📁 Project Structure

```
SP_CUP_Phase_2/
├── Dataset Generation/          # MATLAB scripts for dataset creation
│   ├── train_anechoic.mlx       # Training dataset (anechoic conditions)
│   ├── train_reverb.mlx         # Training dataset (reverberant conditions)
│   ├── test_anechoic.mlx        # Test dataset (anechoic conditions)
│   └── test_reverb.mlx          # Test dataset (reverberant conditions)
│
├── Model Inference/             # Python training and inference scripts
│   ├── train_Conformer.py       # Training script
│   ├── test_Conformer.py        # Evaluation script
│   ├── inference_Conformer.py   # Single-file inference script
│   ├── anechoic_Conformer.pth   # Pretrained model (anechoic)
│   └── reverb_Conformer.pth     # Pretrained model (reverberant)
│
├── Test_Dataset/                # Test datasets (gitignored)
│   ├── anechoic/                # Anechoic test samples
│   └── reverb/                  # Reverberant test samples
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🔧 Requirements

### Python Dependencies

Install all dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

Or install manually with these **stable, tested versions**:

```bash
pip install torch==2.1.0 torchaudio==2.1.0 torchmetrics==1.2.0 numpy==1.26.4 tqdm soundfile
```

| Package | Version | Notes |
|---------|---------|-------|
| Python | >= 3.8 | Tested with 3.10 |
| torch | 2.1.0 | Stable PyTorch |
| torchaudio | 2.1.0 | Audio I/O (avoid 2.10+ which requires torchcodec) |
| torchmetrics | 1.2.0 | PESQ, STOI, SI-SDR metrics |
| numpy | 1.26.4 | Numerical operations |
| soundfile | latest | Fallback audio I/O |
| tqdm | latest | Progress bars |

> ⚠️ **Note:** Avoid torchaudio >= 2.6.0 as it requires `torchcodec` which has compatibility issues.

### MATLAB (for dataset generation)

- MATLAB R2020b or later
- Signal Processing Toolbox

---

## 🚀 Pipeline Overview

### Step 1: Dataset Generation (MATLAB)

Generate synthetic training/test datasets with room acoustics simulation.

```
Dataset Generation/
├── train_anechoic.mlx   → Generates anechoic training samples
├── train_reverb.mlx     → Generates reverberant training samples  
├── test_anechoic.mlx    → Generates anechoic test samples
└── test_reverb.mlx      → Generates reverberant test samples
```

**Output Format** (per sample folder):
```
sample_XXXXX/
├── mixture.wav        # Stereo mixture (target + interferer)
├── target.wav         # Ground-truth target audio
└── meta.json          # Metadata: {"target_angle": 45.0, "interf_angle": 135.0}
```

---

### Step 2: Training

Edit configuration in `train_Conformer.py`:

```python
# Configuration (lines 18-29)
DATASET_ROOT = r"./your_training_dataset"   # Path to dataset
BATCH_SIZE = 4                              # Adjust for GPU memory
LEARNING_RATE = 2e-5                        # Learning rate
N_EPOCHS = 10                               # Training epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resume from checkpoint (or set to None to train from scratch)
RESUME_FROM = "DCCRN_Conformer.pth"  # or None
```

**Run Training:**

```bash
cd "Model Inference"
python train_Conformer.py
```

**Output:**
- `DCCRN_Conformer.pth` - Saved model checkpoint (best validation loss)

---

### Step 3: Evaluation

Evaluate model performance on test dataset with SI-SDR, STOI, and PESQ metrics.

Edit configuration in `test_Conformer.py`:

```python
# Configuration (lines 17-23)
MODEL_PATH = "DCCRN_Conformer.pth"           # Model checkpoint
TEST_DATASET_ROOT = r"D:\test_dataset"       # Path to test dataset
OUTPUT_DIR = "evaluation_Conformer"          # Output directory
```

**Run Evaluation:**

```bash
cd "Model Inference"
python test_Conformer.py
```

**Output:**
- Console report with best SI-SDR, STOI, PESQ scores
- Audio files saved to `evaluation_Conformer/`:
  - `BEST_OVERALL_output.wav` - Model output
  - `BEST_OVERALL_mixture.wav` - Input mixture
  - `BEST_OVERALL_target.wav` - Ground truth

---

### Step 4: Single-File Inference

Process a single audio file with a specified target angle.

**Usage:**

```bash
cd "Model Inference"
python inference_Conformer.py --input mixture.wav --angle 45 --output output.wav
```

**Arguments:**

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input stereo audio file | Required |
| `--angle` | `-a` | Target angle (0-180°) | Required |
| `--output` | `-o` | Output audio file | `output.wav` |
| `--model` | `-m` | Model checkpoint path | `DCCRN_Conformer.pth` |
| `--device` | `-d` | Device (cpu/cuda) | `cpu` |

**Example:**

```bash
# Extract source at 90 degrees using GPU
python inference_Conformer.py -i stereo_mix.wav -a 90 -o extracted.wav -d cuda
```

> **Note:** Audio is automatically trimmed/padded to 3 seconds (model's fixed input size)

---

## 🏗️ Model Architecture

**DCCRN-Conformer** (~10M parameters)

- **Encoder**: Complex 2D convolutions (48 → 96 → 192 → 256 channels)
- **Bottleneck**: Dual-path Conformer (frequency + time processing)
- **Decoder**: Complex transposed convolutions with skip connections
- **Angle Conditioning**: MLP-based angle embedding injection

**Audio Processing:**
- Sample Rate: 16 kHz
- STFT: n_fft=512, hop_length=128
- Fixed Duration: 3 seconds

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio (dB) |
| **STOI** | Short-Time Objective Intelligibility (0-1) |
| **PESQ** | Perceptual Evaluation of Speech Quality (-0.5 to 4.5) |

---

## 📋 Quick Start

```bash
# 1. Clone/navigate to the repo
cd SP_CUP_Phase_2

# 2. (Recommended) Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run inference with pretrained model
cd "Model Inference"
python inference_Conformer.py -i your_audio.wav -a 45 -o output.wav -m reverb_Conformer.pth

# 5. (Optional) Evaluate on test set
python test_Conformer.py

# 6. (Optional) Train your own model
python train_Conformer.py
```

---

## 📝 Notes

- **Stereo Input Required**: Input audio must be stereo (2 channels). Mono files are automatically duplicated.
- **GPU Recommended**: Training and evaluation are significantly faster on CUDA-enabled GPUs.
- **Model Selection**: Use `anechoic_Conformer.pth` for clean audio, `reverb_Conformer.pth` for reverberant environments.

---

## 📄 License

This project is developed for the Signal Processing Cup competition.
