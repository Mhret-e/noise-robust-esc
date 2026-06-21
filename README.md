# Denoiser–Representation Interaction in Noise-Robust ESC

A systematic study of classical (MMSE-LSA, adaptive Wiener filtering) and learned (U-Net) audio enhancement methods across three architecturally distinct environmental sound classification (ESC) pipelines, evaluated on the ESC-50 dataset under additive white Gaussian noise (AWGN) and real-world traffic noise.

**Authors:** Mhretewold Endashaw, Eden Yoseph, Kalkidan Getnet, Misgana Damtew, Imad Zyout
Engineering Technology & Science, Higher Colleges of Technology, Abu Dhabi, UAE

📄 Full paper: [`paper/AEECT2026_Speech_Updated_V1.pdf`](paper/AEECT2026_Speech_Updated_V1.pdf)

---

## Overview

Modern ESC systems degrade significantly under real-world noise. This project investigates whether the choice of denoiser interacts with the downstream audio representation — i.e., whether a given enhancement method helps or hurts depends on *how* the audio is fed into the classifier.

Three pipelines are evaluated:

| Pipeline | Representation | Classifier |
|---|---|---|
| 1 | Raw waveform | 1D CNN (4 conv blocks + SE attention) |
| 2 | MFCC + Δ + ΔΔ (statistical aggregation) | Subspace Discriminant Ensemble |
| 3 | Log-mel spectrogram | CNN14 (AudioSet-pretrained, PANNs) |

Each pipeline is tested with **two enhancement strategies**: a classical DSP method matched to that representation, and a learned U-Net spectral-mask denoiser.

| Pipeline | Classical method | Learned method |
|---|---|---|
| 1D CNN | MMSE-LSA | U-Net |
| MFCC Ensemble | Adaptive Wiener filter | U-Net |
| CNN14 | MMSE-LSA | U-Net |

---

## Repository structure

```
noise-robust-esc/
├── paper/
│   └── AEECT2026_Speech_Updated_V1.pdf
├── notebooks/
│   ├── Data_Exploration.ipynb
│   ├── 1d_cnn/
│   │   ├── 1D_CNN_mmse_lsa.ipynb
│   │   └── 1D_CNN_unet.ipynb
│   ├── mfcc/
│   │   ├── mfcc_wiener_filter.ipynb
│   │   └── mfcc_unet.ipynb
│   └── cnn14/
│       ├── cnn14_mmse_lsa.ipynb
│       └── cnn14_unet.ipynb
├── figures/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

[ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 labeled environmental audio clips, 50 classes, 40 samples/class, 5s clips @ 44.1 kHz, 5 predefined cross-validation folds. Not included in this repo — download separately and update notebook paths.

Real-world traffic noise was sourced from [Freesound.org](https://freesound.org/search/?q=traffic) ("Crowd traffic under bridge in Cairo").

---

## Key results

### Clean audio baseline (5-fold CV)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 1D CNN | 48.34% ± 4.8 | 53.9% | 47.5% | 46.7% |
| MFCC Ensemble | 54.00% ± 0.74 | 57.7% | 55.8% | 54.4% |
| CNN14 (PANNs) | **86.95% ± 2.5** | 89.4% | 87.5% | 86.8% |

### Noise robustness highlights

- **U-Net generalizes best across all pipelines and both noise types**, with traffic-noise gains statistically significant at every tested SNR (p < 0.0001).
- **CNN14 + U-Net** recovers up to **+24.30 points** at −5 dB SNR under traffic noise.
- **MMSE-LSA** helps under stationary Gaussian noise but **degrades CNN14 performance under non-stationary traffic noise** (up to −25.85 points at 0 dB), violating its stationarity assumption.
- **MFCC + Adaptive Wiener filter** gives small, stable gains; **MFCC + MMSE-LSA degrades performance at all tested conditions** — the clearest evidence of denoiser–representation mismatch.

Full per-SNR tables are in the paper (Tables IV–VI) and reproduced in [`results/`](results/).

---

## How to Run

These notebooks were developed and tested in **Google Colab** and are the recommended way to run this project.

### Prerequisites
- A Google account with Google Drive access
- GPU runtime enabled for the CNN14 notebooks: `Runtime → Change runtime type → T4 GPU`
  (CNN14 fine-tuning takes ~30–60 min on a free T4 GPU; CPU-only execution is not recommended)

### Step 1 — Get the dataset
Download [ESC-50](https://github.com/karolpiczak/ESC-50) and upload it to your Google Drive at:
```
MyDrive/DSP Project/esc50_data/ESC-50-master/
├── audio/        ← all 2,000 .wav files
└── meta/
    └── esc50.csv
```

### Step 2 — Upload notebooks to Colab
Upload the `notebooks/` folder contents to Colab, or open them directly from Drive.

### Step 3 — Run in order
1. `Data_Exploration.ipynb`
2. `1d_cnn/1D_CNN_mmse_lsa.ipynb` then `1d_cnn/1D_CNN_unet.ipynb`
3. `mfcc/mfcc_wiener_filter.ipynb` then `mfcc/mfcc_unet.ipynb`
4. `cnn14/cnn14_mmse_lsa.ipynb` then `cnn14/cnn14_unet.ipynb`

> **Running locally?** The notebooks were not built for local execution and currently expect a mounted Google Drive path. They *can* be adapted — swap the `drive.mount()` cell for a local data path and ensure `requirements.txt` is installed — but this hasn't been tested by us, so expect to debug paths and package versions yourself.

Each notebook trains/evaluates one pipeline under one enhancement strategy across SNR levels {−10, −5, 0, 5, 10} dB.

---

## Testing on your own audio file

After training is complete, you can classify a custom `.wav` file:

```python
import librosa
import numpy as np
from google.colab import files

uploaded = files.upload()  # click to upload your .wav file
filename = list(uploaded.keys())[0]
y, sr = librosa.load(filename, sr=22050, mono=True)

# Pad or truncate to 5 seconds
MAX_LEN = 110250
y = np.pad(y, (0, max(0, MAX_LEN - len(y))))[:MAX_LEN]

# 1D CNN
y_input = y[np.newaxis, :, np.newaxis]
pred = final_model.predict(y_input, verbose=0)
print(f"Recognized sound: {le.classes_[np.argmax(pred)]} ({np.max(pred)*100:.1f}% confidence)")

# MFCC Ensemble (alternative)
features = extract_features_from_wave(y)
features_scaled = scaler.transform([features])
print(f"Recognized sound: {model.predict(features_scaled)[0]}")
```

---

## Tools & Technologies

| Category | Tool / Library | Purpose |
|---|---|---|
| Platform | Google Colab | Cloud notebook environment with free GPU |
| Language | Python 3.10+ | All code |
| Audio loading | librosa | Load, resample, normalize audio |
| Audio (Mel) | torchaudio | Mel spectrogram extraction for CNN14 |
| Deep Learning | TensorFlow / Keras | 1D CNN model |
| Deep Learning | PyTorch | CNN14 (PANNs) and U-Net models |
| Classical ML | scikit-learn | Subspace Discriminant Ensemble, scalers, metrics |
| DSP | scipy.signal | Wiener filter |
| DSP | scipy.special | exp1 function for MMSE-LSA gain computation |
| DSP | NumPy FFT | STFT/iSTFT |
| Data | pandas | Metadata loading and management |
| Visualization | matplotlib, seaborn | Plots, spectrograms, confusion matrices |
| Dataset | ESC-50 | 2,000 environmental audio clips, 50 classes |
| Pretrained Model | CNN14 (PANNs) | Pretrained on AudioSet (2M+ clips, 527 classes) |

> Cross-check this list against your notebooks' actual `import` statements before finalizing `requirements.txt`.

---

## Team

| Name | Student ID | Email |
|---|---|---|
| Eden Yoseph | H00540243 | h00540243@hct.ac.ae |
| Kalkidan Getnet | H00540242 | h00540242@hct.ac.ae |
| Mhretewold Endashaw | H00540239 | h00540239@hct.ac.ae |
| Misgana Damtew | H00540241 | h00540241@hct.ac.ae |

**Institution:** Higher Colleges of Technology — Faculty of Engineering Technology, Electrical Engineering Department, Sharjah, UAE
**Supervisor:** Dr. Imad Zyout

---

## Future Work

This study already evaluates both synthetic AWGN and real-world non-stationary traffic noise. Building on these results, future work will:

- Investigate end-to-end joint optimization of the denoising and classification models, rather than training them as separate stages
- Evaluate additional learned enhancement architectures beyond U-Net
- Extend the analysis to larger real-world acoustic datasets (e.g., UrbanSound8K, AudioSet subsets) and a more diverse range of environmental noise conditions (e.g., cafeteria, babble, music) beyond traffic noise
- Explore model compression (distillation, quantization) for edge/embedded deployment
- Extend to polyphonic / overlapping sound event classification
- Investigate multi-channel audio with beamforming as a spatial preprocessing stage

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{endashaw2026denoiser,
  title     = {Denoiser--Representation Interaction in Noise-Robust ESC: A Systematic Study of Classical and Learned Enhancement Methods},
  author    = {Endashaw, Mhretewold and Yoseph, Eden and Getnet, Kalkidan and Damtew, Misgana and Zyout, Imad},
  booktitle = {AEECT 2026},
  year      = {2026}
}
```

> No license is currently attached to this repository — all rights reserved by default. The code is publicly viewable, but reuse or redistribution requires permission from the authors. This may be revisited once the paper is published.
