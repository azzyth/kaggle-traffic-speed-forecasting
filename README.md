# Traffic Speed Forecasting   Kaggle Datathon
Yea this is my first kaggle competition everybody... someone gotta start somewhere right (¬`‸´¬)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

competition link: https://www.kaggle.com/competitions/datathon-task-1/leaderboard

Forecasting **future traffic speed** (km/h) for 1,260 connected road segments, 20 / 40 / 60 minutes ahead, by combining **time-series speed history**, **natural-language event descriptions**, and a **road-network adjacency matrix**.

> **Result:** 234th out of 276 teams · Team **TelyuAlgo**

---

## Overview

Traffic does not move at random. It responds to the road network, to the time of day, and to events unfolding across the city   accidents, closures, weather, gatherings   all described in a running stream of text.

Given one hour of recent speed history across 1,260 connected road segments and a live feed of event text, the goal is to forecast how fast traffic will move **20, 40, and 60 minutes into the future**.

## Ringkasan (Bahasa Indonesia)

Lalu lintas tidak bergerak acak   ia dipengaruhi jaringan jalan, jam, dan kejadian (kecelakaan, penutupan jalan, cuaca, keramaian) yang tertulis dalam teks berita. Dari riwayat kecepatan 1 jam terakhir pada 1.260 ruas jalan + teks kejadian, kita memprediksi kecepatan **20, 40, dan 60 menit ke depan**.

> **Hasil:** peringkat 234 dari 276 tim · Tim **TelyuAlgo**

---

## Task

Predict the traffic speed (km/h) for **every** road segment at three horizons:

| Horizon | Minutes ahead | Submission tag |
|--------:|--------------:|----------------|
| h5      | +20           | `h5`           |
| h10     | +40           | `h10`          |
| h15     | +60           | `h15`          |

Submission format   one prediction per row:

```csv
id,speed
test_00000_h5_r0,44.0
test_00000_h5_r1,73.0
```

## Data

Three sources of signal per sample:

1. **Speed history**   a 15-step window of recent speed readings (1 hour) for all 1,260 road segments.
2. **Event text**   natural-language descriptions of what is happening on the network.
3. **Road network**   a `1260 x 1260` adjacency matrix + road geometry/metadata (`roadclass`, etc.).

## Approach

```
Data loading → Validation & EDA → Feature engineering → Baselines (HGBR) → MLP (PyTorch) → Submission
```

1. **EDA & validation** (`scripts/explore.py`)   inspect `active_mask`, adjacency matrix, road metadata, and the pre-built `test_X_hist` window. The raw network has 1,296 roads filtered down to 1,260 "active" ones via `active_mask`.
2. **Feature engineering**   row-aligned speed matrices, event text series, roadclass majority vote across sub-segments, zero-speed fraction, rush-hour flags; `ColumnTransformer` with median imputation + `StandardScaler`.
3. **Models**
   - `HistGradientBoostingRegressor` (main, sklearn)   memory-friendly tree ensemble on raw features.
   - `TrafficMLP` (PyTorch)   small MLP with **embeddings for `roadclass` and `horizon`**, numeric features standardized via internal buffer; trained on a memory-mapped dataset to stay RAM-friendly.
4. **Submission**   formats each prediction as `test_{sample}_h{horizon}_r{road}`.

## Repository Structure

```
kaggle-traffic-speed-forecasting/
├── scripts/
│   ├── train.py          # main pipeline: EDA → features → HGBR / MLP → submission
│   ├── explore.py        # data sanity checks & shape inspection
│   └── flatten_test.py   # reshape (15,1260) history → flattened CSV
├── assets/
│   └── evaluation_formula.png
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

> ⚠️ The scripts reference absolute Windows paths (e.g. `D:\coding stuff\pandasenv\kaggle\train`).
> Update the `DATA_DIR` / `FILES` paths at the top of `scripts/train.py` before running.

```powershell
pip install -r requirements.txt
python scripts/explore.py
python scripts/train.py
```

## Technologies

- **Python**, NumPy, pandas
- **scikit-learn**   `HistGradientBoostingRegressor`, `ColumnTransformer`, `StandardScaler`
- **PyTorch**   custom MLP with categorical embeddings
- **JSON / NumPy memmap**   memory-efficient loading of large matrices

## Result

**234th out of 276** on the leaderboard (team **TelyuAlgo**).

## License

[MIT](LICENSE)
