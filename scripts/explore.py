import numpy as np
import json
import os
from pathlib import Path

TRAIN_DIR = Path(os.environ.get("TRAFFIC_DATA_DIR",
                                Path(__file__).resolve().parent.parent / "data" / "train"))

# static files
mask = np.load(TRAIN_DIR / "static" / "active_mask.npy")
print("active_mask:", mask.shape, mask.dtype, mask[:10])

mat = np.load(TRAIN_DIR / "static" / "matrix.npy")
print("matrix:", mat.shape, mat.dtype)
print(mat[:5, :5])

with open(TRAIN_DIR / "static" / "Roads1260.json", encoding="utf-8") as f:
    roads = json.load(f)
print("Roads1260 type:", type(roads))
if isinstance(roads, dict):
    print("keys sample:", list(roads.keys())[:5])
    first_key = list(roads.keys())[0]
    print(first_key, "->", roads[first_key])
elif isinstance(roads, list):
    print("len:", len(roads))
    print("item[0]:", roads[0])

# yang kemarin belum sempat dijalankan
print("test_X_hist:", np.load(TRAIN_DIR / "test_X_hist.npy").shape)
print("train_speed_m1:", np.load(TRAIN_DIR / "train_speed_m1_1_11160.npy").shape)
print("train_speed_m2:", np.load(TRAIN_DIR / "train_speed_m2_1_5039.npy").shape)

with open(TRAIN_DIR / "test_texts.json", encoding="utf-8") as f:
    tt = json.load(f)
print("test_texts type/len:", type(tt), len(tt))
print("sample key:", list(tt.keys())[:3])

print(mask.sum())  # kalau hasilnya 1260, dugaan saya benar