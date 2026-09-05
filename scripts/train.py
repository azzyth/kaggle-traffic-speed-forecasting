import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfigurasi path
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"D:\coding stuff\pandasenv\kaggle\train")
STATIC_DIR = DATA_DIR / "static"

FILES = {
    "m1_speed": DATA_DIR / "output_data_m1.csv",
    "m1_text": DATA_DIR / "train_text_m1_1_11160.json",
    "m2_speed": DATA_DIR / "output_data_m2.csv",
    "m2_text": DATA_DIR / "train_text_m2_1_5039.json",
    "sample_submission": DATA_DIR / "sample_submission.csv",
    "test_x_hist": DATA_DIR / "test_X_hist.npy",
    "test_text": DATA_DIR / "test_texts.json",
    "adjacency": STATIC_DIR / "matrix.npy",
    "active_mask": STATIC_DIR / "active_mask.npy",
    "roads_meta": STATIC_DIR / "Roads1260.json",
}
 

# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
def load_speed_matrix(path: Path) -> pd.DataFrame:
    """Load file speed train (feat_0..feat_1259) menjadi DataFrame.

    Asumsi: setiap BARIS = satu timestep, setiap KOLOM = satu ruas jalan.
    Urutan baris diasumsikan berurutan waktu (belum ada kolom timestamp
    eksplisit), asumsi ini divalidasi di tahap EDA.
    """
    df = pd.read_csv(path)
    expected_cols = [f"feat_{i}" for i in range(df.shape[1])]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Urutan/nama kolom tidak sesuai ekspektasi di {path}")
    # samakan penamaan kolom dengan id di sample_submission (r0..r1259)
    df.columns = [f"r{i}" for i in range(df.shape[1])]
    return df


def load_text_series(path: Path, prefix: str) -> list:
    """Load JSON event text TRAIN menjadi list terurut sesuai index baris.

    Format: {"{prefix}_1": "...", "{prefix}_2": "...", ...}, satu teks per
    TIMESTEP -> row-aligned dengan file speed yang berpasangan.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    n = len(raw)
    ordered = []
    for i in range(1, n + 1):
        key = f"{prefix}_{i}"
        if key not in raw:
            raise KeyError(f"Key {key} tidak ditemukan di {path}")
        ordered.append(raw[key])
    return ordered


def load_test_text(path: Path, n_samples: int) -> list:
    """Load JSON event text TEST menjadi list terurut sesuai sample id.

    Format: {"test_00000": "...", "test_00001": "...", ...}, satu teks per
    SAMPLE (bukan per timestep seperti di train).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ordered = []
    for i in range(n_samples):
        key = f"test_{i:05d}"
        if key not in raw:
            raise KeyError(f"Key {key} tidak ditemukan di {path}")
        ordered.append(raw[key])
    return ordered


def load_data(verbose: bool = True) -> dict:
    """Load seluruh data: train (speed + text), test (history array + text),
    dan data statis jaringan jalan (adjacency matrix + metadata + mask).

    Return dict berisi:
        m1_speed, m1_text     -> time series train 1 (11.160 timestep)
        m2_speed, m2_text     -> time series train 2 (5.039 timestep)
        test_x_hist           -> np.array (540, 15, 1260), window history test
                                  siap pakai (SUDAH dalam bentuk 15-langkah,
                                  tidak perlu sliding window manual)
        test_text             -> list 540 teks event, 1 per sample test
        adjacency              -> np.array (1260, 1260) int8, adjacency matrix
        roads_meta             -> list 1260 metadata/geometri ruas jalan
        active_mask             -> np.array (1296,) bool, sum()=1260. Ini index
                                  filter dari jaringan jalan mentah (1296 ruas)
                                  ke subset "aktif" (1260 ruas) yang dipakai di
                                  seluruh data lain (speed, adjacency, roads_meta).
                                  Data lain SUDAH dalam ruang 1260 ter-filter,
                                  jadi mask ini murni informatif, tidak perlu
                                  diterapkan lagi secara manual.
        sample_submission      -> format id yang harus diikuti saat submit
    """
    data = {}

    # --- train ---
    data["m1_speed"] = load_speed_matrix(FILES["m1_speed"])
    data["m1_text"] = load_text_series(FILES["m1_text"], prefix="m1")

    data["m2_speed"] = load_speed_matrix(FILES["m2_speed"])
    data["m2_text"] = load_text_series(FILES["m2_text"], prefix="m2")

    # --- submission format ---
    data["sample_submission"] = pd.read_csv(FILES["sample_submission"])
    sub = data["sample_submission"]
    n_test_sample = sub["id"].str.extract(r"test_(\d+)_")[0].nunique()

    # --- test (siap pakai, tidak perlu window manual) ---
    data["test_x_hist"] = np.load(FILES["test_x_hist"])
    data["test_text"] = load_test_text(FILES["test_text"], n_samples=n_test_sample)

    # --- data statis jaringan jalan ---
    data["adjacency"] = np.load(FILES["adjacency"])
    with open(FILES["roads_meta"], "r", encoding="utf-8") as f:
        data["roads_meta"] = json.load(f)
    data["active_mask"] = np.load(FILES["active_mask"])

    # -------------------------------------------------------------------
    # validasi dasar
    # -------------------------------------------------------------------
    assert len(data["m1_speed"]) == len(data["m1_text"]), (
        f"Jumlah baris m1_speed ({len(data['m1_speed'])}) != "
        f"jumlah entri m1_text ({len(data['m1_text'])})"
    )
    assert len(data["m2_speed"]) == len(data["m2_text"]), (
        f"Jumlah baris m2_speed ({len(data['m2_speed'])}) != "
        f"jumlah entri m2_text ({len(data['m2_text'])})"
    )
    assert data["test_x_hist"].shape[0] == n_test_sample, (
        f"Jumlah sample test_x_hist ({data['test_x_hist'].shape[0]}) != "
        f"jumlah sample unik di sample_submission ({n_test_sample})"
    )
    assert len(data["test_text"]) == n_test_sample, (
        f"Jumlah test_text ({len(data['test_text'])}) != "
        f"jumlah sample unik di sample_submission ({n_test_sample})"
    )
    assert data["adjacency"].shape == (1260, 1260), (
        f"Shape adjacency matrix tidak sesuai ekspektasi: {data['adjacency'].shape}"
    )
    assert len(data["roads_meta"]) == 1260, (
        f"Jumlah entri roads_meta ({len(data['roads_meta'])}) != 1260"
    )

    if verbose:
        horizons = sorted(sub["id"].str.extract(r"_h(\d+)_")[0].unique())
        n_road = sub["id"].str.extract(r"_r(\d+)$")[0].nunique()

        print("=== load_data summary ===")
        print(f"m1_speed        : {data['m1_speed'].shape}  (rows=timestep, cols=road)")
        print(f"m1_text         : {len(data['m1_text'])} entri (1 per timestep)")
        print(f"m2_speed        : {data['m2_speed'].shape}")
        print(f"m2_text         : {len(data['m2_text'])} entri (1 per timestep)")
        print(f"test_x_hist     : {data['test_x_hist'].shape}  (sample, timestep, road)")
        print(f"test_text       : {len(data['test_text'])} entri (1 per sample)")
        print(f"adjacency       : {data['adjacency'].shape}  dtype={data['adjacency'].dtype}")
        print(f"roads_meta      : {len(data['roads_meta'])} entri")
        print(f"active_mask     : {data['active_mask'].shape}  sum={data['active_mask'].sum()}")
        print(f"sample_submission : {sub.shape}")
        print(f"  -> {n_test_sample} sample test unik")
        print(f"  -> horizon unik: {horizons}")
        print(f"  -> jumlah road unik: {n_road}")
        print()
        print(f"CATATAN: active_mask ({data['active_mask'].shape[0]} elemen, "
              f"{data['active_mask'].sum()} aktif) adalah filter dari jaringan")
        print("jalan mentah ke 1260 ruas yang dipakai di seluruh data lain.")
        print("Sudah konsisten -- tidak perlu diterapkan manual lagi.")

    return data


# ---------------------------------------------------------------------------
# data eda
# ---------------------------------------------------------------------------
from collections import Counter


def eda_basic_stats(data):
    print("\n" + "=" * 70)
    print("1. BASIC STATS - SPEED DATA")
    print("=" * 70)
    for name in ["m1_speed", "m2_speed"]:
        arr = data[name].values
        print(f"\n[{name}] shape={arr.shape}")
        print(f"  min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}  std={arr.std():.2f}")
        print(f"  proporsi nilai 0     : {(arr == 0).mean() * 100:.3f}%")
        print(f"  proporsi NaN         : {np.isnan(arr).mean() * 100:.3f}%")
        qs = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        print(f"  percentile [1,5,25,50,75,95,99]: {np.round(qs, 2)}")

    arr = data["test_x_hist"]
    print(f"\n[test_x_hist] shape={arr.shape}")
    print(f"  min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}  std={arr.std():.2f}")
    print(f"  proporsi nilai 0     : {(arr == 0).mean() * 100:.3f}%")
    print(f"  proporsi NaN         : {np.isnan(arr).mean() * 100:.3f}%")


def eda_time_ordering(data):
    print("\n" + "=" * 70)
    print("2. VALIDASI URUTAN WAKTU (m1 & m2)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    for name in ["m1_speed", "m2_speed"]:
        arr = data[name].values.astype(np.float32)
        n = arr.shape[0]
        print(f"\n[{name}] n_timestep={n}")

        # a) autokorelasi rata-rata speed di berbagai lag
        mean_series = arr.mean(axis=1)
        lags = [l for l in [1, 2, 5, 10, 15, 30, 60, 120, 180, 360, 720] if l < n]
        print("  Autokorelasi rata-rata speed per timestep (1.0 = identik):")
        for lag in lags:
            corr = np.corrcoef(mean_series[:-lag], mean_series[lag:])[0, 1]
            print(f"    lag={lag:>4} steps : corr={corr:.4f}")

        # b) selisih antar baris berurutan vs pasangan baris acak
        diffs_seq = np.abs(np.diff(arr, axis=0)).mean()
        idx_a = rng.integers(0, n, size=5000)
        idx_b = rng.integers(0, n, size=5000)
        valid = idx_a != idx_b
        diffs_rand = np.abs(arr[idx_a[valid]] - arr[idx_b[valid]]).mean()
        print(f"  Rata-rata |selisih| antar baris BERURUTAN : {diffs_seq:.3f}")
        print(f"  Rata-rata |selisih| antar baris ACAK      : {diffs_rand:.3f}")
        print(f"  Rasio seq/random (makin kecil makin kuat bukti time-series): {diffs_seq / diffs_rand:.3f}")


def eda_text_analysis(data):
    print("\n" + "=" * 70)
    print("3. ANALISIS TEKS EVENT")
    print("=" * 70)

    def extract_events(text):
        return [p.strip() for p in text.split(".") if p.strip()]

    for name in ["m1_text", "m2_text", "test_text"]:
        texts = data[name]
        n_empty = sum(1 for t in texts if not t.strip())
        lengths = [len(extract_events(t)) for t in texts]
        print(f"\n[{name}] n={len(texts)}")
        print(f"  teks kosong          : {n_empty}")
        print(f"  jumlah event/teks    : min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.2f}")

    print("\n  Top 15 tipe event (kata kunci sebelum ' on ') di train (m1+m2, sampled):")
    counter = Counter()
    for name in ["m1_text", "m2_text"]:
        for t in data[name][::5]:  # subsample tiap 5 baris biar cepat
            for ev in extract_events(t):
                kind = ev.split(" on ")[0].strip() if " on " in ev else ev.strip()
                counter[kind] += 1
    for kind, cnt in counter.most_common(15):
        print(f"    {kind:45s} : {cnt}")

    print("\n  Persistensi event (Jaccard similarity set-event, subsample tiap 10 baris):")
    rng = np.random.default_rng(42)
    for name in ["m1_text", "m2_text"]:
        texts = data[name][::10]
        event_sets = [set(extract_events(t)) for t in texts]
        m = len(event_sets)

        def jaccard(a, b):
            u = a | b
            return len(a & b) / len(u) if u else 1.0

        seq_scores = [jaccard(event_sets[i], event_sets[i + 1]) for i in range(m - 1)]
        idx_a = rng.integers(0, m, size=500)
        idx_b = rng.integers(0, m, size=500)
        rand_scores = [jaccard(event_sets[a], event_sets[b]) for a, b in zip(idx_a, idx_b) if a != b]

        print(f"  [{name}] jaccard berurutan (rata2) = {np.mean(seq_scores):.4f}  "
              f"vs acak (rata2) = {np.mean(rand_scores):.4f}")


def eda_network(data):
    print("\n" + "=" * 70)
    print("4. JARINGAN JALAN (adjacency matrix & metadata)")
    print("=" * 70)

    adj = data["adjacency"]
    print(f"\n[adjacency] shape={adj.shape} dtype={adj.dtype}")
    print(f"  simetris? {np.array_equal(adj, adj.T)}")
    diag = np.diag(adj)
    print(f"  semua diagonal = 1? {(diag == 1).all()}  ({(diag == 1).sum()}/{len(diag)})")
    degree = adj.sum(axis=1) - diag
    print(f"  degree (tetangga, exclude diri sendiri): min={degree.min()} max={degree.max()} mean={degree.mean():.2f}")
    print(f"  jumlah node tanpa tetangga (isolated): {(degree == 0).sum()}")
    print(f"  density matrix: {adj.sum() / (adj.shape[0] * adj.shape[1]):.4f}")

    roads = data["roads_meta"]
    print(f"\n[roads_meta] n={len(roads)}")
    sub_counts = [len(r) if isinstance(r, list) else 1 for r in roads]
    print(f"  jumlah sub-segmen per entry: min={min(sub_counts)} max={max(sub_counts)} mean={np.mean(sub_counts):.2f}")

    roadclasses = Counter()
    lengths = []
    for r in roads:
        segs = r if isinstance(r, list) else [r]
        for seg in segs:
            roadclasses[seg.get("roadclass")] += 1
            lengths.append(seg.get("length", np.nan))
    print(f"  distribusi roadclass: {dict(roadclasses)}")
    lengths = np.array(lengths, dtype=float)
    print(f"  length sub-segmen (meter): min={np.nanmin(lengths):.1f} max={np.nanmax(lengths):.1f} mean={np.nanmean(lengths):.1f}")


def eda_target_feasibility(data):
    print("\n" + "=" * 70)
    print("5. FEASIBILITY WINDOWING UNTUK TARGET TRAIN")
    print("=" * 70)
    for name in ["m1_speed", "m2_speed"]:
        n = len(data[name])
        n_possible_samples = n - 15 - 15 + 1  # window 15 input + horizon terjauh 15
        print(f"[{name}] n_timestep={n} -> jumlah sample training yang bisa dibentuk "
              f"(window=15, horizon maks=15): {n_possible_samples}")


def eda_zero_pattern(data):
    print("\n" + "=" * 70)
    print("6. INVESTIGASI POLA NILAI 0")
    print("=" * 70)

    def zero_run_lengths(col_bool):
        """Hitung panjang setiap 'run' nilai True (zero) yang beruntun."""
        runs = []
        count = 0
        for v in col_bool:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        return runs

    for speed_name, text_name in [("m1_speed", "m1_text"), ("m2_speed", "m2_text")]:
        arr = data[speed_name].values
        n_rows, n_cols = arr.shape
        print(f"\n--- [{speed_name}] ---")

        # a) distribusi zero per-road (kolom)
        road_zero_frac = (arr == 0).mean(axis=0)
        print(f"  Per-road zero fraction: min={road_zero_frac.min():.3f} "
              f"max={road_zero_frac.max():.3f} mean={road_zero_frac.mean():.3f}")
        n_road_50 = (road_zero_frac > 0.5).sum()
        n_road_90 = (road_zero_frac > 0.9).sum()
        n_road_0 = (road_zero_frac == 0).sum()
        print(f"  Road dengan zero_frac > 50%: {n_road_50} | > 90%: {n_road_90} "
              f"| tidak pernah 0 sama sekali: {n_road_0}")
        top_idx = np.argsort(road_zero_frac)[::-1][:10]
        print("  Top 10 road dengan zero_frac tertinggi:")
        for idx in top_idx:
            print(f"    road r{idx:<5d} zero_frac={road_zero_frac[idx]:.3f}")

        # b) distribusi zero per-timestep (baris) -- deteksi mass outage
        row_zero_frac = (arr == 0).mean(axis=1)
        print(f"\n  Per-timestep zero fraction: min={row_zero_frac.min():.3f} "
              f"max={row_zero_frac.max():.3f} mean={row_zero_frac.mean():.3f}")
        n_row_50 = (row_zero_frac > 0.5).sum()
        print(f"  Jumlah timestep dengan >50% road zero bersamaan (mass outage): {n_row_50} / {n_rows}")

        # c) panjang run zero pada 15 road dengan zero_frac tertinggi
        sample_cols = top_idx[:15]
        all_runs = []
        for c in sample_cols:
            all_runs.extend(zero_run_lengths(arr[:, c] == 0))
        if all_runs:
            all_runs = np.array(all_runs)
            print(f"\n  Panjang 'run' zero beruntun (dari 15 road zero_frac tertinggi):")
            print(f"    n_runs={len(all_runs)} min={all_runs.min()} max={all_runs.max()} "
                  f"mean={all_runs.mean():.2f} median={np.median(all_runs):.1f}")
            print(f"    Runs > 15 steps (>1 jam): {(all_runs > 15).sum()} "
                  f"| Runs > 60 steps (>4 jam): {(all_runs > 60).sum()}")

        # d) korelasi jumlah zero per-timestep vs jumlah kata 'closure' di teks pada baris sama
        zero_count_row = (arr == 0).sum(axis=1)
        closure_count = np.array([t.count("closure") for t in data[text_name]])
        corr = np.corrcoef(zero_count_row, closure_count)[0, 1]
        print(f"\n  Korelasi (jumlah road=0 per timestep) vs (jumlah kata 'closure' di teks): {corr:.4f}")
        print("  (mendekati 0 = zero TIDAK berkaitan dgn closure -> indikasi missing/sensor;")
        print("   makin ke arah 1 = zero berkaitan dgn closure -> indikasi sinyal valid)")


def eda_dead_road_crosscheck(data):
    print("\n" + "=" * 70)
    print("7. CROSS-CHECK ROAD 'MATI' (zero_frac=1.0) ANTARA m1, m2, TEST")
    print("=" * 70)

    m1_frac = (data["m1_speed"].values == 0).mean(axis=0)
    m2_frac = (data["m2_speed"].values == 0).mean(axis=0)
    test_arr = data["test_x_hist"].reshape(-1, data["test_x_hist"].shape[-1])
    test_frac = (test_arr == 0).mean(axis=0)

    dead_m1 = set(np.where(m1_frac == 1.0)[0].tolist())
    dead_m2 = set(np.where(m2_frac == 1.0)[0].tolist())
    dead_test = set(np.where(test_frac == 1.0)[0].tolist())

    print(f"  Jumlah road 'mati' (zero_frac=1.0): m1={len(dead_m1)}  m2={len(dead_m2)}  test={len(dead_test)}")
    print(f"  Irisan m1 & m2                     : {len(dead_m1 & dead_m2)}")
    print(f"  Irisan m1 & test                   : {len(dead_m1 & dead_test)}")
    print(f"  Irisan m2 & test                   : {len(dead_m2 & dead_test)}")
    print(f"  Irisan m1 & m2 & test              : {len(dead_m1 & dead_m2 & dead_test)}")
    print(f"  Gabungan semua (union)             : {len(dead_m1 | dead_m2 | dead_test)}")

    # cek zero_frac road dead-m1 di m2, dan sebaliknya (barangkali bukan 1.0 tapi tetap tinggi)
    if dead_m1:
        idx = sorted(dead_m1)[:10]
        print("\n  zero_frac di m2 untuk 10 road yang 'mati' di m1 (cek konsistensi):")
        for i in idx:
            print(f"    road r{i:<5d} zero_frac_m1={m1_frac[i]:.3f}  zero_frac_m2={m2_frac[i]:.3f}  zero_frac_test={test_frac[i]:.3f}")


def run_eda(data):
    eda_basic_stats(data)
    eda_time_ordering(data)
    eda_text_analysis(data)
    eda_network(data)
    eda_target_feasibility(data)
    eda_zero_pattern(data)
    eda_dead_road_crosscheck(data)


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
# Strategi (berdasarkan hasil EDA bagian 6 & 7):
#   - 4 road (r1119, r1189, r1224, r1259) mati total di m1, m2, DAN test ->
#     ditandai sbg fitur statis (dead_all3_flag), model tetap yang memutuskan
#     bobotnya, tapi kita simpan juga zero_frac per-periode sbg fitur kontinu.
#   - Road lain yg "closure periodik" (mis. 9 road dead-di-m1-saja) ditangani
#     via fitur DINAMIS per window: zero_frac, run-length zero berjalan,
#     last value -- bukan hard-coded blacklist.
#   - adjacency TIDAK simetris (graf terarah) -> fitur tetangga pakai
#     out-degree, bukan asumsi simetri.
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

WINDOW = 15
HORIZONS = (5, 10, 15)

EVENT_KEYWORDS = [
    "road closure",
    "a general traffic accident",
    "construction",
    "prohibit left turn",
    "road traffic control",
    "an announcement",
]

NUMERIC_COLS = [
    "last", "mean", "std", "min", "max", "zero_frac", "run_len_zero_end", "trend",
    "neighbor_mean_last", "neighbor_zero_frac_last",
    "length_total", "n_subseg", "out_degree", "in_degree",
    "zero_frac_m1", "zero_frac_m2", "zero_frac_test",
    "text_len", "n_event_approx",
] + [f"kw_{i}" for i in range(len(EVENT_KEYWORDS))]
CATEGORICAL_COLS = ["roadclass", "horizon"]
FLAG_COLS = ["dead_all3_flag", "has_neighbor"]


def build_windows(speed_arr: np.ndarray, window: int = WINDOW, horizons=HORIZONS):
    """Bangun sliding window (X) + target (y) dari matrix speed (n_timestep, n_road).

    X dibuat via stride_tricks (VIEW, bukan copy) supaya hemat memori untuk
    n_timestep besar (m1=11160, m2=5039).

    Return:
        X       : np.ndarray (n_sample, window, n_road)
        y       : dict {horizon: np.ndarray (n_sample, n_road)}
        n_sample: int
    """
    n_timestep, n_road = speed_arr.shape
    max_h = max(horizons)
    n_sample = n_timestep - window - max_h + 1
    assert n_sample > 0, "data terlalu pendek untuk window+horizon yang diminta"

    windows_all = np.lib.stride_tricks.sliding_window_view(speed_arr, window, axis=0)
    windows_all = np.moveaxis(windows_all, -1, 1)  # -> (n_timestep-window+1, window, n_road)
    X = windows_all[:n_sample]

    y = {}
    for h in horizons:
        target_idx = np.arange(n_sample) + window + h - 1
        y[h] = speed_arr[target_idx]  # (n_sample, n_road)

    return X, y, n_sample


def extract_window_features(X: np.ndarray) -> dict:
    """Fitur dinamis per (sample, road) dari window (n_sample, window, n_road).

    Return dict of arrays shape (n_sample, n_road):
        last, mean, std, min, max, zero_frac, run_len_zero_end, trend
    """
    n_sample, window, n_road = X.shape
    feats = {
        "last": X[:, -1, :],
        "mean": X.mean(axis=1),
        "std": X.std(axis=1),
        "min": X.min(axis=1),
        "max": X.max(axis=1),
    }

    is_zero = (X == 0)
    feats["zero_frac"] = is_zero.mean(axis=1)

    # run-length zero yang SEDANG berjalan di akhir window (mundur dari step
    # terakhir, berhenti begitu ketemu nilai non-zero)
    run_len = np.zeros((n_sample, n_road), dtype=np.int16)
    active = np.ones((n_sample, n_road), dtype=bool)
    for t in range(window - 1, -1, -1):
        step_zero = is_zero[:, t, :]
        run_len += (active & step_zero).astype(np.int16)
        active &= step_zero
    feats["run_len_zero_end"] = run_len

    # trend: slope regresi linear sederhana atas `window` titik waktu
    t_idx = np.arange(window, dtype=np.float32)
    t_centered = t_idx - t_idx.mean()
    denom = (t_centered ** 2).sum()
    X_centered = X - X.mean(axis=1, keepdims=True)
    feats["trend"] = np.tensordot(X_centered, t_centered, axes=([1], [0])) / denom

    return {k: v.astype(np.float32) for k, v in feats.items()}


def extract_neighbor_features(last_vals: np.ndarray, adjacency: np.ndarray) -> dict:
    """Fitur agregat tetangga dari nilai speed di langkah TERAKHIR window.

    last_vals : (n_sample, n_road)
    adjacency : (n_road, n_road) int8, graf TERARAH (tidak simetris), diag=1.
    Road tanpa tetangga (out_degree=0) -> NaN (di-impute di preprocessing pipeline).
    """
    adj_no_diag = adjacency.copy()
    np.fill_diagonal(adj_no_diag, 0)
    out_degree = adj_no_diag.sum(axis=1).astype(np.float32)
    safe_degree = np.where(out_degree == 0, 1, out_degree)

    neighbor_mean = (last_vals @ adj_no_diag.T) / safe_degree
    zero_mask = (last_vals == 0).astype(np.float32)
    neighbor_zero_frac = (zero_mask @ adj_no_diag.T) / safe_degree

    no_neighbor = out_degree == 0
    neighbor_mean[:, no_neighbor] = np.nan
    neighbor_zero_frac[:, no_neighbor] = np.nan

    return {
        "neighbor_mean_last": neighbor_mean.astype(np.float32),
        "neighbor_zero_frac_last": neighbor_zero_frac.astype(np.float32),
    }


def build_static_road_features(roads_meta: list, adjacency: np.ndarray,
                                zero_frac_m1: np.ndarray, zero_frac_m2: np.ndarray,
                                zero_frac_test: np.ndarray) -> pd.DataFrame:
    """Fitur statis per road: metadata geometri + posisi graf + histori zero
    lintas-periode (m1/m2/test) yang sudah divalidasi di EDA bagian 7."""
    n_road = len(roads_meta)
    roadclass = np.zeros(n_road, dtype=np.int16)
    length_total = np.zeros(n_road, dtype=np.float32)
    n_subseg = np.zeros(n_road, dtype=np.int16)

    for i, r in enumerate(roads_meta):
        segs = r if isinstance(r, list) else [r]
        classes = [s.get("roadclass") for s in segs]
        lens = [s.get("length", 0.0) for s in segs]
        vals, counts = np.unique(classes, return_counts=True)
        roadclass[i] = vals[np.argmax(counts)]  # roadclass mayoritas antar sub-segmen
        length_total[i] = float(np.nansum(lens))
        n_subseg[i] = len(segs)

    adj_no_diag = adjacency.copy()
    np.fill_diagonal(adj_no_diag, 0)
    out_degree = adj_no_diag.sum(axis=1).astype(np.float32)
    in_degree = adj_no_diag.sum(axis=0).astype(np.float32)

    dead_all3 = ((zero_frac_m1 == 1.0) & (zero_frac_m2 == 1.0) & (zero_frac_test == 1.0)).astype(np.int8)

    return pd.DataFrame({
        "road_id": np.arange(n_road),
        "roadclass": roadclass,
        "length_total": length_total,
        "n_subseg": n_subseg,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "has_neighbor": (out_degree > 0).astype(np.int8),
        "zero_frac_m1": zero_frac_m1.astype(np.float32),
        "zero_frac_m2": zero_frac_m2.astype(np.float32),
        "zero_frac_test": zero_frac_test.astype(np.float32),
        "dead_all3_flag": dead_all3,
    })


def extract_text_features(texts: list) -> pd.DataFrame:
    """Fitur sederhana per SAMPLE dari teks event (hitung kata kunci kategori
    event yang paling sering muncul, lihat EDA bagian 3). Fitur ini SHARED
    untuk semua road dalam satu sample (teks bukan per-road)."""
    rows = []
    for t in texts:
        row = {f"kw_{i}": t.count(kw) for i, kw in enumerate(EVENT_KEYWORDS)}
        row["text_len"] = len(t)
        row["n_event_approx"] = t.count(" on ")  # heuristik pola "<event> on <lokasi>"
        rows.append(row)
    return pd.DataFrame(rows)


def assemble_training_table(speed_arr: np.ndarray, texts: list, adjacency: np.ndarray,
                             roads_static_df: pd.DataFrame, window: int = WINDOW,
                             horizons=HORIZONS, sample_stride: int = 1,
                             source: str = None) -> pd.DataFrame:
    """Tabel LONG: 1 baris = (sample, road, horizon), siap masuk preprocessing.

    sample_stride: ambil 1 dari setiap N sample utk mengecilkan volume data
    (dev/testing). WAJIB set ke 1 untuk training final -- lihat catatan
    memori di bagian akhir file ini.
    source: label asal data ("m1"/"m2"), dipakai utk time-based split per
    periode (m1 & m2 punya rentang waktu berbeda, tidak boleh dicampur saat
    menentukan cutoff train/val).
    """
    X, y, n_sample = build_windows(speed_arr, window=window, horizons=horizons)

    idx = np.arange(0, n_sample, sample_stride)
    X = X[idx]
    y = {h: y[h][idx] for h in horizons}
    n_sample_used = len(idx)
    n_road = speed_arr.shape[1]

    dyn = extract_window_features(X)
    neigh = extract_neighbor_features(dyn["last"], adjacency)

    # teks yg relevan utk sample ke-i = teks di step TERAKHIR window (paling
    # dekat waktu prediksi), row-aligned dgn speed (lihat load_text_series)
    text_feat = extract_text_features([texts[window - 1 + i] for i in idx])

    base = {
        "sample_idx": np.repeat(idx, n_road),
        "road_id": np.tile(np.arange(n_road), n_sample_used),
    }
    for k, v in dyn.items():
        base[k] = v.reshape(-1)
    for k, v in neigh.items():
        base[k] = v.reshape(-1)

    base_df = pd.DataFrame(base)
    base_df = base_df.merge(roads_static_df, on="road_id", how="left")

    text_feat_rep = text_feat.loc[text_feat.index.repeat(n_road)].reset_index(drop=True)
    base_df = pd.concat([base_df.reset_index(drop=True), text_feat_rep], axis=1)

    frames = []
    for h in horizons:
        f = base_df.copy()
        f["horizon"] = h
        f["y"] = y[h].reshape(-1).astype(np.float32)
        if source is not None:
            f["source"] = source
        frames.append(f)
    return pd.concat(frames, axis=0, ignore_index=True)


def build_test_table(data: dict, roads_static_df: pd.DataFrame) -> pd.DataFrame:
    """Tabel LONG utk test_x_hist, format kolom SAMA dgn assemble_training_table
    tapi tanpa kolom `y` (target belum diketahui) + kolom `id` sesuai format
    sample_submission (test_{sample}_h{horizon}_r{road})."""
    X = data["test_x_hist"]  # (n_sample, window, n_road) -- sudah siap pakai
    n_sample, window, n_road = X.shape
    texts = data["test_text"]

    dyn = extract_window_features(X)
    neigh = extract_neighbor_features(dyn["last"], data["adjacency"])
    text_feat = extract_text_features(texts)  # 1 teks per SAMPLE test (bukan per timestep)

    base = {
        "sample_idx": np.repeat(np.arange(n_sample), n_road),
        "road_id": np.tile(np.arange(n_road), n_sample),
    }
    for k, v in dyn.items():
        base[k] = v.reshape(-1)
    for k, v in neigh.items():
        base[k] = v.reshape(-1)

    base_df = pd.DataFrame(base)
    base_df = base_df.merge(roads_static_df, on="road_id", how="left")
    text_feat_rep = text_feat.loc[text_feat.index.repeat(n_road)].reset_index(drop=True)
    base_df = pd.concat([base_df.reset_index(drop=True), text_feat_rep], axis=1)

    frames = []
    for h in HORIZONS:
        f = base_df.copy()
        f["horizon"] = h
        f["id"] = [f"test_{s:05d}_h{h}_r{r}" for s, r in zip(f["sample_idx"], f["road_id"])]
        frames.append(f)
    return pd.concat(frames, axis=0, ignore_index=True)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer sklearn: numeric -> impute median + StandardScaler,
    categorical (roadclass, horizon) -> OneHotEncoder, flag biner -> passthrough."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("flag", "passthrough", FLAG_COLS),
    ])


def run_preprocessing_demo(data: dict, sample_stride_m1: int = 50, sample_stride_m2: int = 20):
    """Demo end-to-end dgn STRIDE besar (subsample) supaya cepat -- untuk
    validasi bug & cek bentuk output sebelum dijalankan full di data asli.
    WAJIB stride=1 untuk pipeline produksi (lihat catatan memori di bawah)."""
    print("\n" + "=" * 70)
    print("8. DEMO PREPROCESSING (subsampled utk validasi cepat)")
    print("=" * 70)

    m1_frac = (data["m1_speed"].values == 0).mean(axis=0)
    m2_frac = (data["m2_speed"].values == 0).mean(axis=0)
    test_arr = data["test_x_hist"].reshape(-1, data["test_x_hist"].shape[-1])
    test_frac = (test_arr == 0).mean(axis=0)

    roads_static = build_static_road_features(
        data["roads_meta"], data["adjacency"], m1_frac, m2_frac, test_frac
    )
    print("\n[roads_static] shape:", roads_static.shape)
    print(roads_static.head())
    print("\ndead_all3_flag sum:", roads_static["dead_all3_flag"].sum())

    train_m1 = assemble_training_table(
        data["m1_speed"].values, data["m1_text"], data["adjacency"], roads_static,
        sample_stride=sample_stride_m1, source="m1",
    )
    train_m2 = assemble_training_table(
        data["m2_speed"].values, data["m2_text"], data["adjacency"], roads_static,
        sample_stride=sample_stride_m2, source="m2",
    )
    train_full = pd.concat([train_m1, train_m2], axis=0, ignore_index=True)
    print("\n[train_m1] shape:", train_m1.shape, " [train_m2] shape:", train_m2.shape)
    print("[train_full] shape:", train_full.shape)
    print(train_full[NUMERIC_COLS + CATEGORICAL_COLS + FLAG_COLS + ["y"]].describe().T)
    print("\nNaN per kolom (sebelum imputasi):")
    print(train_full[NUMERIC_COLS].isna().sum()[lambda s: s > 0])

    test_table = build_test_table(data, roads_static)
    print("\n[test_table] shape:", test_table.shape)
    print(test_table[["id"] + NUMERIC_COLS[:5]].head())

    pre = build_preprocessor()
    X_train_transformed = pre.fit_transform(
        train_full[NUMERIC_COLS + CATEGORICAL_COLS + FLAG_COLS]
    )
    print("\n[preprocessor] output shape (train, subsampled):", X_train_transformed.shape)
    print("  n_kolom numeric:", len(NUMERIC_COLS))
    print("  n_kolom hasil one-hot (cat):",
          X_train_transformed.shape[1] - len(NUMERIC_COLS) - len(FLAG_COLS))

    X_test_transformed = pre.transform(
        test_table[NUMERIC_COLS + CATEGORICAL_COLS + FLAG_COLS]
    )
    print("[preprocessor] output shape (test)               :", X_test_transformed.shape)

    return train_full, test_table, pre


# ---------------------------------------------------------------------------
# CATATAN SKALA PENUH (baca sebelum menjalankan sample_stride=1)
# ---------------------------------------------------------------------------
# n_sample*n_road*n_horizon full: m1 (~11131*1260*3 ~= 42jt baris) + m2
# (~5010*1260*3 ~= 18.9jt baris) -> tabel long ~61jt baris x ~25 kolom.
# Dengan dtype float32 itu sekitar 6-7 GB di memori -- bisa berat tergantung
# RAM Anda. Kalau OOM saat sample_stride=1, opsi:
#   1) proses per-chunk (mis. per periode m1/m2 terpisah, tulis ke parquet,
#      baru gabung/latih pakai library yg support out-of-core / partial_fit
#      seperti LightGBM dgn Dataset streaming),
#   2) turunkan resolusi (subsample timestep, bukan road -- road harus tetap
#      lengkap 1260 krn submission butuh prediksi semua road),
#   3) pertimbangkan pakai float32/float16 & category dtype pandas utk
#      roadclass/horizon supaya lebih hemat sebelum masuk ColumnTransformer.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MODULE ESTIMATOR
# ---------------------------------------------------------------------------
# Ini problem REGRESI (target speed kontinu, dievaluasi MSE) -- sklearn.cluster
# & classifier tidak relevan. Pemilihan regressor didasarkan pada karakteristik
# data hasil EDA & preprocessing:
#   - ~61jt baris skala penuh, banyak fitur kategorikal (roadclass, horizon)
#     bercampur numerik -> butuh model yang scalable & native handle campuran.
#   - Ada NaN terstruktur (neighbor_* utk 13 road isolated) -> lebih baik model
#     yang native handle missing value drpd hard-impute (impute bisa
#     menyembunyikan sinyal "road ini gak punya tetangga").
#   - Hubungan fitur->target kemungkinan besar NON-LINEAR (mis. run_len_zero_end
#     vs speed bukan linear, ada threshold effect), dan ada interaksi
#     roadclass x zero_frac x jam sibuk -> tree-based ensemble lebih cocok
#     drpd model linear murni.
#   - Data TIME-SERIES dgn autokorelasi sangat tinggi (EDA bag.2) & window
#     antar-sample overlap -> split train/val WAJIB berdasarkan waktu, bukan
#     acak, supaya skor validasi tidak bias optimis (leakage).
#
# -> Model utama: sklearn.ensemble.HistGradientBoostingRegressor
#    (native categorical_features='from_dtype' & native NaN handling,
#     histogram-based jadi scalable ke puluhan juta baris).
# -> Baseline pembanding: DummyRegressor (mean) & Ridge (linear, pakai
#    ColumnTransformer yang sudah dibuat) -- utk mengukur seberapa besar
#    non-linearitas & interaksi benar-benar berkontribusi.
# ---------------------------------------------------------------------------
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def time_based_split(df: pd.DataFrame, val_frac: float = 0.15,
                      gap: int = WINDOW + max(HORIZONS)) -> tuple:
    """Split TRAIN/VAL berdasarkan WAKTU (sample_idx) per `source` (m1/m2),
    BUKAN acak.

    Kenapa wajib: window antar-sample overlap 14/15 langkah (sliding window),
    jadi sample_idx=100 dan sample_idx=101 nyaris identik. Kalau split acak,
    keduanya bisa kepisah ke train & val -> model "menghafal" val dari train
    yg nyaris sama -> skor MSE validasi jadi terlalu optimis (data leakage).

    `gap` step di sekitar titik potong dibuang dari sisi train supaya window
    train paling akhir tidak overlap dgn window val paling awal.
    """
    train_parts, val_parts = [], []
    for src, g in df.groupby("source"):
        max_idx = g["sample_idx"].max()
        cutoff = int(max_idx * (1 - val_frac))
        train_parts.append(g[g["sample_idx"] <= cutoff - gap])
        val_parts.append(g[g["sample_idx"] > cutoff])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


def run_model_baselines(train_full: pd.DataFrame, pre: ColumnTransformer):
    print("\n" + "=" * 70)
    print("9. TIME-BASED SPLIT & BASELINE MODEL")
    print("=" * 70)

    train_df, val_df = time_based_split(train_full, val_frac=0.15)
    print(f"  train: {train_df.shape}  val: {val_df.shape}")
    print("  rentang sample_idx train per source:")
    print(train_df.groupby("source")["sample_idx"].agg(["min", "max"]))
    print("  rentang sample_idx val per source:")
    print(val_df.groupby("source")["sample_idx"].agg(["min", "max"]))

    feat_cols = NUMERIC_COLS + CATEGORICAL_COLS + FLAG_COLS
    y_train, y_val = train_df["y"].values, val_df["y"].values
    results = {}

    # --- 0) baseline naif: prediksi rata-rata global ---
    dummy = DummyRegressor(strategy="mean").fit(train_df[feat_cols], y_train)
    results["dummy_mean"] = mean_squared_error(y_val, dummy.predict(val_df[feat_cols]))

    # --- 1) baseline linear: Ridge, pakai preprocessor (scaled + one-hot) ---
    X_tr = pre.fit_transform(train_df[feat_cols])
    X_val = pre.transform(val_df[feat_cols])
    ridge = Ridge(alpha=1.0).fit(X_tr, y_train)
    results["ridge"] = mean_squared_error(y_val, ridge.predict(X_val))

    # --- 2) model utama: HistGradientBoostingRegressor pada fitur RAW ---
    # Tidak perlu impute/scale/one-hot manual -- HGBR native handle NaN &
    # kategorikal (categorical_features="from_dtype" baca dtype 'category').
    train_raw = train_df[feat_cols].copy()
    val_raw = val_df[feat_cols].copy()
    for c in CATEGORICAL_COLS:
        train_raw[c] = train_raw[c].astype("category")
        val_raw[c] = val_raw[c].astype("category").cat.set_categories(train_raw[c].cat.categories)

    hgbr = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        max_iter=800,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=42,
    )
    hgbr.fit(train_raw, y_train)
    pred_hgbr = hgbr.predict(val_raw)
    results["hist_gb"] = mean_squared_error(y_val, pred_hgbr)

    print("\n  MSE validasi (time-based split, GAP dibuang -> anti-leakage):")
    for name, mse in results.items():
        print(f"    {name:<12s}: {mse:.4f}")

    best_name = min(results, key=results.get)
    print(f"\n  Model terbaik sejauh ini: {best_name}")
    if best_name == "hist_gb":
        print("  MSE per horizon (hist_gb):")
        for h in sorted(val_df["horizon"].unique()):
            mask = (val_df["horizon"] == h).values
            print(f"    h{h}: {mean_squared_error(y_val[mask], pred_hgbr[mask]):.4f}")
        print(f"  n_iter aktual (early stopping): {hgbr.n_iter_}")

    return results, hgbr, ridge


# ---------------------------------------------------------------------------
# FINAL MODEL (full data) & SUBMISSION
# ---------------------------------------------------------------------------
# Hyperparameter difiksasi berdasarkan hasil tuning di subsample (lihat log
# eksperimen): max_iter=500, learning_rate=0.06, sisanya sama dgn baseline
# hist_gb (max_leaf_nodes=31, l2_regularization=0.1). early_stopping tetap
# dibiarkan ON dgn validation_fraction kecil -- ini HANYA dipakai HGBR utk
# menentukan kapan berhenti dini kalau ternyata full data (61jt baris)
# konvergen lebih cepat dari max_iter=500; TIDAK dipakai utk skor evaluasi
# kita (skor evaluasi = leaderboard). Kalau ingin benar2 fix di 500 iterasi
# tanpa early stop, set early_stopping=False saat memanggil fungsi ini.
FINAL_HGBR_PARAMS = dict(
    max_iter=500,
    learning_rate=0.06,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=True,
    n_iter_no_change=15,
    validation_fraction=0.1,
    random_state=42,
)


def train_final_model(train_full: pd.DataFrame,
                       feat_cols: list = None,
                       hgbr_params: dict = None) -> tuple:
    """Fit HistGradientBoostingRegressor final di SELURUH train_full (tidak
    di-split train/val lagi -- val-split hanya dipakai di tahap tuning utk
    mengukur MSE, bukan utk model yang disubmit).

    Return:
        model          : HistGradientBoostingRegressor terlatih
        feat_cols      : kolom fitur yang dipakai (urutan harus sama dgn saat predict)
        cat_categories : dict {kolom_kategorikal: pd.Index kategori}, WAJIB
                         dipakai ulang saat menyiapkan test set (supaya one-hot
                         index kategori dari tree splits konsisten train<->test)
    """
    if feat_cols is None:
        feat_cols = NUMERIC_COLS + CATEGORICAL_COLS + FLAG_COLS
    if hgbr_params is None:
        hgbr_params = FINAL_HGBR_PARAMS

    print("\n" + "=" * 70)
    print("10. TRAIN FINAL MODEL (full data)")
    print("=" * 70)
    print(f"  n_rows train_full : {len(train_full):,}")
    print(f"  params            : {hgbr_params}")

    train_raw = train_full[feat_cols].copy()
    cat_categories = {}
    for c in CATEGORICAL_COLS:
        train_raw[c] = train_raw[c].astype("category")
        cat_categories[c] = train_raw[c].cat.categories

    y = train_full["y"].values.astype(np.float32)

    model = HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        **hgbr_params,
    )
    model.fit(train_raw, y)

    pred_train = model.predict(train_raw)
    mse_train = mean_squared_error(y, pred_train)
    print(f"  n_iter_ aktual    : {model.n_iter_}")
    print(f"  MSE in-sample (train, BUKAN skor final -- indikatif saja): {mse_train:.4f}")

    return model, feat_cols, cat_categories


def generate_submission(data: dict, roads_static: pd.DataFrame, model,
                         feat_cols: list, cat_categories: dict,
                         out_path: str = "submission.csv") -> pd.DataFrame:
    """Prediksi test_x_hist -> susun submission.csv sesuai format id di
    sample_submission (test_{sample}_h{horizon}_r{road})."""
    print("\n" + "=" * 70)
    print("11. GENERATE SUBMISSION")
    print("=" * 70)

    test_table = build_test_table(data, roads_static)
    print(f"  test_table shape  : {test_table.shape}")

    test_raw = test_table[feat_cols].copy()
    for c in CATEGORICAL_COLS:
        test_raw[c] = test_raw[c].astype("category").cat.set_categories(cat_categories[c])
        # kategori yg muncul di test tapi tidak pernah muncul di train -> NaN,
        # biarkan HGBR native-handle (masuk sbg "missing" split), JANGAN drop baris
        n_unseen = test_raw[c].isna().sum() - test_table[c].isna().sum()
        if n_unseen > 0:
            print(f"  WARNING kolom {c}: {n_unseen} baris test punya kategori "
                  f"tak dikenal saat train -> ditandai missing")

    preds = model.predict(test_raw)
    test_table = test_table.copy()
    test_table["speed"] = preds.astype(np.float32)

    # merge ke urutan id sample_submission (jangan asumsikan urutan sama)
    sub = data["sample_submission"][["id"]].copy()
    sub = sub.merge(test_table[["id", "speed"]], on="id", how="left")

    n_missing = sub["speed"].isna().sum()
    assert n_missing == 0, (
        f"{n_missing} id di sample_submission tidak ketemu prediksinya -- "
        "cek format id / kelengkapan test_table"
    )
    assert len(sub) == len(data["sample_submission"]), "jumlah baris submission tidak cocok"

    print(f"  speed pred  : min={sub['speed'].min():.2f} max={sub['speed'].max():.2f} "
          f"mean={sub['speed'].mean():.2f}")
    neg = (sub["speed"] < 0).sum()
    if neg > 0:
        print(f"  WARNING: {neg} prediksi negatif -> clip ke 0")
        sub["speed"] = sub["speed"].clip(lower=0)

    sub.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")
    return sub


# ---------------------------------------------------------------------------
# ENGINE ALTERNATIF: PyTorch (bisa CUDA) -- fitur dirakit ke np.memmap di
# disk, BUKAN pandas, supaya training 61jt baris tidak butuh RAM sebesar
# datanya. Lihat catatan panjang di komentar generate_submission_torch()
# soal kenapa crash sebelumnya murni soal RAM, bukan soal CPU/GPU.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

TORCH_FEAT_COLS = NUMERIC_COLS + FLAG_COLS  # semua fitur kontinu/biner (roadclass & horizon terpisah, lewat embedding)
N_FEAT = len(TORCH_FEAT_COLS)
HORIZON_TO_CODE = {h: i for i, h in enumerate(HORIZONS)}


def _static_arrays_for_memmap(roads_static_df: pd.DataFrame) -> dict:
    """Array per-road (indeks = road_id) siap di-broadcast per chunk, plus
    mapping roadclass -> kode embedding kontigu 0..K-1 (nn.Embedding butuh
    indeks kontigu, roadclass asli nilainya {0,1,2,3,6} -- ada gap)."""
    roadclass_values = sorted(roads_static_df["roadclass"].unique().tolist())
    roadclass_to_code = {v: i for i, v in enumerate(roadclass_values)}

    static_numeric = {
        c: roads_static_df[c].values.astype(np.float32)
        for c in ["length_total", "n_subseg", "out_degree", "in_degree",
                  "zero_frac_m1", "zero_frac_m2", "zero_frac_test"]
    }
    flag_arrays = {c: roads_static_df[c].values.astype(np.float32) for c in FLAG_COLS}
    roadclass_code_by_road = roads_static_df["roadclass"].map(roadclass_to_code).values.astype(np.int8)

    return {
        "static_numeric": static_numeric,
        "flag_arrays": flag_arrays,
        "roadclass_code_by_road": roadclass_code_by_road,
        "roadclass_to_code": roadclass_to_code,
        "n_roadclass": len(roadclass_values),
    }


def _write_source_to_memmap(speed_arr: np.ndarray, texts: list, adjacency: np.ndarray,
                             static: dict, X_mm, y_mm, rc_mm, hz_mm, si_mm,
                             row_offset: int, window: int, horizons, chunk_samples: int,
                             tag: str) -> dict:
    """Tulis fitur utk SATU sumber (m1 atau m2) ke memmap yang SUDAH dibuka,
    mulai dari row_offset, per-chunk sample (RAM terpakai cuma sebesar 1
    chunk, bukan seluruh sumber). Return meta (n_sample, n_rows_written,
    chunk_samples) dipakai belakangan utk hitung batas train/val by time."""
    n_timestep, n_road = speed_arr.shape
    max_h = max(horizons)
    n_sample = n_timestep - window - max_h + 1
    n_horizon = len(horizons)

    row_ptr = row_offset
    for start in range(0, n_sample, chunk_samples):
        end = min(start + chunk_samples, n_sample)
        n_s = end - start

        X_chunk = np.stack([speed_arr[i:i + window] for i in range(start, end)], axis=0)
        dyn = extract_window_features(X_chunk)
        neigh = extract_neighbor_features(dyn["last"], adjacency)
        # road tanpa tetangga -> NaN; isi 0, model sudah tahu lewat flag has_neighbor
        neigh_mean = np.nan_to_num(neigh["neighbor_mean_last"], nan=0.0)
        neigh_zero = np.nan_to_num(neigh["neighbor_zero_frac_last"], nan=0.0)

        texts_chunk = [texts[window - 1 + i] for i in range(start, end)]
        text_feat_df = extract_text_features(texts_chunk)

        parts = []
        for c in NUMERIC_COLS:
            if c in dyn:
                parts.append(dyn[c].reshape(-1))
            elif c == "neighbor_mean_last":
                parts.append(neigh_mean.reshape(-1))
            elif c == "neighbor_zero_frac_last":
                parts.append(neigh_zero.reshape(-1))
            elif c in static["static_numeric"]:
                parts.append(np.tile(static["static_numeric"][c], n_s))
            else:  # text_len, n_event_approx, kw_0..kw_5 -- per sample, broadcast per road
                parts.append(np.repeat(text_feat_df[c].values.astype(np.float32), n_road))
        for c in FLAG_COLS:
            parts.append(np.tile(static["flag_arrays"][c], n_s))
        X_block = np.stack(parts, axis=1).astype(np.float32)  # (n_s*n_road, N_FEAT)
        rc_block = np.tile(static["roadclass_code_by_road"], n_s)
        si_block = np.repeat(np.arange(start, end, dtype=np.int32), n_road)

        for h in horizons:
            target_idx = np.arange(start, end) + window + h - 1
            y_block = speed_arr[target_idx].reshape(-1).astype(np.float32)
            hz_block = np.full(n_s * n_road, HORIZON_TO_CODE[h], dtype=np.int8)

            n_block = n_s * n_road
            X_mm[row_ptr:row_ptr + n_block] = X_block
            y_mm[row_ptr:row_ptr + n_block] = y_block
            rc_mm[row_ptr:row_ptr + n_block] = rc_block
            hz_mm[row_ptr:row_ptr + n_block] = hz_block
            si_mm[row_ptr:row_ptr + n_block] = si_block
            row_ptr += n_block

        if (start // chunk_samples) % 20 == 0:
            print(f"    [{tag}] chunk sample {start}-{end}/{n_sample} "
                  f"({row_ptr - row_offset:,} baris ditulis)")

    return {"n_sample": n_sample, "n_rows": row_ptr - row_offset, "row_offset": row_offset,
            "n_road": n_road, "n_horizon": n_horizon}


def build_train_memmap(data: dict, roads_static_df: pd.DataFrame, out_dir: Path,
                        chunk_samples: int = 200, val_frac: float = 0.15,
                        window: int = WINDOW, horizons=HORIZONS) -> dict:
    """Rakit fitur m1+m2 LANGSUNG ke satu set file memmap gabungan di disk
    (X/y/roadclass/horizon/sample_idx), lalu hitung batas train/val
    berdasarkan WAKTU per sumber (sama prinsipnya dgn time_based_split, tapi
    dibulatkan ke batas chunk supaya tidak perlu index array besar -- cukup
    pakai slice (start,end) per sumber, hemat memori)."""
    print("\n" + "=" * 70)
    print("10T. BANGUN FITUR -> MEMMAP DISK (m1 + m2, tidak lewat pandas)")
    print("=" * 70)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static = _static_arrays_for_memmap(roads_static_df)

    n_road = data["m1_speed"].shape[1]
    n_horizon = len(horizons)
    n_sample_m1 = data["m1_speed"].shape[0] - window - max(horizons) + 1
    n_sample_m2 = data["m2_speed"].shape[0] - window - max(horizons) + 1
    n_rows_m1 = n_sample_m1 * n_road * n_horizon
    n_rows_m2 = n_sample_m2 * n_road * n_horizon
    n_rows_total = n_rows_m1 + n_rows_m2
    print(f"  n_rows_m1={n_rows_m1:,}  n_rows_m2={n_rows_m2:,}  total={n_rows_total:,}")
    est_gb = n_rows_total * (N_FEAT * 4 + 4 + 1 + 1 + 4) / 1e9
    print(f"  perkiraan ukuran file di disk: ~{est_gb:.1f} GB (pastikan cukup ruang di drive ini)")

    paths = {
        "X_path": out_dir / "X_train.f32",
        "y_path": out_dir / "y_train.f32",
        "roadclass_path": out_dir / "roadclass_train.i8",
        "horizon_path": out_dir / "horizon_train.i8",
        "sample_idx_path": out_dir / "sample_idx_train.i32",
    }
    X_mm = np.memmap(paths["X_path"], dtype=np.float32, mode="w+", shape=(n_rows_total, N_FEAT))
    y_mm = np.memmap(paths["y_path"], dtype=np.float32, mode="w+", shape=(n_rows_total,))
    rc_mm = np.memmap(paths["roadclass_path"], dtype=np.int8, mode="w+", shape=(n_rows_total,))
    hz_mm = np.memmap(paths["horizon_path"], dtype=np.int8, mode="w+", shape=(n_rows_total,))
    si_mm = np.memmap(paths["sample_idx_path"], dtype=np.int32, mode="w+", shape=(n_rows_total,))

    meta_m1 = _write_source_to_memmap(
        data["m1_speed"].values, data["m1_text"], data["adjacency"], static,
        X_mm, y_mm, rc_mm, hz_mm, si_mm, row_offset=0,
        window=window, horizons=horizons, chunk_samples=chunk_samples, tag="m1",
    )
    meta_m2 = _write_source_to_memmap(
        data["m2_speed"].values, data["m2_text"], data["adjacency"], static,
        X_mm, y_mm, rc_mm, hz_mm, si_mm, row_offset=n_rows_m1,
        window=window, horizons=horizons, chunk_samples=chunk_samples, tag="m2",
    )
    X_mm.flush(); y_mm.flush(); rc_mm.flush(); hz_mm.flush(); si_mm.flush()

    # batas train/val PER SUMBER berdasarkan waktu (sample_idx), dibulatkan
    # ke kelipatan chunk_samples supaya batasnya persis batas baris di memmap
    gap = window + max(horizons)
    train_ranges, val_ranges = [], []
    for meta in (meta_m1, meta_m2):
        n_sample = meta["n_sample"]
        max_idx = n_sample - 1
        cutoff = int(max_idx * (1 - val_frac))
        train_cut_sample = max(0, ((cutoff - gap) // chunk_samples) * chunk_samples)
        val_start_sample = min(n_sample, ((cutoff + chunk_samples) // chunk_samples) * chunk_samples)

        rows_per_sample = meta["n_road"] * meta["n_horizon"]
        train_ranges.append((meta["row_offset"],
                              meta["row_offset"] + train_cut_sample * rows_per_sample))
        val_ranges.append((meta["row_offset"] + val_start_sample * rows_per_sample,
                            meta["row_offset"] + meta["n_rows"]))

    print(f"  train_ranges (baris memmap): {train_ranges}")
    print(f"  val_ranges   (baris memmap): {val_ranges}")

    return {
        "paths": paths, "n_rows": n_rows_total, "n_feat": N_FEAT,
        "train_ranges": train_ranges, "val_ranges": val_ranges,
        "roadclass_to_code": static["roadclass_to_code"],
        "n_roadclass": static["n_roadclass"], "n_horizon": n_horizon,
    }


def compute_feature_stats(paths: dict, n_rows: int, n_feat: int,
                           sample_size: int = 2_000_000, seed: int = 42):
    """Mean/std per kolom fitur dari SUBSAMPLE acak (index diurutkan dulu
    supaya baca dari memmap tetap relatif berurutan, bukan akses acak murni)
    -- dipakai utk standardisasi input MLP."""
    X_mm = np.memmap(paths["X_path"], dtype=np.float32, mode="r", shape=(n_rows, n_feat))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=min(sample_size, n_rows), replace=False)
    idx.sort()
    sample = np.asarray(X_mm[idx])
    mean = sample.mean(axis=0)
    std = sample.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def iterate_batches(paths: dict, n_rows: int, n_feat: int, row_ranges: list,
                     batch_size: int, block_size: int = 2_000_000,
                     shuffle: bool = True, seed: int = None):
    """Generator mini-batch (numpy) dari memmap. Strategi 'block shuffle':
    baca BLOK besar sekaligus via slice kontigu (I/O berurutan, cepat),
    acak URUTAN blok + urutan baris DI DALAM blok, baru dipecah jadi
    mini-batch. Ini kompromi standar utk training di atas data > RAM --
    shuffle baris acak murni thd memmap 61jt baris akan sangat lambat kalau
    dibaca satu-satu."""
    X_mm = np.memmap(paths["X_path"], dtype=np.float32, mode="r", shape=(n_rows, n_feat))
    y_mm = np.memmap(paths["y_path"], dtype=np.float32, mode="r", shape=(n_rows,))
    rc_mm = np.memmap(paths["roadclass_path"], dtype=np.int8, mode="r", shape=(n_rows,))
    hz_mm = np.memmap(paths["horizon_path"], dtype=np.int8, mode="r", shape=(n_rows,))

    rng = np.random.default_rng(seed)
    blocks = []
    for (start, end) in row_ranges:
        for bstart in range(start, end, block_size):
            bend = min(bstart + block_size, end)
            if bend > bstart:
                blocks.append((bstart, bend))
    if shuffle:
        rng.shuffle(blocks)

    for (bstart, bend) in blocks:
        Xb = np.asarray(X_mm[bstart:bend])
        yb = np.asarray(y_mm[bstart:bend])
        rcb = np.asarray(rc_mm[bstart:bend]).astype(np.int64)
        hzb = np.asarray(hz_mm[bstart:bend]).astype(np.int64)

        order = np.arange(len(Xb))
        if shuffle:
            rng.shuffle(order)

        for s in range(0, len(order), batch_size):
            sel = order[s:s + batch_size]
            yield Xb[sel], rcb[sel], hzb[sel], yb[sel]


if TORCH_AVAILABLE:
    class TrafficMLP(nn.Module):
        """MLP kecil: fitur numerik (distandardisasi via buffer mean/std yg
        disimpan DI DALAM model -- jadi ikut saat model dipindah device/disave)
        + embedding roadclass & horizon, digabung -> beberapa Linear+ReLU."""

        def __init__(self, n_numeric: int, n_roadclass: int, n_horizon: int,
                     feat_mean: np.ndarray, feat_std: np.ndarray,
                     emb_roadclass: int = 4, emb_horizon: int = 2,
                     hidden=(256, 128), dropout: float = 0.1):
            super().__init__()
            self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
            self.register_buffer("feat_std", torch.tensor(feat_std, dtype=torch.float32))
            self.emb_roadclass = nn.Embedding(n_roadclass, emb_roadclass)
            self.emb_horizon = nn.Embedding(n_horizon, emb_horizon)

            in_dim = n_numeric + emb_roadclass + emb_horizon
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, roadclass_code, horizon_code):
            x_num = (x_num - self.feat_mean) / self.feat_std
            e1 = self.emb_roadclass(roadclass_code)
            e2 = self.emb_horizon(horizon_code)
            x = torch.cat([x_num, e1, e2], dim=1)
            return self.mlp(x).squeeze(-1)


def train_torch_model(mm_meta: dict, epochs: int = 3, batch_size: int = 8192,
                       lr: float = 1e-3, block_size: int = 2_000_000,
                       val_every_n_batches: int = 2000, device=None):
    """Training loop MLP di atas memmap. Jalan otomatis di GPU (`cuda`) kalau
    tersedia (device default = DEVICE global = cuda kalau ada), fallback CPU
    kalau tidak. Progress + MSE val dicetak berkala supaya bisa dipantau."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch belum terpasang. Install dulu: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121 "
            "(sesuaikan versi CUDA driver Anda -- cek nvidia-smi)."
        )
    if device is None:
        device = DEVICE

    print("\n" + "=" * 70)
    print("11T. TRAIN MODEL PYTORCH")
    print("=" * 70)
    print(f"  device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    feat_mean, feat_std = compute_feature_stats(mm_meta["paths"], mm_meta["n_rows"], mm_meta["n_feat"])

    model = TrafficMLP(
        n_numeric=mm_meta["n_feat"], n_roadclass=mm_meta["n_roadclass"], n_horizon=mm_meta["n_horizon"],
        feat_mean=feat_mean, feat_std=feat_std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def run_validation():
        model.eval()
        total_se, total_n = 0.0, 0
        with torch.no_grad():
            for Xb, rcb, hzb, yb in iterate_batches(
                mm_meta["paths"], mm_meta["n_rows"], mm_meta["n_feat"],
                mm_meta["val_ranges"], batch_size=65536, block_size=block_size, shuffle=False,
            ):
                x = torch.from_numpy(Xb).to(device)
                rc = torch.from_numpy(rcb).to(device)
                hz = torch.from_numpy(hzb).to(device)
                y = torch.from_numpy(yb).to(device)
                pred = model(x, rc, hz)
                total_se += ((pred - y) ** 2).sum().item()
                total_n += len(yb)
        model.train()
        return total_se / max(total_n, 1)

    n_batch_seen = 0
    for epoch in range(epochs):
        print(f"\n  -- epoch {epoch + 1}/{epochs} --")
        model.train()
        running_loss, running_n = 0.0, 0
        for Xb, rcb, hzb, yb in iterate_batches(
            mm_meta["paths"], mm_meta["n_rows"], mm_meta["n_feat"],
            mm_meta["train_ranges"], batch_size=batch_size, block_size=block_size,
            shuffle=True, seed=epoch,
        ):
            x = torch.from_numpy(Xb).to(device, non_blocking=True)
            rc = torch.from_numpy(rcb).to(device, non_blocking=True)
            hz = torch.from_numpy(hzb).to(device, non_blocking=True)
            y = torch.from_numpy(yb).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x, rc, hz)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(yb)
            running_n += len(yb)
            n_batch_seen += 1

            if n_batch_seen % val_every_n_batches == 0:
                val_mse = run_validation()
                print(f"    batch {n_batch_seen:>7} | train MSE (running) "
                      f"{running_loss / running_n:.4f} | val MSE {val_mse:.4f}")
                running_loss, running_n = 0.0, 0

        val_mse = run_validation()
        print(f"  epoch {epoch + 1} selesai | val MSE {val_mse:.4f}")

    return model


def generate_submission_torch(data: dict, roads_static_df: pd.DataFrame, model,
                               mm_meta: dict, out_path: Path, device=None,
                               batch_size: int = 65536) -> pd.DataFrame:
    """Prediksi test_x_hist pakai model PyTorch -> submission.csv.

    Test set cuma ~2 juta baris (540 sample x 1260 road x 3 horizon), jadi
    AMAN dirakit lewat pandas (build_test_table, sudah teruji sebelumnya) --
    OOM sebelumnya cuma terjadi di train (61jt baris), bukan test."""
    if device is None:
        device = DEVICE

    print("\n" + "=" * 70)
    print("12T. GENERATE SUBMISSION (PyTorch)")
    print("=" * 70)

    test_table = build_test_table(data, roads_static_df)
    print(f"  test_table shape: {test_table.shape}")

    # road tanpa tetangga -> neighbor_mean_last/neighbor_zero_frac_last NaN,
    # isi 0 SAMA seperti perlakuan saat bangun memmap train (_write_source_to_memmap)
    # -- kalau tidak, NaN akan merambat lewat standardisasi & bikin prediksi NaN.
    test_table[["neighbor_mean_last", "neighbor_zero_frac_last"]] = \
        test_table[["neighbor_mean_last", "neighbor_zero_frac_last"]].fillna(0.0)

    X_test = test_table[TORCH_FEAT_COLS].astype(np.float32).values
    assert not np.isnan(X_test).any(), "masih ada NaN di X_test setelah fillna -- cek kolom lain"
    rc_test = test_table["roadclass"].map(mm_meta["roadclass_to_code"]).values.astype(np.int64)
    hz_test = test_table["horizon"].map(HORIZON_TO_CODE).values.astype(np.int64)

    model.eval()
    preds = np.empty(len(test_table), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(test_table), batch_size):
            e = min(s + batch_size, len(test_table))
            x = torch.from_numpy(X_test[s:e]).to(device)
            rc = torch.from_numpy(rc_test[s:e]).to(device)
            hz = torch.from_numpy(hz_test[s:e]).to(device)
            preds[s:e] = model(x, rc, hz).cpu().numpy()

    test_table = test_table.copy()
    test_table["speed"] = preds

    sub = data["sample_submission"][["id"]].copy()
    sub = sub.merge(test_table[["id", "speed"]], on="id", how="left")
    n_missing = sub["speed"].isna().sum()
    assert n_missing == 0, f"{n_missing} id tidak ketemu prediksinya"
    assert len(sub) == len(data["sample_submission"])

    print(f"  speed pred: min={sub['speed'].min():.2f} max={sub['speed'].max():.2f} "
          f"mean={sub['speed'].mean():.2f}")
    neg = (sub["speed"] < 0).sum()
    if neg > 0:
        print(f"  WARNING: {neg} prediksi negatif -> clip ke 0")
        sub["speed"] = sub["speed"].clip(lower=0)

    sub.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")
    return sub


def run_full_pipeline_torch(out_path: Path = None, memmap_dir: Path = None,
                             chunk_samples: int = 200, epochs: int = 3,
                             batch_size: int = 8192, lr: float = 1e-3):
    """Pipeline produksi versi PyTorch: load -> fitur ke memmap disk ->
    train MLP (GPU kalau ada) -> submission. Pakai ini kalau HGBR/sklearn
    OOM di mesin Anda (lihat run_full_pipeline() versi sklearn)."""
    if out_path is None:
        out_path = DATA_DIR / "TelyuAlgo_submission.csv"
    if memmap_dir is None:
        memmap_dir = DATA_DIR / "_memmap_cache"

    data = load_data()
    m1_frac = (data["m1_speed"].values == 0).mean(axis=0)
    m2_frac = (data["m2_speed"].values == 0).mean(axis=0)
    test_arr = data["test_x_hist"].reshape(-1, data["test_x_hist"].shape[-1])
    test_frac = (test_arr == 0).mean(axis=0)
    roads_static = build_static_road_features(
        data["roads_meta"], data["adjacency"], m1_frac, m2_frac, test_frac
    )

    mm_meta = build_train_memmap(data, roads_static, memmap_dir, chunk_samples=chunk_samples)
    model = train_torch_model(mm_meta, epochs=epochs, batch_size=batch_size, lr=lr)
    sub = generate_submission_torch(data, roads_static, model, mm_meta, out_path=out_path)
    return model, sub


def run_full_pipeline(sample_stride_m1: int = 1, sample_stride_m2: int = 1,
                       out_path: Path = None):
    """Pipeline PRODUKSI: load -> fitur full (stride=1 by default) -> train
    final -> submission. Ini yang dipakai utk submit beneran, bukan
    run_preprocessing_demo (yang sengaja di-subsample utk cek cepat).

    PERINGATAN: stride=1 -> ~61jt baris long-table, butuh RAM besar (lihat
    catatan di atas file ini). Turunkan stride kalau OOM, tapi INGAT skor
    leaderboard hanya valid kalau submission akhir pakai stride=1.
    """
    if out_path is None:
        out_path = DATA_DIR / "TelyuAlgo_submission.csv"

    data = load_data()

    m1_frac = (data["m1_speed"].values == 0).mean(axis=0)
    m2_frac = (data["m2_speed"].values == 0).mean(axis=0)
    test_arr = data["test_x_hist"].reshape(-1, data["test_x_hist"].shape[-1])
    test_frac = (test_arr == 0).mean(axis=0)
    roads_static = build_static_road_features(
        data["roads_meta"], data["adjacency"], m1_frac, m2_frac, test_frac
    )

    train_m1 = assemble_training_table(
        data["m1_speed"].values, data["m1_text"], data["adjacency"], roads_static,
        sample_stride=sample_stride_m1, source="m1",
    )
    train_m2 = assemble_training_table(
        data["m2_speed"].values, data["m2_text"], data["adjacency"], roads_static,
        sample_stride=sample_stride_m2, source="m2",
    )
    train_full = pd.concat([train_m1, train_m2], axis=0, ignore_index=True)
    print(f"[run_full_pipeline] train_full shape: {train_full.shape}")

    model, feat_cols, cat_categories = train_final_model(train_full)
    sub = generate_submission(data, roads_static, model, feat_cols, cat_categories,
                               out_path=out_path)
    return model, sub


if __name__ == "__main__":
    # --- mode 1: EDA + baseline tuning di subsample (cepat, utk eksplorasi) ---
    RUN_DEMO = False
    # --- mode 2: pipeline produksi full data -> submission.csv (lambat) ---
    RUN_FULL = True
    # engine "torch"   -> fitur ke memmap disk + MLP (GPU kalau ada), HEMAT RAM
    # engine "sklearn" -> HGBR, versi lama (OOM di 61jt baris kalau RAM terbatas)
    ENGINE = "torch"

    if RUN_DEMO:
        data = load_data()
        run_eda(data)
        train_full, test_table, pre = run_preprocessing_demo(data)
        run_model_baselines(train_full, pre)

    if RUN_FULL:
        if ENGINE == "torch":
            run_full_pipeline_torch(chunk_samples=200, epochs=3, batch_size=8192, lr=1e-3)
        else:
            run_full_pipeline(sample_stride_m1=1, sample_stride_m2=1)