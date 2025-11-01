import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hashlib

# =========================================================
# 0. LOGIN & ROLE
# =========================================================
def make_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# baca dari secrets
auth_conf = st.secrets.get("auth", {})
user_list = auth_conf.get("users", [])
pass_list = auth_conf.get("passwords", [])
role_list = auth_conf.get("roles", [])

USERS = {}
for u, p, r in zip(user_list, pass_list, role_list):
    USERS[u] = {"password": make_hash(p), "role": r}

# fallback kalau secrets kosong / belum diisi di Streamlit Cloud
if not USERS:
    USERS = {
        "admin": {"password": make_hash("admin123"), "role": "admin"},
        "viewer": {"password": make_hash("viewer123"), "role": "viewer"},
    }

def login_block():
    st.title("🔐 Login Aplikasi Prediksi PRB")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in USERS and USERS[u]["password"] == make_hash(p):
            st.session_state["authenticated"] = True
            st.session_state["username"] = u
            st.session_state["role"] = USERS[u]["role"]
            st.rerun()
        else:
            st.error("❌ Username atau password salah")

# init session
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_block()
    st.stop()

# =========================================================
# 1. APP UTAMA
# =========================================================
st.set_page_config(page_title="PRB Stock Predictor", layout="wide")
st.title("Prediksi Kebutuhan Stok PRB (EFTS + PSO)")

# info user di sidebar
st.sidebar.write(f"👤 Login sebagai: **{st.session_state['username']}** ({st.session_state['role']})")
role = st.session_state["role"]

DATA_DIR = Path("./data")

# =====================
# Sidebar: Input & opsi
# =====================
st.sidebar.header("📥 Data Sumber (untuk prediksi)")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) atau CSV", type=["xlsx", "csv"])
sep = st.sidebar.selectbox("Delimiter (jika CSV)", [";", ",", "\\t"], index=0)
sheet_name = st.sidebar.text_input("Sheet name (Excel)", value=None, help="Kosongkan kalau bukan Excel atau sheet pertama.")

st.sidebar.header("⚙️ Pengaturan Prediksi")
uod_method = st.sidebar.selectbox("Metode UoD", ["std", "iqr", "fixed"], index=0)
fixed_d = st.sidebar.number_input("fixed_d (bila method=fixed)", value=100, step=10)

# kalau viewer, jangan izinkan PSO (berat)
if role == "viewer":
    st.sidebar.write("🔒 PSO dikunci untuk viewer")
    run_pso = False
else:
    run_pso = st.sidebar.toggle("Jalankan PSO untuk optimasi boundaries", value=False, help="Hati-hati: komputasi bisa lama.")

num_intervals = st.sidebar.slider("Jumlah interval fuzzy", min_value=5, max_value=12, value=8)
n_particles = st.sidebar.slider("PSO: n_particles", 10, 80, 30)
iters = st.sidebar.slider("PSO: iters", 30, 300, 100)
service_level = st.sidebar.slider("Service level (safety stock)", 0.80, 0.99, 0.95, step=0.01)
lead_time_days = st.sidebar.number_input("Lead time (hari)", min_value=1, value=7, step=1)

st.sidebar.header("🧪 Evaluasi (opsional)")
uploaded_real = st.sidebar.file_uploader("Upload Realisasi (Excel/CSV)", type=["xlsx", "csv"], key="real")

# 👉 sidebar khusus APRIORI (file transaksi terpisah)
st.sidebar.header("🧩 Data Transaksi untuk Apriori")
uploaded_apriori = st.sidebar.file_uploader(
    "Upload transaksi (REFASALSEP, OBAT)",
    type=["xlsx", "csv"],
    key="apriori_file"
)
min_support_apr = st.sidebar.slider("Min support", 0.01, 0.5, 0.02, step=0.01)
min_lift_apr = st.sidebar.slider("Min lift", 1.0, 5.0, 1.2, step=0.1)

# =====================
# Helper load data
# =====================
@st.cache_data
def load_df(file, is_csv, sep, sheet_name):
    if is_csv:
        sep = "\t" if sep == "\\t" else sep
        for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]:
            try:
                file.seek(0)
                return pd.read_csv(file, sep=sep, encoding=enc)
            except UnicodeDecodeError:
                continue
        file.seek(0)
        return pd.read_csv(file, sep=sep, encoding_errors="ignore")
    else:
        return pd.read_excel(file, sheet_name=sheet_name if sheet_name else 0)

@st.cache_data
def load_sample():
    sample_path = DATA_DIR / "contoh_data.xlsx"
    if sample_path.exists():
        return pd.read_excel(sample_path)
    return None

# =====================
# Load data utk prediksi
# =====================
if uploaded:
    is_csv = uploaded.name.lower().endswith(".csv")
    df_raw = load_df(uploaded, is_csv, sep, sheet_name)
else:
    df_raw = load_sample()
    if df_raw is None:
        st.info("Silakan upload file data di sidebar. Kolom wajib: PERIODE, OBAT, JML OBAT SETUJU.")
        st.stop()

# Normalisasi kolom
required_cols = {"PERIODE", "OBAT", "JML OBAT SETUJU"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}. Sesuaikan header file.")
    st.stop()

df = df_raw.copy()
df["PERIODE"] = pd.to_datetime(df["PERIODE"])
df = df.sort_values(["OBAT", "PERIODE"])
st.subheader("🔎 Cuplikan Data Prediksi")
st.dataframe(df.head())

# =====================
# UoD
# =====================
def determine_uod(df, method='std', fixed_d=100):
    grouped = df.groupby('OBAT')["JML OBAT SETUJU"]
    out = []
    for obat, data in grouped:
        min_x, max_x = data.min(), data.max()
        if method == 'fixed':
            d1 = d2 = fixed_d
        elif method == 'iqr':
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            d1 = d2 = 1.5 * iqr
        elif method == 'std':
            std = data.std()
            d1 = d2 = std
        else:
            raise ValueError("method must be std/iqr/fixed")
        u_min = max(0, min_x - d1)
        u_max = max_x + d2
        out.append({"OBAT": obat, "U_MIN": u_min, "U_MAX": u_max})
    return pd.DataFrame(out)

df_uod = determine_uod(df, method=uod_method, fixed_d=fixed_d)
with st.expander("📐 UoD (10 baris)"):
    st.dataframe(df_uod.head(10))

# =====================
# PSO (opsional) + FTS
# =====================
def create_fuzzy_intervals(u_min, u_max, inner_bounds):
    all_bounds = np.concatenate(([u_min], np.sort(inner_bounds), [u_max]))
    return [(all_bounds[i], all_bounds[i+1]) for i in range(len(all_bounds)-1)]

def triangular_membership(x, a, b, c):
    if a <= x <= b:
        return (x - a) / (b - a) if b != a else 0
    elif b < x <= c:
        return (c - x) / (c - b) if c != b else 0
    return 0

def determine_fuzzy_set(value, intervals):
    memberships = []
    for i, (lo, hi) in enumerate(intervals):
        if i == 0:
            a, b, c = lo, hi, hi + (hi - lo)
        elif i == len(intervals) - 1:
            a, b, c = lo - (hi - lo), lo, hi
        else:
            a, b, c = lo, (lo + hi) / 2, hi
        memberships.append(triangular_membership(value, a, b, c))
    return f"A{np.argmax(memberships) + 1}"

def build_flrg(fuzzy_series):
    flrg = {}
    for i in range(len(fuzzy_series) - 1):
        cur_, nxt_ = fuzzy_series[i], fuzzy_series[i + 1]
        flrg.setdefault(cur_, []).append(nxt_)
    return flrg

def defuzzify(predicted_labels, intervals):
    if not predicted_labels:
        return None
    mids = []
    for lab in predicted_labels:
        try:
            idx = int(lab[1:]) - 1
            lo, hi = intervals[idx]
            mids.append((lo + hi) / 2)
        except:
            continue
    return float(np.mean(mids)) if mids else None

def _fitness_boundaries(inner_bounds, data, u_min, u_max, k):
    costs = []
    for row in inner_bounds:
        bounds = np.sort(row)
        intervals = create_fuzzy_intervals(u_min, u_max, bounds)
        fuzz = [determine_fuzzy_set(v, intervals) for v in data]
        if len(fuzz) < 2:
            costs.append(np.inf)
            continue
        y_true = fuzz[1:]
        y_pred = fuzz[:-1]
        to_idx = lambda s: int(s[1:]) if isinstance(s, str) and s.startswith("A") else 0
        y_t = np.array([to_idx(s) for s in y_true], float)
        y_p = np.array([to_idx(s) for s in y_pred], float)
        costs.append(sqrt(mean_squared_error(y_t, y_p)))
    return np.array(costs)

optimal_bounds = {}
if run_pso:
    import pyswarms as ps
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    st.info("Menjalankan PSO… proses bisa lama tergantung data.")
    prog = st.progress(0)
    uniq_obat = df["OBAT"].unique()
    for idx, obat in enumerate(uniq_obat, start=1):
        hist = df[df["OBAT"] == obat]["JML OBAT SETUJU"].values
        info = df_uod[df_uod["OBAT"] == obat]
        if len(hist) < 2 or info.empty:
            continue
        u_min, u_max = info["U_MIN"].iloc[0], info["U_MAX"].iloc[0]
        dim = num_intervals - 1
        lower = np.full(dim, u_min)
        upper = np.full(dim, u_max)
        opt = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dim,
            options=options,
            bounds=(lower, upper),
        )
        cost, pos = opt.optimize(
            _fitness_boundaries,
            iters=iters,
            data=hist,
            u_min=u_min,
            u_max=u_max,
            k=num_intervals,
            verbose=False,
        )
        optimal_bounds[obat] = np.sort(pos)
        prog.progress(idx / len(uniq_obat))
    st.success("PSO selesai.")
else:
    for _, r in df_uod.iterrows():
        obat = r["OBAT"]
        u_min = r["U_MIN"]
        u_max = r["U_MAX"]
        inner = np.linspace(u_min, u_max, num_intervals + 1)[1:-1]
        optimal_bounds[obat] = inner

# =====================
# Fuzzify → FLRG → Prediksi bulan depan
# =====================
pred_rows = []
for obat, sub in df.groupby("OBAT"):
    info = df_uod[df_uod["OBAT"] == obat]
    if info.empty or obat not in optimal_bounds:
        continue
    u_min, u_max = info["U_MIN"].iloc[0], info["U_MAX"].iloc[0]
    intervals = create_fuzzy_intervals(u_min, u_max, optimal_bounds[obat])

    sub = sub.sort_values("PERIODE")
    series = sub["JML OBAT SETUJU"].tolist()
    fuzzy_series = [determine_fuzzy_set(v, intervals) for v in series]
    flrg = build_flrg(fuzzy_series)

    last_fs = fuzzy_series[-1]
    predicted_fs = flrg.get(last_fs, [last_fs])
    pred_value = defuzzify(predicted_fs, intervals)

    next_period = sub["PERIODE"].iloc[-1] + pd.DateOffset(months=1)
    pred_rows.append(
        {
            "OBAT": obat,
            "PERIODE_PREDIKSI": next_period,
            "FUZZY_SET_TERAKHIR": last_fs,
            "FUZZY_PREDIKSI_BULAN_DEPAN": predicted_fs,
            "PREDIKSI_BULAN_DEPAN": pred_value,
        }
    )

df_pred = pd.DataFrame(pred_rows).sort_values("OBAT")
st.subheader("📈 Prediksi Bulan Depan")
st.dataframe(df_pred)

# =====================
# Evaluasi (opsional)
# =====================
def load_real(file):
    is_csv = file.name.lower().endswith(".csv")
    if is_csv:
        return pd.read_csv(file, sep=sep if sep != "\\t" else "\t")
    else:
        return pd.read_excel(file)

if uploaded_real is not None:
    df_real = load_real(uploaded_real).copy()
    if "OBAT" not in df_real.columns or "REALISASI_AKTUAL" not in df_real.columns:
        st.error("File realisasi harus punya kolom 'OBAT' dan 'REALISASI_AKTUAL'.")
    else:
        join = pd.merge(
            df_pred[["OBAT", "PREDIKSI_BULAN_DEPAN"]],
            df_real[["OBAT", "REALISASI_AKTUAL"]],
            on="OBAT",
            how="inner",
        )
        if join.empty:
            st.warning("Tidak ada OBAT yang cocok antara prediksi dan realisasi.")
        else:
            y_true = join["REALISASI_AKTUAL"].astype(float)
            y_pred = join["PREDIKSI_BULAN_DEPAN"].astype(float)
            valid = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
            if valid.any():
                mape = np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100
                rmse = sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
                mae = mean_absolute_error(y_true[valid], y_pred[valid])
                st.subheader("📊 Evaluasi")
                st.write(f"**MAPE:** {mape:.2f}%  |  **RMSE:** {rmse:.2f}  |  **MAE:** {mae:.2f}")
                st.dataframe(join)
            else:
                st.info("Data valid untuk evaluasi tidak cukup (cek nilai 0/NaN).")

# =====================
# Safety Stock & ROP
# =====================
from scipy.stats import norm
def calc_stock_levels(df_hist, lead_time_days, service_level):
    lt_months = lead_time_days / 30.44
    stats = df_hist.groupby("OBAT")["JML OBAT SETUJU"].agg(["mean", "std"]).reset_index()
    stats.rename(columns={"mean": "AVG_MONTHLY", "std": "STD_MONTHLY"}, inplace=True)
    stats["STD_MONTHLY"] = stats["STD_MONTHLY"].fillna(0.0)
    z = norm.ppf(service_level)
    stats["SAFETY_STOCK"] = z * stats["STD_MONTHLY"] * np.sqrt(lt_months)
    avg_daily = stats["AVG_MONTHLY"] / 30.44
    stats["REORDER_POINT"] = (avg_daily * lead_time_days) + stats["SAFETY_STOCK"]
    stats["SAFETY_STOCK"] = stats["SAFETY_STOCK"].clip(lower=0)
    stats["REORDER_POINT"] = stats["REORDER_POINT"].clip(lower=0)
    return stats[["OBAT", "SAFETY_STOCK", "REORDER_POINT"]]

df_stock = calc_stock_levels(df, lead_time_days, service_level)
st.subheader("🏬 Safety Stock & Reorder Point")
st.dataframe(df_stock.head())

# =====================
# Download tombol (prediksi)
# =====================
def to_excel_bytes(df_dict):
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in df_dict.items():
            d.to_excel(xw, sheet_name=name, index=False)
    buf.seek(0)
    return buf

st.download_button(
    "⬇️ Download hasil prediksi (Excel)",
    data=to_excel_bytes({"prediksi": df_pred, "uod": df_uod, "stok": df_stock}),
    file_name="hasil_prediksi_prb.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =========================================================
# 7. APRIORI / BUNDLING OBAT (FILE TERPISAH)
# =========================================================
if uploaded_apriori is not None:
    st.subheader("🧩 Analisis Bundling Obat (Apriori)")

    # baca file apriori
    is_csv_apr = uploaded_apriori.name.lower().endswith(".csv")
    try:
        if is_csv_apr:
            df_apr = pd.read_csv(uploaded_apriori, sep=";", encoding="utf-8")
        else:
            df_apr = pd.read_excel(uploaded_apriori)
    except Exception:
        uploaded_apriori.seek(0)
        df_apr = pd.read_csv(uploaded_apriori, sep=",", encoding_errors="ignore")

    st.write("📄 Cuplikan data transaksi:")
    st.dataframe(df_apr.head())

    if {"REFASALSEP", "OBAT"}.issubset(df_apr.columns):
        df_tx = df_apr[["REFASALSEP", "OBAT"]].dropna()

        # one hot
        basket = pd.crosstab(df_tx["REFASALSEP"], df_tx["OBAT"])
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # import di sini supaya gak error kalau mlxtend belum ada
        from mlxtend.frequent_patterns import apriori, association_rules

        freq_itemsets = apriori(basket, min_support=min_support_apr, use_colnames=True)

        if not freq_itemsets.empty:
            rules = association_rules(freq_itemsets, metric="lift", min_threshold=min_lift_apr)
            rules_1 = rules[rules["antecedents"].apply(lambda x: len(x) == 1)]

            bundling_rows = []
            for obat, grp in rules_1.groupby(rules_1["antecedents"].apply(lambda x: list(x)[0])):
                grp = grp.sort_values("confidence", ascending=False)
                rekom = [list(c)[0] for c in grp["consequents"]]
                bundling_rows.append({
                    "OBAT": obat,
                    "REKOMENDASI_BUNDLING": ", ".join(rekom),
                    "JUMLAH_REKOMENDASI": len(rekom)
                })

            df_bundling = pd.DataFrame(bundling_rows).sort_values(
                "JUMLAH_REKOMENDASI", ascending=False
            )
            st.success("✅ Analisis bundling selesai!")
            st.dataframe(df_bundling)

            # tombol download bundling
            from io import BytesIO
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                df_bundling.to_excel(xw, index=False, sheet_name="Bundling")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Hasil Bundling (Excel)",
                data=buf,
                file_name="hasil_bundling_apriori.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Tidak ada frequent itemset pada threshold tersebut. Coba turunkan min_support.")
    else:
        st.error("File Apriori harus punya kolom **REFASALSEP** dan **OBAT**.")
else:
    st.info("📎 Upload file transaksi (REFASALSEP, OBAT) di sidebar untuk jalankan Apriori.")
