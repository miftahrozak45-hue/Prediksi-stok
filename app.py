import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hashlib
from scipy.stats import norm


# =========================================================
# 1) LOGIN & ROLE (ADMIN / VIEWER) - PAKAI st.secrets
# =========================================================

def make_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# baca konfigurasi dari secrets (streamlit.io)
auth_conf = st.secrets.get("auth", {})
user_list = auth_conf.get("users", [])
pass_list = auth_conf.get("passwords", [])
role_list = auth_conf.get("roles", [])

# bentuk: {"admin": {"password": <hash>, "role": "admin"}, ...}
USERS = {}
for u, p, r in zip(user_list, pass_list, role_list):
    USERS[u] = {"password": make_hash(p), "role": r}

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

# inisialisasi session
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# kalau belum login, jangan lanjut
if not st.session_state["authenticated"]:
    login_block()
    st.stop()


# =========================================================
# 2) APP UTAMA
# =========================================================

st.set_page_config(page_title="PRB Stock Predictor", layout="wide")
st.title("Prediksi Kebutuhan Stok PRB (EFTS + PSO)")

# info user di sidebar + logout
st.sidebar.write(f"👤 {st.session_state.get('username','-')} ({st.session_state.get('role','-')})")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

DATA_DIR = Path("./data")

# =====================
# Sidebar: Input & opsi
# =====================
st.sidebar.header("📥 Data Sumber")

# kalau mau upload HANYA admin, uncomment ini:
# if st.session_state.get("role") != "admin":
#     st.sidebar.info("Upload hanya untuk admin.")
#     uploaded = None
# else:
#     uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) atau CSV", type=["xlsx", "csv"])

uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) atau CSV", type=["xlsx", "csv"])

sep = st.sidebar.selectbox("Delimiter (jika CSV)", [";", ",", "\\t"], index=0)
sheet_name = st.sidebar.text_input("Sheet name (Excel)", value=None, help="Kosongkan kalau bukan Excel atau sheet pertama.")

st.sidebar.header("⚙️ Pengaturan")
uod_method = st.sidebar.selectbox("Metode UoD", ["std", "iqr", "fixed"], index=0)
fixed_d = st.sidebar.number_input("fixed_d (bila method=fixed)", value=100, step=10)
run_pso = st.sidebar.toggle("Jalankan PSO untuk optimasi boundaries", value=False, help="Hati-hati: komputasi bisa lama.")
num_intervals = st.sidebar.slider("Jumlah interval fuzzy", min_value=5, max_value=12, value=8)
n_particles = st.sidebar.slider("PSO: n_particles", 10, 80, 30)
iters = st.sidebar.slider("PSO: iters", 30, 300, 100)
service_level = st.sidebar.slider("Service level (safety stock)", 0.80, 0.99, 0.95, step=0.01)
lead_time_days = st.sidebar.number_input("Lead time (hari)", min_value=1, value=7, step=1)

st.sidebar.header("🧪 Evaluasi (opsional)")
uploaded_real = st.sidebar.file_uploader("Upload Realisasi (Excel/CSV)", type=["xlsx", "csv"], key="real")

# =====================
# Helper load data
# =====================
@st.cache_data
def load_df(file, is_csv, sep, sheet_name):
    if is_csv:
        return pd.read_csv(file, sep=sep if sep != "\\t" else "\t")
    else:
        return pd.read_excel(file, sheet_name=sheet_name if sheet_name else 0)

@st.cache_data
def load_sample():
    sample_path = DATA_DIR / "contoh_data.xlsx"
    if sample_path.exists():
        return pd.read_excel(sample_path)
    return None

# =====================
# Load data
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
required_cols = {"PERIODE","OBAT","JML OBAT SETUJU"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}. Sesuaikan header file.")
    st.stop()

df = df_raw.copy()
df["PERIODE"] = pd.to_datetime(df["PERIODE"])
df = df.sort_values(["OBAT","PERIODE"])
st.subheader("🔎 Cuplikan Data")
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
            iqr = q3-q1
            d1 = d2 = 1.5*iqr
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
        return (x-a)/(b-a) if b!=a else 0
    elif b < x <= c:
        return (c-x)/(c-b) if c!=b else 0
    return 0

def determine_fuzzy_set(value, intervals):
    memberships = []
    for i,(lo,hi) in enumerate(intervals):
        if i==0:
            a, b, c = lo, hi, hi+(hi-lo)
        elif i==len(intervals)-1:
            a, b, c = lo-(hi-lo), lo, hi
        else:
            a, b, c = lo, (lo+hi)/2, hi
        memberships.append(triangular_membership(value,a,b,c))
    return f"A{np.argmax(memberships)+1}"

def build_flrg(fuzzy_series):
    flrg = {}
    for i in range(len(fuzzy_series)-1):
        cur_, nxt_ = fuzzy_series[i], fuzzy_series[i+1]
        flrg.setdefault(cur_, []).append(nxt_)
    return flrg

def defuzzify(predicted_labels, intervals):
    if not predicted_labels:
        return None
    mids = []
    for lab in predicted_labels:
        try:
            idx = int(lab[1:]) - 1
            lo,hi = intervals[idx]
            mids.append((lo+hi)/2)
        except:
            continue
    return float(np.mean(mids)) if mids else None

# PSO util
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
        map_idx = lambda s:int(s[1:]) if isinstance(s,str) and s[0]=='A' else 0
        y_t = np.array([map_idx(s) for s in y_true], float)
        y_p = np.array([map_idx(s) for s in y_pred], float)
        costs.append(sqrt(mean_squared_error(y_t, y_p)))
    return np.array(costs)

optimal_bounds = {}
if run_pso:
    import pyswarms as ps
    options = {'c1':0.5,'c2':0.3,'w':0.9}
    st.info("Menjalankan PSO… proses bisa lama tergantung data.")
    prog = st.progress(0)
    uniq_obat = df["OBAT"].unique()
    for idx, obat in enumerate(uniq_obat, start=1):
        hist = df[df["OBAT"]==obat]["JML OBAT SETUJU"].values
        info = df_uod[df_uod["OBAT"]==obat]
        if len(hist)<2 or info.empty:
            continue
        u_min, u_max = info["U_MIN"].iloc[0], info["U_MAX"].iloc[0]
        dim = num_intervals - 1
        lower = np.full(dim, u_min)
        upper = np.full(dim, u_max)
        opt = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dim,
            options=options,
            bounds=(lower,upper)
        )
        cost, pos = opt.optimize(
            _fitness_boundaries,
            iters=iters,
            data=hist,
            u_min=u_min,
            u_max=u_max,
            k=num_intervals,
            verbose=False
        )
        optimal_bounds[obat] = np.sort(pos)
        prog.progress(idx/len(uniq_obat))
    st.success("PSO selesai.")
else:
    # tanpa PSO: bagi rata
    for _, r in df_uod.iterrows():
        obat = r["OBAT"]; u_min=r["U_MIN"]; u_max=r["U_MAX"]
        inner = np.linspace(u_min, u_max, num_intervals+1)[1:-1]
        optimal_bounds[obat] = inner

# =====================
# Fuzzify → FLRG → Prediksi bulan depan
# =====================
pred_rows = []
for obat, sub in df.groupby("OBAT"):
    info = df_uod[df_uod["OBAT"]==obat]
    if info.empty or obat not in optimal_bounds:
        continue
    u_min, u_max = info["U_MIN"].iloc[0], info["U_MAX"].iloc[0]
    intervals = create_fuzzy_intervals(u_min, u_max, optimal_bounds[obat])

    sub = sub.sort_values("PERIODE")
    series = sub["JML OBAT SETUJU"].tolist()
    fuzzy_series = [determine_fuzzy_set(v, intervals) for v in series]
    flrg = build_flrg(fuzzy_series)

    last_fs = fuzzy_series[-1]
    predicted_fs = flrg.get(last_fs, [last_fs])  # fallback
    pred_value = defuzzify(predicted_fs, intervals)

    next_period = sub["PERIODE"].iloc[-1] + pd.DateOffset(months=1)
    pred_rows.append({
        "OBAT": obat,
        "PERIODE_PREDIKSI": next_period,
        "FUZZY_SET_TERAKHIR": last_fs,
        "FUZZY_PREDIKSI_BULAN_DEPAN": predicted_fs,
        "PREDIKSI_BULAN_DEPAN": pred_value
    })

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
    if "OBAT" not in df_real.columns:
        st.error("File realisasi harus punya kolom 'OBAT' dan 'REALISASI_AKTUAL'.")
    elif "REALISASI_AKTUAL" not in df_real.columns:
        st.error("Kolom 'REALISASI_AKTUAL' tidak ditemukan.")
    else:
        join = pd.merge(
            df_pred[["OBAT","PREDIKSI_BULAN_DEPAN"]],
            df_real[["OBAT","REALISASI_AKTUAL"]],
            on="OBAT",
            how="inner"
        )
        if join.empty:
            st.warning("Tidak ada OBAT yang cocok antara prediksi dan realisasi.")
        else:
            y_true = join["REALISASI_AKTUAL"].astype(float)
            y_pred = join["PREDIKSI_BULAN_DEPAN"].astype(float)
            valid = (y_true!=0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
            if valid.any():
                mape = np.mean(np.abs((y_true[valid]-y_pred[valid])/y_true[valid]))*100
                rmse = sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
                mae  = mean_absolute_error(y_true[valid], y_pred[valid])
                st.subheader("📊 Evaluasi")
                st.write(f"**MAPE:** {mape:.2f}%  |  **RMSE:** {rmse:.2f}  |  **MAE:** {mae:.2f}")
                st.dataframe(join)
            else:
                st.info("Data valid untuk evaluasi tidak cukup (cek nilai 0/NaN).")

# =====================
# Safety Stock & ROP
# =====================
def calc_stock_levels(df_hist, lead_time_days, service_level):
    lt_months = lead_time_days / 30.44
    stats = df_hist.groupby("OBAT")["JML OBAT SETUJU"].agg(["mean","std"]).reset_index()
    stats.rename(columns={"mean":"AVG_MONTHLY","std":"STD_MONTHLY"}, inplace=True)
    stats["STD_MONTHLY"] = stats["STD_MONTHLY"].fillna(0.0)
    z = norm.ppf(service_level)
    stats["SAFETY_STOCK"] = z * stats["STD_MONTHLY"] * np.sqrt(lt_months)
    avg_daily = stats["AVG_MONTHLY"] / 30.44
    stats["REORDER_POINT"] = (avg_daily * lead_time_days) + stats["SAFETY_STOCK"]
    stats["SAFETY_STOCK"] = stats["SAFETY_STOCK"].clip(lower=0)
    stats["REORDER_POINT"] = stats["REORDER_POINT"].clip(lower=0)
    return stats[["OBAT","SAFETY_STOCK","REORDER_POINT"]]

df_stock = calc_stock_levels(df, lead_time_days, service_level)
st.subheader("🏬 Safety Stock & Reorder Point")
st.dataframe(df_stock.head())

# =====================
# Download tombol (KHUSUS ADMIN)
# =====================
def to_excel_bytes(df_dict):
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in df_dict.items():
            d.to_excel(xw, sheet_name=name, index=False)
    buf.seek(0)
    return buf

if st.session_state.get("role") == "admin":
    st.download_button(
        "⬇️ Download hasil (Excel)",
        data=to_excel_bytes({"prediksi": df_pred, "uod": df_uod, "stok": df_stock}),
        file_name="hasil_prediksi_prb.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("👁️ Anda login sebagai *viewer* — fitur download hanya untuk admin.")
