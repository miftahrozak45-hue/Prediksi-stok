import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================================================
# 1) LOGIN & ROLE SYSTEM
# =========================================================
def make_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

auth_conf = st.secrets.get("auth", {})
user_list = auth_conf.get("users", [])
pass_list = auth_conf.get("passwords", [])
role_list = auth_conf.get("roles", [])

USERS = {}
for u, p, r in zip(user_list, pass_list, role_list):
    USERS[u] = {"password": make_hash(p), "role": r}

# Fallback jika secrets kosong
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

# Inisialisasi session
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_block()
    st.stop()

# =========================================================
# 2) HEADER APP
# =========================================================
st.set_page_config(page_title="PRB Stock Predictor", layout="wide")
st.title("📈 Prediksi Kebutuhan Stok PRB (EFTS + PSO)")

st.sidebar.write(f"👤 **Login sebagai:** {st.session_state['username']} ({st.session_state['role']})")

DATA_DIR = Path("./data")

# =========================================================
# 3) UPLOAD DATA
# =========================================================
def load_df(file, is_csv, sep, sheet_name):
    if is_csv:
        sep = "\t" if sep == "\\t" else sep
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                file.seek(0)
                return pd.read_csv(file, sep=sep, encoding=enc)
            except UnicodeDecodeError:
                continue
        file.seek(0)
        return pd.read_csv(file, sep=sep, encoding_errors="ignore", engine="python")
    else:
        return pd.read_excel(file, sheet_name=sheet_name if sheet_name else 0)

st.sidebar.header("📥 Data Sumber")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) atau CSV", type=["xlsx", "csv"])
sep = st.sidebar.selectbox("Delimiter (jika CSV)", [";", ",", "\\t"], index=0)
sheet_name = st.sidebar.text_input("Sheet name (Excel)", value=None)

# =========================================================
# 4) ADMIN / VIEWER MODE
# =========================================================
role = st.session_state["role"]

if role == "admin":
    st.sidebar.success("✅ Mode Admin: dapat mengupload dan memproses data.")
elif role == "viewer":
    st.sidebar.info("👁 Mode Viewer: hanya dapat melihat hasil prediksi.")

if uploaded is None:
    st.info("Silakan upload file sumber data terlebih dahulu untuk melanjutkan.")
    st.stop()

is_csv = uploaded.name.lower().endswith(".csv")
df = load_df(uploaded, is_csv, sep, sheet_name)

st.subheader("📊 Data Awal")
st.dataframe(df.head())

# =========================================================
# 5) CONTOH ANALISIS SEDERHANA (placeholder)
# =========================================================
# Misal: tampilkan statistik sederhana
if "JML OBAT SETUJU" in df.columns:
    stats = df["JML OBAT SETUJU"].describe()
    st.subheader("📈 Statistik Jumlah Obat Disetujui")
    st.write(stats)
else:
    st.warning("Kolom 'JML OBAT SETUJU' tidak ditemukan. Pastikan nama kolom sesuai template data PRB.")

# =========================================================
# 6) ROLE-BASED AKSES (ADMIN ONLY)
# =========================================================
if role == "admin":
    st.subheader("⚙️ Proses Prediksi (Admin Only)")
    if st.button("Jalankan Prediksi Sederhana"):
        # contoh placeholder: rata-rata tiap obat
        hasil = df.groupby("OBAT")["JML OBAT SETUJU"].mean().reset_index()
        hasil.columns = ["OBAT", "PREDIKSI_BULAN_DEPAN"]
        st.success("✅ Prediksi berhasil dihitung!")
        st.dataframe(hasil)

        # Export hasil ke Excel
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            hasil.to_excel(writer, index=False, sheet_name="Prediksi")
        st.download_button(
            label="⬇️ Download hasil prediksi (Excel)",
            data=buffer.getvalue(),
            file_name="hasil_prediksi_prb.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.warning("🔒 Hanya Admin yang dapat menjalankan proses prediksi.")
