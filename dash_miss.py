# dashboard_development_missnomi
#Mukhammad Rekza Mufti

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta

# time-series classical
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# optional libs (wrapped)
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from pmdarima import auto_arima
    _HAS_PMD = True
except Exception:
    _HAS_PMD = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    _HAS_TF = True
except Exception:
    _HAS_TF = False

st.set_page_config(page_title="Dashboard Analisis Data", layout="wide")
st.title("Dashboard Analitik Missnomi")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(gsheet_url):
    return pd.read_csv(gsheet_url)

def beautify_timeseries_plot(ax, title="", ylabel="Value"):
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45)

def evaluate_series(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    rmse = np.sqrt(np.mean((pred - true)**2))
    mae = np.mean(np.abs(pred - true))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((pred - true) / np.where(true==0, np.nan, true))) * 100
    return rmse, mae, mape

def remove_outliers_iqr(series, multiplier=1.5):
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    iqr = q3 - q1
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    series_clipped = np.where(series < low, low, np.where(series > high, high, series))
    return series_clipped

def create_lag_features(df, col='value', lags=[1,7,14,30]):
    df_feat = df.copy()
    for l in lags:
        df_feat[f'lag_{l}'] = df_feat[col].shift(l)
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['day'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat = df_feat.dropna()
    return df_feat

# -----------------------------
# Load Google Sheet
# -----------------------------
GSHEET_URL = st.text_input("Masukkan CSV export Google Sheet URL :",
                           value="https://docs.google.com/spreadsheets/d/1APilL0UzyGIBslMDIPF7B2Ftvo7XK0lo6q_d2IOjW1A/export?format=csv&gid=1146076705")
# --- URL master produk (BARU) ---
GSHEET_URL_MASTER = st.text_input("Masukkan CSV export Google Sheet MASTER PRODUK :",
                           value="https://docs.google.com/spreadsheets/d/1SQ-6kW6YHmqVJHWjE4v5pTGrk10niXl1w-Pvt2N70zs/export?format=csv&gid=0")

try:
    df = load_data(GSHEET_URL)
    df_master = load_data(GSHEET_URL_MASTER)
except Exception as e:
    st.error(f"Gagal load Google Sheet: {e}")
    st.stop()

required_cols = ["Tgl. Pesanan", "Nama Barang", "QTY", "Nominal"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan di sheet: {missing}")
    st.stop()

df["Tgl. Pesanan"] = pd.to_datetime(df["Tgl. Pesanan"], errors="coerce")
df = df.dropna(subset=["Tgl. Pesanan"])

# -----------------------------
# Sidebar UX
# -----------------------------
st.sidebar.header("⚙️ Pengaturan Dashboard")
# Date range
min_date, max_date = df["Tgl. Pesanan"].min(), df["Tgl. Pesanan"].max()
date_range = st.sidebar.date_input("Filter Rentang Tanggal", value=(min_date, max_date))
if len(date_range) != 2:
    st.error("Pilih rentang tanggal dengan benar.")
    st.stop()
start_date, end_date = date_range

# Product selection
products = sorted(df["Nama Barang"].dropna().unique().tolist())
prod_multi = st.sidebar.multiselect("Pilih Nama Barang (boleh lebih dari 1):", options=products, default=None)

# Metric selection
metric_choice = st.sidebar.radio("Pilih metrik:", ["QTY", "Nominal"])

# Preprocessing options
st.sidebar.markdown("### Preprocessing / Cleansing")
apply_outlier = st.sidebar.checkbox("Remove outliers (IQR clipping)", value=True)
apply_log = st.sidebar.checkbox("Apply log1p transform (improves stability)", value=False)
apply_smoothing = st.sidebar.checkbox("Apply smoothing (rolling mean)", value=False)
smoothing_window = st.sidebar.slider("Smoothing window (days)", 3, 30, 7)

# Modeling options
st.sidebar.markdown("### Modeling Options")
enable_auto_model = st.sidebar.checkbox("Enable Auto Model Selection (compare RMSE)", value=True)
include_heavy_models = st.sidebar.checkbox("Include heavy models (XGBoost / LSTM) if available", value=False)
lstm_toggle = st.sidebar.checkbox("Enable LSTM (only if TensorFlow is installed)", value=False)

# Analysis selector
st.sidebar.markdown("### Analisis")
analysis = st.sidebar.radio("Pilih Analisis:", ["Preview Data", "Descriptive", "Correlation", "Forecasting","Sales by Channel","Monitoring Produk","Pareto Produk","Gross Profit & Margin","Klasifikasi Produk"])

# -----------------------------
# Apply filters
# -----------------------------
df_filtered = df[(df["Tgl. Pesanan"] >= pd.to_datetime(start_date)) & (df["Tgl. Pesanan"] <= pd.to_datetime(end_date))]
if prod_multi:
    df_filtered = df_filtered[df_filtered["Nama Barang"].isin(prod_multi)]

st.subheader("Preview Data Setelah Filter")
st.dataframe(df_filtered.head(300))
with st.expander("Metodologi & Rule of Thumb", expanded=False):
        st.markdown("""MUFTIIII  
        """)

# -----------------------------
# Numeric & quick checks
# -----------------------------
if df_filtered.empty:
    st.error("Data kosong setelah filter. Cek rentang tanggal / produk.")
    st.stop()

# -----------------------------
# Descriptive & Correlation
# -----------------------------
if analysis == "Descriptive":
    st.header("Descriptive Analysis")
    st.write(df_filtered.describe(include='all'))
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        chosen = st.multiselect("Pilih kolom numerik untuk plot histogram:", num_cols, default=num_cols[:3])
        for c in chosen:
            fig, ax = plt.subplots()
            ax.hist(df_filtered[c].dropna(), bins=30)
            ax.set_title(f"Distribusi {c}")
            st.pyplot(fig)

elif analysis == "Correlation":
    st.header("Correlation Analysis")
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Perlu minimal 2 kolom numerik.")
    else:
        chosen = st.multiselect("Pilih kolom untuk korelasi:", num_cols, default=num_cols[:5])
        if len(chosen) >= 2:
            corr = df_filtered[chosen].corr()
            st.dataframe(corr)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
            st.pyplot(fig)

elif analysis == "Sales by Channel":

    st.header("Analisis Penjualan per Channel")

    # ===============================
    # STEP 0 — PILIH KOLOM TANGGAL
    # ===============================
    date_col = st.selectbox(
        "Pilih kolom Tanggal",
        df.columns,
        index=df.columns.get_loc("Tgl. Pesanan") if "Tgl. Pesanan" in df.columns else 0
    )

    # Pastikan datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # ===============================
    # STEP 1 — FILTER PERIODE
    # ===============================

    start_date, end_date = st.date_input(
        "Pilih Periode Analisis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter dataframe
    df_filtered = df[
        (df[date_col] >= pd.to_datetime(start_date)) &
        (df[date_col] <= pd.to_datetime(end_date))
    ]

    if df_filtered.empty:
        st.error("Tidak ada data pada periode ini.")
        st.stop()

    # ===============================
    # STEP 2 — PILIH CHANNEL & METRIC
    # ===============================

    channel_col = st.selectbox(
        "Pilih kolom Channel",
        df.columns,
        index=df.columns.get_loc("Channel") if "Channel" in df.columns else 0
    )

    metric_col = st.selectbox(
        "Pilih Metric",
        ["QTY", "Nominal"] if "Nominal" in df.columns else ["QTY"]
    )

    # ===============================
    # STEP 3 — AGREGASI DATA
    # ===============================

    df_channel = (
        df_filtered
        .groupby(channel_col)[metric_col]
        .sum()
        .reset_index()
        .sort_values(metric_col, ascending=False)
    )

    total_value = df_channel[metric_col].sum()

    # ===============================
    # STEP 4 — SHARE % CHANNEL
    # ===============================

    df_channel["Share (%)"] = (
        df_channel[metric_col] / total_value * 100
    ).round(2)

    # ===============================
    # STEP 5 — OUTPUT
    # ===============================

    st.subheader("Ringkasan Penjualan per Channel")
    st.caption(f"Periode: {start_date} s/d {end_date}")
    st.dataframe(df_channel)

    st.subheader("Share Penjualan (%)")
    st.bar_chart(
        df_channel.set_index(channel_col)["Share (%)"]
    )

    # ===============================
    # STEP 6 — INSIGHT OTOMATIS
    # ===============================

    top_channel = df_channel.iloc[0][channel_col]
    top_share = df_channel.iloc[0]["Share (%)"]

    low_channels = df_channel[df_channel["Share (%)"] < 10][channel_col].tolist()

    st.subheader("Insight :")

    st.markdown(f"""
    **Channel Dominan:** `{top_channel}`  
    Kontribusi **{top_share}%** pada periode terpilih.
    """)

    if low_channels:
        st.warning(
            f"Channel dengan kontribusi <10%: {', '.join(low_channels)}."
        )
    else:
        st.success("Distribusi channel sehat.")

elif analysis == "Monitoring Produk":

    st.subheader("Monitoring QTY Sold per Produk per Bulan")

    # =====================
    # FILTER PERIODE
    # =====================
    min_date = df['Tgl. Pesanan'].min()
    max_date = df['Tgl. Pesanan'].max()

    start_date, end_date = st.date_input(
        "Pilih Periode",
        value=[min_date, max_date]
    )

    df_filter = df[
        (df['Tgl. Pesanan'] >= pd.to_datetime(start_date)) &
        (df['Tgl. Pesanan'] <= pd.to_datetime(end_date))
    ].copy()

    # =====================
    # PREPARE DATA BULANAN
    # =====================
    df_filter['year_month'] = df_filter['Tgl. Pesanan'].dt.to_period('M').astype(str)

    pivot_qty = pd.pivot_table(
        df_filter,
        index='Nama Barang',
        columns='year_month',
        values='QTY',
        aggfunc='sum',
        fill_value=0
    )

    pivot_qty['Grand Total'] = pivot_qty.sum(axis=1)

    # =====================
    # TABEL
    # =====================
    st.markdown("### Tabel QTY Sold Bulanan")
    st.dataframe(pivot_qty)

    # =====================
    # PILIH PRODUK UNTUK GRAFIK
    # =====================
    produk_pilihan = st.multiselect(
        "Pilih Produk untuk Grafik",
        options=pivot_qty.index.tolist(),
        default=pivot_qty.sort_values('Grand Total', ascending=False).head(5).index.tolist()
    )
    # =====================
    # GRAFIK
    # =====================
    if produk_pilihan:
        chart_df = pivot_qty.loc[produk_pilihan].drop(columns='Grand Total')

        st.markdown("### Grafik QTY Sold Bulanan")
        st.line_chart(chart_df.T)
    
    st.subheader("Insight :")
    top_product = pivot_qty["Grand Total"].idxmax()
    top_qty = pivot_qty["Grand Total"].max()
    st.markdown(
        f"Produk dengan penjualan tertinggi adalah "
        f"**{top_product}** dengan total **{top_qty:,} pcs**.")
    
    monthly_total = pivot_qty.drop(columns="Grand Total").sum()
    best_month = monthly_total.idxmax()
    best_month_qty = monthly_total.max()

    st.markdown(
        f"Bulan dengan penjualan tertinggi adalah "
        f"**{best_month}** dengan total **{best_month_qty:,} pcs**."
    )
    last_3 = monthly_total.tail(3)
    diff = last_3.diff().dropna()

    if (diff > 0).all():
        trend = "meningkat"
    elif (diff < 0).all():
        trend = "menurun"
    else:
        trend = "fluktuatif"

    st.markdown(
        f"Tren penjualan 3 bulan terakhir menunjukkan pola **{trend}**."
    )
    zero_months = (pivot_qty.drop(columns="Grand Total") == 0).sum(axis=1)
    inactive_count = (zero_months >= 3).sum()

    if inactive_count > 0:
        st.markdown(
            f"Terdapat **{inactive_count} produk** "
            f"dengan penjualan nol di ≥3 bulan."
        )
    else:
        st.markdown(
            "Tidak ditemukan produk dengan penjualan nol berkepanjangan."
        )

elif analysis == "Pareto Produk":

    st.header("Pareto Analisis Produk")

    # =========================
    # FILTER TANGGAL KHUSUS
    # =========================
    min_date_p = df["Tgl. Pesanan"].min()
    max_date_p = df["Tgl. Pesanan"].max()

    start_p, end_p = st.date_input(
        "Pilih Periode Analisis Pareto",
        value=(min_date_p, max_date_p)
    )

    df_pareto_base = df[
        (df["Tgl. Pesanan"] >= pd.to_datetime(start_p)) &
        (df["Tgl. Pesanan"] <= pd.to_datetime(end_p))
    ].copy()

    # Kalau mau tetap hormati filter produk dari sidebar (prod_multi)
    if prod_multi:
        df_pareto_base = df_pareto_base[df_pareto_base["Nama Barang"].isin(prod_multi)]

    if df_pareto_base.empty:
        st.warning("Tidak ada data untuk analisis Pareto pada periode ini.")
        st.stop()

    # =========================
    # PILIH METRIK
    # =========================
    metric_pareto = st.radio(
        "Pilih metrik untuk Pareto:",
        ["QTY", "Nominal"],
        index=0,
        horizontal=True
    )

    # =========================
    # AGREGASI PER PRODUK
    # =========================
    df_pareto = (
        df_pareto_base
        .groupby("Nama Barang")[["QTY", "Nominal"]]
        .sum()
        .reset_index()
    )

    if df_pareto.empty:
        st.warning("Tidak ada data untuk analisis Pareto setelah filter.")
    else:
        df_pareto = df_pareto.sort_values(metric_pareto, ascending=False)

        total_val = df_pareto[metric_pareto].sum()
        df_pareto["Cum_Value"] = df_pareto[metric_pareto].cumsum()
        df_pareto["Cum_%"] = df_pareto["Cum_Value"] / total_val * 100

        st.subheader(f"Tabel Pareto Produk berdasarkan {metric_pareto}")
        st.caption(f"Periode: {start_p} s/d {end_p}")
        st.dataframe(
            df_pareto[["Nama Barang", "QTY", "Nominal", "Cum_%"]]
            .style.format(
                {"QTY": "{:,.0f}", "Nominal": "{:,.0f}", "Cum_%": "{:,.2f}%"}
            )
        )

        # =========================
        # PARETO CHART
        # =========================
        st.subheader("Pareto Chart")
        fig, ax1 = plt.subplots(figsize=(12, 4))

        x = np.arange(len(df_pareto))
        ax1.bar(x, df_pareto[metric_pareto], color="tab:blue")
        ax1.set_xlabel("Produk")
        ax1.set_ylabel(metric_pareto, color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_pareto["Nama Barang"], rotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x, df_pareto["Cum_%"], color="tab:red", marker="o")
        ax2.set_ylabel("Cumulative %", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(0, 110)

        ax2.axhline(80, color="gray", linestyle="--", linewidth=1)

        fig.tight_layout()
        st.pyplot(fig)

        # =========================
        # INSIGHT 80/20
        # =========================
        st.subheader("Insight Pareto")
        top_mask = df_pareto["Cum_%"] <= 80
        top_products = df_pareto.loc[top_mask, "Nama Barang"].tolist()
        top_share = (
            df_pareto.loc[top_mask, "Cum_%"].max()
            if not df_pareto.loc[top_mask, "Cum_%"].empty
            else 0
        )

        if top_products:
            st.markdown(
                f"- Sekitar **{len(top_products)} produk** pertama menyumbang "
                f"±**{top_share:.1f}%** dari total {metric_pareto} pada periode ini.\n"
                f"- Produk kunci: `{', '.join(top_products[:10])}`"
                + (" dan seterusnya..." if len(top_products) > 10 else "")
            )
        else:
            st.markdown(
                "Belum ada produk yang mencapai ambang **80%**; distribusi relatif merata."
            )

elif analysis == "Gross Profit & Margin":
    st.subheader("Gross Profit & Margin Analysis")
    df_gp = df.copy()
    # =========================
    # SAFETY CHECK KOLOM WAJIB
    # =========================
    required_cols = ["Tgl. Pesanan", "QTY", "HPP", "HARGA JUAL"]
    for col in required_cols:
        if col not in df_gp.columns:
            st.error(f"Kolom '{col}' tidak ditemukan di dataset.")
            st.stop()
    # =========================
    # CLEANING & TYPE FIX
    # =========================
    df_gp["Nominal"] = pd.to_numeric(df_gp["Nominal"], errors="coerce").fillna(0)
    df_gp["QTY"] = pd.to_numeric(df_gp["QTY"], errors="coerce").fillna(0)
    df_gp["HPP"] = pd.to_numeric(df_gp["HPP"], errors="coerce").fillna(0)
    df_gp["HARGA JUAL"] = pd.to_numeric(df_gp["HARGA JUAL"], errors="coerce").fillna(0)
    df_gp["Tgl. Pesanan"] = pd.to_datetime(df_gp["Tgl. Pesanan"], errors="coerce")
    df_gp = df_gp.dropna(subset=["Tgl. Pesanan"])
    # =========================
    # DATE FILTER
    # =========================
    min_date = df_gp["Tgl. Pesanan"].min().date()
    max_date = df_gp["Tgl. Pesanan"].max().date()
    start_date, end_date = st.date_input(
        "Filter Tanggal",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    df_gp = df_gp[
        (df_gp["Tgl. Pesanan"].dt.date >= start_date) &
        (df_gp["Tgl. Pesanan"].dt.date <= end_date)
    ]

    if df_gp.empty:
        st.warning("Tidak ada data pada rentang tanggal tersebut.")
        st.stop()

    # =========================
    # METRIK DASAR
    # =========================
    df_gp["Revenue"] = df_gp["Nominal"]
    df_gp["COGS"] = df_gp["QTY"] * df_gp["HPP"]
    df_gp["Gross Profit"] = df_gp["Revenue"] - df_gp["COGS"]

    total_revenue = df_gp["Revenue"].sum()
    total_cogs = df_gp["COGS"].sum()
    total_gp = df_gp["Gross Profit"].sum()

    gross_margin = (total_gp / total_revenue * 100) if total_revenue != 0 else 0

    # =========================
    # KPI DISPLAY
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"Rp {total_revenue:,.0f}")
    col2.metric("Total COGS", f"Rp {total_cogs:,.0f}")
    col3.metric("Gross Profit", f"Rp {total_gp:,.0f}")
    col4.metric("Gross Margin %", f"{gross_margin:.2f}%")

    st.divider()

    # =========================
    # DAILY SUMMARY
    # =========================
    st.subheader("Daily Gross Profit")

    daily = (
        df_gp.groupby(df_gp["Tgl. Pesanan"].dt.date)
        .agg({
            "Revenue": "sum",
            "COGS": "sum",
            "Gross Profit": "sum"
        })
        .reset_index()
    )

    daily["Gross Margin %"] = (
        daily["Gross Profit"] / daily["Revenue"] * 100
    ).replace([float("inf"), -float("inf")], 0).fillna(0)

    st.dataframe(daily, use_container_width=True)

    # =========================
    # MONTHLY SUMMARY
    # =========================
    st.subheader("Monthly Gross Profit")

    df_gp["Month"] = df_gp["Tgl. Pesanan"].dt.to_period("M")

    monthly = (
        df_gp.groupby("Month")
        .agg({
            "Revenue": "sum",
            "COGS": "sum",
            "Gross Profit": "sum"
        })
        .reset_index()
    )

    monthly["Gross Margin %"] = (
        monthly["Gross Profit"] / monthly["Revenue"] * 100
    ).replace([float("inf"), -float("inf")], 0).fillna(0)

    monthly["Month"] = monthly["Month"].astype(str)

    st.dataframe(monthly, use_container_width=True)

    # =========================
    # AUTO INSIGHT SECTION
    # =========================
    st.divider()
    st.subheader("Strategic Insight")

    insight_list = []

    # --- 1. Margin Check ---
    if gross_margin < 20:
        insight_list.append(
            "⚠️ Gross Margin di bawah 20%. Risiko pricing terlalu rendah atau HPP terlalu tinggi."
        )
    elif gross_margin < 30:
        insight_list.append(
            "Gross Margin moderat (20–30%). Masih ada ruang optimasi harga atau efisiensi biaya."
        )
    else:
        insight_list.append(
            "✅ Gross Margin sehat (>30%). Struktur harga relatif aman."
        )

    # --- 2. Daily Volatility ---
    if len(daily) > 3:
        daily_std = daily["Gross Profit"].std()
        daily_mean = daily["Gross Profit"].mean()

        if daily_std > daily_mean:
            insight_list.append(
                "📉 Profit harian sangat fluktuatif. Distribusi penjualan tidak stabil."
            )

    # --- 3. Weakest Day ---
    if not daily.empty:
        weakest_day = daily.loc[daily["Gross Profit"].idxmin()]
        insight_list.append(
            f"🔻 Hari terlemah: {weakest_day['Tgl. Pesanan']} dengan GP Rp {weakest_day['Gross Profit']:,.0f}."
        )

    # --- 4. Monthly Concentration Risk ---
    if len(monthly) > 1:
        top_month_share = monthly["Gross Profit"].max() / monthly["Gross Profit"].sum()

        if top_month_share > 0.5:
            insight_list.append(
                "⚠️ Lebih dari 50% profit berasal dari 1 bulan. Risiko ketergantungan periode tinggi."
            )

    # --- 5. Negative Profit Check ---
    if total_gp < 0:
        insight_list.append(
            "🚨 Total Gross Profit negatif. Model bisnis atau diskon perlu dievaluasi segera."
        )

    # =========================
    # DISPLAY INSIGHT
    # =========================
    for ins in insight_list:
        st.write(ins)
    
    # =========================
    # RULE OF THUMB
    # =========================
    with st.expander("Metodologi & Rule of Thumb", expanded=False):
        st.markdown("""
        - Gross Profit = Revenue - COGS  
        - Gross Margin = Gross Profit / Revenue  
        ⚠️ Ini bukan Net Profit (belum dikurangi ads & operasional)

        ### Rule of Thumb :
        - <20% → Rentan  
        - 20–35% → Normal retail  
        - >40% → Sehat & scalable  
        """)

## Menu klasifikasi ##
elif analysis == "Klasifikasi Produk":

    st.header("Klasifikasi Produk Otomatis (Master + Transaksi)")

    # ==========================================
    # 1. FILTER TANGGAL & LEVEL KLASIFIKASI
    # ==========================================

    min_date = df["Tgl. Pesanan"].min()
    max_date = df["Tgl. Pesanan"].max()

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        start_k, end_k = st.date_input(
            "Periode Analisis Klasifikasi",
            value=(min_date, max_date),
            key="classif_date"
        )
    with col_f2:
        level_class = st.radio(
            "Level klasifikasi:",
            ["Per SKU", "Per Nama Produk"],
            index=0,
            horizontal=True
        )

    # filter transaksi sesuai periode
    df_k = df[
        (df["Tgl. Pesanan"] >= pd.to_datetime(start_k)) &
        (df["Tgl. Pesanan"] <= pd.to_datetime(end_k))
    ].copy()

    if df_k.empty:
        st.warning("Tidak ada transaksi pada periode ini.")
        st.stop()

    # pastikan QTY & Nominal numerik
    for col in ["QTY", "Nominal"]:
        if col in df_k.columns:
            df_k[col] = pd.to_numeric(df_k[col], errors="coerce")

    df_k = df_k.dropna(subset=["QTY", "Nominal"])

    # ==========================================
    # 2. MERGE TRANSAKSI + MASTER SEKALI SAJA
    # ==========================================

    if "SKU" not in df_k.columns or "SKU" not in df_master.columns:
        st.error("Kolom 'SKU' harus ada di transaksi dan master produk.")
        st.stop()

    df_m = df_master.rename(
        columns={
            "Nama Produk": "Nama Produk Master",
            "Category Name": "Category Name",
            "Sell Price": "Sell Price",
            "TANGGAL LAUNCHING": "TANGGAL_LAUNCHING"
        }
    ).copy()

    df_m["TANGGAL_LAUNCHING"] = pd.to_datetime(df_m["TANGGAL_LAUNCHING"], errors="coerce")

    # merge satu kali, setelah itu semua agregasi pakai df_merged
    df_merged = df_k.merge(df_m, on="SKU", how="left")
    df_merged = df_merged.reset_index(drop=True)

    # ====== Cek aja ======
    #st.write("Kolom df_master:", df_master.columns.tolist())
    #st.write("Sample TANGGAL LAUNCHING master:", df_master["TANGGAL LAUNCHING"].head())

   # st.write("Kolom df_merged:", df_merged.columns.tolist())
  #  if "TANGGAL_LAUNCHING" in df_merged.columns:
 #       st.write("Sample TANGGAL_LAUNCHING merged:",
#                 df_merged["TANGGAL_LAUNCHING"].head())#
    # ============================ 
    # ==========================================
    # 3. TENTUKAN LEVEL KLASIFIKASI
    # ==========================================

    if level_class == "Per SKU":
        group_cols = ["SKU"]
        name_col = "Nama Produk Master" if "Nama Produk Master" in df_merged.columns else "Nama Barang"
    else:
        name_col = "Nama Produk Master" if "Nama Produk Master" in df_merged.columns else "Nama Barang"
        group_cols = [name_col]

    # ==========================================
    # 4. AGREGASI METRIK + ATRIBUT SEKALIGUS
    # ==========================================

    total_months = (
        (pd.to_datetime(end_k).to_period("M") - pd.to_datetime(start_k).to_period("M")).n
        + 1
    )

    agg_dict = {
        "total_revenue": ("Nominal", "sum"),
        "total_qty": ("QTY", "sum"),
        "last_sale_date": ("Tgl. Pesanan", "max"),
        "first_sale_date": ("Tgl. Pesanan", "min"),
        "months_sold": ("Tgl. Pesanan", lambda x: x.dt.to_period("M").nunique()),
    }

    # atribut master: ambil nilai pertama (asumsi konsisten per SKU)
    for c in ["Nama Produk Master", "Category Name", "Warna", "Size", "HPP", "Sell Price", "TANGGAL_LAUNCHING"]:
        if c in df_merged.columns:
            agg_dict[c] = (c, "first")

    #df_prod = (
      #  df_merged
       # .groupby(group_cols)
      #  .agg(**agg_dict)
     #   .reset_index()
    #)
    df_prod = (
    df_merged
    .groupby(group_cols, as_index=False)
    .agg(**agg_dict)
    )

    # name_col sudah pasti ada di df_prod (karena join & agg first)
   # df_prod = df_prod.reset_index(drop=True)

    # ==========================================
    # 5. METRIK BANTUAN (AGE, DEAD CUTOFF, THRESHOLD)
    # ==========================================
    #today = df_prod["last_sale_date"].max()
    #dead_cutoff = today - pd.DateOffset(months=6)   # Dead: tidak laku ≥ 6 bulan [web:63]

    #if "TANGGAL_LAUNCHING" in df_prod.columns:
      #  diff = today.to_period("M") - df_prod["TANGGAL_LAUNCHING"].dt.to_period("M")
     #   df_prod["age_months"] = pd.to_numeric(diff, errors="coerce")
    #else:
     #   df_prod["age_months"] = np.nan

    today = df_prod["last_sale_date"].max()
    dead_cutoff = today - pd.DateOffset(months=6)

    if "TANGGAL_LAUNCHING" in df_prod.columns:
        launch = pd.to_datetime(df_prod["TANGGAL_LAUNCHING"], errors="coerce")
        last = pd.to_datetime(today)

        # umur dalam bulan: (year diff * 12) + (month diff)
        age_months = (last.year - launch.dt.year) * 12 + (last.month - launch.dt.month)
        # kalau launching NaT → age_months jadi NaN
        age_months = age_months.where(~launch.isna(), np.nan)

        df_prod["age_months"] = age_months
    else:
        df_prod["age_months"] = np.nan

    p80_rev = df_prod["total_revenue"].quantile(0.8)
    p80_qty = df_prod["total_qty"].quantile(0.8)    # top 20% ≈ kelas A / best seller [web:58][web:61]

    # batas bawah (median) untuk Slow Moving
    p50_rev = df_prod["total_revenue"].quantile(0.5)
    p50_qty = df_prod["total_qty"].quantile(0.5)


    # ==========================================
    # 6. FUNGSI KLASIFIKASI
    # ==========================================
    def classify_row(row):
        # 1) DEAD: tidak laku lama atau belum pernah laku
        if (row["last_sale_date"] < dead_cutoff) or (row["total_qty"] == 0):
            return "Dead"

        # 2) NEW LAUNCHING: umur ≤ 3 bulan (jika Launching ada)
        age = row.get("age_months", np.nan)
        if not np.isnan(age):
            if age <= 3:
                return "New Launching"

        # 3) BEST SELLER: top 20% revenue/qty dan laku di ≥ 50% bulan
        if (row["total_revenue"] >= p80_rev) or (row["total_qty"] >= p80_qty):
            if row["months_sold"] >= max(1, total_months * 0.5):
                return "Best Seller"

        # 4) SLOW MOVING:
        #    - masih ada penjualan, tapi jauh di bawah median
        #      (revenue & qty di bawah p50) ATAU
        #    - hanya terjual di sedikit bulan
        low_value = (row["total_revenue"] < p50_rev) and (row["total_qty"] < p50_qty)
        very_sparse = row["months_sold"] <= max(1, total_months * 0.25)

        if low_value or very_sparse:
            return "Slow Moving"

        # 5) SISANYA = STANDAR (mid-range)
        return "Standar"

    df_prod["CAT_auto"] = df_prod.apply(classify_row, axis=1)

    # ==========================================
    # 7. TABEL HASIL KLASIFIKASI
    # ==========================================

    st.subheader("Tabel Klasifikasi Produk")

    show_cols = []
    if level_class == "Per SKU" and "SKU" in df_prod.columns:
        show_cols.append("SKU")
    if name_col in df_prod.columns:
        show_cols.append(name_col)

    for c in ["Category Name", "Warna", "Size", "HPP", "Sell Price", "TANGGAL_LAUNCHING"]:
        if c in df_prod.columns:
            show_cols.append(c)

    show_cols += [
        "CAT_auto",
        "total_revenue", "total_qty",
        "months_sold", "age_months",
        "first_sale_date", "last_sale_date",
    ]
    show_cols = [c for c in show_cols if c in df_prod.columns]

    st.caption(
        f"Periode: {start_k} s/d {end_k}  |  "
        f"Level: {'Per SKU' if level_class == 'Per SKU' else 'Per Nama Produk'}"
    )

    df_show = df_prod[show_cols].sort_values("total_revenue", ascending=False)
    st.dataframe(df_show)

    # ==========================================
    # SUMMARY PER KELAS KLASIFIKASI (QTY & OMSET)
    # ==========================================
    st.subheader("Summary Per Kelas Klasifikasi (Qty & Omset)")

    class_perf = (
        df_prod
        .groupby("CAT_auto", as_index=False)
        .agg(
            total_qty=("total_qty", "sum"),
            total_revenue=("total_revenue", "sum")
        )
        .sort_values("total_revenue", ascending=False)
    )

    st.dataframe(class_perf)

    st.markdown(
        "- **total_qty** = total QTY terjual untuk semua produk di kelas tersebut.\n"
        "- **total_revenue** = total omset untuk semua produk di kelas tersebut."
    )

    # ==========================================
    # 8. RINGKASAN & GRAFIK DISTRIBUSI KELAS
    # ==========================================

    st.subheader("Ringkasan Jumlah Produk per Kelas")

    class_count = (
        df_prod["CAT_auto"]
        .value_counts()
        .rename_axis("Kelas")
        .reset_index(name="Jumlah Produk")
    )
    st.dataframe(class_count)

    fig_c1, ax_c1 = plt.subplots(figsize=(6, 4))
    ax_c1.bar(class_count["Kelas"], class_count["Jumlah Produk"], color="#1f77b4")
    ax_c1.set_xlabel("Kelas Produk")
    ax_c1.set_ylabel("Jumlah Produk")
    ax_c1.set_title("Distribusi Produk per Kelas (Auto)")
    ax_c1.grid(axis="y", alpha=0.3)
    st.pyplot(fig_c1)

    # ==========================================
    # 9. INSIGHT OTOMATIS
    # ==========================================

    st.subheader("Strategic Insight")

    total_prod = len(df_prod)
    n_best = (df_prod["CAT_auto"] == "Best Seller").sum()
    n_new = (df_prod["CAT_auto"] == "New Launching").sum()
    n_slow = (df_prod["CAT_auto"] == "Slow Moving").sum()
    n_dead = (df_prod["CAT_auto"] == "Dead").sum()
    n_stan = (df_prod["CAT_auto"] == "Standar").sum()

    share_best = n_best / total_prod * 100 if total_prod > 0 else 0
    share_new = n_new / total_prod * 100 if total_prod > 0 else 0
    share_slow = n_slow / total_prod * 100 if total_prod > 0 else 0
    share_dead = n_dead / total_prod * 100 if total_prod > 0 else 0
    share_stan = n_stan / total_prod * 100 if total_prod > 0 else 0

    rev_by_class = (
        df_prod.groupby("CAT_auto")["total_revenue"]
        .sum()
        .reindex(["Best Seller", "New Launching", "Slow Moving", "Dead","Standar"])
        .fillna(0)
    )
    total_rev = rev_by_class.sum()
    best_rev_share = (
        rev_by_class.get("Best Seller", 0) / total_rev * 100
        if total_rev > 0 else 0
    )

    st.markdown(
        f"- Dari total **{total_prod} produk**, klasifikasi otomatis menghasilkan: "
        f"**{n_best} Best Seller ({share_best:.1f}%)**, "
        f"**{n_new} New Launching ({share_new:.1f}%)**, "
        f"**{n_slow} Slow Moving ({share_slow:.1f}%)**, "
        f"dan **{n_dead} Dead ({share_dead:.1f}%)**, "
        f"dan **{n_stan} Standar ({share_stan:.1f}%)**.\n"
        f"- Kelas **Best Seller** menyumbang sekitar **{best_rev_share:.1f}%** dari total revenue periode ini."
    )

    if share_dead > 20:
        st.warning(
            f"Porsi produk yang dikategorikan **Dead** cukup besar ({share_dead:.1f}%). "
            "Perlu evaluasi stok dan strategi clearance / write-off."
        )
    if share_slow > 40:
        st.warning(
            f"Porsi **Slow Moving** cukup tinggi ({share_slow:.1f}%). "
            "Pertimbangkan promo, bundling, atau pengurangan pembelian ulang."
        )
    if share_best < 10:
        st.info(
            f"Produk **Best Seller** masih sedikit ({share_best:.1f}% dari total SKU). "
            "Portofolio terlalu tersebar, atau threshold Best Seller perlu di-adjust."
        )
    if share_stan < 30:
        st.info(
            f"Porsi produk **Standar** relatif kecil ({share_stan:.1f}%). "
            "Komposisi portofolio didominasi oleh ekstrem (Best Seller vs Slow/Dead)."
        )
    # ==========================================
    # 10. METODOLOGI & RULE OF THUMB
    # ==========================================

    with st.expander("Metodologi & Rule of Thumb", expanded=False):
        st.markdown(
            """
            **Metodologi:**
            - Data transaksi digabung dengan *master produk* berdasarkan `SKU`, sehingga atribut seperti
              Category, Warna, Size, HPP, Sell Price, dan Tanggal Launching diambil dari master produk.
            - Metrik penjualan per produk selama periode analisis: `total_revenue`, `total_qty`, 
              `months_sold`, `first_sale_date`, `last_sale_date`.
            - Umur produk (`age_months`) dihitung dari `TANGGAL_LAUNCHING`; produk yang tidak laku dalam
              ≥ 6 bulan terakhir diperlakukan sebagai kandidat **dead stock**.
            - Threshold **Best Seller** memakai percentile 80% (top 20%) dari distribusi revenue/qty, 
              sesuai praktik umum ABC classification.

            **Rule of Thumb Kelas:**
            - **Dead**: tidak ada penjualan dalam 6 bulan terakhir atau `total_qty = 0` di periode analisis.  
            - **New Launching**: umur produk ≤ 3 bulan sejak Tanggal Launching.  
            - **Best Seller**: 
              - Bukan New Launching dan bukan Dead.  
              - Masuk top 20% produk berdasarkan `total_revenue` atau `total_qty`.  
              - Terjual di ≥ 50% bulan dalam periode analisis (stabil).  
            - **Slow Moving**: 
              - Bukan New Launching dan bukan Dead.  
              - Tidak memenuhi kriteria Best Seller (nilai dan frekuensi penjualan relatif rendah).  
            - **Standar**
              - Bukan New Launching dan bukan Dead.
              - Tidak memenuhi kriteria Best Seller dan tidak termasuk Slow Moving.
              - Produk dengan performa mid-range: kontribusinya positif, tapi bukan prioritas utama maupun masalah stok.
            """
        )

    # ==========================================
    # 11. DOWNLOAD HASIL KLASIFIKASI
    # ==========================================

    st.subheader("Download Hasil Klasifikasi")

    export_cols = show_cols.copy()
    if "CAT_auto" not in export_cols:
        export_cols.append("CAT_auto")

    df_export = df_prod[export_cols].sort_values("total_revenue", ascending=False)
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV Klasifikasi Produk",
        data=csv_bytes,
        file_name=f"klasifikasi_produk_{start_k}_sd_{end_k}.csv",
        mime="text/csv"
    )


# -----------------------------
# Forecasting
# -----------------------------
else:
    st.header("Forecasting per Produk / Grup")

    metric = metric_choice
    st.write(f"Metric for forecasting: **{metric}**")

    # Aggregate per day
    df_daily = df_filtered[["Tgl. Pesanan", metric]].groupby("Tgl. Pesanan").sum()
    ts_daily = df_daily.resample("D").sum()

    # Replace zeros (sparse) optionally or keep zeros but ffill option
    zero_count = (ts_daily[metric] == 0).sum()
    st.write(f"Total days: {len(ts_daily)} — days with 0 {metric}: {zero_count}")

    # Optionally replace 0 with NaN then ffill/bfill to reduce sparsity
    if st.sidebar.checkbox("Treat zeros as missing (ffill)", value=True):
        ts_daily[metric] = ts_daily[metric].replace(0, np.nan)
        ts_daily[metric] = ts_daily[metric].ffill().bfill().fillna(0)

    # Outlier removal
    if apply_outlier:
        ts_daily['value_raw'] = ts_daily[metric].values
        ts_daily['value'] = remove_outliers_iqr(ts_daily['value_raw'])
    else:
        ts_daily['value'] = ts_daily[metric].values

    # Log transform
    if apply_log:
        ts_daily['value'] = np.log1p(ts_daily['value'])

    # Smoothing
    if apply_smoothing:
        ts_daily['value'] = ts_daily['value'].rolling(window=smoothing_window, min_periods=1, center=False).mean()

    # Decomposition (if enough points)
    if len(ts_daily) >= 14:
        try:
            decomp = seasonal_decompose(ts_daily['value'], period=7, model='additive', extrapolate_trend='freq')
            st.subheader("Decomposition (additive, period=7)")
            fig_d, axes = plt.subplots(4,1, figsize=(12,8), sharex=True)
            axes[0].plot(decomp.observed); axes[0].set_title("Observed")
            axes[1].plot(decomp.trend); axes[1].set_title("Trend")
            axes[2].plot(decomp.seasonal); axes[2].set_title("Seasonal")
            axes[3].plot(decomp.resid); axes[3].set_title("Residual")
            beautify_timeseries_plot(axes[-1], title="Decomposition (dates)", ylabel=metric)
            st.pyplot(fig_d)
        except Exception as e:
            st.info(f"Decomposition error: {e}")

    # Basic stats & histogram
    st.subheader("Summary Statistik Harian (setelah cleaning)")
    st.write(ts_daily['value'].describe())
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(ts_daily['value'].dropna(), bins=40)
    ax_hist.set_title("Histogram nilai harian")
    st.pyplot(fig_hist)

    # Plot series
    st.subheader("Time Series Harian")
    fig_ts, ax_ts = plt.subplots(figsize=(14,3))
    ax_ts.plot(ts_daily.index, ts_daily['value'], label="Actual")
    beautify_timeseries_plot(ax_ts, title=f"Daily Series - {metric}", ylabel=metric)
    ax_ts.legend()
    st.pyplot(fig_ts)

    # Check for all zeros
    if np.allclose(ts_daily['value'].fillna(0).values, 0):
        st.warning("Nilai target semuanya nol setelah preprocessing — forecasting tidak akurat.")
        st.stop()

    # Train-test split
    train_size = int(len(ts_daily) * 0.8)
    if train_size < 10:
        st.warning("Data terlalu sedikit untuk model yang kompleks. Pertimbangkan agregasi mingguan.")
    train = ts_daily.iloc[:train_size].copy()
    test = ts_daily.iloc[train_size:].copy()

    # Models list (dynamically adjusted)
    base_models = ["ARIMA", "SES", "Holt", "Holt-Winters", "MovingAverage", "Naive", "LinearRegression", "RandomForest"]
    ml_models = []
    if _HAS_XGB and include_heavy_models:
        ml_models.append("XGBoost")
    if include_heavy_models and _HAS_TF:
        ml_models.append("LSTM")
    all_models = base_models + ml_models

    # Allow user to pick models or auto-run
    st.subheader("Model Selection")
    chosen_models = st.multiselect("Pilih model (kosong = jalankan auto-selection)", options=all_models, default=None)
    if not chosen_models:
        chosen_models = all_models if enable_auto_model else base_models

    st.write("Models to run:", chosen_models)

    # Function that runs a single model and returns test_forecast and future_forecast series
    def run_model(name, train_ser, test_ser, period=30):
        """Return (test_forecast_series, future_forecast_series, info_dict)"""
        info = {"model": name}
        try:
            if name == "ARIMA":
                model = ARIMA(train_ser, order=(2,1,2))
                fit = model.fit()
                test_fc = fit.forecast(steps=len(test_ser))
                future_fc = fit.forecast(steps=period)
                test_fc = pd.Series(test_fc, index=test_ser.index)
                future_fc = pd.Series(future_fc, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))
                info['note'] = "ARIMA(2,1,2)"

            elif name == "SES":
                model = SimpleExpSmoothing(train_ser)
                fit = model.fit()
                test_fc = fit.forecast(steps=len(test_ser))
                future_fc = fit.forecast(steps=period)
                test_fc = pd.Series(test_fc, index=test_ser.index)
                future_fc = pd.Series(future_fc, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))

            elif name == "Holt":
                model = Holt(train_ser)
                fit = model.fit()
                test_fc = fit.forecast(steps=len(test_ser))
                future_fc = fit.forecast(steps=period)
                test_fc = pd.Series(test_fc, index=test_ser.index)
                future_fc = pd.Series(future_fc, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))

            elif name == "Holt-Winters":
                sp = 7 if len(train_ser) >= 14 else None
                if sp:
                    model = ExponentialSmoothing(train_ser, trend="add", seasonal="add", seasonal_periods=sp)
                else:
                    model = ExponentialSmoothing(train_ser, trend="add", seasonal=None)
                fit = model.fit()
                test_fc = fit.forecast(steps=len(test_ser))
                future_fc = fit.forecast(steps=period)
                test_fc = pd.Series(test_fc, index=test_ser.index)
                future_fc = pd.Series(future_fc, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))

            elif name == "MovingAverage":
                window = 7
                last_ma = train_ser.rolling(window).mean().iloc[-1]
                test_fc = pd.Series([last_ma]*len(test_ser), index=test_ser.index)
                future_fc = pd.Series([last_ma]*period, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))

            elif name == "Naive":
                last = train_ser.iloc[-1]
                test_fc = pd.Series([last]*len(test_ser), index=test_ser.index)
                future_fc = pd.Series([last]*period, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))

            elif name == "LinearRegression":
                X_train = np.arange(len(train_ser)).reshape(-1,1)
                y_train = train_ser.values
                X_test = np.arange(len(train_ser), len(train_ser)+len(test_ser)).reshape(-1,1)
                lr = LinearRegression(); lr.fit(X_train, y_train)
                test_pred = lr.predict(X_test)
                test_fc = pd.Series(test_pred, index=test_ser.index)
                X_future = np.arange(len(train_ser)+len(test_ser), len(train_ser)+len(test_ser)+period).reshape(-1,1)
                future_fc = pd.Series(lr.predict(X_future), index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))
                info['estimator'] = lr

            elif name == "RandomForest":
                Xy = create_lag_features(pd.DataFrame(train_ser), col='value', lags=[1,7,14])
                X_train = Xy.drop(columns=['value']).values
                y_train = Xy['value'].values
                # build test features by concatenating end of train + test
                merged = pd.concat([train_ser, test_ser])
                merged_feat = create_lag_features(pd.DataFrame(merged), col='value', lags=[1,7,14])
                X_test = merged_feat.drop(columns=['value']).iloc[len(Xy):].values
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X_train, y_train)
                test_pred = rf.predict(X_test)
                test_fc = pd.Series(test_pred, index=test_ser.index)
                # future using iterative prediction
                future_preds = []
                last_window = merged.values.flatten().tolist()
                for _ in range(period):
                    lag1 = last_window[-1]
                    lag7 = last_window[-7] if len(last_window)>=7 else last_window[0]
                    lag14 = last_window[-14] if len(last_window)>=14 else last_window[0]
                    feat = np.array([lag1, lag7, lag14, pd.Timestamp.max.dayofweek, 0, 0]).reshape(1,-1)  # simple placeholder for dows
                    # we will instead rely on model.predict with just lags shape
                    # build feat consistent with training (lag1,lag7,lag14,dayofweek,day,month) -- we approximate day features as zeros
                    try:
                        pred = rf.predict(feat)
                    except Exception:
                        pred = [last_window[-1]]
                    future_preds.append(pred[0])
                    last_window.append(pred[0])
                future_fc = pd.Series(future_preds, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))
                info['estimator'] = rf

            elif name == "XGBoost" and _HAS_XGB:
                X_train = np.arange(len(train_ser)).reshape(-1,1)
                y_train = train_ser.values
                X_test = np.arange(len(train_ser), len(train_ser)+len(test_ser)).reshape(-1,1)
                model = XGBRegressor(n_estimators=200, learning_rate=0.05)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                test_fc = pd.Series(pred, index=test_ser.index)
                X_future = np.arange(len(train_ser)+len(test_ser), len(train_ser)+len(test_ser)+period).reshape(-1,1)
                future_fc = pd.Series(model.predict(X_future), index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))
                info['estimator'] = model

            elif name == "LSTM" and _HAS_TF:
                # build sequences (simple)
                series = np.array(train_ser.fillna(method='ffill').values).flatten()
                window = 14
                Xs, ys = [], []
                for i in range(window, len(series)):
                    Xs.append(series[i-window:i])
                    ys.append(series[i])
                Xs, ys = np.array(Xs), np.array(ys)
                if len(Xs) < 10:
                    raise ValueError("Data too short for LSTM")
                Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
                split = int(len(Xs)*0.8)
                X_train_l, y_train_l = Xs, ys
                # build model
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(window,1)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train_l, y_train_l, epochs=30, batch_size=16, verbose=0)
                # prepare test sequences from the end of train + test
                combined = np.concatenate([train_ser.values.flatten(), test_ser.values.flatten()])
                X_test_seq = []
                for i in range(window, window+len(test_ser)):
                    seq = combined[i-window:i]
                    X_test_seq.append(seq)
                X_test_seq = np.array(X_test_seq).reshape((len(X_test_seq), window,1))
                pred_scaled = model.predict(X_test_seq).flatten()
                test_fc = pd.Series(pred_scaled, index=test_ser.index)
                # future iterative
                last_window = combined[-window:].tolist()
                future_preds = []
                for _ in range(period):
                    arr = np.array(last_window[-window:]).reshape((1,window,1))
                    p = model.predict(arr)[0][0]
                    future_preds.append(p)
                    last_window.append(p)
                future_fc = pd.Series(future_preds, index=pd.date_range(test_ser.index[-1]+timedelta(days=1), periods=period, freq='D'))
                info['estimator'] = model

            else:
                raise ValueError(f"Model {name} not available or not implemented")

            return test_fc, future_fc, info

        except Exception as e:
            return None, None, {"model": name, "error": str(e)}

    # Run models
    period = st.slider("Jumlah hari forecast masa depan:", 7, 180, 30)
    results = []
    progress = st.progress(0)
    for i, m in enumerate(chosen_models):
        progress.progress(int((i+1)/len(chosen_models)*100))
        test_fc, future_fc, info = run_model(m, train['value'], test['value'], period=period)
        if test_fc is None:
            st.warning(f"Model {m} gagal: {info.get('error')}")
            continue
        rmse, mae, mape = evaluate_series(test['value'].values, test_fc.values)
        results.append({"model": m, "rmse": rmse, "mae": mae, "mape": mape, "test_fc": test_fc, "future_fc": future_fc, "info": info})

    progress.empty()

    if not results:
        st.error("Tidak ada model yang berhasil dijalankan.")
        st.stop()

    # Show comparison table
    df_results = pd.DataFrame([{"Model":r["model"], "RMSE":r["rmse"], "MAE":r["mae"], "MAPE":r["mape"]} for r in results]).sort_values("RMSE")
    st.subheader("Perbandingan Model (diurutkan RMSE kecil -> besar)")
    st.dataframe(df_results.style.format({"RMSE":"{:.2f}", "MAE":"{:.2f}", "MAPE":"{:.2f}%"}))

    # Auto-select best
    best = min(results, key=lambda x: x['rmse'])
    st.success(f"Best model (by RMSE): {best['model']} — RMSE: {best['rmse']:.2f}, MAPE: {best['mape']:.2f}%")

    # Plot best model results
    st.subheader(f"Plot hasil model terbaik: {best['model']}")
    fig_b, ax_b = plt.subplots(figsize=(12,4))
    ax_b.plot(train.index, train['value'], label="Train")
    ax_b.plot(test.index, test['value'], label="Test Actual")
    ax_b.plot(best['test_fc'].index, best['test_fc'].values, label=f"Test Forecast ({best['model']})")
    beautify_timeseries_plot(ax_b, title=f"Train/Test vs Forecast - {best['model']}", ylabel=metric)
    ax_b.legend()
    st.pyplot(fig_b)

    # Show future forecast of best
    st.subheader(f"Future Forecast ({best['model']}) next {period} days")
    future_fc = best['future_fc']
    st.dataframe(future_fc.to_frame(name="Forecast"))

    fig_f, ax_f = plt.subplots(figsize=(14,4))
    # plot last N days of actual for context
    n_last = st.slider("Context window (days) for plotting actual:", 30, min(365, len(ts_daily)), 120)
    ts_zoom = ts_daily.tail(n_last)
    ax_f.plot(ts_zoom.index, ts_zoom['value'], label="Actual")
    ax_f.plot(future_fc.index, future_fc.values, label="Future Forecast")
    beautify_timeseries_plot(ax_f, title=f"Actual + Future Forecast ({best['model']})", ylabel=metric)
    ax_f.legend()
    st.pyplot(fig_f)

    # If log transform applied, remind user results are in log-scale
    if apply_log:
        st.warning("Transform log1p diterapkan pada data — hasil forecast dalam skala log1p. Untuk interpretasi, gunakan inverse np.expm1.")
    st.info("by Mukhammad Rekza Mufti-Data Analis")
















































