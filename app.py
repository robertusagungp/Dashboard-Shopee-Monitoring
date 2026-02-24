import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta

# Optional auto refresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except:
    HAS_AUTOREFRESH = False


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Shopee BI Dashboard",
    layout="wide"
)

SHEET_ID = "11YP-YgU6N65Uaq64FLcNa65yWdqW8ABX"
GID = "788089194"

GOOGLE_SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"


# ============================================================
# AUTO REFRESH
# ============================================================

if HAS_AUTOREFRESH:
    st_autorefresh(interval=300000, key="refresh")
else:
    st.caption("Auto refresh aktif (fallback mode)")


# ============================================================
# FORMAT RUPIAH FIX
# ============================================================

def format_rupiah(value):

    try:
        value = float(value)
    except:
        return "Rp0"

    sign = "-" if value < 0 else ""
    value = abs(int(value))

    return f"{sign}Rp{value:,}".replace(",", ".")


# ============================================================
# SAFE PLOTLY FIX
# ============================================================

def make_plotly_safe(df):

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

        elif pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)

        elif isinstance(df[col].dtype, pd.PeriodDtype):
            df[col] = df[col].astype(str)

        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_data():

    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

    df.columns = df.columns.str.strip()

    return df


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(df):

    d = df.copy()

    if "Waktu Pesanan Dibuat" in d.columns:
        d["event_dt"] = pd.to_datetime(d["Waktu Pesanan Dibuat"], errors="coerce")

    if "Total Pembayaran" in d.columns:
        d["revenue"] = pd.to_numeric(d["Total Pembayaran"], errors="coerce")

    if "Jumlah" in d.columns:
        d["qty"] = pd.to_numeric(d["Jumlah"], errors="coerce")

    if "Nama Produk" in d.columns:
        d["product_name"] = d["Nama Produk"]

    if "Status Pesanan" in d.columns:
        d["status"] = d["Status Pesanan"]

    if "No. Pesanan" in d.columns:
        d["order_id"] = d["No. Pesanan"]

    if "Nama Pembeli" in d.columns:
        d["buyer"] = d["Nama Pembeli"]

    if "Kota/Kabupaten" in d.columns:
        d["city"] = d["Kota/Kabupaten"]

    # OPTIONAL fee columns
    fee_map = {
        "Biaya Layanan": "service_fee",
        "Biaya Administrasi": "admin_fee",
        "Voucher Toko": "voucher",
        "Ongkos Kirim": "shipping_cost"
    }

    for col, new_col in fee_map.items():
        if col in d.columns:
            d[new_col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[new_col] = 0

    # ========================================================
    # SHOPEE NET REVENUE FORMULA
    # ========================================================

    d["shopee_net_revenue"] = (
        d["revenue"]
        - d["service_fee"]
        - d["admin_fee"]
        - d["voucher"]
        - d["shipping_cost"]
    )

    d["event_date"] = d["event_dt"].dt.date

    return d


# ============================================================
# METRICS
# ============================================================

@st.cache_data(ttl=300)
def compute_metrics(d):

    return {

        "orders": d["order_id"].nunique(),

        "revenue": d["revenue"].sum(),

        "net_revenue": d["shopee_net_revenue"].sum(),

        "qty": d["qty"].sum(),

        "aov": d["revenue"].sum() / d["order_id"].nunique()

    }


# ============================================================
# REPEAT CUSTOMER
# ============================================================

@st.cache_data(ttl=300)
def repeat_customer(d):

    if "buyer" not in d.columns:
        return 0

    repeat = d.groupby("buyer")["order_id"].nunique()

    repeat_rate = (repeat >= 2).mean()

    return repeat_rate


# ============================================================
# MAIN
# ============================================================

st.title("Shopee Business Intelligence Dashboard")

raw = load_data()

d = preprocess(raw)

metrics = compute_metrics(d)

repeat_rate = repeat_customer(d)


# ============================================================
# KPI
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Orders", f"{metrics['orders']:,}")

col2.metric("Revenue", format_rupiah(metrics["revenue"]))

col3.metric("Shopee Net Revenue", format_rupiah(metrics["net_revenue"]))

col4.metric("Units Sold", f"{int(metrics['qty']):,}")

col5.metric("Repeat Rate", f"{repeat_rate*100:.1f}%")


# ============================================================
# TREND
# ============================================================

st.subheader("Revenue Trend")

trend = d.groupby("event_date").agg(

    revenue=("revenue", "sum"),

    net_revenue=("shopee_net_revenue", "sum")

).reset_index()

trend = make_plotly_safe(trend)

fig = px.line(trend, x="event_date", y="net_revenue")

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

st.subheader("Top Products")

prod = d.groupby("product_name").agg(

    revenue=("revenue", "sum"),

    net_revenue=("shopee_net_revenue", "sum"),

    qty=("qty", "sum")

).sort_values("net_revenue", ascending=False).head(20).reset_index()

prod = make_plotly_safe(prod)

fig2 = px.bar(prod, x="product_name", y="net_revenue")

st.plotly_chart(fig2, use_container_width=True)

st.dataframe(prod)


# ============================================================
# CITY PERFORMANCE
# ============================================================

st.subheader("Top Cities")

city = d.groupby("city").agg(

    net_revenue=("shopee_net_revenue", "sum")

).sort_values("net_revenue", ascending=False).head(20).reset_index()

city = make_plotly_safe(city)

fig3 = px.bar(city, x="city", y="net_revenue")

st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

st.subheader("Recommendations")

prod["score"] = prod["net_revenue"] * prod["qty"]

prod = prod.sort_values("score", ascending=False)

st.write("Scale these products:")

st.dataframe(prod.head(10))

st.write("Low performers:")

st.dataframe(prod.tail(10))


# ============================================================
# RAW DATA
# ============================================================

with st.expander("Raw Data"):

    st.dataframe(d)
