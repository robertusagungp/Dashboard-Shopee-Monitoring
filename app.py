import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Shopee Enterprise BI Dashboard",
    layout="wide"
)

SHEET_ID = "11YP-YgU6N65Uaq64FLcNa65yWdqW8ABX"
GID = "788089194"

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"


# ============================================================
# AUTO REFRESH 5 MINUTES
# ============================================================

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="refresh")
except:
    pass


# ============================================================
# FORMAT RUPIAH
# ============================================================

def format_rupiah(x):

    try:
        x = float(x)
    except:
        return "Rp0"

    sign = "-" if x < 0 else ""

    x = abs(int(round(x)))

    return f"{sign}Rp{x:,}".replace(",", ".")


# ============================================================
# PLOTLY SAFE
# ============================================================

def plotly_safe(df):

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)

        else:
            df[col] = df[col].astype(str)

    df = df.fillna(0)

    return df


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_data():

    df = pd.read_csv(CSV_URL)

    df.columns = df.columns.str.strip()

    return df


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(df):

    d = df.copy()

    # datetime
    d["event_dt"] = pd.to_datetime(
        d.get("Waktu Pesanan Dibuat"),
        errors="coerce"
    )

    d["event_date"] = d["event_dt"].dt.date

    # numeric
    d["revenue"] = pd.to_numeric(
        d.get("Total Pembayaran"),
        errors="coerce"
    )

    d["qty"] = pd.to_numeric(
        d.get("Jumlah"),
        errors="coerce"
    )

    # Shopee cost components
    d["voucher"] = pd.to_numeric(
        d.get("Voucher Toko", 0),
        errors="coerce"
    ).fillna(0)

    d["service_fee"] = pd.to_numeric(
        d.get("Biaya Layanan", 0),
        errors="coerce"
    ).fillna(0)

    d["admin_fee"] = pd.to_numeric(
        d.get("Biaya Administrasi", 0),
        errors="coerce"
    ).fillna(0)

    d["other_fee"] = pd.to_numeric(
        d.get("Biaya Lainnya", 0),
        errors="coerce"
    ).fillna(0)

    d["shipping_cost"] = pd.to_numeric(
        d.get("Ongkos Kirim yang Dibayarkan ke Jasa Kirim", 0),
        errors="coerce"
    ).fillna(0)

    # Shopee NET REVENUE
    d["shopee_net_revenue"] = (
        d["revenue"]
        - d["voucher"]
        - d["service_fee"]
        - d["admin_fee"]
        - d["other_fee"]
        - d["shipping_cost"]
    )

    d["product"] = d.get("Nama Produk")

    d["order_id"] = d.get("No. Pesanan")

    d["buyer"] = d.get("Nama Pembeli")

    d["city"] = d.get("Kota/Kabupaten")

    return d


# ============================================================
# LOAD + PREPROCESS
# ============================================================

raw = load_data()

d = preprocess(raw)


# ============================================================
# KPI
# ============================================================

orders = d["order_id"].nunique()

revenue = d["revenue"].sum()

net_revenue = d["shopee_net_revenue"].sum()

qty = d["qty"].sum()

repeat = d.groupby("buyer")["order_id"].nunique()

repeat_rate = (repeat >= 2).mean()


# ============================================================
# DASHBOARD
# ============================================================

st.title("Shopee Enterprise BI Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Orders", f"{orders:,}")

c2.metric("Revenue", format_rupiah(revenue))

c3.metric("Shopee Net Revenue", format_rupiah(net_revenue))

c4.metric("Units Sold", f"{int(qty):,}")

c5.metric("Repeat Rate", f"{repeat_rate*100:.1f}%")


# ============================================================
# TREND ANALYSIS
# ============================================================

st.subheader("Revenue Trend")

trend = d.groupby("event_date").agg(

    revenue=("revenue", "sum"),

    net_revenue=("shopee_net_revenue", "sum")

).reset_index()

trend = plotly_safe(trend)

fig = px.line(trend, x="event_date", y="net_revenue")

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SKU ANALYSIS
# ============================================================

st.subheader("SKU Performance")

sku = d.groupby("product").agg(

    revenue=("revenue", "sum"),

    net_revenue=("shopee_net_revenue", "sum"),

    qty=("qty", "sum"),

    orders=("order_id", "nunique")

).reset_index()

sku["margin"] = sku["net_revenue"] / sku["revenue"]

sku = sku.sort_values("net_revenue", ascending=False)

sku = plotly_safe(sku)

fig2 = px.bar(sku.head(20), x="product", y="net_revenue")

st.plotly_chart(fig2, use_container_width=True)

st.dataframe(sku)


# ============================================================
# CITY ANALYSIS
# ============================================================

st.subheader("City Performance")

city = d.groupby("city").agg(

    net_revenue=("shopee_net_revenue", "sum")

).reset_index()

city = plotly_safe(city)

fig3 = px.bar(city, x="city", y="net_revenue")

st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# PARETO ANALYSIS
# ============================================================

st.subheader("Pareto Analysis")

sku["cum"] = sku["net_revenue"].cumsum()

sku["cum_pct"] = sku["cum"] / sku["net_revenue"].sum()

st.dataframe(sku.head(20))


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

st.subheader("Recommendation Engine")

sku["score"] = sku["net_revenue"] * sku["qty"]

st.write("Scale these:")

st.dataframe(sku.head(10))

st.write("Consider dropping:")

st.dataframe(sku.tail(10))


# ============================================================
# REPEAT CUSTOMER ANALYSIS
# ============================================================

st.subheader("Repeat Customer")

repeat_df = repeat.reset_index()

repeat_df.columns = ["buyer", "orders"]

repeat_df = repeat_df.sort_values("orders", ascending=False)

st.dataframe(repeat_df.head(20))


# ============================================================
# RAW DATA
# ============================================================

with st.expander("Raw Data"):

    st.dataframe(d)
