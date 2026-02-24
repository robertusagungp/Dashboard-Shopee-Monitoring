import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =============================
# CONFIG
# =============================

st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide"
)

GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "11YP-YgU6N65Uaq64FLcNa65yWdqW8ABX"
    "/export?format=csv&gid=788089194"
)

# =============================
# LOAD DATA
# =============================

@st.cache_data(ttl=300)
def load_data():

    df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

    # normalisasi nama kolom
    df.columns = df.columns.str.strip()

    # convert datetime
    datetime_cols = [
        "Waktu Pesanan Dibuat",
        "Waktu Pesanan Selesai",
        "Waktu Pembayaran Dilakukan"
    ]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # convert numeric
    numeric_cols = [
        "Total Pembayaran",
        "Jumlah",
        "Total Harga Produk"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # filter hanya yang selesai
    if "Status Pesanan" in df.columns:
        df = df[df["Status Pesanan"] == "Selesai"]

    return df


df = load_data()

# =============================
# HEADER
# =============================

st.title("Business Performance Dashboard")
st.caption("Live data from Google Sheet")

# =============================
# SIDEBAR FILTER
# =============================

st.sidebar.header("Filter")

if "Waktu Pesanan Dibuat" in df.columns:

    min_date = df["Waktu Pesanan Dibuat"].min()
    max_date = df["Waktu Pesanan Dibuat"].max()

    date_range = st.sidebar.date_input(
        "Tanggal",
        value=(min_date, max_date)
    )

    if len(date_range) == 2:
        start, end = date_range
        df = df[
            (df["Waktu Pesanan Dibuat"].dt.date >= start)
            &
            (df["Waktu Pesanan Dibuat"].dt.date <= end)
        ]

# =============================
# KPI SECTION
# =============================

total_revenue = df["Total Pembayaran"].sum()
total_orders = df["No. Pesanan"].nunique()
total_units = df["Jumlah"].sum()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"Rp {total_revenue:,.0f}"
)

col2.metric(
    "Total Orders",
    f"{total_orders:,}"
)

col3.metric(
    "Total Units Sold",
    f"{total_units:,}"
)

col4.metric(
    "Avg Order Value",
    f"Rp {avg_order_value:,.0f}"
)

st.divider()

# =============================
# REVENUE TREND
# =============================

st.subheader("Revenue Trend")

if "Waktu Pesanan Dibuat" in df.columns:

    trend = (
        df.groupby(df["Waktu Pesanan Dibuat"].dt.date)
        ["Total Pembayaran"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        trend,
        x="Waktu Pesanan Dibuat",
        y="Total Pembayaran"
    )

    st.plotly_chart(fig, use_container_width=True)

# =============================
# TOP PRODUCT
# =============================

st.subheader("Top Products")

product = (
    df.groupby("Nama Produk")
    .agg(
        revenue=("Total Pembayaran", "sum"),
        units=("Jumlah", "sum"),
        orders=("No. Pesanan", "nunique")
    )
    .sort_values("revenue", ascending=False)
    .head(10)
    .reset_index()
)

fig2 = px.bar(
    product,
    x="Nama Produk",
    y="revenue"
)

st.plotly_chart(fig2, use_container_width=True)

st.dataframe(product)

# =============================
# TOP CITY
# =============================

st.subheader("Top Cities")

city = (
    df.groupby("Kota/Kabupaten")
    .agg(
        revenue=("Total Pembayaran", "sum"),
        orders=("No. Pesanan", "nunique")
    )
    .sort_values("revenue", ascending=False)
    .head(10)
    .reset_index()
)

fig3 = px.bar(
    city,
    x="Kota/Kabupaten",
    y="revenue"
)

st.plotly_chart(fig3, use_container_width=True)

st.dataframe(city)

# =============================
# RECOMMENDATION ENGINE
# =============================

st.subheader("AI Recommendation Engine")

rec = (
    df.groupby("Nama Produk")
    .agg(
        revenue=("Total Pembayaran", "sum"),
        units=("Jumlah", "sum"),
        orders=("No. Pesanan", "nunique")
    )
    .reset_index()
)

rec["avg_units_per_order"] = rec["units"] / rec["orders"]
rec["score"] = rec["revenue"] * rec["avg_units_per_order"]

rec = rec.sort_values("score", ascending=False)

st.write("Products to scale aggressively:")
st.dataframe(rec.head(10))

st.write("Products underperforming:")
st.dataframe(rec.tail(10))

# =============================
# RAW DATA
# =============================

with st.expander("Raw Data"):
    st.dataframe(df)
