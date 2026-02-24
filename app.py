import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Shopee Enterprise BI Dashboard",
    layout="wide"
)

DEFAULT_SHEET_ID = "11YP-YgU6N65Uaq64FLcNa65yWdqW8ABX"
DEFAULT_GID = "788089194"


# ============================================================
# AUTO REFRESH
# ============================================================

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="refresh")
except:
    pass


# ============================================================
# FORMAT RUPIAH DISPLAY
# ============================================================

def rupiah(x):

    try:
        x = float(x)
    except:
        return "Rp0"

    sign = "-" if x < 0 else ""
    x = abs(int(round(x)))

    return f"{sign}Rp{x:,}".replace(",", ".")


# ============================================================
# FIX RUPIAH PARSER (CRITICAL FIX)
# ============================================================

def parse_rupiah(series):

    s = series.astype(str)

    s = (
        s
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0)


def num(df, col):

    if col and col in df.columns:

        return parse_rupiah(df[col])

    return pd.Series(
        0,
        index=df.index,
        dtype="float64"
    )


def txt(df, col):

    if col and col in df.columns:

        return df[col].astype(str)

    return pd.Series("", index=df.index)


def dt(df, col):

    if col and col in df.columns:

        return pd.to_datetime(df[col], errors="coerce")

    return pd.Series(pd.NaT, index=df.index)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load(sheet_id, gid):

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    df = pd.read_csv(url)

    df.columns = df.columns.str.strip()

    return df


# ============================================================
# PREPROCESS
# ============================================================

@st.cache_data(ttl=300)
def preprocess(raw, mapping):

    d = raw.copy()

    d["order_id"] = txt(d, mapping["order_id"])

    d["product"] = txt(d, mapping["product"])

    d["buyer"] = txt(d, mapping["buyer"])

    d["city"] = txt(d, mapping["city"])

    d["event_dt"] = dt(d, mapping["datetime"])

    d["event_date"] = d["event_dt"].dt.date

    d["qty"] = num(d, mapping["qty"])

    d["harga_produk"] = num(d, mapping["harga_produk"])

    d["ongkir_pembeli"] = num(d, mapping["ongkir_pembeli"])

    d["subsidi"] = num(d, mapping["subsidi"])

    d["ongkir_kurir"] = num(d, mapping["ongkir_kurir"])

    d["voucher"] = num(d, mapping["voucher"])

    d["admin"] = num(d, mapping["admin"])

    d["layanan"] = num(d, mapping["layanan"])

    d["lain"] = num(d, mapping["lain"])

    d["cogs"] = num(d, mapping["cogs"])

    # Official Shopee Net Revenue

    d["net_revenue"] = (

        d["harga_produk"]
        + d["ongkir_pembeli"]
        + d["subsidi"]
        - d["ongkir_kurir"]
        - d["voucher"]
        - d["admin"]
        - d["layanan"]
        - d["lain"]

    )

    d["gross_profit"] = d["net_revenue"] - d["cogs"]

    d["margin"] = np.where(

        d["net_revenue"] > 0,
        d["gross_profit"] / d["net_revenue"],
        np.nan

    )

    return d


# ============================================================
# METRICS
# ============================================================

def metrics(d):

    orders = d["order_id"].nunique()

    gross = d["harga_produk"].sum()

    net = d["net_revenue"].sum()

    profit = d["gross_profit"].sum()

    qty = d["qty"].sum()

    repeat = (

        d.groupby("buyer")["order_id"]
        .nunique()

    )

    repeat_rate = (repeat >= 2).mean()

    return {

        "orders": orders,
        "gross": gross,
        "net": net,
        "profit": profit,
        "qty": qty,
        "repeat_rate": repeat_rate

    }


# ============================================================
# UI
# ============================================================

st.title("Shopee Enterprise BI Dashboard")


with st.sidebar:

    st.header("Source")

    sheet_id = st.text_input(
        "Sheet ID",
        DEFAULT_SHEET_ID
    )

    gid = st.text_input(
        "GID",
        DEFAULT_GID
    )


raw = load(sheet_id, gid)

cols = ["(none)"] + list(raw.columns)


with st.sidebar:

    st.header("Column Mapping")

    def pick(name):

        v = st.selectbox(name, cols)

        return None if v == "(none)" else v


    mapping = {

        "order_id": pick("Order ID"),
        "product": pick("Product"),
        "buyer": pick("Buyer"),
        "city": pick("City"),
        "datetime": pick("Datetime"),
        "qty": pick("Quantity"),

        "harga_produk": pick("Harga Produk"),
        "ongkir_pembeli": pick("Ongkir Pembeli"),
        "subsidi": pick("Subsidi Shopee"),
        "ongkir_kurir": pick("Ongkir ke Kurir"),
        "voucher": pick("Voucher"),
        "admin": pick("Admin Fee"),
        "layanan": pick("Service Fee"),
        "lain": pick("Other Fee"),

        "cogs": pick("COGS")

    }


d = preprocess(raw, mapping)

m = metrics(d)


# ============================================================
# KPI
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Orders", f"{m['orders']:,}")

c2.metric("Gross Revenue", rupiah(m["gross"]))

c3.metric("Shopee Net Revenue", rupiah(m["net"]))

c4.metric("Gross Profit", rupiah(m["profit"]))

c5.metric("Repeat Rate", f"{m['repeat_rate']*100:.1f}%")


# ============================================================
# TREND
# ============================================================

st.subheader("Revenue Trend")

trend = (

    d.groupby("event_date")
    .agg(net=("net_revenue", "sum"))
    .reset_index()

)

fig = px.line(trend, x="event_date", y="net")

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

st.subheader("Product Performance")

prod = (

    d.groupby("product")
    .agg(
        net=("net_revenue", "sum"),
        profit=("gross_profit", "sum"),
        qty=("qty", "sum")
    )
    .sort_values("net", ascending=False)

)

st.dataframe(prod)


# ============================================================
# RAW DATA
# ============================================================

with st.expander("Raw Data"):

    st.dataframe(d)
