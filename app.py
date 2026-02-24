import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============ AUTO-REFRESH (5 menit) ============
# Prefer: pip install streamlit-autorefresh
# Docs: https://docs.streamlit.io/develop/api-reference/execution-flow (autorefresh section)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore

    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

# ============ APP CONFIG ============
st.set_page_config(page_title="Shopee BI Dashboard", layout="wide")

DEFAULT_SHEET_ID = "11YP-YgU6N65Uaq64FLcNa65yWdqW8ABX"
DEFAULT_GID = "788089194"


# ============ UTIL: normalize column names ============
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\u00a0", " ")
    return s


def guess_col(columns, candidates):
    """
    Find a column by fuzzy matching candidates.
    Returns original column name or None.
    """
    cols = list(columns)
    norm_map = {_norm(c): c for c in cols}

    # exact candidate match on normalized
    for cand in candidates:
        cand_n = _norm(cand)
        if cand_n in norm_map:
            return norm_map[cand_n]

    # contains match (best-effort)
    for cand in candidates:
        cand_n = _norm(cand)
        for nc, oc in norm_map.items():
            if cand_n in nc:
                return oc

    return None


def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", utc=False)


def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def money_fmt(x):
    try:
        return f"Rp {float(x):,.0f}"
    except Exception:
        return "Rp 0"


# ============ DATA LOADER (enterprise-ish) ============
@st.cache_data(ttl=300, show_spinner=False)
def load_sheet_csv(sheet_id: str, gid: str) -> pd.DataFrame:
    """
    Raw load from Google Sheets CSV export.
    Cached 5 minutes.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(ttl=300, show_spinner=False)
def preprocess(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Convert types, derive common fields (date, hour, dow).
    Cached 5 minutes, keyed by df content + mapping.
    """
    d = df.copy()

    # Rename virtual columns into standard names (do NOT overwrite original; create standardized columns)
    # Standard keys:
    # order_id, order_status, order_created_dt, order_paid_dt, order_completed_dt
    # product_name, qty, revenue, cogs, discount, platform_fee, shipping_fee, other_fee
    # buyer_id, buyer_name, city, province
    def col(k):
        return mapping.get(k) or None

    # datetime parsing
    for k in ["order_created_dt", "order_paid_dt", "order_completed_dt"]:
        c = col(k)
        if c and c in d.columns:
            d[k] = safe_to_datetime(d[c])
        else:
            d[k] = pd.NaT

    # numeric parsing
    for k in ["qty", "revenue", "cogs", "discount", "platform_fee", "shipping_fee", "other_fee"]:
        c = col(k)
        if c and c in d.columns:
            d[k] = safe_to_numeric(d[c])
        else:
            d[k] = np.nan

    # text fields
    for k in ["order_id", "order_status", "product_name", "buyer_id", "buyer_name", "city", "province"]:
        c = col(k)
        if c and c in d.columns:
            d[k] = d[c].astype(str)
        else:
            d[k] = None

    # Choose "event_dt" for time-based analysis: prefer created_dt, else paid_dt, else completed_dt
    d["event_dt"] = d["order_created_dt"]
    d.loc[d["event_dt"].isna(), "event_dt"] = d["order_paid_dt"]
    d.loc[d["event_dt"].isna(), "event_dt"] = d["order_completed_dt"]

    d["event_date"] = d["event_dt"].dt.date
    d["event_month"] = d["event_dt"].dt.to_period("M").astype(str)
    d["event_hour"] = d["event_dt"].dt.hour
    d["event_dow"] = d["event_dt"].dt.day_name()

    # Profit calculations (robust)
    # Profit = revenue - cogs - platform_fee - shipping_fee - other_fee - discount
    # (discount often already netted in revenue in some exports; we keep it explicit, user can set mapping discount=None)
    d["gross_profit"] = d["revenue"] - d["cogs"]
    d["net_profit"] = (
        d["revenue"]
        - d["cogs"]
        - d["platform_fee"]
        - d["shipping_fee"]
        - d["other_fee"]
        - d["discount"]
    )
    d["profit_margin_net"] = np.where(d["revenue"] > 0, d["net_profit"] / d["revenue"], np.nan)

    return d


@st.cache_data(ttl=300, show_spinner=False)
def compute_metrics(d: pd.DataFrame) -> dict:
    """
    Heavy aggregations cached.
    """
    out = {}

    # Base KPIs
    out["orders"] = int(d["order_id"].nunique()) if d["order_id"].notna().any() else int(len(d))
    out["revenue"] = float(np.nansum(d["revenue"]))
    out["units"] = float(np.nansum(d["qty"])) if d["qty"].notna().any() else np.nan
    out["aov"] = out["revenue"] / out["orders"] if out["orders"] > 0 else 0.0
    out["net_profit"] = float(np.nansum(d["net_profit"]))
    out["gross_profit"] = float(np.nansum(d["gross_profit"]))
    out["net_margin"] = out["net_profit"] / out["revenue"] if out["revenue"] > 0 else np.nan

    # Trends (daily)
    if d["event_date"].notna().any():
        out["trend_daily"] = (
            d.groupby("event_date", dropna=True)
            .agg(
                revenue=("revenue", "sum"),
                orders=("order_id", "nunique") if d["order_id"].notna().any() else ("revenue", "size"),
                net_profit=("net_profit", "sum"),
            )
            .reset_index()
        )
    else:
        out["trend_daily"] = pd.DataFrame()

    # Product aggregation
    if d["product_name"].notna().any():
        out["by_product"] = (
            d.groupby("product_name", dropna=True)
            .agg(
                revenue=("revenue", "sum"),
                units=("qty", "sum"),
                orders=("order_id", "nunique") if d["order_id"].notna().any() else ("revenue", "size"),
                net_profit=("net_profit", "sum"),
                net_margin=("profit_margin_net", "mean"),
            )
            .sort_values("revenue", ascending=False)
            .reset_index()
        )
    else:
        out["by_product"] = pd.DataFrame()

    # City aggregation
    if d["city"].notna().any():
        out["by_city"] = (
            d.groupby("city", dropna=True)
            .agg(
                revenue=("revenue", "sum"),
                orders=("order_id", "nunique") if d["order_id"].notna().any() else ("revenue", "size"),
                net_profit=("net_profit", "sum"),
            )
            .sort_values("revenue", ascending=False)
            .reset_index()
        )
    else:
        out["by_city"] = pd.DataFrame()

    # Status mix
    if d["order_status"].notna().any():
        out["status_mix"] = (
            d.groupby("order_status", dropna=True)
            .agg(
                orders=("order_id", "nunique") if d["order_id"].notna().any() else ("revenue", "size"),
                revenue=("revenue", "sum"),
            )
            .sort_values("orders", ascending=False)
            .reset_index()
        )
    else:
        out["status_mix"] = pd.DataFrame()

    # Heatmap dow/hour
    if d["event_dow"].notna().any() and d["event_hour"].notna().any():
        hm = (
            d.groupby(["event_dow", "event_hour"], dropna=True)
            .agg(revenue=("revenue", "sum"))
            .reset_index()
        )
        out["heatmap"] = hm
    else:
        out["heatmap"] = pd.DataFrame()

    return out


# ============ REPEAT CUSTOMER ANALYTICS ============
@st.cache_data(ttl=300, show_spinner=False)
def repeat_customer_analytics(d: pd.DataFrame) -> dict:
    """
    Requires buyer_id OR buyer_name + event_dt.
    Produces:
      - repeat_rate
      - cohort table (month 0..n)
      - RFM snapshot
    """
    res = {
        "repeat_rate": np.nan,
        "repeat_counts": pd.DataFrame(),
        "cohort_retention": pd.DataFrame(),
        "rfm": pd.DataFrame(),
        "top_repeat_buyers": pd.DataFrame(),
    }

    if d["event_dt"].isna().all():
        return res

    # choose buyer key
    if d["buyer_id"].notna().any():
        buyer_key = "buyer_id"
    elif d["buyer_name"].notna().any():
        buyer_key = "buyer_name"
    else:
        return res

    dd = d[[buyer_key, "event_dt", "revenue", "order_id"]].copy()
    dd = dd.dropna(subset=[buyer_key, "event_dt"])

    if dd.empty:
        return res

    # orders per buyer
    if dd["order_id"].notna().any():
        orders_by_buyer = dd.groupby(buyer_key)["order_id"].nunique()
    else:
        orders_by_buyer = dd.groupby(buyer_key).size()

    repeat_buyers = (orders_by_buyer >= 2).sum()
    total_buyers = orders_by_buyer.shape[0]
    res["repeat_rate"] = float(repeat_buyers / total_buyers) if total_buyers > 0 else np.nan

    res["repeat_counts"] = (
        orders_by_buyer.reset_index()
        .rename(columns={0: "orders"})
        .sort_values(by=orders_by_buyer.name or "order_id", ascending=False)
        .head(50)
    )

    # Cohort retention (by first purchase month)
    dd["order_month"] = dd["event_dt"].dt.to_period("M")
    first_month = dd.groupby(buyer_key)["order_month"].min().rename("cohort_month")
    dd = dd.join(first_month, on=buyer_key)
    dd["cohort_index"] = (dd["order_month"] - dd["cohort_month"]).apply(lambda x: x.n)

    cohort_counts = dd.groupby(["cohort_month", "cohort_index"])[buyer_key].nunique().reset_index()
    cohort_pivot = cohort_counts.pivot(index="cohort_month", columns="cohort_index", values=buyer_key).fillna(0)

    # convert to retention %
    cohort_sizes = cohort_pivot[0].replace(0, np.nan)
    retention = cohort_pivot.divide(cohort_sizes, axis=0)

    res["cohort_retention"] = retention.reset_index().rename(columns={"cohort_month": "cohort_month"})

    # RFM (snapshot)
    now = dd["event_dt"].max() + pd.Timedelta(days=1)

    if dd["order_id"].notna().any():
        freq = dd.groupby(buyer_key)["order_id"].nunique().rename("frequency")
    else:
        freq = dd.groupby(buyer_key).size().rename("frequency")

    monetary = dd.groupby(buyer_key)["revenue"].sum(min_count=1).rename("monetary")
    last_purchase = dd.groupby(buyer_key)["event_dt"].max().rename("last_purchase")
    recency_days = (now - last_purchase).dt.days.rename("recency_days")

    rfm = pd.concat([recency_days, freq, monetary], axis=1).reset_index().rename(columns={buyer_key: "buyer"})
    rfm = rfm.sort_values(["frequency", "monetary"], ascending=False)
    res["rfm"] = rfm

    # Top repeat buyers
    res["top_repeat_buyers"] = rfm[rfm["frequency"] >= 2].head(30)

    return res


# ============ RECOMMENDATION ENGINE ============
@st.cache_data(ttl=300, show_spinner=False)
def recommendation_engine(d: pd.DataFrame) -> dict:
    """
    Rule-based, stable, explainable recommendations:
      - Scale winners (high revenue + good margin)
      - Fix margin (high revenue but low/negative margin)
      - Restock suspects (high orders with low avg qty/possible stockout proxy) -> proxy only
      - Long tail (low revenue, consider bundling/stop)
    """
    res = {
        "scale_winners": pd.DataFrame(),
        "fix_margin": pd.DataFrame(),
        "long_tail": pd.DataFrame(),
        "notes": [],
    }

    if d["product_name"].isna().all():
        res["notes"].append("Kolom product_name belum ada / belum dimapping.")
        return res

    g = (
        d.groupby("product_name", dropna=True)
        .agg(
            revenue=("revenue", "sum"),
            orders=("order_id", "nunique") if d["order_id"].notna().any() else ("revenue", "size"),
            units=("qty", "sum"),
            net_profit=("net_profit", "sum"),
        )
        .reset_index()
    )
    g["net_margin"] = np.where(g["revenue"] > 0, g["net_profit"] / g["revenue"], np.nan)
    g["rev_share"] = g["revenue"] / (g["revenue"].sum() if g["revenue"].sum() else np.nan)

    # thresholds adaptive
    rev_q75 = g["revenue"].quantile(0.75) if g["revenue"].notna().any() else 0
    margin_q60 = g["net_margin"].quantile(0.60) if g["net_margin"].notna().any() else 0

    # scale winners: high revenue + decent margin
    scale = g[(g["revenue"] >= rev_q75) & (g["net_margin"] >= margin_q60)].sort_values(
        ["revenue", "net_margin"], ascending=False
    )
    res["scale_winners"] = scale.head(30)

    # fix margin: high revenue but low/negative margin
    fix = g[(g["revenue"] >= rev_q75) & (g["net_margin"] < margin_q60)].sort_values(
        ["revenue", "net_margin"], ascending=[False, True]
    )
    res["fix_margin"] = fix.head(30)

    # long tail: low revenue, low orders
    long_tail = g.sort_values(["revenue", "orders"], ascending=True).head(30)
    res["long_tail"] = long_tail

    # notes
    if d["cogs"].isna().all() and (d["platform_fee"].isna().all() and d["other_fee"].isna().all()):
        res["notes"].append(
            "Profit analytics masih proxy karena COGS/Fee belum dimapping. Map kolom COGS/Fee di sidebar supaya margin akurat."
        )

    return res


# ============ UI: Sidebar ============
st.title("Shopee-level Business Intelligence Dashboard")
st.caption("Live dari Google Sheet (CSV export). Stabil, explainable, siap deploy.")

# Auto refresh every 5 minutes (300,000 ms)
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh_5m")
else:
    # fallback (doesn't rerun server state as cleanly, but avoids dependency)
    st.components.v1.html(
        """
        <script>
          setTimeout(function() { window.location.reload(); }, 300000);
        </script>
        """,
        height=0,
    )

with st.sidebar:
    st.header("Data Source (Google Sheet)")
    sheet_id = st.text_input("Sheet ID", value=DEFAULT_SHEET_ID)
    gid = st.text_input("GID (worksheet)", value=DEFAULT_GID)

    st.caption("Jika error HTTP 403/404: Share sheet → Anyone with the link → Viewer")

    st.divider()
    st.header("Enterprise Cache Control")
    st.caption("Cache TTL = 5 menit. Kamu bisa force refresh manual.")
    if st.button("Force Refresh Now (clear cache)"):
        st.cache_data.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()

    st.divider()
    st.header("Kolom Mapping (anti-error)")
    st.caption("Kalau nama kolom beda, pilih yang benar. Ini bikin app tahan banting.")


# ============ LOAD RAW ============
try:
    raw = load_sheet_csv(sheet_id, gid)
except Exception as e:
    st.error("Gagal load Google Sheet.")
    st.info(
        "Cek ini:\n"
        "1) Link sheet bisa diakses publik (Viewer)\n"
        "2) Sheet ID & GID benar\n"
        "3) Tidak ada prompt login Google\n\n"
        f"Detail error (ringkas): {type(e).__name__}"
    )
    st.stop()

if raw.empty:
    st.warning("Data kosong.")
    st.stop()

# ============ MAPPING GUESS ============
cols = list(raw.columns)

default_map = {
    "order_id": guess_col(cols, ["No. Pesanan", "Order ID", "order_id", "no pesanan"]),
    "order_status": guess_col(cols, ["Status Pesanan", "status", "order status"]),
    "order_created_dt": guess_col(cols, ["Waktu Pesanan Dibuat", "created time", "order created", "tanggal pesanan"]),
    "order_paid_dt": guess_col(cols, ["Waktu Pembayaran Dilakukan", "paid time", "payment time"]),
    "order_completed_dt": guess_col(cols, ["Waktu Pesanan Selesai", "completed time", "finish time"]),
    "product_name": guess_col(cols, ["Nama Produk", "Product", "product name", "nama item"]),
    "qty": guess_col(cols, ["Jumlah", "Qty", "quantity", "jumlah item"]),
    "revenue": guess_col(cols, ["Total Pembayaran", "Revenue", "total payment", "total pembayaran"]),
    # Profit components (optional)
    "cogs": guess_col(cols, ["COGS", "HPP", "modal", "harga modal"]),
    "discount": guess_col(cols, ["Diskon", "Discount", "voucher", "promo"]),
    "platform_fee": guess_col(cols, ["Biaya Admin", "Platform Fee", "fee", "service fee"]),
    "shipping_fee": guess_col(cols, ["Ongkir", "Shipping", "shipping fee"]),
    "other_fee": guess_col(cols, ["Biaya Lain", "Other Fee", "penalty", "biaya tambahan"]),
    # Customer & geo
    "buyer_id": guess_col(cols, ["Buyer ID", "ID Pembeli", "customer_id", "userid"]),
    "buyer_name": guess_col(cols, ["Nama Pembeli", "Buyer Name", "customer", "pembeli"]),
    "city": guess_col(cols, ["Kota/Kabupaten", "City", "kota", "kabupaten"]),
    "province": guess_col(cols, ["Provinsi", "Province", "provinsi"]),
}

with st.sidebar:
    mapping = {}
    for k, label in [
        ("order_id", "Order ID / No. Pesanan"),
        ("order_status", "Status Pesanan"),
        ("order_created_dt", "Waktu Pesanan Dibuat"),
        ("order_paid_dt", "Waktu Pembayaran Dilakukan (opsional)"),
        ("order_completed_dt", "Waktu Pesanan Selesai (opsional)"),
        ("product_name", "Nama Produk"),
        ("qty", "Jumlah / Qty"),
        ("revenue", "Total Pembayaran (Revenue)"),
        ("cogs", "COGS/HPP (opsional)"),
        ("discount", "Diskon/Voucher (opsional)"),
        ("platform_fee", "Biaya Admin/Platform Fee (opsional)"),
        ("shipping_fee", "Ongkir/Shipping Fee (opsional)"),
        ("other_fee", "Biaya Lain (opsional)"),
        ("buyer_id", "Buyer ID (opsional)"),
        ("buyer_name", "Nama Pembeli (opsional)"),
        ("city", "Kota/Kabupaten (opsional)"),
        ("province", "Provinsi (opsional)"),
    ]:
        options = ["(none)"] + cols
        default = default_map.get(k)
        idx = options.index(default) if default in options else 0
        chosen = st.selectbox(label, options=options, index=idx, key=f"map_{k}")
        mapping[k] = None if chosen == "(none)" else chosen

# ============ PREPROCESS ============
d = preprocess(raw, mapping)

# Optional filter: completed only (if status mapped)
with st.sidebar:
    st.divider()
    st.header("Filter")
    use_status_filter = st.checkbox("Filter status (Completed/Selesai only)", value=True)
    if use_status_filter and d["order_status"].notna().any():
        completed_values = ["selesai", "completed", "done", "pesanan selesai"]
        mask = d["order_status"].astype(str).str.lower().str.strip().isin(completed_values)
        d = d[mask].copy()

    # Date filter
    if d["event_dt"].notna().any():
        min_dt = d["event_dt"].min()
        max_dt = d["event_dt"].max()
        date_range = st.date_input("Tanggal (event)", value=(min_dt.date(), max_dt.date()))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            d = d[(d["event_dt"].dt.date >= start) & (d["event_dt"].dt.date <= end)].copy()

# Guard
if d.empty:
    st.warning("Setelah filter, data jadi kosong. Coba longgarkan filter/status/tanggal.")
    st.stop()

# ============ METRICS ============
m = compute_metrics(d)

# ============ TOP KPIs ============
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue", money_fmt(m["revenue"]))
k2.metric("Orders", f"{m['orders']:,}")
k3.metric("Units", f"{int(m['units']):,}" if np.isfinite(m["units"]) else "—")
k4.metric("AOV", money_fmt(m["aov"]))
k5.metric("Net Profit", money_fmt(m["net_profit"]))

# Profit margin
pm1, pm2, pm3 = st.columns(3)
pm1.metric("Gross Profit", money_fmt(m["gross_profit"]))
pm2.metric("Net Margin", f"{(m['net_margin']*100):.2f}%" if np.isfinite(m["net_margin"]) else "—")
pm3.metric("Net Profit / Order", money_fmt(m["net_profit"] / m["orders"] if m["orders"] else 0))

st.divider()

# ============ SECTION: Trend ============
st.subheader("1) Performance Trend")
trend = m["trend_daily"]
if not trend.empty:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(trend, x="event_date", y="revenue", title="Revenue per Day")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.line(trend, x="event_date", y="net_profit", title="Net Profit per Day")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Trend tidak tersedia (kolom waktu belum dimapping).")

# ============ SECTION: SKU / City ============
st.subheader("2) SKU & Geo Performance")
c1, c2 = st.columns(2)
with c1:
    byp = m["by_product"]
    if not byp.empty:
        topn = st.slider("Top N Products", 5, 50, 15)
        show = byp.head(topn)
        fig = px.bar(show, x="product_name", y="revenue", title=f"Top {topn} Products by Revenue")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(show, use_container_width=True)
    else:
        st.info("Produk belum tersedia (map kolom Nama Produk).")

with c2:
    byc = m["by_city"]
    if not byc.empty:
        topc = st.slider("Top N Cities", 5, 50, 15)
        showc = byc.head(topc)
        figc = px.bar(showc, x="city", y="revenue", title=f"Top {topc} Cities by Revenue")
        st.plotly_chart(figc, use_container_width=True)
        st.dataframe(showc, use_container_width=True)
    else:
        st.info("City belum tersedia (map kolom Kota/Kabupaten).")

# ============ SECTION: BI - Status / Heatmap / Pareto ============
st.subheader("3) Shopee-level BI Views")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Status Mix**")
    sm = m["status_mix"]
    if not sm.empty:
        fig = px.pie(sm, values="orders", names="order_status", title="Order Status Share (by orders)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sm, use_container_width=True)
    else:
        st.caption("Status tidak tersedia / tidak dimapping.")

with c2:
    st.markdown("**Day/Hour Heatmap (Revenue)**")
    hm = m["heatmap"]
    if not hm.empty:
        # order dow for nicer display
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hm["event_dow"] = pd.Categorical(hm["event_dow"], categories=dow_order, ordered=True)
        pivot = hm.pivot_table(index="event_dow", columns="event_hour", values="revenue", aggfunc="sum").fillna(0)
        fig = px.imshow(pivot, aspect="auto", title="Revenue Heatmap (DOW x Hour)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Heatmap butuh kolom waktu valid.")

with c3:
    st.markdown("**Pareto (80/20) Products**")
    byp = m["by_product"]
    if not byp.empty:
        p = byp.copy()
        p["cum_rev"] = p["revenue"].cumsum()
        total = p["revenue"].sum()
        p["cum_share"] = p["cum_rev"] / total if total else np.nan
        p80 = p[p["cum_share"] <= 0.8].shape[0]
        st.metric("Products needed for 80% revenue", f"{p80}")
        st.dataframe(p[["product_name", "revenue", "cum_share"]].head(30), use_container_width=True)
    else:
        st.caption("Butuh kolom Nama Produk.")

st.divider()

# ============ SECTION: Profit Analytics ============
st.subheader("4) Profit Analytics (Gross/Net + Breakdown)")

# breakdown view
fee_cols = ["cogs", "discount", "platform_fee", "shipping_fee", "other_fee"]
bd = {k: float(np.nansum(d[k])) for k in fee_cols if d[k].notna().any()}
bd["revenue"] = float(np.nansum(d["revenue"]))

bdf = pd.DataFrame(
    [{"component": k, "value": v} for k, v in bd.items()]
).sort_values("value", ascending=False)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Profit Summary**")
    st.write(
        f"- Revenue: **{money_fmt(m['revenue'])}**\n"
        f"- Gross Profit: **{money_fmt(m['gross_profit'])}**\n"
        f"- Net Profit: **{money_fmt(m['net_profit'])}**\n"
        f"- Net Margin: **{(m['net_margin']*100):.2f}%**" if np.isfinite(m["net_margin"]) else "- Net Margin: **—**"
    )

    missing_profit_inputs = []
    if d["cogs"].isna().all():
        missing_profit_inputs.append("COGS/HPP")
    if d["platform_fee"].isna().all() and d["other_fee"].isna().all() and d["shipping_fee"].isna().all():
        missing_profit_inputs.append("Fee (Admin/Ongkir/Lainnya)")

    if missing_profit_inputs:
        st.warning(
            "Profit masih proxy / kurang akurat karena belum ada kolom: "
            + ", ".join(missing_profit_inputs)
            + ". Map kolomnya di sidebar kalau ada."
        )

with c2:
    st.markdown("**Cost/Fee Breakdown**")
    fig = px.bar(bdf, x="component", y="value", title="Components Breakdown")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bdf, use_container_width=True)

st.divider()

# ============ SECTION: Repeat Customer ============
st.subheader("5) Repeat Customer Analytics")

rc = repeat_customer_analytics(d)

c1, c2 = st.columns(2)
with c1:
    st.metric(
        "Repeat Buyer Rate",
        f"{(rc['repeat_rate']*100):.2f}%" if np.isfinite(rc["repeat_rate"]) else "—",
    )
    if np.isfinite(rc["repeat_rate"]):
        st.caption("Repeat buyer = buyer dengan >= 2 order dalam periode filter.")
    if not rc["top_repeat_buyers"].empty:
        st.markdown("**Top Repeat Buyers (RFM view)**")
        st.dataframe(rc["top_repeat_buyers"], use_container_width=True)
    else:
        st.info("Repeat analytics butuh buyer_id atau buyer_name yang valid + waktu transaksi.")

with c2:
    st.markdown("**Cohort Retention (by first purchase month)**")
    cohort = rc["cohort_retention"]
    if not cohort.empty:
        # display as heatmap
        cohort2 = cohort.set_index("cohort_month")
        fig = px.imshow(cohort2, aspect="auto", title="Retention % (cohort x months since first purchase)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Cohort butuh buyer + event_dt valid.")

st.divider()

# ============ SECTION: Recommendation Engine ============
st.subheader("6) Actionable Recommendations (Explainable)")

rec = recommendation_engine(d)

if rec["notes"]:
    for n in rec["notes"]:
        st.info(n)

tabs = st.tabs(["Scale Winners", "Fix Margin", "Long Tail"])

with tabs[0]:
    st.markdown("**Scale Winners** (SKU besar & margin relatif bagus) → push ads, tambah stok, bundling upsell")
    if rec["scale_winners"].empty:
        st.caption("Belum ada hasil (butuh product_name + revenue).")
    else:
        st.dataframe(rec["scale_winners"], use_container_width=True)

with tabs[1]:
    st.markdown("**Fix Margin** (SKU besar tapi margin rendah/negatif) → audit COGS, harga, fee, diskon, ongkir")
    if rec["fix_margin"].empty:
        st.caption("Belum ada hasil (butuh profit inputs untuk lebih akurat).")
    else:
        st.dataframe(rec["fix_margin"], use_container_width=True)

with tabs[2]:
    st.markdown("**Long Tail** (SKU kontribusi kecil) → bundling, clearance, stop produksi, atau perbaiki listing")
    if rec["long_tail"].empty:
        st.caption("Belum ada hasil.")
    else:
        st.dataframe(rec["long_tail"], use_container_width=True)

# ============ RAW DATA PREVIEW ============
with st.expander("Raw Data (preview)"):
    st.dataframe(raw.head(200), use_container_width=True)

with st.expander("Processed Data (preview)"):
    st.dataframe(d.head(200), use_container_width=True)

# Footer
st.caption(
    "Auto-refresh: 5 menit. Cache TTL: 5 menit. "
    "Jika ada error HTTP, hampir pasti karena permission Google Sheet belum public viewer."
)
