import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# OPTIONAL AUTO REFRESH
# ============================================================
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Shopee BI Dashboard", layout="wide")

DEFAULT_SHEET_ID = "11YP-YgU6N65Uaq64FLcNa65yWdqW8ABX"
DEFAULT_GID = "788089194"


# ============================================================
# UTIL
# ============================================================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def guess_col(columns, candidates):
    cols = list(columns)
    norm_map = {_norm(c): c for c in cols}

    for cand in candidates:
        cn = _norm(cand)
        if cn in norm_map:
            return norm_map[cn]

    for cand in candidates:
        cn = _norm(cand)
        for k, v in norm_map.items():
            if cn in k:
                return v
    return None


def format_rupiah(value):
    try:
        value = float(value)
    except Exception:
        return "Rp0"
    sign = "-" if value < 0 else ""
    value = abs(int(round(value)))
    # 160710 -> 160,710 -> 160.710
    return f"{sign}Rp{value:,}".replace(",", ".")


def make_plotly_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plotly on Streamlit Cloud can crash on Period/Categorical/NaT/inf.
    This forces JSON-safe types.
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif isinstance(df[col].dtype, pd.PeriodDtype):
            df[col] = df[col].astype(str)
        elif pd.api.types.is_categorical_dtype(df[col]):  # type: ignore
            df[col] = df[col].astype(str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        else:
            # keep as string but avoid None
            df[col] = df[col].astype(str)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_sheet_csv(sheet_id: str, gid: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    return df


def safe_series_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")


def safe_series_text(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col and col in df.columns:
        return df[col].astype(str).fillna("")
    return pd.Series("", index=df.index, dtype="object")


def safe_series_datetime(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col and col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


@st.cache_data(ttl=300, show_spinner=False)
def preprocess(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    d = raw.copy()

    # Standardized fields
    d["order_id"] = safe_series_text(d, mapping.get("order_id"))
    d["order_status"] = safe_series_text(d, mapping.get("order_status"))
    d["product_name"] = safe_series_text(d, mapping.get("product_name"))
    d["buyer_id"] = safe_series_text(d, mapping.get("buyer_id"))
    d["buyer_name"] = safe_series_text(d, mapping.get("buyer_name"))
    d["city"] = safe_series_text(d, mapping.get("city"))
    d["province"] = safe_series_text(d, mapping.get("province"))

    d["order_created_dt"] = safe_series_datetime(d, mapping.get("order_created_dt"))
    d["order_paid_dt"] = safe_series_datetime(d, mapping.get("order_paid_dt"))
    d["order_completed_dt"] = safe_series_datetime(d, mapping.get("order_completed_dt"))

    # choose event_dt priority: created -> paid -> completed
    d["event_dt"] = d["order_created_dt"]
    d.loc[d["event_dt"].isna(), "event_dt"] = d["order_paid_dt"]
    d.loc[d["event_dt"].isna(), "event_dt"] = d["order_completed_dt"]
    d["event_date"] = d["event_dt"].dt.date
    d["event_month"] = d["event_dt"].dt.to_period("M").astype(str)
    d["event_hour"] = d["event_dt"].dt.hour
    d["event_dow"] = d["event_dt"].dt.day_name()

    # Core numeric inputs
    d["harga_produk"] = safe_series_numeric(d, mapping.get("harga_produk"))
    d["qty"] = safe_series_numeric(d, mapping.get("qty"))

    # Shopee official net revenue components
    # Net = harga_produk + ongkir_pembeli + subsidi_ongkir - ongkir_ke_kurir - voucher_penjual - biaya_admin - biaya_layanan - biaya_lainnya
    d["ongkir_pembeli"] = safe_series_numeric(d, mapping.get("ongkir_pembeli"))
    d["subsidi_ongkir_shopee"] = safe_series_numeric(d, mapping.get("subsidi_ongkir_shopee"))
    d["ongkir_ke_kurir"] = safe_series_numeric(d, mapping.get("ongkir_ke_kurir"))
    d["voucher_penjual"] = safe_series_numeric(d, mapping.get("voucher_penjual"))
    d["biaya_admin"] = safe_series_numeric(d, mapping.get("biaya_admin"))
    d["biaya_layanan"] = safe_series_numeric(d, mapping.get("biaya_layanan"))
    d["biaya_lainnya"] = safe_series_numeric(d, mapping.get("biaya_lainnya"))

    d["shopee_net_revenue"] = (
        d["harga_produk"]
        + d["ongkir_pembeli"]
        + d["subsidi_ongkir_shopee"]
        - d["ongkir_ke_kurir"]
        - d["voucher_penjual"]
        - d["biaya_admin"]
        - d["biaya_layanan"]
        - d["biaya_lainnya"]
    )

    # Profit analytics (optional)
    d["cogs"] = safe_series_numeric(d, mapping.get("cogs"))
    d["gross_profit"] = d["shopee_net_revenue"] - d["cogs"]
    d["gross_margin"] = np.where(d["shopee_net_revenue"] > 0, d["gross_profit"] / d["shopee_net_revenue"], np.nan)

    # Buyer key for repeat analytics
    if (d["buyer_id"].astype(str).str.len() > 0).any():
        d["buyer_key"] = d["buyer_id"]
    else:
        d["buyer_key"] = d["buyer_name"]

    return d


@st.cache_data(ttl=300, show_spinner=False)
def compute_metrics(d: pd.DataFrame) -> dict:
    orders = int(d["order_id"].nunique()) if d["order_id"].astype(str).str.len().gt(0).any() else int(len(d))
    harga_produk_sum = float(d["harga_produk"].sum())
    net_sum = float(d["shopee_net_revenue"].sum())
    qty_sum = float(d["qty"].sum())
    aov = net_sum / orders if orders else 0.0
    gp = float(d["gross_profit"].sum())
    gm = gp / net_sum if net_sum else np.nan

    return {
        "orders": orders,
        "harga_produk": harga_produk_sum,
        "shopee_net_revenue": net_sum,
        "qty": qty_sum,
        "aov": aov,
        "gross_profit": gp,
        "gross_margin": gm,
    }


@st.cache_data(ttl=300, show_spinner=False)
def repeat_customer_analytics(d: pd.DataFrame) -> dict:
    res = {
        "repeat_rate": np.nan,
        "top_repeat_buyers": pd.DataFrame(),
        "cohort_retention": pd.DataFrame(),
        "rfm": pd.DataFrame(),
    }

    if d["event_dt"].isna().all():
        return res

    dd = d[["buyer_key", "event_dt", "shopee_net_revenue", "order_id"]].copy()
    dd = dd.dropna(subset=["buyer_key", "event_dt"])
    dd = dd[dd["buyer_key"].astype(str).str.len() > 0]

    if dd.empty:
        return res

    # frequency
    if dd["order_id"].astype(str).str.len().gt(0).any():
        freq = dd.groupby("buyer_key")["order_id"].nunique()
    else:
        freq = dd.groupby("buyer_key").size()

    repeat_rate = float((freq >= 2).mean()) if len(freq) else np.nan
    res["repeat_rate"] = repeat_rate

    # RFM
    now = dd["event_dt"].max() + pd.Timedelta(days=1)
    monetary = dd.groupby("buyer_key")["shopee_net_revenue"].sum()
    last_purchase = dd.groupby("buyer_key")["event_dt"].max()
    recency_days = (now - last_purchase).dt.days

    rfm = pd.DataFrame({
        "buyer": monetary.index.astype(str),
        "recency_days": recency_days.values,
        "frequency": freq.reindex(monetary.index).values,
        "monetary": monetary.values,
    }).sort_values(["frequency", "monetary"], ascending=False)

    res["rfm"] = rfm
    res["top_repeat_buyers"] = rfm[rfm["frequency"] >= 2].head(30)

    # Cohort retention (month-based)
    dd["order_month"] = dd["event_dt"].dt.to_period("M")
    first_month = dd.groupby("buyer_key")["order_month"].min().rename("cohort_month")
    dd = dd.join(first_month, on="buyer_key")
    dd["cohort_index"] = (dd["order_month"] - dd["cohort_month"]).apply(lambda x: x.n)

    cohort_counts = dd.groupby(["cohort_month", "cohort_index"])["buyer_key"].nunique().reset_index()
    cohort_pivot = cohort_counts.pivot(index="cohort_month", columns="cohort_index", values="buyer_key").fillna(0)
    cohort_sizes = cohort_pivot[0].replace(0, np.nan)
    retention = cohort_pivot.divide(cohort_sizes, axis=0)

    retention = retention.reset_index()
    retention["cohort_month"] = retention["cohort_month"].astype(str)
    res["cohort_retention"] = retention

    return res


@st.cache_data(ttl=300, show_spinner=False)
def recommendation_engine(d: pd.DataFrame) -> dict:
    res = {"scale_winners": pd.DataFrame(), "fix_margin": pd.DataFrame(), "long_tail": pd.DataFrame(), "notes": []}

    if (d["product_name"].astype(str).str.len() == 0).all():
        res["notes"].append("Kolom produk belum dimapping.")
        return res

    g = (
        d.groupby("product_name", dropna=True)
        .agg(
            net_revenue=("shopee_net_revenue", "sum"),
            orders=("order_id", "nunique"),
            units=("qty", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        .reset_index()
    )

    g["gross_margin"] = np.where(g["net_revenue"] > 0, g["gross_profit"] / g["net_revenue"], np.nan)

    rev_q75 = g["net_revenue"].quantile(0.75) if len(g) else 0
    margin_q60 = g["gross_margin"].quantile(0.60) if g["gross_margin"].notna().any() else 0

    res["scale_winners"] = g[(g["net_revenue"] >= rev_q75) & (g["gross_margin"] >= margin_q60)].sort_values(
        ["net_revenue", "gross_margin"], ascending=False
    ).head(30)

    res["fix_margin"] = g[(g["net_revenue"] >= rev_q75) & (g["gross_margin"] < margin_q60)].sort_values(
        ["net_revenue", "gross_margin"], ascending=[False, True]
    ).head(30)

    res["long_tail"] = g.sort_values(["net_revenue", "orders"], ascending=True).head(30)

    if d["cogs"].sum() == 0:
        res["notes"].append("COGS/HPP belum dimapping → gross profit & margin masih proxy (anggap COGS=0).")

    return res


# ============================================================
# UI
# ============================================================
st.title("Shopee-level Business Intelligence Dashboard")
st.caption("Live dari Google Sheet. Custom mapping kolom. Anti error. Auto refresh 5 menit.")

# Auto refresh 5 minutes
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh_5m")
else:
    st.components.v1.html(
        "<script>setTimeout(()=>{window.location.reload();}, 300000);</script>",
        height=0,
    )

with st.sidebar:
    st.header("Data Source")
    sheet_id = st.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
    gid = st.text_input("Worksheet GID", value=DEFAULT_GID)
    st.caption("Jika HTTPError: Share sheet → Anyone with the link → Viewer")

    st.divider()
    st.header("Cache")
    if st.button("Force refresh (clear cache)"):
        st.cache_data.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()

# Load raw
try:
    raw = load_sheet_csv(sheet_id, gid)
except Exception as e:
    st.error("Gagal load Google Sheet (CSV export).")
    st.info("Cek permission share (Viewer), Sheet ID, dan GID.")
    st.write(f"Error type: {type(e).__name__}")
    st.stop()

if raw.empty:
    st.warning("Data kosong.")
    st.stop()

cols = list(raw.columns)

# Default mapping guesses (Shopee export biasanya mirip, tapi kamu bisa custom)
default_map = {
    "order_id": guess_col(cols, ["No. Pesanan", "Order ID", "no pesanan"]),
    "order_status": guess_col(cols, ["Status Pesanan", "status pesanan", "status"]),
    "order_created_dt": guess_col(cols, ["Waktu Pesanan Dibuat", "Tanggal Pesanan", "created"]),
    "order_paid_dt": guess_col(cols, ["Waktu Pembayaran Dilakukan", "paid"]),
    "order_completed_dt": guess_col(cols, ["Waktu Pesanan Selesai", "completed", "selesai"]),
    "product_name": guess_col(cols, ["Nama Produk", "product name", "produk"]),
    "qty": guess_col(cols, ["Jumlah", "qty", "quantity"]),
    # Shopee official components
    "harga_produk": guess_col(cols, ["Harga Produk", "Subtotal Pesanan", "Total Pembayaran", "Total Harga Produk"]),
    "ongkir_pembeli": guess_col(cols, ["Ongkir Dibayar Pembeli", "Ongkir Pembeli"]),
    "subsidi_ongkir_shopee": guess_col(cols, ["Potongan Ongkos Kirim dari Shopee", "Subsidi Ongkir", "Subsidi Shopee"]),
    "ongkir_ke_kurir": guess_col(cols, ["Ongkos Kirim yang Dibayarkan ke Jasa Kirim", "Ongkir ke Jasa Kirim"]),
    "voucher_penjual": guess_col(cols, ["Voucher Toko", "Voucher Toko yang ditanggung Penjual"]),
    "biaya_admin": guess_col(cols, ["Biaya Administrasi", "Admin Fee"]),
    "biaya_layanan": guess_col(cols, ["Biaya Layanan", "Service Fee"]),
    "biaya_lainnya": guess_col(cols, ["Biaya Lainnya", "Other Fee"]),
    # Optional profit input
    "cogs": guess_col(cols, ["COGS", "HPP", "Harga Modal", "modal"]),
    # Customer
    "buyer_id": guess_col(cols, ["Buyer ID", "ID Pembeli", "customer_id"]),
    "buyer_name": guess_col(cols, ["Nama Pembeli", "buyer name", "pembeli"]),
    "city": guess_col(cols, ["Kota/Kabupaten", "City", "kota"]),
    "province": guess_col(cols, ["Provinsi", "Province", "provinsi"]),
}

# Mapping UI
with st.sidebar:
    st.header("Mapping Kolom")
    st.caption("Pilih kolom yang sesuai. Kalau tidak ada, pilih (none).")

    options = ["(none)"] + cols
    mapping = {}

    def pick(key, label):
        default = default_map.get(key)
        idx = options.index(default) if default in options else 0
        chosen = st.selectbox(label, options=options, index=idx, key=f"map_{key}")
        mapping[key] = None if chosen == "(none)" else chosen

    # Core identity/time
    pick("order_id", "Order ID / No. Pesanan")
    pick("order_status", "Status Pesanan")
    pick("order_created_dt", "Waktu Pesanan Dibuat")
    pick("order_paid_dt", "Waktu Pembayaran (opsional)")
    pick("order_completed_dt", "Waktu Pesanan Selesai (opsional)")

    # Dimensions
    pick("product_name", "Nama Produk")
    pick("qty", "Jumlah/QTY")
    pick("buyer_id", "Buyer ID (opsional)")
    pick("buyer_name", "Nama Pembeli (opsional)")
    pick("city", "Kota/Kabupaten (opsional)")
    pick("province", "Provinsi (opsional)")

    st.divider()
    st.subheader("Shopee Net Revenue Components (Official)")
    pick("harga_produk", "Harga Produk / Subtotal Pesanan")
    pick("ongkir_pembeli", "Ongkir Dibayar Pembeli (opsional)")
    pick("subsidi_ongkir_shopee", "Subsidi Ongkir dari Shopee (opsional)")
    pick("ongkir_ke_kurir", "Ongkir Dibayar ke Jasa Kirim (opsional)")
    pick("voucher_penjual", "Voucher Toko Ditanggung Penjual (opsional)")
    pick("biaya_admin", "Biaya Administrasi (opsional)")
    pick("biaya_layanan", "Biaya Layanan (opsional)")
    pick("biaya_lainnya", "Biaya Lainnya (opsional)")

    st.divider()
    st.subheader("Profit Input (Optional)")
    pick("cogs", "COGS/HPP (opsional)")

# Preprocess
d = preprocess(raw, mapping)

# Filters
with st.sidebar:
    st.header("Filter")
    only_completed = st.checkbox("Hanya status selesai/completed", value=False)
    if only_completed and (d["order_status"].astype(str).str.len() > 0).any():
        status = d["order_status"].astype(str).str.lower().str.strip()
        d = d[status.isin(["selesai", "completed", "done", "pesanan selesai"])].copy()

    if d["event_dt"].notna().any():
        min_dt = d["event_dt"].min().date()
        max_dt = d["event_dt"].max().date()
        dr = st.date_input("Rentang tanggal", value=(min_dt, max_dt))
        if isinstance(dr, tuple) and len(dr) == 2:
            start, end = dr
            d = d[(d["event_dt"].dt.date >= start) & (d["event_dt"].dt.date <= end)].copy()

if d.empty:
    st.warning("Data kosong setelah filter.")
    st.stop()

# Metrics
m = compute_metrics(d)

# ================= KPIs =================
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Orders", f"{m['orders']:,}")
k2.metric("Harga Produk (Gross)", format_rupiah(m["harga_produk"]))
k3.metric("Shopee Net Revenue", format_rupiah(m["shopee_net_revenue"]))
k4.metric("Units", f"{int(m['qty']):,}" if np.isfinite(m["qty"]) else "—")
k5.metric("AOV (Net/Order)", format_rupiah(m["aov"]))
k6.metric("Gross Profit (proxy)", format_rupiah(m["gross_profit"]))

st.divider()

# ================= Trend =================
st.subheader("1) Trend")
trend = (
    d.groupby("event_date", dropna=True)
    .agg(
        harga_produk=("harga_produk", "sum"),
        shopee_net_revenue=("shopee_net_revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        orders=("order_id", "nunique"),
    )
    .reset_index()
)
trend_safe = make_plotly_safe(trend)

c1, c2 = st.columns(2)
with c1:
    fig = px.line(trend_safe, x="event_date", y="shopee_net_revenue", title="Shopee Net Revenue per Day")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = px.line(trend_safe, x="event_date", y="orders", title="Orders per Day")
    st.plotly_chart(fig2, use_container_width=True)

# ================= SKU / City =================
st.subheader("2) SKU & City Performance")

by_sku = (
    d.groupby("product_name", dropna=True)
    .agg(
        net_revenue=("shopee_net_revenue", "sum"),
        gross=("harga_produk", "sum"),
        units=("qty", "sum"),
        orders=("order_id", "nunique"),
        gross_profit=("gross_profit", "sum"),
    )
    .reset_index()
)
by_sku["gross_margin"] = np.where(by_sku["net_revenue"] > 0, by_sku["gross_profit"] / by_sku["net_revenue"], np.nan)
by_sku = by_sku.sort_values("net_revenue", ascending=False)

by_city = (
    d.groupby("city", dropna=True)
    .agg(
        net_revenue=("shopee_net_revenue", "sum"),
        orders=("order_id", "nunique"),
    )
    .reset_index()
).sort_values("net_revenue", ascending=False)

c1, c2 = st.columns(2)
with c1:
    topn = st.slider("Top N SKU", 5, 50, 15)
    sku_show = make_plotly_safe(by_sku.head(topn))
    fig = px.bar(sku_show, x="product_name", y="net_revenue", title=f"Top {topn} SKU by Net Revenue")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(by_sku.head(topn), use_container_width=True)

with c2:
    topc = st.slider("Top N City", 5, 50, 15)
    city_show = make_plotly_safe(by_city.head(topc))
    fig = px.bar(city_show, x="city", y="net_revenue", title=f"Top {topc} City by Net Revenue")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(by_city.head(topc), use_container_width=True)

st.divider()

# ================= BI Views =================
st.subheader("3) BI Views (Shopee-style)")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Status Mix**")
    sm = (
        d.groupby("order_status", dropna=True)
        .agg(orders=("order_id", "nunique"), net_revenue=("shopee_net_revenue", "sum"))
        .reset_index()
        .sort_values("orders", ascending=False)
    )
    if len(sm):
        sm_safe = make_plotly_safe(sm)
        fig = px.pie(sm_safe, values="orders", names="order_status", title="Order Share by Status")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Status belum dimapping / kosong.")

with c2:
    st.markdown("**Day/Hour Heatmap (Net Revenue)**")
    if d["event_dt"].notna().any():
        hm = (
            d.groupby(["event_dow", "event_hour"], dropna=True)
            .agg(net_revenue=("shopee_net_revenue", "sum"))
            .reset_index()
        )
        if len(hm):
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            hm["event_dow"] = pd.Categorical(hm["event_dow"], categories=dow_order, ordered=True)
            pivot = hm.pivot_table(index="event_dow", columns="event_hour", values="net_revenue", aggfunc="sum").fillna(0)
            # Use numpy for imshow safe
            fig = px.imshow(pivot.values, aspect="auto", title="Heatmap Net Revenue (DOW x Hour)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Heatmap tidak ada data.")
    else:
        st.caption("Heatmap butuh kolom waktu.")

with c3:
    st.markdown("**Pareto 80/20 (SKU)**")
    if len(by_sku):
        p = by_sku.copy()
        total = p["net_revenue"].sum()
        p["cum_rev"] = p["net_revenue"].cumsum()
        p["cum_share"] = np.where(total > 0, p["cum_rev"] / total, np.nan)
        p80 = int((p["cum_share"] <= 0.8).sum())
        st.metric("SKU untuk 80% revenue", f"{p80}")
        st.dataframe(p[["product_name", "net_revenue", "cum_share"]].head(25), use_container_width=True)
    else:
        st.caption("Butuh produk & revenue.")

st.divider()

# ================= Profit Analytics =================
st.subheader("4) Profit Analytics")

bd = pd.DataFrame(
    [
        ("Harga Produk", float(d["harga_produk"].sum())),
        ("Ongkir Pembeli", float(d["ongkir_pembeli"].sum())),
        ("Subsidi Ongkir Shopee", float(d["subsidi_ongkir_shopee"].sum())),
        ("Ongkir ke Kurir", -float(d["ongkir_ke_kurir"].sum())),
        ("Voucher Penjual", -float(d["voucher_penjual"].sum())),
        ("Biaya Admin", -float(d["biaya_admin"].sum())),
        ("Biaya Layanan", -float(d["biaya_layanan"].sum())),
        ("Biaya Lainnya", -float(d["biaya_lainnya"].sum())),
        ("= Shopee Net Revenue", float(d["shopee_net_revenue"].sum())),
        ("COGS/HPP", -float(d["cogs"].sum())),
        ("= Gross Profit", float(d["gross_profit"].sum())),
    ],
    columns=["component", "value"],
)
bd_safe = make_plotly_safe(bd)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Breakdown Komponen**")
    fig = px.bar(bd_safe, x="component", y="value", title="Net Revenue Breakdown")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown("**Tabel Komponen**")
    bd_display = bd.copy()
    bd_display["value_rp"] = bd_display["value"].apply(format_rupiah)
    st.dataframe(bd_display[["component", "value_rp"]], use_container_width=True)

st.divider()

# ================= Repeat Customer =================
st.subheader("5) Repeat Customer Analytics")
rc = repeat_customer_analytics(d)

c1, c2 = st.columns(2)
with c1:
    st.metric(
        "Repeat Buyer Rate",
        f"{(rc['repeat_rate']*100):.2f}%" if np.isfinite(rc["repeat_rate"]) else "—",
    )
    if not rc["top_repeat_buyers"].empty:
        st.markdown("**Top Repeat Buyers (RFM)**")
        st.dataframe(rc["top_repeat_buyers"], use_container_width=True)
    else:
        st.caption("Butuh buyer_id atau buyer_name untuk repeat analytics.")

with c2:
    st.markdown("**Cohort Retention**")
    cohort = rc["cohort_retention"]
    if not cohort.empty:
        cohort2 = cohort.set_index("cohort_month")
        cohort2 = cohort2.drop(columns=[c for c in cohort2.columns if c == "cohort_month"], errors="ignore")
        # plotly safe via numpy array
        fig = px.imshow(cohort2.values, aspect="auto", title="Retention Heatmap (Cohort x Month Index)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Cohort butuh buyer + tanggal transaksi.")

st.divider()

# ================= Recommendation Engine =================
st.subheader("6) Recommendation Engine (Actionable)")
rec = recommendation_engine(d)

if rec["notes"]:
    for n in rec["notes"]:
        st.info(n)

t1, t2, t3 = st.tabs(["Scale Winners", "Fix Margin", "Long Tail"])
with t1:
    st.write("SKU besar & margin relatif bagus → scale ads, tambah stok, bundling")
    st.dataframe(rec["scale_winners"], use_container_width=True)
with t2:
    st.write("SKU besar tapi margin rendah → audit fee/voucher/ongkir/COGS, adjust pricing")
    st.dataframe(rec["fix_margin"], use_container_width=True)
with t3:
    st.write("SKU kontribusi kecil → bundling/clearance/stop, perbaiki listing")
    st.dataframe(rec["long_tail"], use_container_width=True)

# ================= Raw Preview =================
with st.expander("Raw Data (preview)"):
    st.dataframe(raw.head(200), use_container_width=True)

with st.expander("Processed Data (preview)"):
    prev = d.head(200).copy()
    # add formatted columns for quick scan
    for col in ["harga_produk", "shopee_net_revenue", "gross_profit"]:
        if col in prev.columns:
            prev[col + "_rp"] = prev[col].apply(format_rupiah)
    st.dataframe(prev, use_container_width=True)

st.caption("Auto-refresh 5 menit. Cache TTL 5 menit. Mapping kolom di sidebar supaya sesuai export Shopee kamu.")
