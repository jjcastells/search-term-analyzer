import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime

# =====================
# Config
# =====================
st.set_page_config(page_title="BidForest Mini — Search Term Waste Detector", page_icon="🧪", layout="wide")
st.title("🧪 Search Term Waste Detector")
st.caption(
    "Sube un Search Term Report agregado (CSV) y te muestro los términos con muchos clics y 0 compras. "
    "Sin IDs, sin bulk, sin líos. Ideal para higiene rápida."
)

DEFAULT_MIN_CLICKS = 50

# =====================
# Helpers (tu stack)
# =====================
def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(ch)
    )

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _clean(x):
        x = str(x)
        x = x.replace("\ufeff", "")   # BOM
        x = x.replace("\u200b", "")   # zero-width
        x = x.replace("\xa0", " ")    # NBSP
        x = re.sub(r"\s+", " ", x)    # colapsa espacios
        return x.strip()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col if str(c) != "nan"]).strip() for col in df.columns]

    df.columns = [_clean(c) for c in df.columns]
    return df

def to_float_euaware(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if (s.str.contains(",").mean() > 0.5):
        s = s.str.replace(r"[^\d,.\-]", "", regex=True)
        s = s.str.replace(".", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(r"[^\d.\-]", "", regex=True)
        s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def find_col(df: pd.DataFrame, options):
    if isinstance(options, str):
        options = [options]

    def norm(x: str) -> str:
        x = str(x)
        x = x.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
        x = re.sub(r"\s+", " ", x).strip().lower()
        x = strip_accents(x)
        return x

    cols = list(df.columns)
    cols_norm = {norm(c): c for c in cols}

    for opt in options:
        optn = norm(opt)
        if optn in cols_norm:
            return cols_norm[optn]
        # fallback: algunos reportes tienen sufijos
        for cn, original in cols_norm.items():
            if cn.startswith(optn + "."):
                return original
    return None

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        return pd.read_csv(file, dtype=str)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=";", dtype=str)

def sanitize_sheet_name(name: str) -> str:
    cleaned = re.sub(r'[:\\/?*\[\]]', '-', name)
    cleaned = cleaned.strip() or "WASTE"
    return cleaned[:31]

# =====================
# Upload
# =====================
uploaded_file = st.file_uploader("📤 Sube tu Search Term Report (CSV)", type=["csv"])
if uploaded_file is None:
    st.stop()

df = load_csv(uploaded_file)
if df is None or df.empty:
    st.error("No se pudo leer el CSV o está vacío.")
    st.stop()

df = clean_columns(df)

st.subheader("📄 Vista previa")
st.dataframe(df.head(30), use_container_width=True)

# =====================
# Column mapping (STR agregado ES/EN)
# =====================
# Buscamos las columnas típicas de STR (agregado)
col_term = find_col(df, [
    "Término de búsqueda de los clientes",
    "Término de búsqueda",
    "Search Term",
    "Customer Search Term",
    "Search term"
])

col_clicks = find_col(df, ["Clics", "Clicks"])
col_purchases = find_col(df, [
    "Compras",
    "Pedidos",
    "Orders",
    "Purchases",
    "7 Day Total Orders (#)",  # por si viene en EN viejo
])

col_sales = find_col(df, [
    "Ventas (EUR)", "Ventas(EUR)", "Ventas", "Sales",
    "7 Day Total Sales", "14 Day Total Sales", "Total Sales"
])

col_cost = find_col(df, [
    "Coste total (EUR)", "Coste(EUR)", "Coste", "Gasto", "Spend",
    "Cost", "Total Spend"
])

# Validaciones mínimas para tu regla simplificada
missing = []
if not col_term: missing.append("Search Term / Término de búsqueda")
if not col_clicks: missing.append("Clicks / Clics")
if not (col_purchases or col_sales): missing.append("Compras/Pedidos u opcionalmente Ventas")

if missing:
    st.error("Faltan columnas necesarias: " + ", ".join(missing))
    st.info("Columnas detectadas: " + " | ".join(df.columns.astype(str).tolist()))
    st.stop()

# Renombrado canónico interno
rename_map = {col_term: "Search Term", col_clicks: "Clicks"}
if col_purchases: rename_map[col_purchases] = "Purchases"
if col_sales: rename_map[col_sales] = "Sales"
if col_cost: rename_map[col_cost] = "Spend"

df = df.rename(columns=rename_map)

# Parse métricas
df["Clicks"] = pd.to_numeric(df["Clicks"], errors="coerce").fillna(0.0)
if "Purchases" in df.columns:
    df["Purchases"] = pd.to_numeric(df["Purchases"], errors="coerce").fillna(0.0)
else:
    # Si no hay compras, inferimos "Purchases" por Sales>0 (defensivo)
    df["Sales"] = to_float_euaware(df["Sales"])
    df["Purchases"] = np.where(df["Sales"] > 0, 1.0, 0.0)

if "Sales" in df.columns:
    df["Sales"] = to_float_euaware(df["Sales"])
else:
    df["Sales"] = 0.0

if "Spend" in df.columns:
    df["Spend"] = to_float_euaware(df["Spend"])
else:
    df["Spend"] = 0.0

# Limpieza Search Term
df["Search Term"] = df["Search Term"].astype(str).str.strip()

# =====================
# Inputs (mínimos)
# =====================
st.divider()
st.subheader("⚙️ Configuración (mínima)")

c1, c2 = st.columns([1, 1])
with c1:
    min_clicks = st.number_input(
        "Mínimo de clics (evidencia)",
        min_value=1, max_value=100000,
        value=int(DEFAULT_MIN_CLICKS),
        step=5
    )

run = st.button("🚀 Detectar desperdicio", use_container_width=True)
if not run:
    st.stop()

# =====================
# Core rule: Clicks >= N AND Purchases = 0
# =====================
mask = (df["Clicks"] >= float(min_clicks)) & (df["Purchases"] <= 0) & (df["Search Term"] != "")
df_out = df[mask].copy()

# Agrupar por término (STR agregado a veces repite filas por campaña/adgroup)
# Para lead magnet: agrupamos y sumamos, así la lista es "única" y fácil de actuar
grp = df_out.groupby("Search Term", dropna=False)[["Clicks", "Purchases", "Sales", "Spend"]].sum().reset_index()

# Añadimos “motivo” + recomendación
grp["Recomendación"] = f"Considerar como negativa ({neg_match})"
grp["Motivo"] = grp["Clicks"].astype(int).astype(str) + f"+ clics y 0 compras"

# Orden por Clicks desc (tu foco CR) y luego por Spend desc
grp = grp.sort_values(["Clicks", "Spend"], ascending=[False, False])

# =====================
# Resumen
# =====================
st.divider()
st.subheader("✅ Resultados")

k1, k2, k3 = st.columns(3)
k1.metric("Términos detectados", f"{len(grp)}")
k2.metric("Clicks (detectados)", f"{int(grp['Clicks'].sum())}")
k3.metric("Spend (detectado)", f"{grp['Spend'].sum():,.2f} €")

show_cols = ["Search Term", "Clicks", "Spend", "Sales", "Motivo", "Recomendación"]
st.dataframe(grp[show_cols].head(300), use_container_width=True)

st.caption(
    "Este listado está pensado para acción rápida: revisa y aplica negativas manualmente en Amazon Ads. "
    "Si quieres aplicación automática con IDs, esa sería la versión PRO."
)

# =====================
# Export informativo (CSV)
# =====================
st.divider()
st.subheader("💾 Export (CSV informativo)")

export_df = grp[show_cols].copy()
export_df["Clicks"] = export_df["Clicks"].astype(int)

csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Descargar SearchTerm_Waste.csv",
    data=csv_bytes,
    file_name=f"SearchTerm_Waste_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime="text/csv",
)

# Extra: copiar términos (para pegar rápido)
terms_text = "\n".join(export_df["Search Term"].astype(str).tolist())
st.text_area("📋 Copiar lista de términos (uno por línea)", value=terms_text, height=200)
