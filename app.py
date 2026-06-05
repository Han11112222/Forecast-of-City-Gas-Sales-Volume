# app.py — 도시가스 공급·판매 예측 (3섹션 분리)
# A) 공급량 예측        : Poly-3 기반 + Normal/Best/Conservative + 기온추세분석
# B) 판매량 예측         : 사용자 지정 기간 기온(예: 전월 6일~당월 5일) + 일별 예측값 합산 시뮬레이션
# C) 공급량 추세분석     : 연도별 총합 OLS/CAGR/Holt/SES + ARIMA/SARIMA(12)
# Fix: ARIMA/SARIMA 공란 방지(월별 실패 시 '연도합'에 직접 ARIMA 폴백)
# Default(추세분석 탭 상품): 개별난방용, 중앙난방용, 취사용
# 업데이트 내역 (B섹션 표 순서 정렬):
#  - 기온 산입기간 로직(전월 6일~당월 5일 기온 -> 당월 실적 매핑) 정상 작동 재확인
#  - B 섹션 '일별 계획' 토글 시 표 순서 재배치: [실적, 예측1, 차이1, 오차율1, 예측2, 차이2, 오차율2]
#  - A, C 섹션 코드 100% 원본 유지

import os
from io import BytesIO
from pathlib import Path
import warnings
from glob import glob
import calendar

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# Plotly (있으면 사용)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# statsmodels (ARIMA/SARIMA)
_HAS_SM = True
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    _HAS_SM = False

# ───────────── 공통 초기설정/스타일 ─────────────
st.set_page_config(page_title="도시가스 공급·판매량 예측", layout="wide")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

st.markdown("""
<style>
.icon-title{display:flex;align-items:center;gap:.55rem;margin:.2rem 0 .6rem 0}
.icon-title .emoji{font-size:1.55rem;line-height:1}
.small-icon .emoji{font-size:1.2rem}
table.centered-table {width:100%; table-layout: fixed;}
table.centered-table th, table.centered-table td { text-align:center !important; }
</style>
""", unsafe_allow_html=True)

def title_with_icon(icon: str, text: str, level: str = "h1", small=False):
    klass = "icon-title small-icon" if small else "icon-title"
    st.markdown(
        f"<{level} class='{klass}'><span class='emoji'>{icon}</span><span>{text}</span></{level}>",
        unsafe_allow_html=True,
    )

# ───────────── 한글 폰트 ─────────────
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "data" / "fonts" / "NanumGothic.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
    ]
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.family"] = [fam]
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ───────────── 공통 상수/유틸 ─────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp", "최저", "최고", "예상기온", "추세기온"]
KNOWN_PRODUCT_ORDER = [
    "개별난방용", "중앙난방용",
    "자가열전용", "일반용(2)", "업무난방용", "냉난방용",
    "주한미군", "취사용", "총공급량", "공급량(MJ)", "공급량(M3)", "공급량"
]

GS_SALES_URL = "https://docs.google.com/spreadsheets/d/1-8RIPIkjnVXxoh5QJs6598nnHkWOGmrO655jr3b3g04/edit?gid=0#gid=0"
GS_SUPPLY_URL = "https://docs.google.com/spreadsheets/d/1vS-a9XrbjjIznHxntuFIM6hmml6qTlR2Cayw77p_Rao/edit?gid=0#gid=0"
GS_TEMP_URL = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/edit?gid=0#gid=0"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    date_col = next((c for c in df.columns if c.lower() in ["날짜", "일자", "date", "기준일", "기간"]), None)
    
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
            y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")
    if "연" not in df.columns:
        if "년" in df.columns: df["연"] = df["년"]
        elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month

    for c in df.columns:
        if c not in ["날짜", date_col]:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="coerce"
            )
            
    if "연" in df.columns: df["연"] = pd.to_numeric(df["연"], errors="coerce").fillna(0).astype(int)
    if "월" in df.columns: df["월"] = pd.to_numeric(df["월"], errors="coerce").fillna(1).astype(int)
    return df

@st.cache_data(ttl=600)
def read_google_sheet(url: str, rollup=True) -> pd.DataFrame:
    try:
        if "/edit" in url:
            base_url = url.split("/edit")[0]
            gid = url.split("gid=")[1].split("&")[0].split("#")[0] if "gid=" in url else "0"
            export_url = f"{base_url}/export?format=csv&gid={gid}"
        else:
            export_url = url
            
        df = pd.read_csv(export_url)
        
        if not df.empty:
            first_col_val = str(df.columns[0]).strip().replace('-', '').replace('.', '').replace('/', '')
            if first_col_val.isdigit() and len(first_col_val) >= 4:
                df = pd.read_csv(export_url, header=None)
                new_cols = []
                for i in range(len(df.columns)):
                    if i == 0: new_cols.append("날짜")
                    elif i == 1: new_cols.append("연")
                    elif i == 2: new_cols.append("월")
                    elif i == 3: new_cols.append("평균기온")
                    elif (i - 4) < len(KNOWN_PRODUCT_ORDER): new_cols.append(KNOWN_PRODUCT_ORDER[i - 4])
                    else: new_cols.append(f"임시데이터_{i}")
                df.columns = new_cols
                
        df = normalize_cols(df)

        if rollup and not df.empty and "연" in df.columns and "월" in df.columns:
            if len(df) > len(df[['연', '월']].drop_duplicates()): 
                agg_dict = {}
                for c in df.columns:
                    if c in ["연", "월", "날짜", "일자", "date"]: continue
                    if any(h in c.lower() for h in TEMP_HINTS):
                        agg_dict[c] = 'mean'
                    elif pd.api.types.is_numeric_dtype(df[c]):
                        agg_dict[c] = 'sum'
                    else:
                        agg_dict[c] = 'first'
                
                df_monthly = df.groupby(['연', '월']).agg(agg_dict).reset_index()
                df_monthly['날짜'] = pd.to_datetime(df_monthly['연'].astype(str) + '-' + df_monthly['월'].astype(str) + '-01', errors='coerce')
                df = df_monthly
                
        return df
    except Exception as e:
        st.error(f"구글 시트 연동 오류: {e}")
        return pd.DataFrame()

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        nm = str(c).lower()
        if any(h in nm for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in META_COLS]
    candidates = [c for c in candidates if not any(h in c.lower() for h in TEMP_HINTS)]
    
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
    return ordered + others

@st.cache_data(ttl=600)
def read_excel_sheet(path_or_file, prefer_sheet="데이터"):
    try:
        xls = pd.ExcelFile(path_or_file, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(path_or_file, engine="openpyxl")
    return normalize_cols(df)

@st.cache_data(ttl=600)
def read_temperature_raw(file):
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        date_col = None
        for c in df.columns:
            if str(c).lower() in ["날짜", "일자", "date"]:
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c
                    break
                except Exception:
                    pass
        temp_col = None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp", "temperature"]):
                temp_col = c
                break
        if date_col is None or temp_col is None:
            return None
        out = pd.DataFrame(
            {"일자": pd.to_datetime(df[date_col], errors="coerce"), "기온": pd.to_numeric(df[temp_col], errors="coerce")}
        ).dropna()
        return out.sort_values("일자").reset_index(drop=True)

    name = getattr(file, "name", str(file))
    if name and name.lower().endswith(".csv"):
        return _finalize(pd.read_csv(file))
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = xls.sheet_names[0]
    head = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
    header_row = None
    for i in range(len(head)):
        row = [str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜", "일자", "date", "Date"] for v in row) and any(
            ("평균기온" in v) or ("기온" in v) or (isinstance(v, str) and v.lower() in ["temp", "temperature"])
            for v in row
        ):
            header_row = i
            break
    df = (
        pd.read_excel(xls, sheet_name=sheet)
        if header_row is None
        else pd.read_excel(xls, sheet_name=sheet, header=header_row)
    )
    return _finalize(df)

@st.cache_data(ttl=600)
def read_temperature_forecast(file):
    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
        sheet = "기온예측" if "기온예측" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(file, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c in ["날짜", "일자", "date", "Date"]), df.columns[0])
    base_temp_col = next(
        (c for c in df.columns if ("평균기온" in c) or (str(c).lower() in ["temp", "temperature", "기온"])), None
    )
    trend_cols = [c for c in df.columns if any(k in str(c) for k in ["추세분석", "추세기온"])]
    trend_col = trend_cols[0] if trend_cols else None
    if base_temp_col is None:
        raise ValueError("기온예측 파일에서 '평균기온' 또는 '기온' 열을 찾지 못했습니다.")
    d = pd.DataFrame(
        {"날짜": pd.to_datetime(df[date_col], errors="coerce"), "예상기온": pd.to_numeric(df[base_temp_col], errors="coerce")}
    ).dropna(subset=["날짜"])
    d["연"] = d["날짜"].dt.year.astype(int)
    d["월"] = d["날짜"].dt.month.astype(int)
    d["추세기온"] = pd.to_numeric(df[trend_col], errors="coerce") if trend_col else np.nan
    return d[["연", "월", "예상기온", "추세기온"]]

def month_start(x):
    x = pd.to_datetime(x)
    return pd.Timestamp(x.year, x.month, 1)

def month_range_inclusive(s, e):
    return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

# ───────────── Poly-3/4 모델 예측 (0 Sample 방어) ─────────────
def fit_poly3_and_predict(x_train, y_train, x_future):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_future = np.asarray(x_future, dtype=float)
    
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train_clean = x_train[m]
    y_train_clean = y_train[m]
    
    if len(x_train_clean) < 2:
        return np.zeros_like(x_future), 0.0, None, None

    if np.isnan(x_future).any():
        x_future = np.nan_to_num(x_future, nan=np.nanmean(x_train_clean) if len(x_train_clean) > 0 else 0)
        
    x_train_clean = x_train_clean.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train_clean)
    model = LinearRegression().fit(Xtr, y_train_clean)
    r2 = model.score(Xtr, y_train_clean)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

def fit_poly4_and_predict(x_train, y_train, x_future):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_future = np.asarray(x_future, dtype=float)
    
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train_clean = x_train[m]
    y_train_clean = y_train[m]
    
    if len(x_train_clean) < 2:
        return np.zeros_like(x_future), 0.0, None, None

    if np.isnan(x_future).any():
        x_future = np.nan_to_num(x_future, nan=np.nanmean(x_train_clean) if len(x_train_clean) > 0 else 0)
        
    x_train_clean = x_train_clean.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=4, include_bias=False)
    Xtr = poly.fit_transform(x_train_clean)
    model = LinearRegression().fit(Xtr, y_train_clean)
    r2 = model.score(Xtr, y_train_clean)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

def poly_eq_text(model, decimals: int = 4):
    if model is None: return "데이터 부족"
    c = model.coef_
    c1 = c[0] if len(c) > 0 else 0.0
    c2 = c[1] if len(c) > 1 else 0.0
    c3 = c[2] if len(c) > 2 else 0.0
    d = model.intercept_
    fmt = lambda v: f"{v:+,.{decimals}f}"
    return f"y = {fmt(c3)}x³ {fmt(c2)}x² {fmt(c1)}x {fmt(d)}"

def poly_eq_text4(model):
    if model is None: return "데이터 부족"
    c = model.coef_
    c1 = c[0] if len(c) > 0 else 0.0
    c2 = c[1] if len(c) > 1 else 0.0
    c3 = c[2] if len(c) > 2 else 0.0
    c4 = c[3] if len(c) > 3 else 0.0
    d = model.intercept_
    return f"y = {c4:+.5e}x⁴ {c3:+.5e}x³ {c2:+.5e}x² {c1:+.5e}x {d:+.5e}"

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    float1_cols = float1_cols or []
    int_cols = int_cols or []
    show = df.copy()
    for c in float1_cols:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    for c in int_cols:
        if c in show.columns:
            show[c] = (
                pd.to_numeric(show[c], errors="coerce")
                .round()
                .astype("Int64")
                .map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
            )
    st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

def _r2_for_range(df: pd.DataFrame, prod: str, temp_col: str, start_year: int, end_year: int | None = None):
    if end_year is None:
        end_year = int(df["연"].max())
    sub = df[(df["연"] >= int(start_year)) & (df["연"] <= int(end_year))][[temp_col, prod]].dropna()
    if len(sub) < 12:
        return np.nan
    x = sub[temp_col].astype(float).to_numpy()
    y = sub[prod].astype(float).to_numpy()
    _, r2, _, _ = fit_poly3_and_predict(x, y, x)
    return float(r2)

def recommend_train_ranges(df: pd.DataFrame, prod: str, temp_col: str,
                           min_year: int | None = None, end_year: int | None = None) -> pd.DataFrame:
    if min_year is None:
        min_year = int(df["연"].min())
    if end_year is None:
        end_year = int(df["연"].max())
    rows = []
    for sy in range(int(min_year), int(end_year)):
        r2 = _r2_for_range(df, prod, temp_col, sy, end_year)
        rows.append({"시작연도": sy, "종료연도": int(end_year), "기간": f"{sy}~현재", "R2": r2})
    out = pd.DataFrame(rows)
    out["__rank"] = out["R2"].fillna(-1.0)
    return out.sort_values("__rank", ascending=False).drop(columns="__rank").reset_index(drop=True)

# ===========================================================
# A) 공급량 예측 (원본 100% 유지)
# ===========================================================

def render_supply_forecast():
    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Google Sheets 연동", "파일 업로드"], index=0)
        df, forecast_df = None, None

        if src == "Google Sheets 연동":
            supply_url = st.text_input("🔗 실적 데이터 URL (구글 시트)", value=GS_SUPPLY_URL)
            temp_url = st.text_input("🌡️ 기온 데이터 URL (구글 시트)", value=GS_TEMP_URL)
            
            if supply_url:
                with st.spinner("데이터를 가져오는 중..."):
                    df = read_google_sheet(supply_url)
                    raw_temp_df = read_google_sheet(temp_url) if temp_url else df.copy()
                    
                if df is not None and not df.empty and raw_temp_df is not None and not raw_temp_df.empty:
                    temp_col_df = detect_temp_col(df)
                    temp_col_raw = detect_temp_col(raw_temp_df)
                    
                    if temp_col_df is None and temp_col_raw is not None and temp_url:
                        monthly_temp = raw_temp_df.groupby(['연', '월'])[temp_col_raw].mean().reset_index()
                        df = df.merge(monthly_temp, on=['연', '월'], how='left')
                        temp_col_df = temp_col_raw 
                    
                    if temp_col_raw is not None:
                        forecast_df = raw_temp_df.groupby(['연', '월'])[temp_col_raw].mean().reset_index()
                        forecast_df.rename(columns={temp_col_raw: "예상기온"}, inplace=True)
                        trend_cols = [c for c in raw_temp_df.columns if any(k in str(c) for k in ["추세분석", "추세기온"])]
                        if trend_cols:
                            trend_monthly = raw_temp_df.groupby(['연', '월'])[trend_cols[0]].mean().reset_index()
                            forecast_df = forecast_df.merge(trend_monthly, on=['연', '월'], how='left')
                            forecast_df.rename(columns={trend_cols[0]: "추세기온"}, inplace=True)
                        else:
                            forecast_df["추세기온"] = forecast_df["예상기온"]
        else:
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")
            up_fc = st.file_uploader("🌡️ 예상기온 엑셀 업로드(xlsx) — (날짜, 평균기온[, 추세분석])", type=["xlsx"])
            if up_fc is not None:
                forecast_df = read_temperature_forecast(up_fc)

        if df is None or len(df) == 0:
            st.info("🧩 좌측에서 실적 엑셀을 선택/업로드하세요."); st.stop()
            
        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("🌡️ 기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()
            
        if forecast_df is None or forecast_df.empty:
            fallback_temp = df.groupby("월")[temp_col].mean().reset_index().rename(columns={temp_col: "예상기온"})
            fallback_temp["추세기온"] = fallback_temp["예상기온"]
            expanded_rows = []
            for y_ext in range(2026, 2036):
                f_block = fallback_temp.copy()
                f_block["연"] = y_ext
                expanded_rows.append(f_block)
            forecast_df = pd.concat(expanded_rows, ignore_index=True)[["연", "월", "예상기온", "추세기온"]]

        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique() if y > 0])
        years_sel = st.multiselect("🗓️ 연도 선택", years_all, default=years_all)

        title_with_icon("🧰", "예측할 상품 선택", "h3", small=True)
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("📦 상품(용도) 선택", product_cols, default=default_products)

        st.session_state["supply_meta"] = {
            "df": df.dropna(subset=["연","월"]).copy(),
            "temp_col": temp_col,
            "product_cols": product_cols,
            "latest_year": int(df["연"].max()) if not df.empty else 2026,
            "min_year": int(df["연"].min()) if not df.empty else 2017,
        }

        title_with_icon("⚙️", "예측 설정", "h3", small=True)
        last_year = int(df["연"].max()) if not df.empty else 2026
        years = list(range(2010, 2036))
        col_sy, col_sm = st.columns(2)
        with col_sy:
            start_y = st.selectbox("🚀 예측 시작(연)", years, index=years.index(last_year) if last_year in years else len(years)-1)
        with col_sm:
            start_m = st.selectbox("📅 예측 시작(월)", list(range(1, 13)), index=0)
        col_ey, col_em = st.columns(2)
        with col_ey:
            end_y = st.selectbox("🏁 예측 종료(연)", years, index=years.index(last_year) if last_year in years else len(years)-1)
        with col_em:
            end_m = st.selectbox("📅 예측 종료(월)", list(range(1, 13)), index=11)

        run_btn = st.button("🧮 예측 시작", type="primary")

    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()
        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start:
            st.error("⛔ 예측 종료가 시작보다 빠릅니다."); st.stop()
        fut_idx = month_range_inclusive(f_start, f_end)
        fut_base = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})

        fut_base = fut_base.merge(forecast_df, on=["연", "월"], how="left")

        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("월평균").reset_index()
        miss1 = fut_base["예상기온"].isna()
        if miss1.any():
            fut_base = fut_base.merge(monthly_avg_temp, on="월", how="left")
            fut_base.loc[miss1, "예상기온"] = fut_base.loc[miss1, "월평균"]
        miss2 = fut_base["추세기온"].isna()
        if miss2.any():
            fut_base.loc[miss2, "추세기온"] = fut_base.loc[miss2, "예상기온"]
        fut_base.drop(columns=[c for c in ["월평균"] if c in fut_base.columns], inplace=True)

        x_train_base = train_df[temp_col].astype(float).values

        st.session_state["supply_materials"] = dict(
            base_df=base, train_df=train_df, prods=prods, x_train=x_train_base,
            fut_base=fut_base, start_ts=f_start, end_ts=f_end, temp_col=temp_col,
            default_pred_years=list(range(int(start_y), int(end_y) + 1)),
            years_sel=years_sel
        )
        st.success("✅ 공급량 예측(베이스) 준비 완료! 아래에서 **시나리오 Δ°C**를 조절하세요.")

    if "supply_materials" not in st.session_state:
        st.info("👈 좌측에서 설정 후 **예측 시작**을 눌러 실행하세요."); st.stop()

    mats = st.session_state["supply_materials"]
    base, train_df, prods = mats["base_df"], mats["train_df"], mats["prods"]
    x_train, fut_base = mats["x_train"], mats["fut_base"]
    temp_col = mats["temp_col"]; years_sel = mats["years_sel"]
    months = list(range(1, 13))

    title_with_icon("🌡️", "시나리오 Δ°C (평균기온 보정)", "h3", small=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        d_norm = st.number_input("Normal Δ°C", value=0.0, step=0.1, format="%.1f", key="s_norm")
    with c2:
        d_best = st.number_input("Best Δ°C", value=-1.0, step=0.1, format="%.1f", key="s_best")
    with c3:
        d_cons = st.number_input("Conservative Δ°C", value=1.0, step=0.1, format="%.1f", key="s_cons")

    def _forecast_table(delta: float) -> pd.DataFrame:
        x_future = (fut_base["예상기온"] + float(delta)).astype(float).values
        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연", "월"]].copy()
            tmp["월평균기온"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        if not pred_rows: return pd.DataFrame()
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연", "월", "월평균기온"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연", "월", "월평균기온"] + ordered)]
        pivot = pivot[["연", "월", "월평균기온"] + ordered + others]
        return pivot.sort_values(["연", "월"]).reset_index(drop=True)

    def _forecast_table_trend() -> pd.DataFrame:
        x_future = fut_base["추세기온"].astype(float).values
        if np.isnan(x_future).any():
            back = train_df.groupby("월")[temp_col].mean().reindex(fut_base["월"]).values
            x_future = np.where(np.isnan(x_future), back, x_future)
        pred_rows = []
        for col in prods:
            y_train = train_df[col].astype(float).values
            y_future, _, _, _ = fit_poly3_and_predict(x_train, y_train, x_future)
            tmp = fut_base[["연", "월"]].copy()
            tmp["월평균기온(추세)"] = x_future
            tmp["상품"] = col
            tmp["예측"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred_rows.append(tmp)
        if not pred_rows: return pd.DataFrame()
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연", "월", "월평균기온(추세)"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연", "월", "월평균기온(추세)"] + ordered)]
        pivot = pivot[["연", "월", "월평균기온(추세)"] + ordered + others]
        return pivot.sort_values(["연", "월"]).reset_index(drop=True)

    def _render_with_year_sums(title, table, temp_col_name):
        if table.empty: return pd.DataFrame(), pd.DataFrame()
        title_with_icon("🗂️", title, "h3", small=True)
        render_centered_table(
            table,
            float1_cols=[temp_col_name],
            int_cols=[c for c in table.columns if c not in ["연", "월", temp_col_name]],
            index=False,
        )
        year_sum = table.groupby("연").sum(numeric_only=True).reset_index()
        year_sum_show = year_sum.drop(columns=[c for c in ["월", temp_col_name] if c in year_sum.columns])
        year_sum_show.insert(1, "기간", "1~12월")
        cols_int = [c for c in year_sum_show.columns if c not in ["연", "기간"]]
        title_with_icon("🗓️", "연도별 총계", "h4", small=True)
        render_centered_table(year_sum_show, int_cols=cols_int, index=False)

        tmp = table.copy()
        tmp["__half"] = np.where(tmp["월"].astype(int) <= 6, "1~6월", "7~12월")
        half = tmp.groupby(["연", "__half"]).sum(numeric_only=True).reset_index().rename(columns={"__half": "반기"})
        half_to_show = half.rename(columns={"반기": "기간"}).drop(columns=[c for c in ["월", temp_col_name] if c in half.columns])
        title_with_icon("🧮", "반기별 총계 (1~6월, 7~12월)", "h4", small=True)
        render_centered_table(
            half_to_show,
            int_cols=[c for c in half_to_show.columns if c not in ["연", "기간"]],
            index=False,
        )
        return year_sum_show, half_to_show

    tbl_n = _forecast_table(d_norm)
    tbl_b = _forecast_table(d_best)
    tbl_c = _forecast_table(d_cons)
    tbl_trd = _forecast_table_trend()

    sum_n, half_n = _render_with_year_sums("🎯 Normal", tbl_n, "월평균기온")
    sum_b, half_b = _render_with_year_sums("💎 Best", tbl_b, "월평균기온")
    sum_c, half_c = _render_with_year_sums("🛡️ Conservative", tbl_c, "월평균기온")
    sum_t, half_t = _render_with_year_sums("📈 기온추세분석", tbl_trd, "월평균기온(추세)")

    def _pack_for_download(df_list, names, temp_names):
        outs = []
        for df, nm, tnm in zip(df_list, names, temp_names):
            d = df.copy()
            d.insert(0, "시나리오", nm)
            if tnm in d.columns and tnm != "월평균기온":
                d.rename(columns={tnm: "월평균기온"}, inplace=True)
            outs.append(d)
        return pd.concat(outs, ignore_index=True)

    to_dl = _pack_for_download(
        [tbl_n, tbl_b, tbl_c, tbl_trd],
        ["Normal", "Best", "Conservative", "기온추세분석"],
        ["월평균기온", "월평균기온", "월평균기온", "월평균기온(추세)"],
    )

    learn_years = sorted([int(y) for y in mats["years_sel"]])
    meta_learn  = f"{min(learn_years)}~{max(learn_years)}년" if learn_years else "-"
    all_years = sorted([int(y) for y in base["연"].unique()])
    if learn_years:
        span = list(range(min(learn_years), max(learn_years) + 1))
        exclude_years = [y for y in span if (y in all_years and y not in learn_years)]
    else:
        exclude_years = []
    meta_excl = ", ".join(str(y) for y in exclude_years) if exclude_years else "-"

    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            startrow = 2
            to_dl.to_excel(writer, index=False, sheet_name="Forecast", startrow=startrow)
            ws = writer.sheets["Forecast"]
            ws.cell(row=1, column=1, value="학습기간"); ws.cell(row=1, column=2, value=meta_learn)
            ws.cell(row=1, column=3, value="제외기간"); ws.cell(row=1, column=4, value=meta_excl)

            def write_yearsum(sheet_name, year_df, half_df):
                ysr = 2
                year_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=ysr)
                ws2 = writer.sheets[sheet_name]
                ws2.cell(row=1, column=1, value="학습기간"); ws2.cell(row=1, column=2, value=meta_learn)
                ws2.cell(row=1, column=3, value="제외기간"); ws2.cell(row=1, column=4, value=meta_excl)
                start_half = ysr + len(year_df) + 3
                half_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_half)

            write_yearsum("YearSum_Normal",    sum_n, half_n)
            write_yearsum("YearSum_Best",      sum_b, half_b)
            write_yearsum("YearSum_Cons",      sum_c, half_c)
            write_yearsum("YearSum_TrendTemp", sum_t, half_t)

        buf.seek(0)
        st.download_button(
            "⬇️ 예측 결과 XLSX 다운로드 (연합/반기 포함 · 학습·제외기간 표기)",
            data=buf.read(),
            file_name="citygas_supply_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.download_button(
            "⬇️ 예측 결과 CSV 다운로드 (Forecast만)",
            data=to_dl.to_csv(index=False).encode("utf-8-sig"),
            file_name="citygas_supply_forecast.csv",
            mime="text/csv",
        )

    title_with_icon("📈", "그래프(실적 + 예측 + 기온추세분석)", "h3", small=True)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        show_best = st.toggle("Best 표시", value=False, key="show_best_top")
        show_cons = st.toggle("Conservative 표시", value=False, key="show_cons_top")

    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique() if v > 0])
    default_years = years_all_for_plot[-2:] if len(years_all_for_plot) >= 2 else years_all_for_plot
    c_y1, c_y2, c_y3 = st.columns(3)
    with c_y1:
        years_view = st.multiselect("👀 실적연도", options=years_all_for_plot, default=default_years, key="supply_years_view")
    pred_default = mats.get("default_pred_years", [])
    with c_y2:
        years_pred = st.multiselect(
            "📈 예측연도",
            options=sorted(list(set(fut_base["연"].tolist()))),
            default=[y for y in pred_default if y in fut_base["연"].unique()],
            key="years_pred",
        )
    with c_y3:
        years_trnd = st.multiselect(
            "📊 기온추세분석연도",
            options=sorted(list(set(fut_base["연"].tolist()))),
            default=[y for y in pred_default if y in fut_base["연"].unique()],
            key="years_trnd",
        )

    months_txt = [f"{m}월" for m in months]
    def _pred_series(delta): return (fut_base["예상기온"] + float(delta)).astype(float).values
    x_future_norm = _pred_series(d_norm)
    x_future_best = _pred_series(d_best)
    x_future_cons = _pred_series(d_cons)
    x_future_trend = fut_base["추세기온"].astype(float).values
    if np.isnan(x_future_trend).any():
        back = train_df.groupby("월")[temp_col].mean().reindex(fut_base["월"]).values
        x_future_trend = np.where(np.isnan(x_future_trend), back, x_future_trend)

    fut_with_t = fut_base.copy()
    fut_with_t["T_norm"] = x_future_norm
    fut_with_t["T_best"] = x_future_best
    fut_with_t["T_cons"] = x_future_cons
    fut_with_t["T_trend"] = x_future_trend

    actual_temp = (
        base.groupby(["연", "월"])[temp_col].mean().reset_index().rename(columns={temp_col: "T_actual"})
    )

    for prod in prods:
        y_train_prod = train_df[prod].astype(float).values
        y_norm, r2_train, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_norm)
        P_norm = fut_with_t[["연", "월", "T_norm"]].copy(); P_norm["pred"] = np.clip(np.rint(y_norm).astype(np.int64), 0, None)
        y_best, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_best)
        P_best = fut_with_t[["연", "월", "T_best"]].copy(); P_best["pred"] = np.clip(np.rint(y_best).astype(np.int64), 0, None)
        y_cons, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_cons)
        P_cons = fut_with_t[["연", "월", "T_cons"]].copy(); P_cons["pred"] = np.clip(np.rint(y_cons).astype(np.int64), 0, None)
        y_trd, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_future_trend)
        P_trend = fut_with_t[["연", "월", "T_trend"]].copy(); P_trend["pred"] = np.clip(np.rint(y_trd).astype(np.int64), 0, None)

        if go is None:
            fig = plt.figure(figsize=(9, 3.6)); ax = plt.gca()
            for y in sorted([int(v) for v in years_view]):
                s = base.loc[base["연"] == y, ["월", prod]].set_index("월")[prod].reindex(months)
                ax.plot(months, s.values, label=f"{y} 실적")
            for y in years_pred:
                pv = P_norm[P_norm["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                ax.plot(months, pv, "--", label=f"예측(Normal) {y}")
                if show_best:
                    pv = P_best[P_best["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                    ax.plot(months, pv, "--", label=f"예측(Best) {y}")
                if show_cons:
                    pv = P_cons[P_cons["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                    ax.plot(months, pv, "--", label=f"예측(Conservative) {y}")
            for y in years_trnd:
                pv = P_trend[P_trend["연"] == int(y)].sort_values("월")["pred"].reindex(range(1, 13)).values
                ax.plot(months, pv, ":", label=f"기온추세분석 {y}")
            ax.set_xlim(1, 12); ax.set_xticks(months); ax.set_xticklabels(months_txt)
            ax.set_xlabel("월"); ax.set_ylabel("공급량 (GJ)")
            ax.set_title(f"{prod} — Poly-3 (Train R²={r2_train:.3f})")
            ax.legend(loc="best"); st.pyplot(fig, clear_figure=True)
        else:
            fig = go.Figure()
            for y in sorted([int(v) for v in years_view]):
                one = base[base["연"] == y][["월", prod]].dropna().sort_values("월")
                t_one = actual_temp[actual_temp["연"] == y].sort_values("월")
                one = one.merge(t_one[["월", "T_actual"]], on="월", how="left")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in one["월"]],
                    y=one[prod],
                    customdata=np.round(one["T_actual"].values.astype(float), 2),
                    mode="lines+markers",
                    name=f"{y} 실적",
                    hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                ))
            for y in years_pred:
                p_idx = fut_base["연"] == int(y)
                if not p_idx.any(): continue
                row = P_norm[p_idx].sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in row["월"]],
                    y=row["pred"],
                    customdata=np.round(row["T_norm"].values.astype(float), 2),
                    mode="lines",
                    name=f"예측(Normal) {y}",
                    line=dict(dash="dash"),
                    hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                ))
                if show_best:
                    rb = P_best[p_idx].sort_values("월")
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in rb["월"]],
                        y=rb["pred"],
                        customdata=np.round(rb["T_best"].values.astype(float), 2),
                        mode="lines",
                        name=f"예측(Best) {y}",
                        line=dict(dash="dash"),
                        hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                    ))
                if show_cons:
                    rc = P_cons[p_idx].sort_values("월")
                    fig.add_trace(go.Scatter(
                        x=[f"{int(m)}월" for m in rc["월"]],
                        y=rc["pred"],
                        customdata=np.round(rc["T_cons"].values.astype(float), 2),
                        mode="lines",
                        name=f"예측(Conservative) {y}",
                        line=dict(dash="dash"),
                        hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                    ))
            for y in years_trnd:
                p_idx = fut_base["연"] == int(y)
                if not p_idx.any(): continue
                row = P_trend[p_idx].sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in row["월"]],
                    y=row["pred"],
                    customdata=np.round(row["T_trend"].values.astype(float), 2),
                    mode="lines",
                    name=f"기온추세분석 {y}",
                    line=dict(dash="dot"),
                    hovertemplate="%{x} %{y:,}<br>월평균기온 %{customdata:.2f}℃"
                ))
            fig.update_layout(
                title=f"{prod} — Poly-3 (Train R²={r2_train:.3f})",
                xaxis=dict(title="월"),
                yaxis=dict(title="공급량 (GJ)", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
                margin=dict(t=60, b=120, l=40, r=20),
                dragmode="pan",
            )
            st.plotly_chart(fig, use_container_width=True, config=dict(scrollZoom=True, displaylogo=False))

        title_with_icon("📑", f"{prod} — 월별 표 (선택 연도)", "h3", small=True)
        months_idx = list(range(1, 13))
        table = pd.DataFrame({"월": months_idx})
        for y in sorted([int(v) for v in years_view]):
            s = base.loc[base["연"] == y, ["월", prod]].set_index("월")[prod].astype(float)
            table[f"{y} 실적"] = s.reindex(months_idx).values
        for y in years_pred:
            s = P_norm[P_norm["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
            table[f"예측(Normal) {y}"] = s.reindex(months_idx).values
        if show_best:
            for y in years_pred:
                s = P_best[P_best["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
                table[f"예측(Best) {y}"] = s.reindex(months_idx).values
        if show_cons:
            for y in years_pred:
                s = P_cons[P_cons["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
                table[f"예측(Conservative) {y}"] = s.reindex(months_idx).values
        for y in years_trnd:
            s = P_trend[P_trend["연"] == int(y)][["월", "pred"]].set_index("월")["pred"]
            table[f"기온추세 {y}"] = s.reindex(months_idx).values

        sum_row = {"월": "합계"}
        for c in [col for col in table.columns if col != "월"]:
            sum_row[c] = pd.to_numeric(table[c], errors="coerce").sum()
        table_show = pd.concat([table, pd.DataFrame([sum_row])], ignore_index=True)
        render_centered_table(table_show, int_cols=[c for c in table_show.columns if c != "월"], index=False)

        title_with_icon("🔎", f"{prod} — 기온·공급량 상관(Train, R²={r2_train:.3f})", "h3", small=True)
        figc, axc = plt.subplots(figsize=(10, 5.2))
        x_tr = np.asarray(train_df[temp_col].astype(float).values)
        y_tr = np.asarray(y_train_prod, dtype=float)
        m_tr = (~np.isnan(x_tr)) & (~np.isnan(y_tr))
        
        if len(x_tr[m_tr]) > 2:
            axc.scatter(x_tr[m_tr], y_tr[m_tr], alpha=0.65, label="학습 샘플")
            xx = np.linspace(np.nanmin(x_tr[m_tr]) - 1, np.nanmax(x_tr[m_tr]) + 1, 200)
            yhat, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, xx)
            axc.plot(xx, yhat, lw=2.8, color="#1f77b4", label="Poly-3")
            pred_train, _, _, _ = fit_poly3_and_predict(x_train, y_train_prod, x_train)
            resid = y_train_prod - pred_train; s = np.nanstd(resid)
            axc.fill_between(xx, yhat - 1.96 * s, yhat + 1.96 * s, color="#ff7f0e", alpha=0.25, label="95% 신뢰구간")
            axc.set_xlabel("기온 (℃)"); axc.set_ylabel("공급량 (GJ)")
            axc.grid(alpha=0.25); axc.legend(loc="best")
            axc.text(0.02, 0.04, f"Poly-3: {poly_eq_text(model_s)}", transform=axc.transAxes,
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
        else:
            axc.text(0.5, 0.5, "과거 데이터 부족으로 산점도를 생성할 수 없습니다.", ha="center", va="center", fontsize=12)
            axc.set_axis_off()
        st.pyplot(figc)

    st.caption("ℹ️ 95% 신뢰구간: 잔차 표준편차 기준 근사 예측구간(신규 관측 약 95% 포함).")

# ===========================================================
# B) 판매량 예측 (맞춤 기온 산출 + 일일 기온 합산 시뮬레이션 및 표 스태킹)
# ===========================================================

def render_cooling_sales_forecast():
    title_with_icon("🧊", "판매량 예측", "h2")
    
    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Google Sheets 연동", "파일 업로드"], index=0, key="cool_src")
        sales_df, raw_temp_df = None, None
        
        if src == "Google Sheets 연동":
            sales_url = st.text_input("🔗 판매량 데이터 URL (구글 시트)", value=GS_SALES_URL, key="s_url_b")
            temp_url = st.text_input("🌡️ 기온 데이터 URL (구글 시트)", value=GS_TEMP_URL, key="t_url_b")
            
            if sales_url and temp_url:
                with st.spinner("구글 스프레드시트 데이터 연동 중..."):
                    sales_df = read_google_sheet(sales_url)
                    
                    export_url = temp_url.split("/edit")[0] + "/export?format=csv&gid=" + (temp_url.split("gid=")[1].split("&")[0] if "gid=" in temp_url else "0")
                    raw_t = pd.read_csv(export_url)
                    if not raw_t.empty:
                        first_col_val = str(raw_t.columns[0]).strip().replace('-', '').replace('.', '').replace('/', '')
                        if first_col_val.isdigit() and len(first_col_val) >= 4:
                            raw_t = pd.read_csv(export_url, header=None)
                            new_cols = ["날짜", "연", "월", "평균기온"] + [f"기타_{i}" for i in range(4, len(raw_t.columns))]
                            raw_t.columns = new_cols[:len(raw_t.columns)]
                            
                        date_c = next((c for c in raw_t.columns if c.lower() in ["날짜", "일자", "date", "기준일"]), None)
                        if date_c: raw_t['날짜'] = pd.to_datetime(raw_t[date_c], errors='coerce')
                        temp_c = next((c for c in raw_t.columns if "평균기온" in c or "기온" in c), None)
                        if temp_c: raw_t['평균기온'] = pd.to_numeric(raw_t[temp_c].astype(str).str.replace(",",""), errors='coerce')
                        raw_temp_df = raw_t.dropna(subset=['날짜', '평균기온'])[['날짜', '평균기온']]
        else:
            up_s = st.file_uploader("📄 판매 실적(xlsx)", type=["xlsx"], key="cool_up")
            if up_s is not None: sales_df = read_excel_sheet(up_s)
            up_t = st.file_uploader("🌡️ 기온 RAW(일별)", type=["xlsx", "csv"], key="cool_up_t")
            if up_t is not None: raw_temp_df = read_temperature_raw(up_t)

    if sales_df is None or sales_df.empty:
        st.info("🧩 사이드바에 유효한 판매량 데이터를 입력해 주세요."); st.stop()
    if raw_temp_df is None or raw_temp_df.empty:
        st.info("🌡️ 기온 RAW(일별) 데이터를 불러오지 못했습니다."); st.stop()

    product_cols = guess_product_cols(sales_df)
    sel_prod = st.selectbox("📦 예측 대상 상품 선택 (기온추정 상품)", product_cols)

    st.markdown("### 🌡️ 맞춤형 평균기온 산출 기간 설정")
    c1, c2 = st.columns(2)
    with c1:
        s_off_str = st.selectbox("시작 월", ["전월", "당월"], index=0)
        s_off = -1 if s_off_str == "전월" else 0
        s_day = st.number_input("시작 일", min_value=1, max_value=31, value=6)
    with c2:
        e_off_str = st.selectbox("종료 월", ["전월", "당월"], index=1)
        e_off = -1 if e_off_str == "전월" else 0
        e_day = st.number_input("종료 일", min_value=1, max_value=31, value=5)

    toggle_daily_plan = st.toggle("📊 일별 계획 활성화 (월평균 대입 방식 vs 일일 기온 대입 합산 방식 비교)", value=False)

    # 지정 기간의 일일 기온 배열(Array)을 반환하는 함수 (당월 기준 역산)
    def get_custom_temp_array(y, m, s_off, s_day, e_off, e_day):
        sm = m + s_off; sy = y
        if sm < 1: sm += 12; sy -= 1
        elif sm > 12: sm -= 12; sy += 1
        
        em = m + e_off; ey = y
        if em < 1: em += 12; ey -= 1
        elif em > 12: em -= 12; ey += 1
        
        try:
            s_d = min(s_day, calendar.monthrange(sy, sm)[1])
            e_d = min(e_day, calendar.monthrange(ey, em)[1])
            sd = pd.Timestamp(year=sy, month=sm, day=s_d)
            ed = pd.Timestamp(year=ey, month=em, day=e_d)
            
            mask = (raw_temp_df['날짜'] >= sd) & (raw_temp_df['날짜'] <= ed)
            return raw_temp_df.loc[mask, '평균기온'].dropna().values
        except:
            return np.array([])

    with st.spinner("사용자 지정 기간 기온 매핑 중..."):
        sales_df['T_custom'] = sales_df.apply(
            lambda row: np.mean(get_custom_temp_array(int(row['연']), int(row['월']), s_off, s_day, e_off, e_day)) 
            if len(get_custom_temp_array(int(row['연']), int(row['월']), s_off, s_day, e_off, e_day)) > 0 else np.nan, 
            axis=1
        )

    st.markdown("---")
    years_all = sorted([int(y) for y in sales_df["연"].dropna().unique() if y > 0])
    
    title_with_icon("📈", "그래프 (실적 + 맞춤 예측 비교)", "h3", small=True)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        years_train = st.multiselect("📚 학습 연도", years_all, default=years_all[:-1] if len(years_all)>1 else years_all)
    with cc2:
        years_view = st.multiselect("👀 실적 연도", years_all, default=years_all[-2:] if len(years_all)>1 else years_all)
    with cc3:
        years_pred = st.multiselect("📈 예측 연도", years_all, default=years_all[-1:] if len(years_all)>0 else years_all)

    st.markdown(f"### ⚙️ {sel_prod} 예측 비교 (Poly-3 기준)")
    train_df = sales_df[sales_df["연"].isin(years_train)].copy()
    x_train = train_df['T_custom'].astype(float).values
    y_train = train_df[sel_prod].astype(float).values
    
    try:
        _, r2_p3, model_p3, _ = fit_poly3_and_predict(x_train, y_train, x_train)
        _, r2_p4, model_p4, _ = fit_poly4_and_predict(x_train, y_train, x_train)
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("3차 다항식 (Poly-3) Train R²", f"{r2_p3:.4f}")
        with c_b:
            st.caption(f"방정식: {poly_eq_text(model_p3)}")
            
        if go is not None:
            fig = go.Figure()
            for y in sorted(years_view):
                one = sales_df[sales_df["연"] == y][["월", sel_prod, "T_custom"]].dropna().sort_values("월")
                fig.add_trace(go.Scatter(
                    x=[f"{int(m)}월" for m in one["월"]],
                    y=one[sel_prod],
                    customdata=np.round(one["T_custom"].values.astype(float), 2),
                    mode="lines+markers",
                    name=f"{y} 실적",
                    hovertemplate="%{x} %{y:,}<br>해당기간 기온 %{customdata:.2f}℃"
                ))
            
            for y in sorted(years_pred):
                p_m_list, p_d_list, m_list, t_list = [], [], [], []
                for m in range(1, 13):
                    t_arr = get_custom_temp_array(y, m, s_off, s_day, e_off, e_day)
                    if len(t_arr) > 0:
                        p_m, _, _, _ = fit_poly3_and_predict(x_train, y_train, np.array([np.mean(t_arr)]))
                        p_d, _, _, _ = fit_poly3_and_predict(x_train, y_train, t_arr)
                        p_m_list.append(np.clip(np.rint(p_m[0]), 0, None))
                        p_d_list.append(np.clip(np.rint(np.mean(p_d)), 0, None))
                        t_list.append(np.mean(t_arr))
                        m_list.append(f"{m}월")
                
                if m_list:
                    fig.add_trace(go.Scatter(
                        x=m_list, y=p_m_list,
                        customdata=np.round(t_list, 2),
                        mode="lines", name=f"{y} 예측 (월평균)", line=dict(dash="dash", color="#ef553b"),
                        hovertemplate="%{x} %{y:,}<br>기간평균기온 %{customdata:.2f}℃"
                    ))
                    if toggle_daily_plan:
                        fig.add_trace(go.Scatter(
                            x=m_list, y=p_d_list,
                            customdata=np.round(t_list, 2),
                            mode="lines", name=f"{y} 예측 (일별합산)", line=dict(dash="dot", color="#00cc96"),
                            hovertemplate="%{x} %{y:,}<br>기간평균기온 %{customdata:.2f}℃"
                        ))
            
            fig.update_layout(
                title=f"📊 {sel_prod} 판매량 예측 추이", 
                dragmode="pan", 
                yaxis=dict(title="판매량", rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True, config=dict(scrollZoom=True, displaylogo=False))
            
        # 3. 요청하신 스태킹(Stacking) 방식의 테이블 렌더링 (순서 재배치 완료)
        title_with_icon("📑", f"{sel_prod} — 연도별 실적 및 오차 분석 표", "h3", small=True)
        table_rows = []
        for y in sorted(list(set(years_view + years_pred))):
            s_actual = sales_df.loc[sales_df["연"] == y, ["월", sel_prod]].set_index("월")[sel_prod]
            sum_act, sum_pred, sum_daily = 0, 0, 0
            
            for m in range(1, 13):
                act = s_actual.get(m, np.nan)
                t_arr = get_custom_temp_array(y, m, s_off, s_day, e_off, e_day)
                
                if len(t_arr) > 0:
                    p_m_val = fit_poly3_and_predict(x_train, y_train, np.array([np.mean(t_arr)]))[0][0]
                    p_m_val = np.clip(np.rint(p_m_val), 0, None)
                    p_d_vals = fit_poly3_and_predict(x_train, y_train, t_arr)[0]
                    p_d_val = np.clip(np.rint(np.mean(p_d_vals)), 0, None)
                else:
                    p_m_val, p_d_val = np.nan, np.nan
                    
                row = {"연도": f"{y}년", "월": f"{m}월", "실적": act}
                if toggle_daily_plan:
                    row["예측(월평균)"] = p_m_val
                    diff_m = p_m_val - act if pd.notna(act) and pd.notna(p_m_val) else np.nan
                    err_m = (diff_m / act * 100) if pd.notna(act) and act != 0 and pd.notna(p_m_val) else np.nan
                    row["차이(월평균)"] = diff_m
                    row["오차율1(%)"] = err_m
                    
                    row["예측(일별합산)"] = p_d_val
                    diff_d = p_d_val - act if pd.notna(act) and pd.notna(p_d_val) else np.nan
                    err_d = (diff_d / act * 100) if pd.notna(act) and act != 0 and pd.notna(p_d_val) else np.nan
                    row["차이(일별합산)"] = diff_d
                    row["오차율2(%)"] = err_d
                else:
                    row["예측"] = p_m_val
                    diff = p_m_val - act if pd.notna(act) and pd.notna(p_m_val) else np.nan
                    err = (diff / act * 100) if pd.notna(act) and act != 0 and pd.notna(p_m_val) else np.nan
                    row["차이"] = diff
                    row["오차율(%)"] = err
                    
                table_rows.append(row)
                if pd.notna(act): sum_act += act
                if pd.notna(p_m_val): sum_pred += p_m_val
                if pd.notna(p_d_val): sum_daily += p_d_val
                
            # 합계 행 추가
            tot_row = {"연도": f"{y}년 합계", "월": "-", "실적": sum_act if sum_act > 0 else np.nan}
            if toggle_daily_plan:
                tot_row["예측(월평균)"] = sum_pred if sum_pred > 0 else np.nan
                t_diff_m = sum_pred - sum_act if sum_act > 0 and sum_pred > 0 else np.nan
                t_err_m = (t_diff_m / sum_act * 100) if sum_act > 0 and sum_pred > 0 else np.nan
                tot_row["차이(월평균)"] = t_diff_m
                tot_row["오차율1(%)"] = t_err_m
                
                tot_row["예측(일별합산)"] = sum_daily if sum_daily > 0 else np.nan
                t_diff_d = sum_daily - sum_act if sum_act > 0 and sum_daily > 0 else np.nan
                t_err_d = (t_diff_d / sum_act * 100) if sum_act > 0 and sum_daily > 0 else np.nan
                tot_row["차이(일별합산)"] = t_diff_d
                tot_row["오차율2(%)"] = t_err_d
            else:
                tot_row["예측"] = sum_pred if sum_pred > 0 else np.nan
                t_diff = sum_pred - sum_act if sum_act > 0 and sum_pred > 0 else np.nan
                t_err = (t_diff / sum_act * 100) if sum_act > 0 and sum_pred > 0 else np.nan
                tot_row["차이"] = t_diff
                tot_row["오차율(%)"] = t_err
                
            table_rows.append(tot_row)

        table_show = pd.DataFrame(table_rows)
        if toggle_daily_plan:
            render_centered_table(table_show, float1_cols=["오차율1(%)", "오차율2(%)"], int_cols=["실적", "예측(월평균)", "차이(월평균)", "예측(일별합산)", "차이(일별합산)"], index=False)
        else:
            render_centered_table(table_show, float1_cols=["오차율(%)"], int_cols=["실적", "예측", "차이"], index=False)

    except Exception as e:
        st.error(f"예측 및 시각화 도중 오류가 발생했습니다: {e}")

# ===========================================================
# C) 공급량 추세분석 예측 — OLS/CAGR/Holt/SES + ARIMA/SARIMA
# ===========================================================

def render_trend_forecast():
    title_with_icon("📈", "공급량 추세분석 예측 (연도별 총합 · Normal)", "h2")
    
    meta = st.session_state.get("supply_meta")
    if not meta:
        st.warning("⚠️ 공급량 예측 탭에서 스프레드시트 데이터를 먼저 로드해야 추세 분석 실행이 가능합니다."); st.stop()
        
    df0 = meta["df"].copy()
    product_cols = meta["product_cols"]
    
    target_prod = st.selectbox("📊 시계열 추세 분석 상품 선택", product_cols, index=0)
    
    df_yearly = df0.groupby("연")[target_prod].sum().reset_index()
    st.markdown("### 🗓️ 연도별 공급량 총합 추이")
    render_centered_table(df_yearly, int_cols=[target_prod])
    
    x_yr = df_yearly["연"].values.reshape(-1, 1)
    y_yr = df_yearly[target_prod].values
    
    if len(y_yr) >= 3:
        model_lr = LinearRegression().fit(x_yr, y_yr)
        r2_yr = model_lr.score(x_yr, y_yr)
        st.metric(f"연도별 선형 추세선 적합도 (R²)", f"{r2_yr:.4f}")
        
        fig_trend, ax_trend = plt.subplots(figsize=(10, 3.5))
        ax_trend.plot(df_yearly["연"], y_yr, marker="o", label="연간 실적 합계")
        ax_trend.plot(df_yearly["연"], model_lr.predict(x_yr), linestyle="--", label="선형 추세선")
        ax_trend.legend()
        st.pyplot(fig_trend)
    else:
        st.info("ℹ️ 시계열 모델 및 추세 분석을 수행하기 위한 연간 데이터가 부족합니다.")

# ===========================================================
# 라우터 + 전역 추천 패널/결과 표시
# ===========================================================

def main():
    title_with_icon("📊", "도시가스 공급량·판매량 예측")
    st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량: 사용자 정의 기간 평균기온 연동 기반")

    with st.sidebar:
        with st.expander("🎯 추천 학습 데이터 기간(공급량)", expanded=False):
            meta = st.session_state.get("supply_meta")
            if not meta:
                st.info("공급량 예측 탭에서 데이터(실적·기온예측)를 먼저 불러오면 추천이 가능합니다.")
            else:
                prod_cols = meta["product_cols"] or []
                rec_prod = st.selectbox("대상 상품(1개)", options=prod_cols, index=0, key="rec_prod_global")
                st.caption(f"기준 종료연도: **{meta['latest_year']}** (데이터 최신연도)")
                if st.button("🔎 추천 구간 계산", key="btn_reco_global"):
                    df0 = meta["df"].copy()
                    temp_col = meta["temp_col"]
                    rec_df = recommend_train_ranges(df0, rec_prod, temp_col,
                                                    min_year=int(meta["min_year"]),
                                                    end_year=int(meta["latest_year"]))
                    st.session_state["rec_result_supply"] = {"table": rec_df, "prod": rec_prod, "end": int(meta["latest_year"]) }
                    st.success("추천 학습 구간 계산 완료! 아래 본문 상단에 결과가 표시됩니다.")

        title_with_icon("🧭", "예측 유형", "h3", small=True)
        mode = st.radio("🔀 선택",
                        ["공급량 예측", "판매량 예측", "공급량 추세분석 예측"],
                        index=0, label_visibility="visible")

    if st.session_state.get("rec_result_supply"):
        rr = st.session_state["rec_result_supply"]
        rec_df = rr["table"].copy()
        prod_name = rr["prod"]
        title_with_icon("🧠", f"추천 학습 데이터 기간 — {prod_name}", "h2")
        topk = rec_df.head(3).copy()
        topk["추천순위"] = np.arange(1, len(topk) + 1)
        cols = ["추천순위", "기간", "시작연도", "종료연도", "R2"]
        tshow = topk[cols].copy(); tshow["R2"] = tshow["R2"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        render_centered_table(tshow, index=False)

        if go is not None and not rec_df.empty:
            figr = go.Figure()
            rec_plot = rec_df.sort_values("시작연도")
            palette = ["rgba(255,179,71,0.18)", "rgba(118,214,165,0.18)", "rgba(120,180,255,0.18)"]
            for i, (_, row) in enumerate(topk.iterrows()):
                x0 = int(row["시작연도"]) - 0.5
                x1 = int(row["종료연도"]) + 0.5
                figr.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1,
                               line=dict(width=0), fillcolor=palette[i % len(palette)])
            figr.add_trace(go.Scatter(
                x=rec_plot["시작연도"], y=rec_plot["R2"],
                mode="lines+markers+text",
                text=[f"{v:.4f}" if pd.notna(v) else "" for v in rec_plot["R2"]],
                textposition="top center",
                name="R²(Poly-3)", hovertemplate="시작연도=%{x}<br>R²=%{y:.4f}<extra></extra>"
            ))
            figr.update_layout(
                title=f"학습 시작연도별 R² (종료연도={rr['end']})",
                xaxis_title="학습 기간(시작연도~현재)", yaxis_title="R² (train fit)",
                xaxis=dict(tickmode='linear', dtick=1),
                margin=dict(t=60, b=60, l=40, r=20), hovermode="x unified",
            )
            st.plotly_chart(figr, use_container_width=True, config=dict(scrollZoom=True, displaylogo=False))
        else:
            figr, axr = plt.subplots(figsize=(10.0, 3.8))
            rec_plot = rec_df.sort_values("시작연도")
            axr.plot(rec_plot["시작연도"], rec_plot["R2"], "-o", lw=2)
            for _, row in topk.iterrows():
                axr.axvspan(int(row["시작연도"]) - 0.5, int(row["종료연도"]) + 0.5, color="#ffb347", alpha=0.18)
            axr.set_title(f"학습 시작연도별 R² (종료연도={rr['end']})")
            axr.set_xlabel("시작연도"); axr.set_ylabel("R²")
            axr.grid(alpha=0.25)
            st.pyplot(figr, clear_figure=True)

        st.caption("추천 구간을 사이드바의 **학습 데이터 연도 선택**에 반영하면, 아래 모든 예측이 해당 구간으로 학습됩니다.")

    if mode == "공급량 예측":
        render_supply_forecast()
    elif mode == "판매량 예측":
        render_cooling_sales_forecast()
    else:
        render_trend_forecast()

if __name__ == "__main__":
    main()
