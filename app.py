# app.py — 도시가스 공급·판매 예측 (DSE Sales Analytics Dashboard)
# A) 공급량 예측        : Poly-3 기반 + Normal/Best/Conservative + 기온추세분석
# B) 판매량 예측(냉방용) : 전월16~당월15 평균기온 + Poly-3/4 비교
# C) 공급량 추세분석     : 연도별 총합 OLS/CAGR/Holt/SES + ARIMA/SARIMA(12)
# Fix: ARIMA/SARIMA 공란 방지(월별 실패 시 '연도합'에 직접 ARIMA 폴백)
# Default(추세분석 탭 상품): 개별난방용, 중앙난방용, 취사용
# 추가 적용 사항:
#  - 모든 Raw 데이터 및 기온 소스를 외부 파일 없이 구글 스프레드시트 주소에서 직접 원스톱 로드
#  - 예상 기온 데이터 유실 시 과거 동월 평균 기온으로 자동 대체하는 안정화 로직 내장
#  - 추천 학습기간 하이라이트: 시작~종료 전체(rect) 채우기
#  - 추천 R² 표기 소수 4자리
#  - 추천 범위 계산에서 종료연도와 같은 시작연도 제외(range(min_year, end_year))
#  - Plotly 그래프 scrollZoom=True 적용

import os
from io import BytesIO
from pathlib import Path
import warnings
from glob import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# Plotly 설정
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
st.set_page_config(page_title="DSE Sales Analytics Dashboard", layout="wide")
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

# ───────────── 구글 스프레드시트 연동 유틸 ─────────────
@st.cache_data(ttl=600)
def read_google_sheet(url: str) -> pd.DataFrame:
    """구글 스프레드시트 URL을 판다스 데이터프레임으로 변환"""
    try:
        if "/edit" in url:
            base_url = url.split("/edit")[0]
            if "gid=" in url:
                gid = url.split("gid=")[1].split("&")[0].split("#")[0]
                export_url = f"{base_url}/export?format=csv&gid={gid}"
            else:
                export_url = f"{base_url}/export?format=csv"
        else:
            export_url = url
        df = pd.read_csv(export_url)
        return normalize_cols(df)
    except Exception as e:
        st.error(f"구글 스프레드시트를 읽어오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 전역 기본 스프레드시트 주소 설정
GS_SALES_URL = "https://docs.google.com/spreadsheets/d/1-8RIPIkjnVXxoh5QJs6598nnHkWOGmrO655jr3b3g04/edit?gid=0#gid=0"
GS_SUPPLY_URL = "https://docs.google.com/spreadsheets/d/1vS-a9XrbjjIznHxntuFIM6hmml6qTlR2Cayw77p_Rao/edit?gid=0#gid=0"

# ───────────── 한글 폰트 설정 ─────────────
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

# ───────────── 공통 상수 및 데이터 전처리 유틸 ─────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp", "예상기온", "추세기온"]
KNOWN_PRODUCT_ORDER = [
    "개별난방용", "중앙난방용",
    "자가열전용", "일반용(2)", "업무난방용", "냉난방용",
    "주한미군", "취사용", "총공급량",
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
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
        if "년" in df.columns:
            df["연"] = df["년"]
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year
            
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
        
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="ignore",
            )
    return df

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
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others = [c for c in candidates if c not in ordered]
    return ordered + others

def month_start(x):
    x = pd.to_datetime(x)
    return pd.Timestamp(x.year, x.month, 1)

def month_range_inclusive(s, e):
    return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

# Poly-3/4 모델 예측 함수
def fit_poly3_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측치(NaN)가 포함되어 있습니다.")
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

def fit_poly4_and_predict(x_train, y_train, x_future):
    m = (~np.isnan(x_train)) & (~np.isnan(y_train))
    x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측치(NaN)가 포함되어 있습니다.")
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=4, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression().fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2, model, poly

def poly_eq_text(model, decimals: int = 4):
    c = model.coef_
    c1 = c[0] if len(c) > 0 else 0.0
    c2 = c[1] if len(c) > 1 else 0.0
    c3 = c[2] if len(c) > 2 else 0.0
    d = model.intercept_
    fmt = lambda v: f"{v:+,.{decimals}f}"
    return f"y = {fmt(c3)}x³ {fmt(c2)}x² {fmt(c1)}x {fmt(d)}"

def poly_eq_text4(model):
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

# 추천 학습기간 스코어 계산 유틸
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
# A) 공급량 예측 섹션
# ===========================================================

def render_supply_forecast():
    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Google Sheets 자동 연동", "수동 파일 업로드"], index=0)
        df, forecast_df = None, None

        if src == "Google Sheets 자동 연동":
            supply_url = st.text_input("🔗 공급량 스프레드시트 URL", value=GS_SUPPLY_URL)
            if supply_url:
                with st.spinner("구글 스프레드시트에서 데이터를 실시간으로 가져오는 중..."):
                    df = read_google_sheet(supply_url)
                    
                # 별도 파일 없이 공급량 시트 내의 기온 컬럼 혹은 내장된 연월 데이터 기반 추세 기온 자동 추출
                if df is not None and not df.empty:
                    temp_col_found = detect_temp_col(df)
                    if temp_col_found:
                        # 미래 예측을 위한 베이스 기온 데이터프레임 자동 구성
                        d_temp = df[["연", "월", temp_col_found]].copy().rename(columns={temp_col_found: "예상기온"})
                        d_temp["추세기온"] = d_temp["예상기온"]
                        forecast_df = d_temp.groupby(["연", "월"]).mean().reset_index()
        else:
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx)", type=["xlsx"])
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")
            up_fc = st.file_uploader("🌡️ 예상기온 엑셀 업로드(xlsx)", type=["xlsx"])
            if up_fc is not None:
                forecast_df = read_temperature_forecast(up_fc)

        if df is None or len(df) == 0:
            st.info("🧩 실적 데이터를 구글 시트 URL로 연결하거나 엑셀 파일을 업로드해 주세요."); st.stop()

        # 기온 예측 정보가 빈 경우 과거 데이터 기반 동월 평균 자동 폴백 스키마 생성
        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("🌡️ 데이터 내에서 기온 관련 열(평균기온/기온)을 식별할 수 없습니다."); st.stop()
            
        if forecast_df is None or forecast_df.empty:
            fallback_temp = df.groupby("월")[temp_col].mean().reset_index().rename(columns={temp_col: "예상기온"})
            fallback_temp["추세기온"] = fallback_temp["예상기온"]
            # 2026년부터 2035년까지의 연도 확장 스케줄러 자동 적용
            expanded_rows = []
            for y_ext in range(2026, 2036):
                f_block = fallback_temp.copy()
                f_block["연"] = y_ext
                expanded_rows.append(f_block)
            forecast_df = pd.concat(expanded_rows, ignore_index=True)[["연", "월", "예상기온", "추세기온"]]

        title_with_icon("📚", "학습 데이터 연도 선택", "h3", small=True)
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("🗓️ 연도 선택", years_all, default=years_all)

        title_with_icon("🧰", "예측할 상품 선택", "h3", small=True)
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        prods = st.multiselect("📦 상품(용도) 선택", product_cols, default=default_products)

        st.session_state["supply_meta"] = {
            "df": df.dropna(subset=["연","월"]).copy(),
            "temp_col": temp_col,
            "product_cols": product_cols,
            "latest_year": int(df["연"].max()),
            "min_year": int(df["연"].min()),
        }

        title_with_icon("⚙️", "예측 설정", "h3", small=True)
        last_year = int(df["연"].max())
        years = list(range(2010, 2036))
        col_sy, col_sm = st.columns(2)
        with col_sy:
            start_y = st.selectbox("🚀 예측 시작(연)", years, index=years.index(last_year))
        with col_sm:
            start_m = st.selectbox("📅 예측 시작(월)", list(range(1, 13)), index=0)
        col_ey, col_em = st.columns(2)
        with col_ey:
            end_y = st.selectbox("🏁 예측 종료(연)", years, index=years.index(last_year))
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
        st.success("✅ 공급량 데이터 로드 및 분석 완료!")

    if "supply_materials" not in st.session_state:
        st.info("👈 좌측 사이드바 설정 영역에서 **예측 시작** 버튼을 클릭해 주세요."); st.stop()

    mats = st.session_state["supply_materials"]
    base, train_df, prods = mats["base_df"], mats["train_df"], mats["prods"]
    x_train, fut_base = mats["x_train"], mats["fut_base"]
    temp_col = mats["temp_col"]; years_sel = mats["years_sel"]
    months = list(range(1, 13))

    title_with_icon("🌡️", "시나리오 Δ°C 보정 시뮬레이션", "h3", small=True)
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
        pred_all = pd.concat(pred_rows, ignore_index=True)
        pivot = pred_all.pivot_table(index=["연", "월", "월평균기온(추세)"], columns="상품", values="예측").reset_index()
        ordered = [c for c in KNOWN_PRODUCT_ORDER if c in pivot.columns]
        others = [c for c in pivot.columns if c not in (["연", "월", "월평균기온(추세)"] + ordered)]
        pivot = pivot[["연", "월", "월평균기온(추세)"] + ordered + others]
        return pivot.sort_values(["연", "월"]).reset_index(drop=True)

    def _render_with_year_sums(title, table, temp_col_name):
        title_with_icon("🗂️", title, "h3", small=True)
        render_centered_table(table, float1_cols=[temp_col_name], int_cols=[c for c in table.columns if c not in ["연", "월", temp_col_name]], index=False)
        year_sum = table.groupby("연").sum(numeric_only=True).reset_index()
        year_sum_show = year_sum.drop(columns=[c for c in ["월", temp_col_name] if c in year_sum.columns])
        year_sum_show.insert(1, "기간", "1~12월")
        title_with_icon("🗓️", "연도별 총계", "h4", small=True)
        render_centered_table(year_sum_show, int_cols=[c for c in year_sum_show.columns if c not in ["연", "기간"]], index=False)

        tmp = table.copy()
        tmp["__half"] = np.where(tmp["월"].astype(int) <= 6, "1~6월", "7~12월")
        half = tmp.groupby(["연", "__half"]).sum(numeric_only=True).reset_index().rename(columns={"__half": "반기"})
        half_to_show = half.rename(columns={"반기": "기간"}).drop(columns=[c for c in ["월", temp_col_name] if c in half.columns])
        title_with_icon("🧮", "반기별 총계", "h4", small=True)
        render_centered_table(half_to_show, int_cols=[c for c in half_to_show.columns if c not in ["연", "기간"]], index=False)
        return year_sum_show, half_to_show

    tbl_n = _forecast_table(d_norm)
    tbl_b = _forecast_table(d_best)
    tbl_c = _forecast_table(d_cons)
    tbl_trd = _forecast_table_trend()

    sum_n, half_n = _render_with_year_sums("🎯 Normal 시나리오 결과", tbl_n, "월평균기온")
    sum_b, half_b = _render_with_year_sums("💎 Best 시나리오 결과", tbl_b, "월평균기온")
    sum_c, half_c = _render_with_year_sums("🛡️ Conservative 시나리오 결과", tbl_c, "월평균기온")
    sum_t, half_t = _render_with_year_sums("📈 기온추세분석 결과", tbl_trd, "월평균기온(추세)")

    # 시각화 차트 스크립트 구성
    title_with_icon("📈", "인터랙티브 시각화 분석 차트", "h3", small=True)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        show_best = st.toggle("Best 시나리오 라인 표시", value=False)
        show_cons = st.toggle("Conservative 시나리오 라인 표시", value=False)

    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique()])
    default_years = years_all_for_plot[-2:] if len(years_all_for_plot) >= 2 else years_all_for_plot
    c_y1, c_y2 = st.columns(2)
    with c_y1:
        years_view = st.multiselect("👀 조회 대상 실적연도 선택", options=years_all_for_plot, default=default_years)
    pred_default = mats.get("default_pred_years", [])
    with c_y2:
        years_pred = st.multiselect("📈 조회 대상 예측연도 선택", options=sorted(list(set(fut_base["연"].tolist()))), default=[y for y in pred_default if y in fut_base["연"].unique()])

    actual_temp = base.groupby(["연", "월"])[temp_col].mean().reset_index().rename(columns={temp_col: "T_actual"})

    for prod in prods:
        y_train_prod = train_df[prod].astype(float).values
        y_norm, r2_train, model_s, _ = fit_poly3_and_predict(x_train, y_train_prod, (fut_base["예상기온"] + float(d_norm)).astype(float).values)
        
        if go is not None:
            fig = go.Figure()
            for y in sorted([int(v) for v in years_view]):
                one = base[base["연"] == y][["월", prod]].dropna().sort_values("월")
                t_one = actual_temp[actual_temp["연"] == y].sort_values("월")
                one = one.merge(t_one[["월", "T_actual"]], on="월", how="left")
                fig.add_trace(go.Scatter(x=[f"{int(m)}월" for m in one["월"]], y=one[prod], mode="lines+markers", name=f"{y} 실적"))
            
            for y in years_pred:
                p_idx = fut_base["연"] == int(y)
                fig.add_trace(go.Scatter(x=[f"{int(m)}월" for m in fut_base[p_idx]["월"]], y=np.clip(np.rint(y_norm[p_idx]), 0, None), mode="lines", name=f"{y} 예측(Normal)", line=dict(dash="dash")))
            
            fig.update_layout(title=f"📊 {prod} 공급량 추이 분석 (Train R²={r2_train:.4f})", dragmode="pan")
            st.plotly_chart(fig, use_container_width=True, config=dict(scrollZoom=True, displaylogo=False))


# ===========================================================
# B) 판매량 예측 섹션 (냉방용)
# ===========================================================

def render_cooling_sales_forecast():
    title_with_icon("🧊", "판매량 예측(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준", "h2")
    
    with st.sidebar:
        title_with_icon("📥", "판매량 데이터 로드", "h3", small=True)
        sales_url = st.text_input("🔗 판매량 스프레드시트 URL", value=GS_SALES_URL)
        sales_df = None
        if sales_url:
            with st.spinner("구글 스프레드시트에서 판매량 실적을 가져오는 중..."):
                sales_df = read_google_sheet(sales_url)

    if sales_df is None or sales_df.empty:
        st.info("🧩 판매량 분석을 위해 좌측 사이드바에 유효한 구글 스프레드시트 주소를 입력해 주세요."); st.stop()

    temp_col = detect_temp_col(sales_df)
    if temp_col is None:
        st.error("🌡️ 판매량 데이터셋 내에서 기온 정보를 식별할 수 없습니다. 컬럼명을 점검하세요."); st.stop()

    product_cols = guess_product_cols(sales_df)
    sel_prod = st.selectbox("📦 분석 및 예측 대상 상품 선택", product_cols)

    # Poly-3 및 Poly-4 차수 비교 연산 로직 자동 수행
    st.markdown(f"### ⚙️ {sel_prod} 차수별 적합도 비교 (Poly-3 vs Poly-4)")
    x_data = sales_df[temp_col].astype(float).values
    y_data = sales_df[sel_prod].astype(float).values
    
    try:
        _, r2_p3, model_p3, _ = fit_poly3_and_predict(x_data, y_data, x_data)
        _, r2_p4, model_p4, _ = fit_poly4_and_predict(x_data, y_data, x_data)
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("3차 다항식 결정계수 (Poly-3 R²)", f"{r2_p3:.4f}")
            st.caption(f"방정식식: {poly_eq_text(model_p3)}")
        with c_b:
            st.metric("4차 다항식 결정계수 (Poly-4 R²)", f"{r2_p4:.4f}")
            st.caption(f"방정식: {poly_eq_text4(model_p4)}")
            
        # 데이터 분포 산점도 시각화
        fig_cool, ax_cool = plt.subplots(figsize=(10, 4))
        ax_cool.scatter(x_data, y_data, alpha=0.5, label="실적 실시간 샘플")
        x_domain = np.linspace(x_data.min()-1, x_data.max()+1, 100)
        y_p3_dom, _, _, _ = fit_poly3_and_predict(x_data, y_data, x_domain)
        ax_cool.plot(x_domain, y_p3_dom, color="red", label="Poly-3 Fit")
        ax_cool.legend()
        st.pyplot(fig_cool)
    except Exception as e:
        st.error(f"다항식 적합 중 요류가 발생했습니다: {e}")


# ===========================================================
# C) 공급량 추세분석 예측 섹션 (시계열 모델 결합)
# ===========================================================

def render_trend_forecast():
    title_with_icon("📈", "공급량 추세분석 예측 (연도별 총합 분석 패널)", "h2")
    
    # 세션 상태 공유 매커니즘 작동
    meta = st.session_state.get("supply_meta")
    if not meta:
        st.warning("⚠️ 공급량 예측 탭에서 스프레드시트 데이터를 먼저 로드해야 추세 분석 실행이 가능합니다."); st.stop()
        
    df0 = meta["df"].copy()
    product_cols = meta["product_cols"]
    
    target_prod = st.selectbox("📊 시계열 추세 분석 상품 선택", product_cols, index=0)
    
    # 연도합 계산 집계 처리
    df_yearly = df0.groupby("연")[target_prod].sum().reset_index()
    st.markdown("### 🗓️ 연도별 공급량 총합 추이")
    render_centered_table(df_yearly, int_cols=[target_prod])
    
    # 기본적인 선형 추세 모델링 시뮬레이션
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
# 마스터 라우터 시스템 메인 엔트리 구조
# ===========================================================

def main():
    title_with_icon("📊", "DSE Sales Analytics Dashboard")
    st.caption("공급량·판매량 분석 알고리즘 엔진 — 기온 매핑 다항 모델 및 다변량 시계열 모듈 결합")

    with st.sidebar:
        with st.expander("🎯 추천 학습 데이터 기간 산출 연산기", expanded=False):
            meta = st.session_state.get("supply_meta")
            if not meta:
                st.info("공급량 예측 탭에서 구글 시트 데이터 로드 완료 후 사용 가능합니다.")
            else:
                prod_cols = meta["product_cols"] or []
                rec_prod = st.selectbox("분석 타겟 상품 선택", options=prod_cols, index=0)
                if st.button("🔎 추천 구간 계산 개시"):
                    df0 = meta["df"].copy()
                    temp_col = meta["temp_col"]
                    rec_df = recommend_train_ranges(df0, rec_prod, temp_col, min_year=int(meta["min_year"]), end_year=int(meta["latest_year"]))
                    st.session_state["rec_result_supply"] = {"table": rec_df, "prod": rec_prod, "end": int(meta["latest_year"]) }
                    st.success("추천 학습 최적 구간 매핑 성공!")

        title_with_icon("🧭", "분석 대시보드 유형 선택", "h3", small=True)
        mode = st.radio("🔀 대시보드 전환", ["공급량 예측 분석", "판매량 예측 분석 (냉방용)", "공급량 중장기 추세 분석"], index=0)

    # 전역 추천 정보 레이아웃 최상단 노출 처리
    if st.session_state.get("rec_result_supply"):
        rr = st.session_state["rec_result_supply"]
        rec_df = rr["table"].copy()
        title_with_icon("🧠", f"최적 최적 학습 데이터 추천 기간 리포트 — {rr['prod']}", "h2")
        topk = rec_df.head(3).copy()
        topk["추천순위"] = np.arange(1, len(topk) + 1)
        tshow = topk[["추천순위", "기간", "시작연도", "종료연도", "R2"]].copy()
        tshow["R2"] = tshow["R2"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        render_centered_table(tshow, index=False)
        st.markdown("---")

    # 분기 라우팅 매핑
    if mode == "공급량 예측 분석":
        render_supply_forecast()
    elif mode == "판매량 예측 분석 (냉방용)":
        render_cooling_sales_forecast()
    else:
        render_trend_forecast()

if __name__ == "__main__":
    main()
