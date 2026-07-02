# -*- coding: utf-8 -*-
"""
리스크 패리티 + CAPM 포트폴리오 대시보드 (Streamlit UI)
------------------------------------------------------------
실행:
    pip install streamlit yfinance numpy pandas scipy altair
    streamlit run portfolio_app.py

브라우저가 자동으로 열립니다. 왼쪽 사이드바에서 종목코드/설정을 입력하세요.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# 앞서 만든 계산 로직 재사용
from portfolio_risk_parity_capm import (
    download_prices,
    download_prices_range,
    to_returns,
    risk_parity_weights,
    capm_expected_returns,
    estimate_cov,
    get_names,
)

st.set_page_config(page_title="리스크 패리티 포트폴리오", page_icon="📊", layout="wide")


# ============================================================
# 캐시: 동일 입력은 다시 다운로드하지 않음
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_prices(tickers, period, interval):
    return download_prices(tickers, period, interval)


@st.cache_data(show_spinner=False, ttl=86400)
def get_names_cached(tickers):
    # tickers는 캐시를 위해 tuple로 전달
    return get_names(list(tickers))


@st.cache_data(show_spinner=False, ttl=3600)
def get_prices_range(tickers, start, end):
    return download_prices_range(list(tickers), start, end)


@st.cache_data(show_spinner=False, ttl=21600)
def get_screen_universe(markets, size_top):
    from stock_screener import fetch_universe, add_price_factors
    return add_price_factors(fetch_universe(tuple(markets), size_top=size_top))


@st.cache_data(show_spinner=False, ttl=21600)
def get_us_screen_universe(index_name):
    from stock_screener import fetch_us_universe, add_price_factors
    return add_price_factors(fetch_us_universe(index_name))


# ============================================================
# 사이드바 (입력)
# ============================================================
# 스크리너의 '이 종목들로 포트폴리오 구성' 클릭 시 예약된 값을
# 위젯 생성 전에 반영 (생성 후에는 수정 불가라 이 위치여야 함)
if "pending_tickers" in st.session_state:
    st.session_state["tickers_input"] = st.session_state.pop("pending_tickers")
if "pending_market" in st.session_state:
    st.session_state["market_index_input"] = st.session_state.pop("pending_market")
if "pending_currency" in st.session_state:
    st.session_state["currency_input"] = st.session_state.pop("pending_currency")

st.sidebar.header("⚙️ 설정")

tickers_raw = st.sidebar.text_area(
    "종목코드 (쉼표 또는 줄바꿈으로 구분)",
    value="AAPL, MSFT, NVDA, JPM, XOM",
    height=110,
    help="미국주식: AAPL · 한국주식: 005930.KS · 코스닥: 247540.KQ",
    key="tickers_input",
)

market_index = st.sidebar.selectbox(
    "시장지수 (CAPM 벤치마크)",
    options=["^GSPC", "^IXIC", "^KS11", "^KQ11", "^DJI"],
    index=0,
    help="^GSPC=S&P500 · ^IXIC=나스닥 · ^KS11=KOSPI · ^KQ11=KOSDAQ",
    key="market_index_input",
)

col_a, col_b = st.sidebar.columns(2)
period = col_a.selectbox("기간", ["1y", "2y", "3y", "5y", "10y", "max"], index=2)
interval = col_b.selectbox("주기", ["1d", "1wk"], index=0)
annualize = 252 if interval == "1d" else 52

cov_method_label = st.sidebar.radio(
    "공분산 추정 방식",
    ["EWMA (지수가중)", "표본공분산"],
    help="EWMA는 최근 변동성에 더 큰 가중치를 줘 국면 변화를 빠르게 반영합니다.",
)
cov_method = "ewma" if cov_method_label.startswith("EWMA") else "sample"
ewma_lambda = 0.94
if cov_method == "ewma":
    ewma_lambda = st.sidebar.slider(
        "EWMA λ (감쇠계수)", 0.80, 0.99, 0.94, 0.01,
        help="1에 가까울수록 과거를 오래 반영. 일간 0.94 / 주간 0.97 권장.",
    )

rf = st.sidebar.slider("무위험수익률 (연율 %)", 0.0, 10.0, 3.5, 0.1) / 100

mkt_mode = st.sidebar.radio(
    "시장 기대수익률",
    ["과거 평균으로 자동 추정", "직접 입력"],
    help="실무에선 직접 입력(예: 8~9%)이 더 안정적입니다.",
)
market_return = None
if mkt_mode == "직접 입력":
    market_return = st.sidebar.slider("시장 기대수익률 (연율 %)", 0.0, 20.0, 9.0, 0.5) / 100

st.sidebar.divider()
st.sidebar.subheader("💰 투자금 배분")
currency = st.sidebar.radio("통화", ["₩ 원", "$ 달러"], horizontal=True,
                            key="currency_input")
cur_sym = "₩" if currency.startswith("₩") else "$"
invest_amount = st.sidebar.number_input(
    f"총 투자액 ({cur_sym})",
    min_value=0, value=10_000_000, step=1_000_000, format="%d",
    help="입력한 금액을 비중대로 나눠 종목별 배분액과 예상 매수 수량을 계산합니다.",
)

run = st.sidebar.button("🚀 분석 실행", type="primary", use_container_width=True)


# ============================================================
# 본문
# ============================================================
st.title("📊 리스크 패리티 + CAPM 포트폴리오")
st.caption("종목별 위험 기여도를 균등화해 비중을 산정하고, CAPM으로 기대수익률을 도출합니다.")

# ============================================================
# 종목 스크리너 (팩터 기반 자동 선별)
# ============================================================
with st.expander("🔎 종목 스크리너 — 어떤 종목을 살지 모르겠다면 (한국 주식 자동 선별)"):
    st.caption("검증된 투자 팩터 점수로 종목을 자동 선별합니다. "
               "데이터 — 한국: 네이버 금융(PER·ROE) / 미국: Finviz(PER·ROE) "
               "+ 야후 파이낸스(6개월 수익률).")

    s1, s2, s3 = st.columns(3)
    scr_market = s1.selectbox(
        "시장",
        ["코스피", "코스닥", "코스피+코스닥",
         "미국 S&P500", "미국 나스닥100", "미국 다우30"],
    )
    is_us = scr_market.startswith("미국")
    if is_us:
        scr_size = None
        s2.selectbox("종목 범위", ["지수 구성종목 전체"], disabled=True,
                     help="미국은 선택한 지수의 구성종목 전체가 대상입니다.")
    else:
        scr_size = s2.selectbox("종목 범위",
                                ["대형주 (시총 상위 200)", "중대형주 (상위 500)"])
    scr_topn = s3.slider("뽑을 종목 수", 5, 20, 10)

    preset = st.radio(
        "전략 유형",
        ["🏦 중장기 가치+우량 (분기~연 단위 보유)",
         "⚡ 단기 월간 로테이션 (한 달마다 종목 교체)"],
        horizontal=True,
    )
    is_short = preset.startswith("⚡")

    st.write("**사용할 팩터** — 체크한 항목의 순위 점수를 평균해 종합점수를 만듭니다")
    selected_factors = []
    if not is_short:
        fc1, fc2, fc3 = st.columns(3)
        if fc1.checkbox("저PER (저평가)", True,
                        help="번 돈에 비해 주가가 싼 종목. 적자기업은 자동 제외"):
            selected_factors.append("저PER")
        if fc2.checkbox("고ROE (우량)", True,
                        help="자기자본 대비 수익성이 높은 기업"):
            selected_factors.append("고ROE")
        if fc3.checkbox("모멘텀 (6개월)", True,
                        help="최근 6개월간 주가 흐름이 좋은 종목"):
            selected_factors.append("모멘텀(6개월)")
    else:
        fc1, fc2, fc3 = st.columns(3)
        if fc1.checkbox("모멘텀 (3개월)", True,
                        help="최근 3개월 상승 흐름. 단기 지속성이 검증된 구간"):
            selected_factors.append("모멘텀(3개월)")
        if fc2.checkbox("52주 신고가 근접", True,
                        help="1년 최고가에 가까운 종목이 계속 오르는 경향(조지-황 효과)"):
            selected_factors.append("52주 신고가 근접")
        if fc3.checkbox("거래대금 증가", False,
                        help="최근 20일 거래대금이 이전보다 늘어난 종목 (관심 유입, 노이즈도 큼)"):
            selected_factors.append("거래대금 증가")
        st.info("💡 **사용법**: 매달 1일(또는 정한 날)에 같은 조건으로 다시 돌려서, "
                "새 목록에서 빠진 종목은 팔고 새로 들어온 종목만 사는 **월간 리밸런싱** 방식입니다. "
                "매매가 잦아 수수료·세금 부담이 크고, 중장기 전략보다 손실 위험도 큽니다. "
                "참고로 1개월 모멘텀 단독은 오히려 반전되는 경향이 있어 3개월을 기본으로 했어요.")

    if st.button("🔎 종목 찾기", type="primary"):
        if not selected_factors:
            st.error("팩터를 1개 이상 선택해주세요.")
        else:
            from stock_screener import screen_stocks
            try:
                with st.spinner("종목을 선별하는 중... (처음엔 30초~1분 걸릴 수 있어요)"):
                    if is_us:
                        idx_key = {"미국 S&P500": "SP500",
                                   "미국 나스닥100": "NASDAQ100",
                                   "미국 다우30": "DJIA"}[scr_market]
                        uni = get_us_screen_universe(idx_key)
                        st.session_state["screen_bench"] = {
                            "SP500": "^GSPC", "NASDAQ100": "^IXIC", "DJIA": "^DJI",
                        }[idx_key]
                        st.session_state["screen_currency"] = "$ 달러"
                    else:
                        markets = {"코스피": ("KOSPI",), "코스닥": ("KOSDAQ",),
                                   "코스피+코스닥": ("KOSPI", "KOSDAQ")}[scr_market]
                        size_top = 200 if scr_size.startswith("대형주") else 500
                        uni = get_screen_universe(markets, size_top)
                        st.session_state["screen_bench"] = "^KS11"
                        st.session_state["screen_currency"] = "₩ 원"
                    st.session_state["screen_result"] = screen_stocks(
                        uni, selected_factors, top_n=scr_topn,
                    )
                    st.session_state.pop("diverse_result", None)  # 이전 조합 결과 초기화
            except Exception as e:
                st.error(f"선별 중 오류가 발생했습니다: {e}")

    scr_res = st.session_state.get("screen_result")
    if scr_res is not None and len(scr_res) > 0:
        st.dataframe(scr_res, use_container_width=True)
        st.caption("종합점수 = 선택한 팩터별 백분위 순위(0~100)의 평균. 높을수록 조건에 잘 맞는 종목.")
        if st.button("📌 이 종목들로 포트폴리오 구성", type="primary"):
            st.session_state["pending_tickers"] = ", ".join(scr_res["티커"])
            st.session_state["pending_market"] = \
                st.session_state.get("screen_bench", "^KS11")
            st.session_state["pending_currency"] = \
                st.session_state.get("screen_currency", "₩ 원")
            st.session_state["analysis_requested"] = True
            st.rerun()

        # --- 리스크 분산 조합 (상관관계 최소 부분집합) ---
        st.markdown("---")
        st.markdown("**🧩 리스크 분산 조합** — 뽑힌 종목 중 서로 가장 다르게 움직이는 "
                    "(상관관계가 낮은) 조합만 추려 분산 효과를 극대화합니다.")
        if len(scr_res) <= 5:
            st.caption("⚠️ 후보가 5개 이하면 조합을 추릴 여지가 없어요. "
                       "위의 '뽑을 종목 수'를 10개 이상으로 늘려서 다시 찾아보세요.")
        else:
            dv1, dv2 = st.columns([1, 2])
            max_k = min(8, len(scr_res) - 1)
            div_k = dv1.slider("조합 종목 수", 3, max_k, min(5, max_k), key="div_k")
            if dv2.button("🧩 분산 조합 찾기", key="find_div"):
                try:
                    with st.spinner("1년치 주가로 상관관계를 계산하는 중..."):
                        from stock_screener import min_corr_subset
                        px_div = get_prices(tuple(scr_res["티커"]), "1y", "1d")
                        picked, avg_c, overall_c = min_corr_subset(
                            to_returns(px_div), k=div_k
                        )
                        scr_names = dict(zip(scr_res["티커"], scr_res["종목명"]))
                        st.session_state["diverse_result"] = {
                            "tickers": picked,
                            "names": [scr_names.get(t, t) for t in picked],
                            "avg": avg_c,
                            "overall": overall_c,
                        }
                except Exception as e:
                    st.error(f"상관관계 계산 중 오류가 발생했습니다: {e}")

        div_res = st.session_state.get("diverse_result")
        if div_res:
            dm1, dm2 = st.columns(2)
            dm1.metric("후보 전체 평균 상관계수", f"{div_res['overall']:.2f}")
            dm2.metric(f"선택 {len(div_res['tickers'])}종목 평균 상관계수",
                       f"{div_res['avg']:.2f}",
                       delta=f"{div_res['avg'] - div_res['overall']:+.2f}",
                       delta_color="inverse",
                       help="상관계수가 낮을수록 서로 다르게 움직여 분산 효과가 큽니다.")
            div_table = pd.DataFrame({
                "티커": div_res["tickers"],
                "종목명": div_res["names"],
            }, index=pd.RangeIndex(1, len(div_res["tickers"]) + 1, name="No"))
            st.dataframe(div_table, use_container_width=True)
            if st.button("📌 이 분산 조합으로 포트폴리오 구성", type="primary",
                         key="apply_div"):
                st.session_state["pending_tickers"] = ", ".join(div_res["tickers"])
                st.session_state["pending_market"] = \
                    st.session_state.get("screen_bench", "^KS11")
                st.session_state["pending_currency"] = \
                    st.session_state.get("screen_currency", "₩ 원")
                st.session_state["analysis_requested"] = True
                st.rerun()

# 버튼 클릭 상태를 세션에 기억 → 이후 다른 버튼/입력을 눌러도 결과가 유지됨
if run:
    st.session_state["analysis_requested"] = True
if not st.session_state.get("analysis_requested"):
    st.info("왼쪽 사이드바에서 종목코드와 설정을 입력한 뒤 **분석 실행**을 눌러주세요.")
    st.stop()

# 티커 파싱
tickers = [t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()]
if len(tickers) < 2:
    st.error("종목을 2개 이상 입력해주세요.")
    st.stop()

try:
    with st.spinner("가격 데이터를 받아오는 중..."):
        prices = get_prices(tickers, period, interval)
        mkt_price = get_prices(market_index, period, interval)

    # 일부 티커가 누락되면 알림
    valid = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"데이터를 못 받은 종목 (제외됨): {', '.join(missing)}")
    if len(valid) < 2:
        st.error("유효한 종목이 2개 미만입니다. 티커를 확인하세요.")
        st.stop()

    asset_ret = to_returns(prices)[valid]
    mkt_ret = to_returns(mkt_price)
    idx = asset_ret.index.intersection(mkt_ret.index)
    asset_ret, mkt_ret = asset_ret.loc[idx], mkt_ret.loc[idx]

    # --- 리스크 패리티 ---
    cov_annual = estimate_cov(asset_ret, cov_method, ewma_lambda, annualize)
    weights = risk_parity_weights(cov_annual)
    w = pd.Series(weights, index=valid, name="비중")

    # --- CAPM ---
    betas, capm_ret, mkt_exp = capm_expected_returns(
        asset_ret, mkt_ret, rf, market_return, annualize
    )

    # --- 포트폴리오 지표 ---
    port_var = weights @ cov_annual @ weights
    port_vol = float(np.sqrt(port_var))
    port_exp_ret = float((w * capm_ret).sum())
    sharpe = (port_exp_ret - rf) / port_vol
    rc = (weights * (cov_annual @ weights)) / port_var

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
    st.stop()

# --- 상단 요약 지표 ---
st.subheader("포트폴리오 요약")
cov_caption = (f"공분산: EWMA (λ={ewma_lambda})" if cov_method == "ewma"
               else "공분산: 표본공분산")
st.caption(f"{cov_caption} · 기간 {period} · 주기 {interval}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("기대수익률 (CAPM)", f"{port_exp_ret:.2%}")
m2.metric("연변동성 (리스크)", f"{port_vol:.2%}")
m3.metric("샤프지수 (ex-ante)", f"{sharpe:.2f}")
m4.metric("시장 기대수익률", f"{mkt_exp:.2%}")

st.divider()

# --- 결과 테이블 + 차트 ---
with st.spinner("종목명을 조회하는 중..."):
    name_map = get_names_cached(tuple(valid))

table = pd.DataFrame({
    "종목명": pd.Series(name_map),
    "비중": w,
    "위험기여도": pd.Series(rc, index=valid),
    "베타": betas,
    "기대수익률(CAPM)": capm_ret,
})
table.index.name = "티커"

left, right = st.columns([1, 1])

with left:
    st.subheader("종목별 결과")
    st.dataframe(
        table.style.format({
            "비중": "{:.2%}",
            "위험기여도": "{:.2%}",
            "베타": "{:.3f}",
            "기대수익률(CAPM)": "{:.2%}",
        }),
        use_container_width=True,
    )
    csv = table.to_csv().encode("utf-8-sig")
    st.download_button("📥 CSV 다운로드", csv, "portfolio.csv", "text/csv")

with right:
    st.subheader("비중 (리스크 패리티)")
    pie_df = w.reset_index()
    pie_df.columns = ["종목", "비중"]
    pie_df["종목명"] = pie_df["종목"].map(name_map)
    donut = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=60)
        .encode(
            theta="비중:Q",
            color=alt.Color("종목:N", legend=alt.Legend(title=None)),
            tooltip=["종목", "종목명", alt.Tooltip("비중:Q", format=".2%")],
        )
    )
    st.altair_chart(donut, use_container_width=True)

# --- 투자금 배분 ---
if invest_amount > 0:
    st.divider()
    st.subheader("💰 투자금 배분")

    def _money(x):
        return f"{cur_sym}{x:,.0f}" if cur_sym == "₩" else f"{cur_sym}{x:,.2f}"

    last_prices = prices[valid].iloc[-1]            # 가장 최근 종가
    alloc = w * invest_amount                       # 비중대로 나눈 배분 금액
    shares = np.floor(alloc / last_prices)          # 정수 주 (내림)
    actual = shares * last_prices                   # 실제 매수 금액

    invest_table = pd.DataFrame({
        "종목명": pd.Series(name_map),
        "비중": w,
        "배분금액": alloc,
        "현재가": last_prices,
        "예상수량(주)": shares.astype(int),
        "실매수금액": actual,
    })
    invest_table.index.name = "티커"

    st.dataframe(
        invest_table.style.format({
            "비중": "{:.2%}",
            "배분금액": _money,
            "현재가": _money,
            "예상수량(주)": "{:,d}",
            "실매수금액": _money,
        }),
        use_container_width=True,
    )

    spent = float(actual.sum())
    a1, a2, a3 = st.columns(3)
    a1.metric("총 투자액", _money(invest_amount))
    a2.metric("실제 매수 합계", _money(spent))
    a3.metric("잔여 현금 (단주)", _money(invest_amount - spent))

    inv_csv = invest_table.to_csv().encode("utf-8-sig")
    st.download_button("📥 배분표 CSV 다운로드", inv_csv,
                       "allocation.csv", "text/csv", key="alloc_csv")

    st.caption(
        "※ '예상수량'은 현재가 기준 정수 매수(소수점 버림) 가정입니다. "
        "정수로만 살 수 있어 배분금액과 실매수금액에 약간 차이(단주)가 납니다. "
        "투자액 통화와 종목 거래 통화가 다르면(예: 원화 입력 + 해외종목) 수량은 참고용입니다."
    )

st.divider()

# --- 기대수익률 막대 + 상관관계 히트맵 ---
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("종목별 기대수익률 (CAPM)")
    bar_df = capm_ret.reset_index()
    bar_df.columns = ["종목", "기대수익률"]
    bar_df["종목명"] = bar_df["종목"].map(name_map)
    bar = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X("종목:N", sort="-y"),
            y=alt.Y("기대수익률:Q", axis=alt.Axis(format="%")),
            color=alt.Color("종목:N", legend=None),
            tooltip=["종목", "종목명", alt.Tooltip("기대수익률:Q", format=".2%")],
        )
    )
    st.altair_chart(bar, use_container_width=True)

with c2:
    st.subheader("상관관계")
    corr = asset_ret.corr()
    corr.index.name = "종목A"
    corr.columns.name = "종목B"
    corr_df = corr.reset_index().melt(
        id_vars="종목A", var_name="종목B", value_name="상관계수"
    )
    heat = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x="종목A:N",
            y="종목B:N",
            color=alt.Color("상관계수:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["종목A", "종목B", alt.Tooltip("상관계수:Q", format=".2f")],
        )
    )
    st.altair_chart(heat, use_container_width=True)

# ============================================================
# 백테스트: 과거에 사고팔았다면?
# ============================================================
st.divider()
st.subheader("⏱️ 과거에 사고팔았다면? (기간 수익률)")
st.caption("매수일에 사서 매도일에 판 것으로 가정하고 실제 과거 주가로 수익률을 계산합니다. "
           "휴장일을 고르면 가장 가까운 거래일로 자동 조정됩니다. (종가 기준·배당 반영)")

_today = date.today()
with st.form("backtest_form"):
    f1, f2 = st.columns(2)
    buy_date = f1.date_input("매수일", value=_today - timedelta(days=365),
                             min_value=date(2000, 1, 1), max_value=_today)
    sell_date = f2.date_input("매도일", value=_today,
                              min_value=date(2000, 1, 1), max_value=_today)
    f3, f4 = st.columns(2)
    bt_mode = f3.radio("배분 방식", ["리스크 패리티 비중", "균등 비중"],
                       help="위에서 계산된 리스크 패리티 비중대로 살지, 모든 종목을 똑같이 살지 선택")
    bt_amount = f4.number_input(
        f"투자액 ({cur_sym})", min_value=0,
        value=int(invest_amount) if invest_amount > 0 else 10_000_000,
        step=1_000_000, format="%d",
    )
    bt_submit = st.form_submit_button("📈 수익률 계산", type="primary",
                                      use_container_width=True)

if bt_submit:
    def _money2(x):
        return f"{cur_sym}{x:,.0f}" if cur_sym == "₩" else f"{cur_sym}{x:,.2f}"

    if buy_date >= sell_date:
        st.error("매도일은 매수일보다 뒤여야 합니다.")
    elif bt_amount <= 0:
        st.error("투자액을 입력해주세요.")
    else:
        try:
            with st.spinner("과거 주가를 받아오는 중..."):
                px = get_prices_range(
                    tuple(valid), str(buy_date), str(sell_date + timedelta(days=1))
                )
            px = px[[t for t in valid if t in px.columns]]

            after = px.index[px.index >= pd.Timestamp(buy_date)]
            before = px.index[px.index <= pd.Timestamp(sell_date)]
            if len(after) == 0 or len(before) == 0 or after[0] >= before[-1]:
                st.error("해당 기간에 거래일이 부족합니다. 날짜를 다시 확인해주세요.")
                st.stop()
            buy_day, sell_day = after[0], before[-1]
            buy_px, sell_px = px.loc[buy_day], px.loc[sell_day]

            # 매수일에 데이터가 없는 종목(상장 전 등)은 제외
            ok = buy_px.notna() & sell_px.notna()
            excluded = list(ok.index[~ok])
            bt_tickers = list(ok.index[ok])
            if excluded:
                st.warning(f"매수일에 데이터가 없어 제외된 종목: {', '.join(excluded)}")
            if len(bt_tickers) == 0:
                st.error("계산 가능한 종목이 없습니다.")
                st.stop()

            # 비중: 리스크 패리티(재정규화) 또는 균등
            if bt_mode.startswith("리스크"):
                w_bt = w.reindex(bt_tickers)
                w_bt = w_bt / w_bt.sum()
            else:
                w_bt = pd.Series(1.0 / len(bt_tickers), index=bt_tickers)

            alloc_bt = w_bt * bt_amount
            shares_bt = np.floor(alloc_bt / buy_px[bt_tickers])
            invested = shares_bt * buy_px[bt_tickers]
            final_val = shares_bt * sell_px[bt_tickers]
            total_in, total_out = float(invested.sum()), float(final_val.sum())
            if total_in <= 0:
                st.error("투자액이 너무 적어 1주도 살 수 없습니다. 금액을 늘려주세요.")
                st.stop()

            profit = total_out - total_in
            ret = total_out / total_in - 1.0
            days = (sell_day - buy_day).days

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("총 수익률", f"{ret:+.2%}")
            r2.metric("평가손익", _money2(profit), delta=f"{ret:+.2%}")
            r3.metric("투자원금(실매수)", _money2(total_in))
            if days >= 30:
                cagr = (1.0 + ret) ** (365.0 / days) - 1.0
                r4.metric("연환산 수익률", f"{cagr:+.2%}")
            else:
                r4.metric("보유 기간", f"{days}일")

            st.caption(f"실제 체결 가정일 — 매수: {buy_day.date()} / 매도: {sell_day.date()}"
                       f" (보유 {days}일)")

            bt_table = pd.DataFrame({
                "종목명": pd.Series({t: name_map.get(t, t) for t in bt_tickers}),
                "비중": w_bt,
                "수량(주)": shares_bt.astype(int),
                "매수가": buy_px[bt_tickers],
                "매도가": sell_px[bt_tickers],
                "수익률": sell_px[bt_tickers] / buy_px[bt_tickers] - 1.0,
                "평가손익": final_val - invested,
            })
            bt_table.index.name = "티커"
            st.dataframe(
                bt_table.style.format({
                    "비중": "{:.2%}",
                    "수량(주)": "{:,d}",
                    "매수가": _money2,
                    "매도가": _money2,
                    "수익률": "{:+.2%}",
                    "평가손익": _money2,
                }),
                use_container_width=True,
            )
            bt_csv = bt_table.to_csv().encode("utf-8-sig")
            st.download_button("📥 백테스트 CSV 다운로드", bt_csv,
                               "backtest.csv", "text/csv", key="bt_csv")
        except Exception as e:
            st.error(f"계산 중 오류가 발생했습니다: {e}")

st.caption("⚠️ 과거 데이터 기반 추정치이며 투자 권유가 아닙니다. CAPM 기대수익률은 가정에 민감합니다.")
