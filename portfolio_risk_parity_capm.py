# -*- coding: utf-8 -*-
"""
리스크 패리티(Risk Parity) 비중 산정 + CAPM 기대수익률 도출
------------------------------------------------------------
- 종목코드(티커)를 입력하면 과거 가격 데이터를 받아
  1) 리스크 패리티 기반으로 종목별 비중을 산정하고
  2) CAPM 기반으로 종목별/포트폴리오 기대수익률을 도출합니다.

필요 패키지:
    pip install yfinance numpy pandas scipy

티커 표기 예시:
    - 미국주식:  AAPL, MSFT, NVDA ...
    - 한국주식:  005930.KS (삼성전자), 000660.KS (SK하이닉스), 035720.KS (카카오)
                코스닥은 .KQ  예) 247540.KQ (에코프로비엠)
    - 시장지수:  ^GSPC (S&P500),  ^KS11 (KOSPI),  ^IXIC (나스닥)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


# ============================================================
# 1. 설정 (이 부분만 바꾸면 됩니다)
# ============================================================
CONFIG = {
    # 포트폴리오에 담을 종목코드
    "tickers": ["AAPL", "MSFT", "NVDA", "JPM", "XOM"],

    # CAPM 베타 산정에 쓸 시장지수 (벤치마크)
    "market_index": "^GSPC",          # 한국 포트폴리오면 "^KS11"

    # 데이터 기간
    "period": "3y",                   # "1y", "3y", "5y", "10y", "max"
    "interval": "1d",                 # "1d"(일간) / "1wk"(주간)

    # 연율화 계수 (일간=252, 주간=52)
    "annualize": 252,

    # 공분산 추정 방식: "ewma"(지수가중) 또는 "sample"(단순 표본공분산)
    "cov_method": "ewma",
    # EWMA 감쇠계수 λ (1에 가까울수록 과거를 오래 반영)
    #   일간 권장 0.94, 주간/월간 권장 0.97 (RiskMetrics)
    "ewma_lambda": 0.94,

    # CAPM 파라미터 (연율 기준)
    "risk_free_rate": 0.035,          # 무위험수익률 (예: 국채 3.5%)
    # 시장 기대수익률: None 이면 시장지수 과거 평균으로 추정,
    # 숫자를 넣으면 그 값을 사용 (예: 0.09 → 연 9%)
    "market_return": None,
}


# ============================================================
# 2. 데이터 수집
# ============================================================
def download_prices(tickers, period, interval):
    """야후 파이낸스에서 조정종가를 받아 DataFrame으로 반환."""
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,     # 배당/액면분할 반영된 조정종가
        progress=False,
    )["Close"]

    # 단일 종목이면 Series 로 오므로 DataFrame 으로 변환
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

    data = data.dropna(how="all").ffill().dropna()
    if data.empty:
        raise ValueError("가격 데이터를 받지 못했습니다. 티커/기간을 확인하세요.")
    return data


def download_prices_range(tickers, start, end, interval="1d"):
    """
    시작~종료일 구간의 조정종가를 반환.
    야후 규칙상 end 는 '미포함'이므로 호출부에서 end+1일을 넘기는 것을 권장.
    상장 전이라 데이터가 없는 종목은 앞부분이 NaN 으로 남는다(호출부에서 처리).
    """
    data = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False,
    )["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])
    data = data.dropna(how="all").ffill()
    if data.empty:
        raise ValueError("해당 기간의 가격 데이터를 받지 못했습니다. 날짜/티커를 확인하세요.")
    return data


def to_returns(prices):
    """단순 일간(주간) 수익률."""
    return prices.pct_change().dropna()


def get_names(tickers):
    """
    티커 → 종목명 매핑(dict)을 반환.
      - 한국 상장 종목(.KS=코스피, .KQ=코스닥)은 pykrx로 한글명 조회
      - 그 외(미국 등)는 야후 파이낸스로 영어명 조회
    조회에 실패한 종목은 티커를 그대로 이름으로 사용한다.
    (종목 수가 많으면 다소 느릴 수 있음 — 앱에서는 결과를 캐시함)
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # pykrx는 선택적 의존성: 없으면 한국 종목도 야후(영어)로 폴백
    try:
        from pykrx import stock as _krx
    except Exception:
        _krx = None

    names = {}
    for t in tickers:
        name = None
        is_kr = t.upper().endswith((".KS", ".KQ"))

        # 1) 한국 종목 → pykrx 한글명
        if is_kr and _krx is not None:
            code = t.split(".")[0]
            try:
                kn = _krx.get_market_ticker_name(code)
                if isinstance(kn, str) and kn.strip():
                    name = kn.strip()
            except Exception:
                name = None

        # 2) 해외 종목 또는 한글 조회 실패 → 야후 파이낸스
        if name is None:
            try:
                info = yf.Ticker(t).info
                name = info.get("longName") or info.get("shortName")
            except Exception:
                name = None

        names[t] = name if name else t
    return names


# ============================================================
# 2-b. 공분산 추정 (표본 / EWMA)
# ============================================================
def ewma_cov(returns, lam=0.94, annualize=252):
    """
    지수가중이동평균(EWMA) 공분산 행렬 (연율화) — RiskMetrics 방식.
    최근 수익률에 더 큰 가중치를 부여해 변동성 국면 변화를 빠르게 반영한다.

    lam (λ) : 감쇠계수 (0<λ<1). 1에 가까울수록 과거를 오래 반영.
              일간 0.94, 주간/월간 0.97 권장.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("ewma_lambda(λ)는 0과 1 사이여야 합니다.")

    X = returns.values
    T, n = X.shape
    if T < 2:
        raise ValueError("EWMA 추정에 데이터가 부족합니다.")

    # 가중치: 가장 최근 관측치가 가장 큼, 합 = 1
    ages = np.arange(T - 1, -1, -1)        # 최근=0, 과거일수록 커짐
    weights = (1.0 - lam) * lam ** ages
    weights /= weights.sum()

    # 가중 평균으로 중심화 후 가중 공분산 계산
    mu = weights @ X
    Xc = X - mu
    cov = (Xc * weights[:, None]).T @ Xc

    return cov * annualize


def estimate_cov(returns, method="ewma", lam=0.94, annualize=252):
    """공분산 추정 디스패처: 'ewma' 또는 'sample'."""
    if method == "ewma":
        return ewma_cov(returns, lam, annualize)
    elif method == "sample":
        return returns.cov().values * annualize
    raise ValueError(f"알 수 없는 cov_method: {method} (ewma/sample 중 선택)")


# ============================================================
# 3. 리스크 패리티 비중 산정
# ============================================================
def risk_parity_weights(cov):
    """
    각 종목의 '위험 기여도(Risk Contribution)'가 동일해지도록 비중을 최적화.
    제약: 비중 합 = 1, 비중 >= 0 (롱 온리)
    """
    n = cov.shape[0]

    def risk_contributions(w):
        port_var = w @ cov @ w                 # 포트폴리오 분산
        marginal = cov @ w                     # 한계 위험 기여
        rc = w * marginal                      # 위험 기여도 (합 = port_var)
        return rc, port_var

    def objective(w):
        rc, port_var = risk_contributions(w)
        target = port_var / n                  # 목표: 모두 동일
        return np.sum((rc - target) ** 2)

    # 초기값: 역변동성(inverse-volatility) 비중 → 수렴이 빠름
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    x0 = inv_vol / inv_vol.sum()

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((1e-6, 1.0) for _ in range(n))

    result = minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        print(f"[경고] 최적화 미수렴: {result.message} (초기값 사용)")
        return x0
    return result.x


# ============================================================
# 4. CAPM 기대수익률
# ============================================================
def capm_expected_returns(asset_returns, market_returns, rf, market_return, ann):
    """
    E[Ri] = Rf + βi * (E[Rm] - Rf)
    βi = Cov(Ri, Rm) / Var(Rm)
    rf, market_return : 연율 기준
    """
    mkt = market_returns.squeeze()
    var_m = mkt.var()

    # 시장 기대수익률 미지정 시 과거 평균으로 추정 (연율화)
    if market_return is None:
        market_return = mkt.mean() * ann

    betas, exp_rets = {}, {}
    for col in asset_returns.columns:
        cov_im = asset_returns[col].cov(mkt)
        beta = cov_im / var_m
        betas[col] = beta
        exp_rets[col] = rf + beta * (market_return - rf)

    return (
        pd.Series(betas, name="beta"),
        pd.Series(exp_rets, name="capm_expected_return"),
        market_return,
    )


# ============================================================
# 5. 실행 / 리포트
# ============================================================
def build_portfolio(cfg):
    tickers = cfg["tickers"]
    ann = cfg["annualize"]

    # --- 데이터 ---
    prices = download_prices(tickers, cfg["period"], cfg["interval"])
    mkt_price = download_prices(cfg["market_index"], cfg["period"], cfg["interval"])

    asset_ret = to_returns(prices)[tickers]      # 컬럼 순서 보존
    mkt_ret = to_returns(mkt_price)

    # 날짜 정렬 (교집합)
    idx = asset_ret.index.intersection(mkt_ret.index)
    asset_ret, mkt_ret = asset_ret.loc[idx], mkt_ret.loc[idx]

    # --- 리스크 패리티 ---
    cov_annual = estimate_cov(
        asset_ret, cfg["cov_method"], cfg["ewma_lambda"], ann
    )
    weights = risk_parity_weights(cov_annual)
    w = pd.Series(weights, index=tickers, name="weight")

    # --- CAPM ---
    betas, capm_ret, mkt_exp = capm_expected_returns(
        asset_ret, mkt_ret, cfg["risk_free_rate"], cfg["market_return"], ann
    )

    # --- 포트폴리오 지표 ---
    port_var = weights @ cov_annual @ weights
    port_vol = np.sqrt(port_var)
    port_exp_ret = float((w * capm_ret).sum())

    # 위험 기여도(검증용): 합 = 1
    rc = (weights * (cov_annual @ weights)) / port_var

    table = pd.DataFrame({
        "weight": w,
        "risk_contribution": pd.Series(rc, index=tickers),
        "beta": betas,
        "capm_expected_return": capm_ret,
    })

    # --- 출력 ---
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n" + "=" * 60)
    print(" 리스크 패리티 비중 + CAPM 기대수익률")
    print("=" * 60)
    cov_label = (f"EWMA(λ={cfg['ewma_lambda']})"
                 if cfg["cov_method"] == "ewma" else "표본공분산")
    print(f" 기간: {cfg['period']} / 공분산: {cov_label}"
          f" / 무위험수익률: {cfg['risk_free_rate']:.2%}"
          f" / 시장 기대수익률: {mkt_exp:.2%}\n")
    print(table.to_string(formatters={
        "weight": "{:.2%}".format,
        "risk_contribution": "{:.2%}".format,
        "beta": "{:.3f}".format,
        "capm_expected_return": "{:.2%}".format,
    }))
    print("-" * 60)
    print(f" 포트폴리오 기대수익률 (CAPM)   : {port_exp_ret:.2%}")
    print(f" 포트폴리오 연변동성(리스크)     : {port_vol:.2%}")
    sharpe = (port_exp_ret - cfg["risk_free_rate"]) / port_vol
    print(f" 샤프지수 (사전적/ex-ante)      : {sharpe:.3f}")
    print("=" * 60 + "\n")

    return table, port_exp_ret, port_vol


if __name__ == "__main__":
    build_portfolio(CONFIG)
