# -*- coding: utf-8 -*-
"""
팩터 기반 한국 주식 스크리너
------------------------------------------------------------
데이터 소스:
  - 네이버 금융 시가총액 페이지: 종목명 / 시가총액 / PER / ROE
  - 야후 파이낸스: 최근 6개월 수익률 (모멘텀)

팩터 (백분위 순위 0~100 평균 = 종합점수):
  - 저PER : 이익 대비 저평가 (낮을수록 좋음, 적자기업 제외)
  - 고ROE : 수익성 우량 (높을수록 좋음)
  - 모멘텀: 최근 6개월 수익률 (높을수록 좋음)
"""

import re
import time
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

_SUFFIX = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}
_SOSOK = {"KOSPI": 0, "KOSDAQ": 1}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 미국: Finviz 스크리너의 지수 필터 이름
_US_INDEX_FILTER = {
    "SP500": "S&P 500",
    "NASDAQ100": "NASDAQ 100",
    "DJIA": "DJIA",
}

def _fetch_market_pages(market, n_stocks):
    """네이버 금융 시가총액 목록에서 시총 상위 n_stocks개(보통주만)를 수집."""
    rows = []
    page = 1
    max_pages = (n_stocks // 50) * 3 + 5          # 우선주 등 걸러낼 여유분
    while len(rows) < n_stocks and page <= max_pages:
        url = (f"https://finance.naver.com/sise/sise_market_sum.naver"
               f"?sosok={_SOSOK[market]}&page={page}")
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="type_2")
        if table is None:
            break
        # 헤더(N, 종목명, 현재가, ..., PER, ROE, 토론실)와 데이터 행의 td가
        # 같은 순서로 정렬되므로 헤더 인덱스를 그대로 사용
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        try:
            i_cap = headers.index("시가총액")
            i_per = headers.index("PER")
            i_roe = headers.index("ROE")
        except ValueError:
            raise RuntimeError("네이버 금융 페이지 구조가 바뀐 것 같습니다.")

        found_any = False
        for tr in table.find_all("tr"):
            a = tr.find("a", class_="tltle")
            if a is None:
                continue
            found_any = True
            m = re.search(r"code=(\d{6})", a.get("href", ""))
            if not m:
                continue
            code = m.group(1)
            if not code.endswith("0"):             # 보통주만 (우선주 제외)
                continue
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) != len(headers):           # 구분선 등 형식이 다른 행은 제외
                continue

            def num(s):
                s = s.replace(",", "")
                try:
                    return float(s)
                except ValueError:
                    return np.nan                   # 'N/A' 등

            rows.append({
                "코드": code,
                "종목명": a.get_text(strip=True),
                "시가총액(억원)": num(tds[i_cap]),
                "PER": num(tds[i_per]),
                "ROE": num(tds[i_roe]),
                "_시장": market,
            })
            if len(rows) >= n_stocks:
                break
        if not found_any:
            break
        page += 1
        time.sleep(0.2)                             # 서버 부담 완화

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("네이버 금융에서 데이터를 받지 못했습니다. 잠시 후 다시 시도해주세요.")
    return df.set_index("코드")


def fetch_universe(markets=("KOSPI",), size_top=200):
    """
    시장별로 시총 상위 종목을 모아 하나의 유니버스로 반환.
    여러 시장이면 각 시장 상위를 합친 뒤 시총순으로 다시 잘라낸다.
    """
    frames = [_fetch_market_pages(m, size_top) for m in markets]
    df = pd.concat(frames)
    df = df.sort_values("시가총액(억원)", ascending=False).head(size_top)
    df["티커"] = [c + _SUFFIX[m] for c, m in zip(df.index, df["_시장"])]
    return df


def _to_percent(s):
    """
    ROE 등 비율 컬럼을 % 단위 숫자로 통일.
    (finviz 파서 버전에 따라 0.23 같은 소수 또는 '23.45%' 문자열로 올 수 있음)
    """
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False)
    v = pd.to_numeric(s, errors="coerce")
    if v.abs().median() < 1.5:        # 소수(0.23)로 판단되면 %로 환산
        v = v * 100.0
    return v


def fetch_us_universe(index_name="SP500"):
    """
    Finviz 스크리너에서 미국 지수 구성종목의 이름/시총/PER(Overview)과
    ROE(Financial)를 수집해 표준 컬럼으로 반환. 인덱스는 티커.
    """
    from finvizfinance.screener.overview import Overview
    from finvizfinance.screener.financial import Financial

    filt = {"Index": _US_INDEX_FILTER[index_name]}

    ov = Overview()
    ov.set_filter(filters_dict=filt)
    df_o = ov.screener_view(order="Market Cap.", ascend=False, verbose=0)

    fin = Financial()
    fin.set_filter(filters_dict=filt)
    df_f = fin.screener_view(order="Market Cap.", ascend=False, verbose=0)

    if df_o is None or df_o.empty:
        raise RuntimeError("Finviz에서 데이터를 받지 못했습니다. 잠시 후 다시 시도해주세요.")

    # 페이지 수집/병합 과정에서 같은 종목이 중복될 수 있어 제거
    df_o = df_o.drop_duplicates(subset="Ticker")
    df_f = df_f.drop_duplicates(subset="Ticker")
    df = df_o.merge(df_f[["Ticker", "ROE"]], on="Ticker", how="left")

    cap = pd.to_numeric(df["Market Cap"], errors="coerce")
    out = pd.DataFrame({
        "종목명": df["Company"],
        "PER": pd.to_numeric(df["P/E"], errors="coerce"),
        "ROE": _to_percent(df["ROE"]),
        "시가총액($B)": (cap / 1e9).round(1),
        "티커": df["Ticker"],
        "_시장": "US",
    })
    out.index = df["Ticker"]
    out.index.name = "코드"
    return out


def rerank_universe_asof(df, as_of, size_top):
    """
    현재 유니버스의 시가총액을 as_of 시점 주가 비율(당시가/현재가)로 되돌려
    '당시 시총 순위'를 근사 재구성하고 상위 size_top개를 반환.
      - as_of에 상장돼 있지 않던 종목(당시 가격 없음)은 자동 제외
      - 주가 비율 근사라 증자/자사주 소각 등 주식 수 변화는 반영 못 함
      - 당시 상장돼 있었으나 이후 상장폐지된 종목은 후보에 넣을 수 없음 (한계)
    """
    tickers = list(df["티커"])
    end_ts = pd.Timestamp(as_of)

    px_then = yf.download(
        tickers,
        start=(end_ts - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=False, progress=False,
    )["Close"]
    px_now = yf.download(tickers, period="5d", interval="1d",
                         auto_adjust=False, progress=False)["Close"]
    if isinstance(px_then, pd.Series):
        px_then = px_then.to_frame(name=tickers[0])
    if isinstance(px_now, pd.Series):
        px_now = px_now.to_frame(name=tickers[0])

    ratio = {}
    for t in tickers:
        p0 = px_then[t].dropna() if t in px_then.columns else pd.Series(dtype=float)
        p1 = px_now[t].dropna() if t in px_now.columns else pd.Series(dtype=float)
        ratio[t] = (p0.iloc[-1] / p1.iloc[-1]) \
            if len(p0) > 0 and len(p1) > 0 and p1.iloc[-1] > 0 else np.nan

    out = df.copy()
    cap_col = next((c for c in out.columns if c.startswith("시가총액")), None)
    if cap_col is None:
        raise ValueError("시가총액 컬럼이 없습니다.")
    out[cap_col] = out[cap_col] * out["티커"].map(ratio)
    out = out.dropna(subset=[cap_col])
    out = out.sort_values(cap_col, ascending=False).head(size_top)
    out[cap_col] = out[cap_col].round(1 if cap_col.endswith("($B)") else 0)
    return out


def add_price_factors(df, period="1y", as_of=None):
    """
    야후 파이낸스에서 1년치 주가/거래량을 일괄 다운로드해 가격 기반 팩터를 추가.
      - 1/3/6개월 수익률(%)
      - 52주고점대비(%)  : 현재가가 52주 최고가에서 얼마나 떨어져 있나 (0에 가까울수록 신고가 근접)
      - 변동성(%)        : 최근 3개월 일간수익률 표준편차의 연율화
      - 거래대금비율     : 최근 20일 평균 거래대금 / 직전 60일 평균 (1보다 크면 관심 유입)

    as_of: 'YYYY-MM-DD' 문자열/날짜를 주면 그 시점까지의 데이터만으로 계산
           (과거 시점 스크리닝 재현용 — 해당일 이후 정보는 전혀 쓰지 않음)
    """
    tickers = list(df["티커"])
    if as_of is None:
        raw = yf.download(tickers, period=period, interval="1d",
                          auto_adjust=True, progress=False)
    else:
        end_ts = pd.Timestamp(as_of)
        start = (end_ts - pd.Timedelta(days=370)).strftime("%Y-%m-%d")
        end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # 야후 end 미포함
        raw = yf.download(tickers, start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
    close, vol = raw["Close"], raw["Volume"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
        vol = vol.to_frame(name=tickers[0])

    r1, r3, r6, near, sig, turn = {}, {}, {}, {}, {}, {}
    for t in tickers:
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        n = len(s)
        if n < 25:
            r1[t] = r3[t] = r6[t] = near[t] = sig[t] = turn[t] = np.nan
            continue
        last = s.iloc[-1]
        r1[t] = (last / s.iloc[-22] - 1.0) * 100.0
        r3[t] = (last / s.iloc[-64] - 1.0) * 100.0 if n >= 64 else np.nan
        r6[t] = (last / s.iloc[-127] - 1.0) * 100.0 if n >= 127 else np.nan
        near[t] = (last / s.max() - 1.0) * 100.0
        sig[t] = s.pct_change().tail(63).std() * np.sqrt(252) * 100.0

        if t in vol.columns:
            money = (close[t] * vol[t]).dropna()
            if len(money) >= 80:
                recent = money.tail(20).mean()
                prior = money.iloc[-80:-20].mean()
                turn[t] = recent / prior if prior > 0 else np.nan
            else:
                turn[t] = np.nan
        else:
            turn[t] = np.nan

    out = df.copy()
    out["1개월수익률(%)"] = out["티커"].map(r1)
    out["3개월수익률(%)"] = out["티커"].map(r3)
    out["6개월수익률(%)"] = out["티커"].map(r6)
    out["52주고점대비(%)"] = out["티커"].map(near)
    out["변동성(%)"] = out["티커"].map(sig)
    out["거래대금비율"] = out["티커"].map(turn)
    return out


# 하위 호환용 별칭
def add_momentum(df, period="1y"):
    return add_price_factors(df, period)


def min_corr_subset(returns, k=5):
    """
    일간 수익률 DataFrame에서 '평균 쌍별 상관계수'가 가장 낮은 k개 종목 조합을
    완전탐색으로 찾는다. 후보가 20개면 C(20,5)=15,504가지라 수 초 안에 끝난다.

    반환: (선택된 티커 리스트, 선택 조합의 평균 상관, 전체 후보의 평균 상관)
    """
    # 데이터가 너무 짧은 종목(신규상장 등)은 상관 추정이 불안정하므로 제외
    rets = returns.dropna(axis=1, thresh=max(30, len(returns) // 2))
    corr = rets.corr()
    tickers = list(corr.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("상관관계를 계산할 종목이 부족합니다.")

    c = corr.values
    iu_all = np.triu_indices(n, 1)
    overall_avg = float(np.nanmean(c[iu_all]))

    if n <= k:
        return tickers, overall_avg, overall_avg

    iu_k = np.triu_indices(k, 1)
    best_idx, best_avg = None, np.inf
    for combo in combinations(range(n), k):
        sub = c[np.ix_(combo, combo)]
        avg = np.nanmean(sub[iu_k])
        if avg < best_avg:
            best_avg, best_idx = avg, combo

    return [tickers[i] for i in best_idx], float(best_avg), overall_avg


# 팩터 정의: 이름 -> (데이터 컬럼, 높을수록 좋은지, 양수만 유효한지)
FACTORS = {
    "저PER":          ("PER", False, True),
    "고ROE":          ("ROE", True, False),
    "모멘텀(6개월)":   ("6개월수익률(%)", True, False),
    "모멘텀(3개월)":   ("3개월수익률(%)", True, False),
    "모멘텀(1개월)":   ("1개월수익률(%)", True, False),
    "52주 신고가 근접": ("52주고점대비(%)", True, False),
    "거래대금 증가":    ("거래대금비율", True, False),
}


def screen_stocks(df, selected, top_n=10):
    """
    선택한 팩터(FACTORS 키 목록)의 백분위 순위를 평균해 상위 top_n 종목 반환.
    선택 팩터 값이 없는 종목(적자기업의 PER 등)과 스팩/리츠는 제외.
    """
    if not selected:
        raise ValueError("팩터를 1개 이상 선택하세요.")

    d = df.copy()
    d = d[~d.index.duplicated(keep="first")]      # 혹시 남은 중복 방어
    d = d[~d["종목명"].str.contains("스팩|리츠", na=False)]

    ranks = {}
    for name in selected:
        if name not in FACTORS:
            raise ValueError(f"알 수 없는 팩터: {name}")
        col, higher_is_better, positive_only = FACTORS[name]
        if col not in d.columns:
            raise ValueError(f"'{name}' 팩터에 필요한 데이터({col})가 없습니다.")
        v = d[col]
        if positive_only:
            v = v.where(v > 0)
        ranks[name] = v.rank(pct=True, ascending=higher_is_better)

    rank_df = pd.DataFrame(ranks).dropna()
    d = d.loc[rank_df.index]
    d["종합점수"] = (rank_df.mean(axis=1) * 100.0).round(1)
    d = d.sort_values("종합점수", ascending=False).head(top_n)

    # 표시 열: 기본 정보 + 선택한 팩터의 데이터 열 + 시가총액
    factor_cols = []
    for name in selected:
        col = FACTORS[name][0]
        if col not in factor_cols:
            factor_cols.append(col)
    cap_cols = [c for c in d.columns if c.startswith("시가총액")]
    cols = ["티커", "종목명", "종합점수"] + factor_cols + cap_cols

    res = d[cols].copy()
    for col in factor_cols:
        res[col] = res[col].round(2 if col == "거래대금비율" else 1)
    if "시가총액(억원)" in res.columns:
        res["시가총액(억원)"] = res["시가총액(억원)"].astype("Int64")
    res.index = pd.RangeIndex(1, len(res) + 1, name="순위")
    return res
