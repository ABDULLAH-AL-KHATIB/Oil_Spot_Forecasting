"""IRFA corrected Brent/WTI forecasting workflow.

DEFAULT MODE = AUDIT: reproduces the reported metrics and forecast tests from
frozen archived predictions supplied in the supplementary package. This is the
exact route for verifying the estimates printed in the paper.

OPTIONAL MODE = REFIT: re-estimates the six model families from the frozen aligned
price snapshot (or fresh EIA-origin DataHub files), then writes metrics, forecast
tests, event-window diagnostics, delayed P&L diagnostics, block-bootstrap
intervals. SHAP attribution is excluded from this release because the archived attribution outputs could not be reconciled. The workflow records software
versions, uses market-appropriate targets and retains negative WTI prices.
Because model libraries and archived feature-engineering code can differ across
environments, REFIT is a new estimation run, not a bitwise replay of the frozen
estimates.

No LSTM or LightGBM results are reported. The main specification is an
observation-date exercise; historical publication timestamps are unavailable.
"""
from __future__ import annotations
import os, sys, json, hashlib, zipfile, platform, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

# -------------------------- User configuration --------------------------
MODE = os.environ.get("IRFA_MODE", "AUDIT").upper()   # AUDIT or REFIT
REFRESH_LIVE_DATA = False                              # True fetches live CSVs
SEED = 42
MIN_TRAIN = 1000
NW_LAGS = 5
BOOTSTRAP_REPS = 1000
BLOCK_LENGTH = 5
TRANSACTION_COST_BPS = [5, 10, 20]
BRENT_URL = "https://datahub.io/core/oil-prices/_r/-/data/brent-daily.csv"
WTI_URL = "https://datahub.io/core/oil-prices/_r/-/data/wti-daily.csv"
EVENTS = {
    "Abqaiq attack": ("2019-09-16", "U.S. EIA (2019)"),
    "Russia-Ukraine invasion": ("2022-02-24", "U.S. EIA (2022)"),
    "Israel-Gaza attack": ("2023-10-09", "IEA (2023)"),
    "Iran-Hormuz escalation": ("2026-03-04", "Congressional Research Service (2026a, 2026b)"),
}
SEED = int(SEED)
np.random.seed(SEED)

# Paths: place this script in the supplied Colab folder. In Colab, upload or
# unzip the complete submission archive and set ROOT to its extracted folder.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = Path(os.environ.get("IRFA_ROOT", HERE.parent))
SUPP = ROOT / "Supplementary" / "Corrected_Run_2026-10-30"
OUT = ROOT / "Colab_Reproduction_Output"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------- Utilities -------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def hac_mean_test(d: np.ndarray, lags: int = NW_LAGS, two_sided: bool = True):
    """HAC test for E[d]=0; HLN adjustment for one-step forecast horizon.

    Sign convention in the paper: d = candidate loss - benchmark loss.
    Negative mean d favours the candidate; positive mean d favours benchmark.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return {"n": n, "mean_diff": np.nan, "hac_stat": np.nan, "p_value": np.nan}
    fit = sm.OLS(d, np.ones((n, 1))).fit()
    cov = cov_hac(fit, nlags=min(lags, n - 1))
    se = float(np.sqrt(max(cov[0, 0], 0)))
    stat0 = float(d.mean() / se) if se > 0 else np.nan
    hln = float(np.sqrt((n - 1) / n))  # h=1 Harvey-Leybourne-Newbold factor
    stat = stat0 * hln
    p = float(2 * stats.t.sf(abs(stat), df=n - 1)) if two_sided else float(stats.t.sf(stat, df=n - 1))
    return {"n": n, "mean_diff": float(d.mean()), "hac_stat": stat, "p_value": p}

def holm_adjust(frame: pd.DataFrame, family_cols):
    out = frame.copy()
    out["p_holm"] = np.nan
    for _, idx in out.groupby(family_cols, dropna=False).groups.items():
        vals = out.loc[idx, "p_value"].to_numpy(float)
        ok = np.isfinite(vals)
        if ok.any():
            out.loc[np.asarray(list(idx))[ok], "p_holm"] = multipletests(vals[ok], method="holm")[1]
    return out

def moving_block_mean_ci(x, reps=BOOTSTRAP_REPS, block=BLOCK_LENGTH, seed=SEED):
    """Vectorised circular moving-block bootstrap; same RNG draw order as the loop form."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 30: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); n = len(x); nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(reps, -1)[:, :n]
    vals = x[idx].mean(axis=1)
    return tuple(np.quantile(vals, [.025, .975]).tolist())

def metrics_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Recalculate target-space and reconstructed price-space forecast metrics."""
    rows=[]
    for (market, model, spec), g in pred.groupby(["market", "model", "specification"]):
        y=g.actual_target.to_numpy(float); f=g.pred_target.to_numpy(float)
        # Direction uses non-zero realised target and non-neutral candidate forecast.
        active=(np.abs(y)>1e-12)&(np.abs(f)>1e-12)
        da=float(np.mean(np.sign(y[active])==np.sign(f[active]))) if active.any() else np.nan
        origin=g.origin_price.to_numpy(float)
        actual_price=g.actual_next_price.to_numpy(float)
        pred_price=origin*np.exp(f) if market=="Brent" else origin+f
        price_err=actual_price-pred_price
        is_zero_change_benchmark = model in {"RandomWalk", "RandomWalk_LaggedCross"}
        neutral_share = np.nan if is_zero_change_benchmark else float(np.mean(np.abs(f)<=1e-12))
        rows.append({"market":market,"model":model,"specification":spec,"n":len(g),
                     "target_variable":g.target_variable.iloc[0],"target_unit":g.target_unit.iloc[0],
                     "target_RMSE":float(np.sqrt(np.mean((y-f)**2))),
                     "target_MAE":float(np.mean(np.abs(y-f))),
                     "directional_accuracy":da,"directional_n":int(active.sum()),
                     "directional_coverage":float(active.sum()/len(g)),
                     "neutral_forecast_share":neutral_share,
                     "price_RMSE_USD_per_bbl":float(np.sqrt(np.mean(price_err**2))),
                     "price_MAE_USD_per_bbl":float(np.mean(np.abs(price_err))),
                     "mean_forecast_error":float(np.mean(f-y))})
    return pd.DataFrame(rows)

def build_dm_family(pred: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for market in sorted(pred.market.unique()):
        for spec in ["main", "lagged_crossmarket"]:
            sub=pred[(pred.market==market)&(pred.specification==spec)]
            if sub.empty: continue
            wide=sub.pivot(index="target_date",columns="model",values=["actual_target","pred_target"])
            y=wide["actual_target"].iloc[:,0].to_numpy(float)
            base=wide["pred_target"]["RandomWalk" if spec=="main" else "RandomWalk_LaggedCross"].to_numpy(float)
            candidates=[m for m in wide["pred_target"].columns if m not in ["RandomWalk","RandomWalk_LaggedCross","HistoricalMean","HistoricalMean_LaggedCross"]]
            # Main point-loss family, and lagged-only family (only Ridge/XGBoost).
            for model in candidates:
                f=wide["pred_target"][model].to_numpy(float)
                res=hac_mean_test((y-f)**2-(y-base)**2)
                rows.append({"market":market,"specification":spec,"model_vs":"Zero-change benchmark","model":model,
                             "loss":"squared_target_error","sign_convention":"candidate loss minus benchmark loss; negative favours candidate",**res,"common_active_n":np.nan})
            # Directional candidate vs historical mean on same non-zero realised days.
            hm_name="HistoricalMean" if spec=="main" else "HistoricalMean_LaggedCross"
            hm=wide["pred_target"][hm_name].to_numpy(float)
            active=(np.abs(y)>1e-12)&(np.abs(hm)>1e-12)
            for model in candidates:
                f=wide["pred_target"][model].to_numpy(float)
                both=active&(np.abs(f)>1e-12)
                if both.sum()<30: continue
                loss_m=(np.sign(f[both])!=np.sign(y[both])).astype(float)
                loss_h=(np.sign(hm[both])!=np.sign(y[both])).astype(float)
                res=hac_mean_test(loss_m-loss_h)
                rows.append({"market":market,"specification":spec,"model_vs":"Historical-mean directional benchmark","model":model,
                             "loss":"directional_0_1_loss_common_active_days","sign_convention":"candidate directional loss minus benchmark loss; negative favours candidate",**res,"common_active_n":int(both.sum())})
    out=pd.DataFrame(rows)
    out=holm_adjust(out,["market","specification","loss"])
    return out

def build_clark_west_ar5(pred: pd.DataFrame) -> pd.DataFrame:
    """One-sided Clark-West adjusted test for AR(5) versus zero change only."""
    rows=[]
    for market in sorted(pred.market.unique()):
        sub=pred[(pred.market==market)&(pred.specification=="main")]
        wide=sub.pivot(index="target_date",columns="model",values=["actual_target","pred_target"])
        y=wide["actual_target"].iloc[:,0].to_numpy(float)
        f0=wide["pred_target"]["RandomWalk"].to_numpy(float)
        f1=wide["pred_target"]["AR5"].to_numpy(float)
        e0=y-f0; e1=y-f1
        # Positive adjusted differential favours the larger AR(5) model.
        adjusted=e0**2-(e1**2-(f0-f1)**2)
        res=hac_mean_test(adjusted,two_sided=False)
        rows.append({"market":market,"model":"AR5","model_vs":"Zero-change benchmark",
                     "family":"AR5 only; separate nested-linear family","n":res["n"],
                     "mean_adjusted_diff":res["mean_diff"],"hac_stat":res["hac_stat"],
                     "p_one_sided":res["p_value"]})
    return pd.DataFrame(rows)


def build_event_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct exploratory pre-event and event-to-plus-20 metrics."""
    rows=[]
    main=pred[pred.specification=="main"].copy()
    for market in sorted(main.market.unique()):
        market_panel=main[main.market==market]
        unique_dates=sorted(pd.to_datetime(market_panel.target_date).unique())
        for event,(anchor_txt,source) in EVENTS.items():
            anchor=pd.Timestamp(anchor_txt)
            pre_dates=[d for d in unique_dates if d<anchor][-20:]
            post_dates=[d for d in unique_dates if d>=anchor][:21]
            for period,dates in [("pre_event_20",pre_dates),("event_to_plus20",post_dates)]:
                sub=market_panel[market_panel.target_date.isin(dates)]
                for model,g in sub.groupby("model"):
                    y=g.actual_target.to_numpy(float); f=g.pred_target.to_numpy(float)
                    active=(np.abs(y)>1e-12)&(np.abs(f)>1e-12)
                    da=float(np.mean(np.sign(y[active])==np.sign(f[active]))) if active.any() else np.nan
                    rows.append({"market":market,"event":event,"event_anchor":anchor.strftime("%Y-%m-%d"),
                                 "source":source,"period":period,"model":model,"n":len(g),
                                 "target_variable":g.target_variable.iloc[0],"target_unit":g.target_unit.iloc[0],
                                 "target_RMSE":float(np.sqrt(np.mean((y-f)**2))),
                                 "target_MAE":float(np.mean(np.abs(y-f))),
                                 "directional_accuracy":da,"directional_n":int(active.sum())})
    return pd.DataFrame(rows)


def build_one_barrel_pnl(pred: pd.DataFrame) -> pd.DataFrame:
    """Delayed one-barrel spot P&L proxy with entry, rebalance and terminal-exit costs.

    The signal applied to target interval t is the previous interval's forecast.
    Turnover includes the initial entry from flat, all subsequent position changes,
    and liquidation of the final position at the last target price.
    """
    rows=[]
    main=pred[pred.specification=="main"]
    for (market,model),g in main.groupby(["market","model"]):
        g=g.sort_values("target_date").copy()
        # One-observation delayed position: previous forecast's sign.
        pos=np.sign(g.pred_target.shift(1).fillna(0).to_numpy(float))
        entry_rebalance_turnover=np.abs(np.diff(np.r_[0.0,pos]))
        origin=g.origin_price.to_numpy(float)
        actual_next=g.actual_next_price.to_numpy(float)
        if market=="Brent":
            realised_dollar_move=origin*np.expm1(g.actual_target.to_numpy(float))
        else:
            realised_dollar_move=g.actual_target.to_numpy(float)
        for bps in TRANSACTION_COST_BPS:
            rate=bps*1e-4
            entry_rebalance_cost=rate*np.abs(origin)*entry_rebalance_turnover
            net=pos*realised_dollar_move-entry_rebalance_cost
            # Close any final open position at the last observed target price.
            terminal_turnover=abs(float(pos[-1])) if len(pos) else 0.0
            terminal_exit_cost=rate*abs(float(actual_next[-1]))*terminal_turnover if len(pos) else 0.0
            if len(net):
                net[-1]-=terminal_exit_cost
            total_turnover=entry_rebalance_turnover.copy()
            if len(total_turnover):
                total_turnover[-1]+=terminal_turnover
            cum=np.cumsum(net)
            high_water=np.maximum.accumulate(np.r_[0.0,cum])[1:]
            drawdown=cum-high_water
            sd=float(np.std(net,ddof=1))
            sharpe=float(np.mean(net)/sd*np.sqrt(252)) if sd>0 else np.nan
            rows.append({"market":market,"model":model,"specification":"main","cost_bps":bps,
                         "n":len(g),"cumulative_net_PnL_USD_per_bbl":float(net.sum()),
                         "annualized_PnL_sharpe":sharpe,
                         "max_peak_to_trough_PnL_drawdown_USD_per_bbl":float(drawdown.min()),
                         "annualized_turnover_per_unit_position":float(total_turnover.sum()/len(total_turnover)*252) if len(total_turnover) else np.nan,
                         "mean_daily_net_PnL_USD_per_bbl":float(net.mean()),
                         "initial_entry_included":True,"terminal_liquidation_included":True,
                         "terminal_exit_turnover":terminal_turnover,
                         "terminal_exit_cost_USD_per_bbl":terminal_exit_cost,
                         "cost_definition":"entry/rebalance cost = cost_bps x 1e-4 x abs(origin spot price) x turnover; final liquidation cost = cost_bps x 1e-4 x abs(last target spot price) x abs(final position)"})
    return pd.DataFrame(rows)

def build_point_bootstrap_ci(pred: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    main=pred[pred.specification=="main"]
    for market in sorted(main.market.unique()):
        sub=main[main.market==market]
        wide=sub.pivot(index="target_date",columns="model",values=["actual_target","pred_target"])
        y=wide["actual_target"].iloc[:,0].to_numpy(float)
        base=wide["pred_target"]["RandomWalk"].to_numpy(float)
        for model in ["AR5","HistoricalMean","RandomForest","Ridge","XGBoost"]:
            f=wide["pred_target"][model].to_numpy(float)
            d=(y-f)**2-(y-base)**2
            lo,hi=moving_block_mean_ci(d)
            rows.append({"market":market,"model_vs":"RandomWalk","model":model,
                         "mean_squared_loss_difference":float(np.mean(d)),
                         "block_bootstrap_95_low":lo,"block_bootstrap_95_high":hi,
                         "sign_convention":"candidate loss minus benchmark loss; negative favours candidate"})
    return pd.DataFrame(rows)


def build_directional_bootstrap_ci(pred: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    main=pred[pred.specification=="main"]
    for (market,model),g in main.groupby(["market","model"]):
        g=g.sort_values("target_date")
        y=g.actual_target.to_numpy(float); f=g.pred_target.to_numpy(float)
        eligible=(np.abs(y)>1e-12)&(np.abs(f)>1e-12)
        hit=(np.sign(y)==np.sign(f))&eligible
        n=len(y); denominator=int(eligible.sum())
        da=float(hit.sum()/denominator) if denominator else np.nan
        if denominator:
            rng=np.random.default_rng(SEED); nblocks=int(np.ceil(n/BLOCK_LENGTH))
            starts=rng.integers(0,n,size=(BOOTSTRAP_REPS,nblocks))
            idx=(starts[:,:,None]+np.arange(BLOCK_LENGTH)[None,None,:])%n
            idx=idx.reshape(BOOTSTRAP_REPS,-1)[:,:n]
            den=eligible[idx].sum(axis=1); hits=hit[idx].sum(axis=1)
            vals=np.divide(hits,den,out=np.full(BOOTSTRAP_REPS,np.nan),where=den>0)
            lo,hi=np.nanquantile(vals,[.025,.975])
        else: lo=hi=np.nan
        rows.append({"market":market,"model":model,"directional_accuracy":da,
                     "block_bootstrap_95_low":float(lo) if np.isfinite(lo) else np.nan,
                     "block_bootstrap_95_high":float(hi) if np.isfinite(hi) else np.nan,
                     "n_total":n,"directional_n":denominator,
                     "directional_coverage":float(denominator/n),"tie_share":float(1-denominator/n),
                     "neutral_forecast_rule":"excluded from direction denominator"})
    return pd.DataFrame(rows)


def compare_archived_table(calculated: pd.DataFrame, archived_path: Path, keys, numeric_cols, output_path: Path,
                           tolerance: float=1e-10) -> pd.DataFrame:
    """Fail closed if a recalculated table differs from the archived table."""
    archived=pd.read_csv(archived_path)
    merged=calculated.merge(archived,on=keys,how="outer",suffixes=("_calc","_arch"),indicator=True)
    if not (merged["_merge"]=="both").all():
        raise AssertionError(f"Row/key mismatch while checking {archived_path.name}: {merged['_merge'].value_counts().to_dict()}")
    report=merged[keys].copy()
    maxdiff=0.0
    for col in numeric_cols:
        a=pd.to_numeric(merged[f"{col}_calc"],errors="coerce").to_numpy(float)
        b=pd.to_numeric(merged[f"{col}_arch"],errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(a),np.isnan(b)):
            raise AssertionError(f"NaN-pattern mismatch in {archived_path.name}:{col}")
        valid=np.isfinite(a)&np.isfinite(b)
        diff=np.zeros(len(a)); diff[valid]=np.abs(a[valid]-b[valid])
        report[f"{col}_abs_diff"]=diff
        if diff.max(initial=0)>maxdiff: maxdiff=float(diff.max())
    report["max_abs_difference"] = report[[c for c in report.columns if c.endswith("_abs_diff")]].max(axis=1)
    report.to_csv(output_path,index=False)
    if maxdiff>tolerance:
        raise AssertionError(f"Archived output mismatch for {archived_path.name}; max absolute difference={maxdiff}")
    return report

def verify_frozen_inputs():
    predfile=SUPP/"out_of_sample_predictions_all_markets.csv"
    pricefile=ROOT/"Supplementary"/"Raw_Data"/"aligned_EIA_Brent_WTI_spot_prices.csv"
    if not predfile.exists():
        raise FileNotFoundError(f"Missing frozen forecast panel: {predfile}. Extract the supplied submission ZIP first.")
    if not pricefile.exists():
        # fallback: both original run subfolders contain aligned data
        pricefile=SUPP/"WTI_run"/"aligned_EIA_Brent_WTI_spot_prices_corrected.csv"
    pred=pd.read_csv(predfile, parse_dates=["origin_date","target_date"])
    prices=pd.read_csv(pricefile, parse_dates=["Date"])
    assert len(prices)==8425, f"Expected 8,425 aligned prices; got {len(prices)}"
    neg=prices.loc[prices.WTI<0]
    assert len(neg)==1 and float(neg.WTI.iloc[0])==-36.98 and neg.Date.iloc[0]==pd.Timestamp("2020-04-20")
    assert len(pred)==145900, f"Expected 145,900 forecast rows; got {len(pred)}"
    assert pred.target_date.nunique()==7295
    assert not pred.duplicated(["market","model","specification","target_date"]).any()
    dates={m:set(g.target_date) for m,g in pred.groupby("market")}
    assert dates["Brent"]==dates["WTI"], "Brent/WTI target dates differ"
    assert pred.groupby(["market","model","specification"]).size().eq(7295).all()
    assert (pred.target_date>pred.origin_date).all()
    print("AUDIT PASSED: 8,425 aligned prices; WTI negative observation retained; 7,295 common targets per market/model/specification.")
    print("Forecast dates:",pred.target_date.min().date(),"to",pred.target_date.max().date())
    return pred, prices

# ---------------------- Optional transparent re-fit ----------------------
def download_prices():
    import requests
    def read(url, name):
        r=requests.get(url,timeout=60); r.raise_for_status()
        df=pd.read_csv(pd.io.common.BytesIO(r.content))
        df["Date"]=pd.to_datetime(df["Date"]); df["Price"]=pd.to_numeric(df["Price"],errors="coerce")
        return df.dropna(subset=["Date","Price"]).drop_duplicates("Date").set_index("Date")["Price"].rename(name)
    b=read(BRENT_URL,"Brent"); w=read(WTI_URL,"WTI")
    # Crucial: do not filter out non-positive prices.
    return pd.concat([b,w],axis=1,join="inner").dropna().sort_index().rename_axis("Date").reset_index()

def make_features(prices: pd.DataFrame, market: str):
    """Transparent feature recipe for an optional new refit.

    Brent own move = log return; WTI own move = dollar difference. Cross-market
    moves are dollar differences; spread is Brent minus WTI in USD/bbl.
    """
    x=prices.set_index("Date").sort_index()
    other="WTI" if market=="Brent" else "Brent"
    own_move=np.log(x[market]).diff() if market=="Brent" else x[market].diff()
    other_move=x[other].diff()
    spread=x["Brent"]-x["WTI"]
    f=pd.DataFrame(index=x.index)
    f["target_move_0"]=own_move
    f["other_move_0"]=other_move
    for k in [1,2,3,4,5,10]:
        f[f"target_move_lag{k}"]=own_move.shift(k)
        f[f"other_move_lag{k}"]=other_move.shift(k)
    f["spread_0"]=spread
    for k in [1,2,5]: f[f"spread_lag{k}"]=spread.shift(k)
    for w in [5,10,20,60]:
        f[f"target_vol_{w}"]=own_move.rolling(w).std()
        f[f"other_vol_{w}"]=other_move.rolling(w).std()
        f[f"other_vol_lag1_{w}"]=other_move.shift(1).rolling(w).std()
        f[f"target_abs_move_mean_{w}"]=own_move.abs().rolling(w).mean()
    for w in [5,20]: f[f"target_momentum_{w}"]=own_move.rolling(w).sum()
    f["target_price_t"]=x[market]
    f["other_price_t"]=x[other]
    f["actual_next_price"]=x[market].shift(-1)
    f["actual_target"]=own_move.shift(-1)
    f["origin_date"]=f.index
    f["target_date"]=pd.Series(f.index,index=f.index).shift(-1)
    f=f.replace([np.inf,-np.inf],np.nan).dropna().copy()
    f["target_date"]=pd.to_datetime(f.target_date)
    f["origin_date"]=pd.to_datetime(f.origin_date)
    return f

def run_refit(prices: pd.DataFrame):
    """Re-estimate the six models; a fresh run, not a bitwise replay."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    outputs=[]
    params={
      "Brent":{"Ridge":{"alpha":50.0},"RandomForest":{"n_estimators":30,"max_depth":10,"min_samples_leaf":5,"max_features":0.8},"XGBoost":{"n_estimators":30,"max_depth":2,"learning_rate":0.04,"subsample":0.8,"colsample_bytree":0.8,"reg_lambda":5.0}},
      "WTI":{"Ridge":{"alpha":50.0},"RandomForest":{"n_estimators":30,"max_depth":10,"min_samples_leaf":5,"max_features":0.8},"XGBoost":{"n_estimators":30,"max_depth":4,"learning_rate":0.04,"subsample":0.8,"colsample_bytree":0.8,"reg_lambda":5.0}}
    }
    for market in ["Brent","WTI"]:
        f=make_features(prices,market)
        # Annual forecast blocks, retaining the target-date convention.
        f["forecast_period"]=f.target_date.dt.to_period("Y").dt.start_time
        features=[c for c in f.columns if c not in {"target_date","origin_date","target_price_t","other_price_t","actual_next_price","actual_target","forecast_period"}]
        # Match the archived lagged-cross-market selector exactly: remove only
        # contemporaneous other movement, current spread and unlagged other-vol
        # summaries; retain lagged other movements, lagged other-vol summaries
        # (other_vol_lag1_*), and lagged spread terms.
        lag_features=[]
        for c in features:
            if c in {"other_move_0", "spread_0"}:
                continue
            if c.startswith("other_vol_") and not c.startswith("other_vol_lag1_"):
                continue
            lag_features.append(c)
        first_tuned=True
        for period, block in f.groupby("forecast_period",sort=True):
            train=f[f.target_date<period]; test=block
            if len(train)<MIN_TRAIN: continue
            ytr=train.actual_target.astype(float)
            xtr=train[features].astype(float); xte=test[features].astype(float)
            # Training-only fixed hyperparameters from archived CV selection.
            pred_map={
              "RandomWalk":np.zeros(len(test)),
              "HistoricalMean":np.repeat(ytr.mean(),len(test)),
              # The lag-only sensitivity family retains the same two simple
              # benchmarks on the identical forecast dates for audit balance.
              "RandomWalk_LaggedCross":np.zeros(len(test)),
              "HistoricalMean_LaggedCross":np.repeat(ytr.mean(),len(test)),
            }
            ar_cols=["target_move_0"]+[f"target_move_lag{k}" for k in [1,2,3,4]]
            ar=Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=1e-6))]);ar.fit(train[ar_cols],ytr);pred_map["AR5"]=ar.predict(test[ar_cols])
            rr=Pipeline([("scale",StandardScaler()),("model",Ridge(**params[market]["Ridge"]))]);rr.fit(xtr,ytr);pred_map["Ridge"]=rr.predict(xte)
            rf=RandomForestRegressor(random_state=SEED,n_jobs=1,**params[market]["RandomForest"]);rf.fit(xtr,ytr);pred_map["RandomForest"]=rf.predict(xte)
            xb=XGBRegressor(objective="reg:squarederror",random_state=SEED,n_jobs=1,tree_method="hist",verbosity=0,**params[market]["XGBoost"]);xb.fit(xtr,ytr);pred_map["XGBoost"]=xb.predict(xte)
            # lagged-only Ridge and XGB sensitivity
            xtr_l=train[lag_features].astype(float);xte_l=test[lag_features].astype(float)
            rl=Pipeline([("scale",StandardScaler()),("model",Ridge(**params[market]["Ridge"]))]);rl.fit(xtr_l,ytr);pred_map["Ridge_LaggedCross"]=rl.predict(xte_l)
            xl=XGBRegressor(objective="reg:squarederror",random_state=SEED,n_jobs=1,tree_method="hist",verbosity=0,**params[market]["XGBoost"]);xl.fit(xtr_l,ytr);pred_map["XGBoost_LaggedCross"]=xl.predict(xte_l)
            for model,pp in pred_map.items():
                spec="lagged_crossmarket" if model.endswith("LaggedCross") else "main"
                outputs.append(pd.DataFrame({"market":market,"model":model,"specification":spec,"forecast_period":period.strftime('%Y-%m-%d'),"origin_date":test.origin_date.values,"target_date":test.target_date.values,"origin_price":test.target_price_t.values,"actual_next_price":test.actual_next_price.values,"actual_target":test.actual_target.values,"pred_target":np.asarray(pp),"target_variable":"log_return" if market=="Brent" else "dollar_change","target_unit":"log-return units" if market=="Brent" else "USD per barrel"}))
    result=pd.concat(outputs,ignore_index=True)
    result.to_csv(OUT/"out_of_sample_predictions_refit.csv",index=False)
    metrics=metrics_from_predictions(result);metrics.to_csv(OUT/"metrics_refit.csv",index=False)
    # Full inferential and diagnostic outputs for the new forecast panel.
    dm=build_dm_family(result); dm.to_csv(OUT/"DM_HAC_tests_refit.csv",index=False)
    cw=build_clark_west_ar5(result); cw.to_csv(OUT/"Clark_West_AR5_refit.csv",index=False)
    events=build_event_metrics(result); events.to_csv(OUT/"event_window_metrics_refit.csv",index=False)
    pnl=build_one_barrel_pnl(result); pnl.to_csv(OUT/"one_barrel_PnL_refit.csv",index=False)
    boot=build_point_bootstrap_ci(result); boot.to_csv(OUT/"block_bootstrap_loss_CI_refit.csv",index=False)
    dirci=build_directional_bootstrap_ci(result); dirci.to_csv(OUT/"directional_accuracy_block_CI_refit.csv",index=False)
    # SHAP attribution is intentionally not generated in this release.
    metadata={"mode":"REFIT","run_utc":datetime.now(timezone.utc).isoformat(),
              "first_common_date":str(prices.Date.min().date()),"latest_common_date":str(prices.Date.max().date()),
              "n_aligned_prices":int(len(prices)),"n_predictions":int(len(result)),
              "forecast_dates":{"first":str(result.target_date.min().date()),"last":str(result.target_date.max().date())},
              "targets":{"Brent":"next-observation log return","WTI":"next-observation dollar change (USD/bbl)"},
              "models":["RandomWalk","HistoricalMean","AR5","Ridge","RandomForest","XGBoost"],
              "refit_frequency":"annual expanding window","minimum_training_observations":MIN_TRAIN,
              "software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__}}
    (OUT/"analysis_run_metadata_refit.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("REFIT COMPLETE: predictions, metrics, DM/HAC/HLN+Holm, AR(5) Clark-West, event windows, delayed P&L, bootstrap intervals are in",OUT)
    return result

# --------------------------------- Main -----------------------------------
def main():
    if MODE not in {"AUDIT","REFIT"}: raise ValueError("IRFA_MODE must be AUDIT or REFIT")
    if MODE=="AUDIT":
        pred,prices=verify_frozen_inputs()
        # Main metrics: both target units and reconstructed price-level errors.
        met=metrics_from_predictions(pred)
        met.to_csv(OUT/"metrics_recalculated_from_frozen_predictions.csv",index=False)
        metric_check=compare_archived_table(
            met, SUPP/"metrics_by_market_corrected.csv",
            ["market","model","specification"],
            ["n","target_RMSE","target_MAE","directional_accuracy","directional_n",
             "directional_coverage","neutral_forecast_share","price_RMSE_USD_per_bbl",
             "price_MAE_USD_per_bbl","mean_forecast_error"],
            OUT/"metric_reconciliation.csv")
        # Paired HAC/HLN tests and their pre-defined Holm families.
        tests=build_dm_family(pred);tests.to_csv(OUT/"DM_HAC_tests_recalculated.csv",index=False)
        dm_check=compare_archived_table(
            tests, SUPP/"DM_HAC_tests_corrected.csv",
            ["market","specification","model_vs","model","loss"],
            ["n","mean_diff","hac_stat","p_value","p_holm"],
            OUT/"DM_test_reconciliation.csv")
        # Clark-West is restricted to the AR(5) nested linear comparison.
        cw=build_clark_west_ar5(pred);cw.to_csv(OUT/"Clark_West_recalculated.csv",index=False)
        cw_check=compare_archived_table(
            cw, SUPP/"Clark_West_AR5_only.csv", ["market","model","model_vs"],
            ["n","mean_adjusted_diff","hac_stat","p_one_sided"],
            OUT/"Clark_West_reconciliation.csv")
        # Rebuild the event diagnostics and delayed spot P&L from frozen forecasts.
        events=build_event_metrics(pred);events.to_csv(OUT/"event_window_metrics_recalculated.csv",index=False)
        event_check=compare_archived_table(
            events, SUPP/"event_window_metrics_corrected.csv",
            ["market","event","period","model"],
            ["n","target_RMSE","target_MAE","directional_accuracy","directional_n"],
            OUT/"event_window_reconciliation.csv")
        pnl=build_one_barrel_pnl(pred);pnl.to_csv(OUT/"one_barrel_PnL_recalculated.csv",index=False)
        pnl_check=compare_archived_table(
            pnl, SUPP/"one_barrel_PnL_diagnostic.csv",
            ["market","model","specification","cost_bps"],
            ["n","cumulative_net_PnL_USD_per_bbl","annualized_PnL_sharpe",
             "max_peak_to_trough_PnL_drawdown_USD_per_bbl",
             "annualized_turnover_per_unit_position","mean_daily_net_PnL_USD_per_bbl"],
            OUT/"one_barrel_PnL_reconciliation.csv")
        # Recompute moving-block intervals with the archived block length/seed.
        boot=build_point_bootstrap_ci(pred);boot.to_csv(OUT/"block_bootstrap_loss_CI_recalculated.csv",index=False)
        boot_check=compare_archived_table(
            boot, SUPP/"block_bootstrap_loss_CI.csv", ["market","model_vs","model"],
            ["mean_squared_loss_difference","block_bootstrap_95_low","block_bootstrap_95_high"],
            OUT/"block_bootstrap_reconciliation.csv")
        dirci=build_directional_bootstrap_ci(pred);dirci.to_csv(OUT/"directional_accuracy_block_CI_recalculated.csv",index=False)
        dir_check=compare_archived_table(
            dirci, SUPP/"directional_accuracy_block_CI_corrected.csv", ["market","model"],
            ["directional_accuracy","block_bootstrap_95_low","block_bootstrap_95_high",
             "n_total","directional_n","directional_coverage","tie_share"],
            OUT/"directional_bootstrap_reconciliation.csv")
        # Version, scope, checks and limits are explicit in the audit record.
        version={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
                 "pandas":pd.__version__,"statsmodels":sm.__version__ if hasattr(sm,'__version__') else 'see importlib.metadata',
                 "mode":"AUDIT","run_utc":datetime.now(timezone.utc).isoformat(),
                 "frozen_latest_target":str(pred.target_date.max().date()),"forecast_rows":int(len(pred)),
                 "exact_numeric_checks_passed":["metrics and price reconstruction","DM/HAC/HLN and Holm",
                     "Clark-West AR(5)","event-window metrics","one-barrel P&L","point-loss bootstrap",
                     "directional bootstrap"],
                 "SHAP":"excluded from reported results because archived quarter/annual window metadata and WTI attribution could not be independently reconciled"}
        try:
            from importlib.metadata import version as pkgver
            for pkg in ["scikit-learn","xgboost","scipy"]:
                try: version[pkg]=pkgver(pkg)
                except: pass
        except: pass
        (OUT/"audit_environment.json").write_text(json.dumps(version,indent=2))
        summary={"metric_max_abs_diff":float(metric_check.max_abs_difference.max()),
                 "dm_max_abs_diff":float(dm_check.max_abs_difference.max()),
                 "cw_max_abs_diff":float(cw_check.max_abs_difference.max()),
                 "event_max_abs_diff":float(event_check.max_abs_difference.max()),
                 "pnl_max_abs_diff":float(pnl_check.max_abs_difference.max()),
                 "point_bootstrap_max_abs_diff":float(boot_check.max_abs_difference.max()),
                 "directional_bootstrap_max_abs_diff":float(dir_check.max_abs_difference.max()),
                 "status":"PASS"}
        (OUT/"audit_summary.json").write_text(json.dumps(summary,indent=2))
        print("AUDIT PASSED: archived metrics, DM/HAC/HLN+Holm, Clark-West, event windows, P&L and both bootstrap tables reproduce within tolerance.")
        print("SHAP is excluded from the reported manuscript results because archived window metadata and WTI attribution were not independently reconciled.")
    else:
        if REFRESH_LIVE_DATA:
            prices=download_prices()
        else:
            fp=ROOT/"Supplementary"/"Raw_Data"/"aligned_EIA_Brent_WTI_spot_prices.csv"
            if not fp.exists(): raise FileNotFoundError(f"Frozen aligned EIA spot-price file not found: {fp}")
            prices=pd.read_csv(fp,parse_dates=["Date"])
        neg=prices.loc[prices.WTI<0]
        print("Negative WTI observations retained:",neg.to_dict("records"))
        result=run_refit(prices)
        # Save code/data hash and environment information for new runs.
        info={"retrieved_utc":datetime.now(timezone.utc).isoformat(),"mode":"REFIT","latest_common_date":str(pd.to_datetime(prices.Date).max().date()),"n_prices":len(prices),"negative_WTI_rows":neg.to_dict("records"),"target_definition":{"Brent":"next common observation log return","WTI":"next common observation dollar change, USD/bbl"},"min_train":MIN_TRAIN,"refit":"annual expanding","NW_lags":NW_LAGS,"seed":SEED,"note":"Fresh estimation; not expected to be byte-identical to archived run."}
        (OUT/"refit_run_metadata.json").write_text(json.dumps(info,indent=2))
    print("Output directory:",OUT)

if __name__=="__main__": main()
