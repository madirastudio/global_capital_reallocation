import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ---------------- DATA LOAD ---------------- #

def load_data():
    bank = pd.read_csv("data/fundamentals.csv", parse_dates=["quarter"])
    macro = pd.read_csv("data/macro.csv", parse_dates=["quarter"])
    prices = pd.read_csv("data/prices.csv", parse_dates=["quarter"])

    df = (
        bank.merge(macro, on="quarter", how="left")
            .merge(prices, on=["bank", "quarter"], how="left")
            .sort_values(["bank", "quarter"])
    )
    return df

# ---------------- FEATURES ---------------- #

def engineer_features(df):
    df["loan_growth_yoy"] = df.groupby("bank")["loans"].pct_change(4) * 100
    df["deposit_growth_yoy"] = df.groupby("bank")["deposits"].pct_change(4) * 100

    df["credit_cost"] = df["provisions"] / df["avg_loans"] * 100
    df["pb"] = df["price"] / df["bvps"]
    df["term_spread"] = df["gsec_10y"] - df["repo"]

    return df

# ---------------- MODEL 1: NIM ---------------- #

def pooled_nim_model(df):
    model = smf.ols(
        "nim ~ repo + term_spread + deposit_growth_yoy + C(bank)",
        data=df
    ).fit(cov_type="HC3")
    return model

def bankwise_nim_models(df):
    out = {}
    for b in df.bank.unique():
        m = smf.ols(
            "nim ~ repo + term_spread + deposit_growth_yoy",
            data=df[df.bank == b]
        ).fit(cov_type="HC3")
        out[b] = m
    return out

# ---------------- MODEL 2: CREDIT COST ---------------- #

def credit_cost_reversion(df):
    stats = (
        df.groupby("bank")["credit_cost"]
          .agg(["mean", "std", "last"])
          .reset_index()
    )
    stats["z_score"] = (stats["last"] - stats["mean"]) / stats["std"]
    return stats

# ---------------- MODEL 3: ROA → VALUATION ---------------- #

def valuation_model(df):
    model = smf.ols(
        "pb ~ roa + loan_growth_yoy + repo",
        data=df
    ).fit(cov_type="HC3")
    return model

# ---------------- SCENARIOS ---------------- #

def earnings_stress(df, nim_shock, cc_shock):
    df = df.copy()
    df["roa_stressed"] = df["roa"] + 0.4 * nim_shock - 0.6 * cc_shock
    return df

# ---------------- CHARTS ---------------- #

def plot_nim(df):
    plt.figure(figsize=(8,4))
    for b in df.bank.unique():
        sub = df[df.bank == b]
        plt.plot(sub.quarter, sub.nim, label=b)
    plt.legend()
    plt.title("NIM Trend")
    plt.tight_layout()
    plt.savefig("report/charts/nim_trend.png")
    plt.close()
