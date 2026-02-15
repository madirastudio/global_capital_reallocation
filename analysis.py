
---

# analysis.py

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -------------------------------
# 1. LOAD + MERGE
# -------------------------------

def load_data():
    equity = pd.read_csv("data/global_equity.csv", parse_dates=["date"])
    macro = pd.read_csv("data/macro.csv", parse_dates=["date"])
    flows = pd.read_csv("data/flows.csv", parse_dates=["date"])

    df = equity.merge(macro, on="date", how="left")
    df = df.merge(flows, on=["date", "region"], how="left")

    return df


# -------------------------------
# 2. VALUATION DISPERSION
# -------------------------------

def compute_dispersion(df):
    disp = (
        df.groupby(["date", "region"])["pe_ratio"]
        .std()
        .reset_index()
        .rename(columns={"pe_ratio": "valuation_dispersion"})
    )
    return disp


# -------------------------------
# 3. EARNINGS SENSITIVITY
# -------------------------------

def earnings_sensitivity(df):
    results = {}

    for region in df["region"].unique():
        sub = df[df["region"] == region].dropna()

        X = sub[["rate_change", "inflation", "gdp_surprise"]]
        X = sm.add_constant(X)
        y = sub["earnings_growth"]

        model = sm.OLS(y, X).fit()
        results[region] = model

    return results


# -------------------------------
# 4. ROLLING BETAS
# -------------------------------

def rolling_beta(df, window=36):
    df = df.sort_values("date")

    betas = []

    for region in df["region"].unique():
        sub = df[df["region"] == region].copy()

        for i in range(window, len(sub)):
            window_df = sub.iloc[i-window:i]

            X = sm.add_constant(window_df["rate_change"])
            y = window_df["returns"]

            model = sm.OLS(y, X).fit()

            betas.append({
                "date": sub.iloc[i]["date"],
                "region": region,
                "beta_rate": model.params["rate_change"]
            })

    return pd.DataFrame(betas)


# -------------------------------
# 5. REGIME CLASSIFICATION
# -------------------------------

def classify_regime(df):

    conditions = [
        df["rate_change"] > 0.5,
        df["rate_change"] < -0.5
    ]

    choices = ["tightening", "easing"]

    df["regime"] = np.select(conditions, choices, default="neutral")

    return df


# -------------------------------
# 6. CAPITAL REALLOCATION
# -------------------------------

def capital_reallocation(df):

    realloc = (
        df.groupby(["date", "region"])["flow"]
        .sum()
        .groupby(level=1)
        .diff()
        .reset_index()
        .rename(columns={"flow": "flow_change"})
    )

    return realloc


# -------------------------------
# 7. SCENARIO SIMULATION
# -------------------------------

def scenario_analysis(df, rate_shock=1.0):

    df = df.copy()
    df["shock_rate"] = df["rate_change"] + rate_shock

    scenario = (
        df.groupby("region")["shock_rate"]
        .mean()
        .reset_index()
    )

    return scenario


# -------------------------------
# 8. PLOT
# -------------------------------

def plot_dispersion(disp):

    plt.figure()
    for region in disp["region"].unique():
        sub = disp[disp["region"] == region]
        plt.plot(sub["date"], sub["valuation_dispersion"], label=region)

    plt.legend()
    plt.title("Valuation Dispersion by Region")
    plt.savefig("report/charts/valuation_dispersion.png")
    plt.close()
