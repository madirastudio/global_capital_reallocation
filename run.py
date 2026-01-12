from analysis import *

df = load_data()
df = engineer_features(df)

# Models
nim_pooled = pooled_nim_model(df)
nim_bankwise = bankwise_nim_models(df)
cc_stats = credit_cost_reversion(df)
val_model = valuation_model(df)

# Scenario
latest = df.sort_values("quarter").groupby("bank").tail(1)
stress = earnings_stress(latest, nim_shock=-0.4, cc_shock=0.5)

# Outputs
print(nim_pooled.summary())
print(cc_stats)
print(val_model.summary())

plot_nim(df)
stress.to_excel("report/scenario_output.xlsx", index=False)
