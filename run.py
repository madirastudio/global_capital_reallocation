from analysis import *

def main():

    df = load_data()

    df = classify_regime(df)

    disp = compute_dispersion(df)
    plot_dispersion(disp)

    sens = earnings_sensitivity(df)
    betas = rolling_beta(df)
    realloc = capital_reallocation(df)

    scenario = scenario_analysis(df, rate_shock=1.0)

    print("\nEarnings Sensitivities\n")
    for region, model in sens.items():
        print(f"\n--- {region} ---")
        print(model.summary())

    print("\nScenario Output\n")
    print(scenario.head())


if __name__ == "__main__":
    main()
