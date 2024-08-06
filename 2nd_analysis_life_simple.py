# %% [Markdown]
# # Analysis of the different basic retirement variants
# %% [Markdown]
# # Analysis with ideal markets
# %%
results_dir = "results"
# %%
from simulation import BasicLifeSimulation

sim_class = BasicLifeSimulation
bs = sim_class()

bs.simulate()

bs.plot(title=f"{sim_class.__name__}", plot_dir="results/basic_life")

bs.evolution.astype("float").round(2).to_csv(
    f"results/basic_life/{sim_class.__name__}.csv"
)

# %%
from simulation import EarlyRetirementSimulation

sim_class = EarlyRetirementSimulation
retirement_age = 50
sim_class.retirement_age = retirement_age

bs = sim_class()

bs.simulate()

bs.plot(
    title=f"{sim_class.__name__}__{retirement_age}",
    plot_dir=f"results/early_retirement",
)

bs.evolution.astype("float").round(2).to_csv(
    f"results/early_retirement/{sim_class.__name__}__{retirement_age}.csv"
)

# %%
from simulation import FIRELifeSimulation

for fire_percent in [2, 3, 4, 6, 8]:
    sim_class = FIRELifeSimulation
    sim_class.fire_percent = fire_percent

    bs = sim_class()

    bs.simulate()

    bs.plot(title=f"{sim_class.__name__}_{fire_percent}", plot_dir="results/FIRE")

    bs.evolution.astype("float").round(2).to_csv(
        f"results/FIRE/{sim_class.__name__}__{fire_percent}.csv"
    )
