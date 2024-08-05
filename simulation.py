from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Callable, Union
import pandas as pd
import numpy as np
from strategy import BasicInvestmentStrategy
from plot import combined_plot


@dataclass
class Asset:
    name: str
    value: float = 0
    relative_value: Tuple[str, float] = False
    growth_relative: float = 1
    growth_fixed: float = 0
    leech_growth_relative: Tuple[str, float] = None
    leech_growth_fixed: Tuple[str, float] = None
    yearly_balance: bool = False
    active: bool = False


# An AssestCondition is a function that takes all the assets of a strategy and returns a boolean. If it's true, the related Asset will be added
AssetCondition = Callable


class LifeSimulation(ABC):

    year_balance_columns: List[str] = ["Balance", "Inputs", "Outputs"]
    # total_balance_columns: List[str] = ['Cash', 'Investments']
    summary_columns = year_balance_columns  # + total_balance_columns

    def __init__(
        self,
        assets: List[Tuple[Union[int, AssetCondition], Tuple[Asset, dict]]] = None,
        life_expectancy: int = 80,
        investment_strategy: BasicInvestmentStrategy = None,
    ) -> None:

        self.assets: List[Tuple[Union[int, AssetCondition], Tuple[Asset, dict]]] = (
            assets
        )
        self.evolution: pd.DataFrame = None
        self.life_expectancy: int = life_expectancy
        self.investment_strategy = BasicInvestmentStrategy(
            cash_target=30_000, before_investment_ratio=0.25
        )
        if investment_strategy is not None:
            self.investment_strategy = investment_strategy

        super().__init__()

    def simulate(self):

        self.evolution = pd.DataFrame(
            index=range(self.life_expectancy),
            columns=[
                *self.summary_columns,
                *np.unique([a.name for y, a, b in self.assets]),
            ],
        )

        for r in self.evolution.index:
            self.simulate_year(r)

    def simulate_year(self, r):

        # Write assets for current year
        for a_year, asset, add in self.assets:

            # add or remove Asset
            activate = False
            if isinstance(a_year, Callable):
                activate = a_year(r=r)
            else:
                activate = a_year == r
            if activate:
                if add is True:
                    self.evolution.loc[r, asset.name] = asset.value
                if add is False:
                    self.evolution.loc[r, asset.name] = np.NaN

            # update Asset (only if not removed in this same year)
            # (add and asset.name == "retirement") or not (activate and add is False):
            active = False
            if add:
                if isinstance(a_year, Callable):
                    active = a_year(r=r)
                else:
                    active = r >= a_year
            if active:
                # Backward update
                if r > 0:
                    if not pd.isnull(self.evolution.loc[r - 1, asset.name]):
                        self.evolution.loc[r, asset.name] = (
                            self.evolution.loc[r - 1, asset.name]
                            * asset.growth_relative
                            + asset.growth_fixed
                        )

                    if asset.leech_growth_relative is not None:
                        leech_col, leech_rel = asset.leech_growth_relative
                        leech_value = self.evolution.loc[r - 1, leech_col] * leech_rel
                        self.evolution.loc[r, asset.name] = (
                            self.evolution.loc[r, asset.name] + leech_value
                        )
                        self.evolution.loc[r, leech_col] = (
                            self.evolution.loc[r, leech_col] - leech_value
                        )

                if asset.leech_growth_fixed is not None:
                    leech_col, leech_value = asset.leech_growth_fixed
                    self.evolution.loc[r, asset.name] = (
                        self.evolution.loc[r, asset.name] + leech_value
                    )
                    self.evolution.loc[r, leech_col] = (
                        self.evolution.loc[r, leech_col] - leech_value
                    )

        # Yearly Summary
        self.generate_yearly_summary(r)

        # Invest (into next year)
        self.evolution = self.investment_strategy.invest(self.evolution, r)

    def generate_yearly_summary(
        self, r
    ):  # TODO: differentiate between assets that directly add to cash vs are liquid (can be sold) vs not

        self.evolution.loc[r, ["Balance", "Inputs", "Outputs"]] = 0

        asset_columns = self.yearly_balance_columns()

        pos_asset_mask = self.evolution.loc[r, asset_columns] > 0
        if sum(pos_asset_mask) > 0:
            self.evolution.loc[r, "Inputs"] = self.evolution.loc[
                r, pos_asset_mask.loc[pos_asset_mask].index
            ].sum()
        neg_asset_mask = self.evolution.loc[r, asset_columns] < 0
        if sum(neg_asset_mask) > 0:
            self.evolution.loc[r, "Outputs"] = self.evolution.loc[
                r, neg_asset_mask.loc[neg_asset_mask].index
            ].sum()

        self.evolution.loc[r, "Balance"] = (
            self.evolution.loc[r, "Inputs"] + self.evolution.loc[r, "Outputs"]
        )

    def yearly_balance_assets(self) -> List[str]:
        return [a.name for _, a, _ in self.assets if a.yearly_balance]

    def yearly_balance_columns(self) -> List[str]:
        yearly_balance_assets = self.yearly_balance_assets()
        return [c for c in self.evolution.columns if c in yearly_balance_assets]


class BasicLifeSimulation(LifeSimulation):
    """
    The basic financial strategy for a life is the following:
        - Work 1st job (1.2x cost_of_life) @ 22 years of age
        - Work 2nd job (2.0x cost_of_life) @ 25 years of age
        - Family (3.0x cost of life) @ 30 years of age
        - Work 3nd job (3.0x cost_of_life) @ 35 years of age
        - Family out @ 50 years of age
        - Retire (remove work_3 + 4% rule) @ 65 years of age
    """

    median_salary: float = 45_000
    cost_of_life: float = 0.8 * median_salary
    family_cost: float = 1.0 * cost_of_life

    yearly_inflation: Union[float, List[float], pd.Series] = 1 + 2.5 / 100
    yearly_salary_raise: Union[float, List[float], pd.Series] = 1 + 3.5 / 100
    yearly_investment_growth: Union[float, List[float], pd.Series] = 1.1

    cash = Asset(name="cash", value=0)
    investments = Asset(
        name="investments", value=0, growth_relative=yearly_investment_growth
    )

    work_1 = Asset(
        name="work_1",
        value=1.2 * cost_of_life,
        growth_relative=yearly_salary_raise,
        yearly_balance=True,
    )
    savings_ratio_1 = 0.50
    expenses_1 = Asset(
        name="expenses_1",
        value=-(1 - savings_ratio_1) * work_1.value,
        growth_relative=yearly_inflation,
        yearly_balance=True,
    )

    work_2 = Asset(
        name="work_2",
        value=2.0 * cost_of_life,
        growth_relative=yearly_salary_raise,
        yearly_balance=True,
    )
    savings_ratio_2 = 0.40
    expenses_2 = Asset(
        name="expenses_2",
        value=-(1 - savings_ratio_2) * work_2.value,
        growth_relative=yearly_inflation,
        yearly_balance=True,
    )

    work_3 = Asset(
        name="work_3",
        value=3.0 * cost_of_life,
        growth_relative=yearly_salary_raise,
        yearly_balance=True,
    )
    savings_ratio_3 = 0.40
    expenses_3 = Asset(
        name="expenses_3",
        value=-(1 - savings_ratio_3) * work_3.value,
        growth_relative=yearly_inflation,
        yearly_balance=True,
    )

    family = Asset(
        name="family",
        value=-family_cost,
        growth_relative=yearly_inflation,
        yearly_balance=True,
    )

    retirement = Asset(
        name="retirement",
        value=0,
        leech_growth_relative=("investments", 4 / 100),
        yearly_balance=True,
    )

    def __init__(
        self,
        assets: List[Tuple[Union[int, AssetCondition], Tuple[Asset, dict]]] = None,
        life_expectancy: int = 80,
        investment_strategy: BasicInvestmentStrategy = None,
    ) -> None:
        if assets is None:
            assets = self.get_assets()
        super().__init__(
            assets=assets,
            life_expectancy=life_expectancy,
            investment_strategy=investment_strategy,
        )

    def get_assets(self) -> list:
        return [
            (0, self.cash, True),
            (0, self.investments, True),
            (22, self.work_1, True),
            (22, self.expenses_1, True),
            (25, self.work_1, False),
            (25, self.expenses_1, False),
            (25, self.work_2, True),
            (25, self.expenses_2, True),
            (30, self.family, True),
            (35, self.work_2, False),
            (35, self.expenses_2, False),
            (35, self.work_3, True),
            (35, self.expenses_3, True),
            (50, self.family, False),
            (65, self.work_3, False),
            (65, self.retirement, True),
        ]

    def yearly_assets(self) -> List[Asset]:
        assets = []
        a_names = []
        for _, a, _ in self.get_assets():
            if not a.yearly_balance:
                continue
            if a.name in a_names:
                continue
            a_names.append(a.name)
            assets.append(a)
        return assets

    def not_yearly_assets(self) -> List[Asset]:
        assets = []
        a_names = []
        for _, a, _ in self.get_assets():
            if a.yearly_balance:
                continue
            if a.name in a_names:
                continue
            a_names.append(a.name)
            assets.append(a)
        return assets

    def plot(self, title: str = None, plot_dir: str = "./", **kwargs):

        general_status_series = (
            ("Balance", "line", "tab:green", dict(logy=True)),
            *[
                (a.name, "line", None, dict(logy=True))
                for a in self.not_yearly_assets()
            ],
        )
        yearly_balance_series = (
            # ("Inputs", "line", "tab:green", {}),
            # ("Outputs", "line", "tab:red", {}),
            *[
                (a.name, "line", None, {})
                for a in self.yearly_assets()
                if a.name != "retirement"
            ],
        )

        combined_data = (
            self.evolution,
            (general_status_series, yearly_balance_series),
        )
        combined_plot(
            *combined_data,
            title=title,
            path=plot_dir,
            dropna=False,
            **kwargs,
        )


class EarlyRetirementSimulation(BasicLifeSimulation):

    retirement_age: int = 55

    def get_assets(self) -> list:
        return [
            (0, self.cash, True),
            (0, self.investments, True),
            (22, self.work_1, True),
            (22, self.expenses_1, True),
            (26, self.work_1, False),
            (26, self.expenses_1, False),
            (25, self.work_2, True),
            (25, self.expenses_2, True),
            (30, self.family, True),
            (36, self.work_2, False),
            (36, self.expenses_2, False),
            (35, self.work_3, True),
            (35, self.expenses_3, True),
            (50, self.family, False),
            (self.retirement_age, self.work_3, False),
            (self.retirement_age, self.retirement, True),
        ]


