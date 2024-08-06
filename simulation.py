from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Callable, Union
import pandas as pd
import numpy as np
from strategy import InvestmentRetirementStrategy
from plot import combined_plot, save_plot


@dataclass
class Asset:
    name: str
    value: float = 0
    relative_value: Tuple[str, float] = False
    growth_relative: float = 1
    growth_fixed: float = 0
    leech_growth_relative: Tuple[str, float] = None
    leech_growth_fixed: Tuple[str, float] = None
    leech_growth_tax: float = 0
    leech_growth_fee: float = 0

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
        strategy: InvestmentRetirementStrategy = None,
    ) -> None:

        self.assets: List[Tuple[Union[int, AssetCondition], Tuple[Asset, dict]]] = (
            assets
        )
        self.evolution: pd.DataFrame = None
        self.life_expectancy: int = life_expectancy
        self.strategy = InvestmentRetirementStrategy(
            cash_target=30_000, before_investment_ratio=0.25, investing_gains_tax=0.21
        )
        if strategy is not None:
            self.strategy = strategy

        super().__init__()

    def simulate(self):

        self.evolution = pd.DataFrame(
            index=range(self.life_expectancy),
            columns=[
                *self.summary_columns,
                *np.unique([a.name for y, a, b in self.assets]),
            ],
            dtype=float,
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
                leech_relative_value = None
                if r > 0:
                    if not pd.isnull(self.evolution.loc[r - 1, asset.name]):
                        self.evolution.loc[r, asset.name] = (
                            self.evolution.loc[r - 1, asset.name]
                            * asset.growth_relative
                            + asset.growth_fixed
                        )

                    if asset.leech_growth_relative is not None:
                        leech_col, leech_rel = asset.leech_growth_relative
                        leech_relative_value = (
                            self.evolution.loc[r - 1, leech_col] * leech_rel
                        )
                        leech_relative_net_value = leech_relative_value * min(
                            1, max(0, (1 - asset.leech_growth_tax))
                        )
                        leech_relative_net_value = min(
                            leech_relative_net_value,
                            max(0, leech_relative_net_value - asset.leech_growth_fee),
                        )
                        self.evolution.loc[r, asset.name] = (
                            self.evolution.loc[r, asset.name] + leech_relative_net_value
                        )
                        self.evolution.loc[r, leech_col] = (
                            self.evolution.loc[r, leech_col] - leech_relative_value
                        )

                leech_fixed_value = None
                if asset.leech_growth_fixed is not None:
                    leech_col, leech_fixed_value = asset.leech_growth_fixed
                    leech_fixed_net_value = leech_fixed_value * min(
                        1, max(0, (1 - asset.leech_growth_tax))
                    )
                    leech_fixed_net_value = min(
                        leech_fixed_net_value,
                        max(0, leech_fixed_net_value - asset.leech_growth_fee),
                    )
                    self.evolution.loc[r, asset.name] = (
                        self.evolution.loc[r, asset.name] + leech_fixed_net_value
                    )
                    self.evolution.loc[r, leech_col] = (
                        self.evolution.loc[r, leech_col] - leech_fixed_value
                    )

        # Yearly Summary
        self.generate_yearly_summary(r)

        # Invest (into next year)
        self.evolution = self.strategy.apply(self.evolution, r)

    def generate_yearly_summary(
        self, r
    ):  # TODO: differentiate between assets that directly add to cash vs are liquid (can be sold) vs not

        asset_columns = self.yearly_balance_columns()

        pos_asset_mask = self.evolution.loc[r, asset_columns] > 0
        inputs = 0
        if sum(pos_asset_mask) > 0:
            inputs = self.evolution.loc[
                r, pos_asset_mask.loc[pos_asset_mask].index
            ].sum()
            self.evolution.loc[r, "Inputs"] = inputs
        neg_asset_mask = self.evolution.loc[r, asset_columns] < 0
        outputs = 0
        if sum(neg_asset_mask) > 0:
            outputs = self.evolution.loc[
                r, neg_asset_mask.loc[neg_asset_mask].index
            ].sum()
            self.evolution.loc[r, "Outputs"] = outputs

        if self.evolution.loc[r, ["Inputs", "Outputs"]].isna().sum() < 2:
            self.evolution.loc[r, "Balance"] = inputs + outputs

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

    cash = Asset(name="cash", value=np.NaN)
    investments = Asset(
        name="investments", value=np.NaN, growth_relative=yearly_investment_growth
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
        name="investment_drawbacks",
        value=np.NaN,
        growth_relative=0,
        yearly_balance=True,
    )

    def __init__(
        self,
        assets: List[Tuple[Union[int, AssetCondition], Tuple[Asset, dict]]] = None,
        life_expectancy: int = 80,
        strategy: InvestmentRetirementStrategy = None,
    ) -> None:
        if assets is None:
            assets = self.get_assets()
        super().__init__(
            assets=assets,
            life_expectancy=life_expectancy,
            strategy=strategy,
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

    def asset_cols(self) -> List[str]:
        return [a.name for _, a, _ in self.get_assets()]

    def work_cols(self) -> List[str]:
        return [a.name for a in self.yearly_assets() if a.name.startswith("work")]

    def working_mask(self) -> pd.Series:
        return ~self.evolution.loc[:, self.work_cols()].isna().all(axis=1)

    def investment_drawing_mask(self) -> pd.Series:
        return self.evolution.investment_drawbacks > 0

    def kpis(self):
        return {
            "start_working_age": self.start_working_age(),
            "end_working_age": self.end_working_age(),
            "start_drawing_investments_age": self.start_drawing_investments_age(),
            "life_expectancy": self.life_expectancy,
            "education_years": self.education_years(),
            "working_years": self.working_years(),
            "retirement_years": self.retirement_years(),
            "ending_net_worth": self.ending_net_worth(),
            "min_cash_and_year": self.min_cash_and_year(),
            "min_balance_and_year": self.min_balance_and_year(),
            "max_balance_and_year": self.max_balance_and_year(),
            "max_outputs_and_year": self.max_outputs_and_year(),
        }

    def start_working_age(self) -> int:
        working_mask = self.working_mask()
        return working_mask[working_mask].index[0]

    def end_working_age(self) -> int:
        working_mask = self.working_mask()
        return working_mask[working_mask].index[-1]

    def start_drawing_investments_age(self) -> int:
        investment_drawing_mask = self.investment_drawing_mask()
        if investment_drawing_mask.sum() == 0:
            return np.NaN
        return investment_drawing_mask[investment_drawing_mask].index[0]

    def education_years(self) -> float:
        return self.start_working_age() - 0

    def working_years(self) -> float:
        return self.end_working_age() - self.start_working_age()

    def retirement_years(self) -> float:
        return len(self.evolution) - self.end_working_age()

    def ending_net_worth(self) -> float:
        return self.evolution.iloc[-1].loc[self.asset_cols()].sum()

    def min_cash_and_year(self) -> Tuple[int, float]:
        evolution = self.evolution
        return evolution.loc[:, "cash"].idxmin(), evolution.loc[:, "cash"].min()

    def min_balance_and_year(
        self,
    ) -> Tuple[int, float]:
        evolution = self.evolution
        return evolution.loc[:, "Balance"].idxmin(), evolution.loc[:, "Balance"].min()

    def max_balance_and_year(self) -> Tuple[int, float]:
        evolution = self.evolution
        return evolution.loc[:, "Balance"].idxmax(), evolution.loc[:, "Balance"].max()

    def max_outputs_and_year(self) -> Tuple[int, float]:
        evolution = self.evolution
        return evolution.loc[:, "Outputs"].idxmin(), evolution.loc[:, "Outputs"].min()

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
            *[(a.name, "line", None, {}) for a in self.yearly_assets()],
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

    def plot_years(
        self,
        title: str = None,
        kind: str = "bar",
        plot_dir: str = "./",
        **kwargs,
    ):
        for r in self.evolution.index:
            self.plot_year(r, title, kind, plot_dir, **kwargs)

    def plot_year(
        self,
        r: int,
        title: str = None,
        kind: str = "bar",
        plot_dir: str = "./",
        **kwargs,
    ):
        year_to_plot = self.evolution.loc[r].dropna()
        if year_to_plot.empty:
            return
        year_to_plot.plot(kind=kind, **kwargs)
        save_plot(path=plot_dir, title=title, prefix=f"year_{r}")


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


class FIRELifeSimulation(BasicLifeSimulation):

    fire_percent: float = 3.0

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
            (self.ready_for_fire, self.work_3, False),
            (self.ready_for_fire, self.retirement, True),
        ]

    def ready_for_fire(self, r) -> bool:
        if r == 0:
            return False
        if any(
            pd.isnull(v)
            for v in self.evolution.loc[r - 1, ["Outputs", "investments"]].values
        ):
            return False
        if any(
            pd.isnull(v)
            for v in self.evolution.loc[r - 1, ["Outputs", "investments"]].values
        ):
            return False
        safe_withdrawal = (
            self.evolution.loc[r - 1, "investments"] * self.fire_percent / 100
        )
        return -self.evolution.loc[r - 1, "Outputs"] < safe_withdrawal
