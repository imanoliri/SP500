from abc import ABC
from dataclasses import dataclass
import pandas as pd


@dataclass
class InvestmentStrategy(ABC):
    """
    Modifies the current year with investment assets.
    """

    def invest(evolution: pd.DataFrame, r: int):
        raise NotImplementedError


@dataclass
class BasicInvestmentStrategy(InvestmentStrategy):
    """
    Invest only a small ratio until a desired cash level is reached, after that invest all further savings.
    """

    cash_target: float = 10_000
    before_investment_ratio: float = 0.25
    investing_tax: float = 0
    investing_fee: float = 0

    def invest(self, evolution: pd.DataFrame, r: int):

        if pd.isnull(evolution.loc[r, "Balance"]):
            return evolution

        if evolution.loc[r, "Balance"] == 0:
            if pd.isnull(evolution.loc[r, "cash"]):
                evolution.loc[r, "cash"] = 0
            evolution.loc[r, "cash"] = evolution.loc[r, "cash"]
            return evolution

        if evolution.loc[r, "Balance"] < 0:
            if pd.isnull(evolution.loc[r, "cash"]):
                evolution.loc[r, "cash"] = 0
            evolution.loc[r, "cash"] = (
                evolution.loc[r, "cash"] + evolution.loc[r, "Balance"]
            )
            return evolution

        if evolution.loc[r, "Balance"] > 0:
            if pd.isnull(evolution.loc[r, "cash"]):
                evolution.loc[r, "cash"] = 0
            if evolution.loc[r, "cash"] < self.cash_target:
                invested_ammount = (
                    evolution.loc[r, "Balance"] * self.before_investment_ratio
                )
                invested_ammount = (
                    invested_ammount * (1 - self.investing_tax) - self.investing_fee
                )
                saved_ammount = evolution.loc[r, "Balance"] - invested_ammount

                if pd.isnull(evolution.loc[r, "investments"]):
                    evolution.loc[r, "investments"] = 0
                evolution.loc[r, "investments"] = (
                    evolution.loc[r, "investments"] + invested_ammount
                )
                evolution.loc[r, "cash"] = evolution.loc[r, "cash"] + saved_ammount

                return evolution

            if evolution.loc[r, "cash"] >= self.cash_target:
                invested_ammount = evolution.loc[r, "Balance"]
                invested_ammount = (
                    invested_ammount * (1 - self.investing_tax) - self.investing_fee
                )
                saved_ammount = 0

                if pd.isnull(evolution.loc[r, "investments"]):
                    evolution.loc[r, "investments"] = 0
                evolution.loc[r, "investments"] = (
                    evolution.loc[r, "investments"] + invested_ammount
                )
                evolution.loc[r, "cash"] = evolution.loc[r, "cash"] + saved_ammount

                return evolution
        raise ValueError("Couldn't handle the investing situation.")
