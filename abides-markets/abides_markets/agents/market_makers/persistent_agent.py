from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ...messages.marketdata import MarketDataMsg, L2SubReqMsg
from ...messages.query import QuerySpreadResponseMsg
from ...orders import Side
from ..trading_agent import TradingAgent


class PersistentAgent(TradingAgent):
    """
    Trading Agent that wakes up every minute and places a random
    Simple Trading Agent that compares the 20 past mid-price observations with the 50 past observations and places a
    buy limit order if the 20 mid-price average >= 50 mid-price average or a
    sell limit order if the 20 mid-price average < 50 mid-price average
    """

    def __init__(
        self,
        id: int,
        symbol,
        starting_cash,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        min_size=20,
        max_size=50,

        epsilon: float = -0.5,     # same fixed pricing parameters for bid and ask
        pct_to_hedge: float = 0.2,  # fixed percentage of inventory to hedge

        wake_up_freq: NanosecondTime = str_to_ns("60s"),
        poisson_arrival=True,
        order_size_model=None,
        subscribe=False,
        log_orders=False,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size

        self.epsilon = epsilon,
        self.pct_to_hedge = pct_to_hedge,

        self.size = (
            self.random_state.randint(self.min_size, self.max_size)
            if order_size_model is None
            else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list: List[float] = []
#        self.avg_20_list: List[float] = []
#        self.avg_50_list: List[float] = []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=int(10e9),
                    depth=1,
                )
            )
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"
        elif can_trade and not self.subscribe:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Random agent actions are determined randomly in the range [self.min_epsilon, self.max_epsilon]"""
        super().receive_message(current_time, sender_id, message)

        holdings = self.get_holdings(self.symbol)

        if holdings > 0:
            self.place_market_order(self.symbol,
                                    quantity=int(self.pct_to_hedge * holdings),
                                    side=Side.ASK,
                                    )
        else:
            self.place_market_order(self.symbol,
                                    quantity=int(self.pct_to_hedge * abs(holdings)),
                                    side=Side.BID
                                    )

        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):

            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

            self.place_limit_order(self.symbol,
                                   quantity=self.size,
                                   side=Side.BID,
                                   limit_price=bid * (1+self.epsilon))
            self.place_limit_order(self.symbol,
                                   quantity=self.size,
                                   side=Side.ASK,
                                   limit_price=ask * (1+self.epsilon))
            self.set_wakeup(current_time + self.get_wake_frequency())
            self.state = "AWAITING_WAKEUP"
        elif (
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and isinstance(message, MarketDataMsg)
        ):
            bids, asks = self.known_bids[self.symbol], self.known_asks[self.symbol]
            if bids and asks:
                self.place_limit_order(self.symbol,
                                       quantity=self.size,
                                       side=Side.BID,
                                       limit_price=bids[0][0] * (1 + self.epsilon))
                self.place_limit_order(self.symbol,
                                       quantity=self.size,
                                       side=Side.ASK,
                                       limit_price=asks[0][0] * (1 + self.epsilon))
            self.state = "AWAITING_MARKET_DATA"

    """
    def place_orders(self, bid: int, ask: int) -> None:
       """ """Momentum Agent actions logic""" """"
        if bid and ask:
            self.mid_list.append((bid + ask) / 2)
            if len(self.mid_list) > 20:
                self.avg_20_list.append(
                    MomentumAgent.ma(self.mid_list, n=20)[-1].round(2)
                )
            if len(self.mid_list) > 50:
                self.avg_50_list.append(
                    MomentumAgent.ma(self.mid_list, n=50)[-1].round(2)
                )
            if len(self.avg_20_list) > 0 and len(self.avg_50_list) > 0:
                if self.order_size_model is not None:
                    self.size = self.order_size_model.sample(
                        random_state=self.random_state
                    )

                if self.size > 0:
                    if self.avg_20_list[-1] >= self.avg_50_list[-1]:
                        self.place_limit_order(
                            self.symbol,
                            quantity=self.size,
                            side=Side.BID,
                            limit_price=ask,
                        )
                    else:
                        self.place_limit_order(
                            self.symbol,
                            quantity=self.size,
                            side=Side.ASK,
                            limit_price=bid,
                        )
    """

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

    """ @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n
        """
