def osmium_fair_value_arb(state: Status, fair_price):
        """

        This strategy identifies arbitrage opportunities by comparing 
        the current best bid and ask prices against a known fair price.
        It attempts to sell when bids are overpriced (above 9999) or buy
        when asks are underpriced (below 10001). If no clear arbitrage 
        is available, it performs market making.

        Args:
            state (Status): The market state for ash coated osmium.
            fair_price (float): Theoretical fair price (currently unused).
        """

        orders = []

        # arbitrage opportunity on the bids so sell here
        if state.best_bid > 9999:
            if state.second_bid is not None and state.second_bid > 9999 and state.rt_position > 0:
                orders.append(Order(state.product, state.second_bid, -state.possible_sell_amt))
                if state.maxamt_bidprc + 1 < state.second_bid:
                    orders.append(Order(state.product, state.maxamt_bidprc + 1, state.possible_buy_amt))
                else:
                    orders.append(Order(state.product, state.second_bid-1, state.possible_buy_amt))
            else:
                orders.append(Order(state.product, state.best_bid, -state.best_bid_amount))
                remaining_order = state.possible_sell_amt - state.best_bid_amount
                if remaining_order > 0:
                    orders.append(Order(state.product, state.best_ask-1, - remaining_order))
                
                if state.second_bid is not None and state.second_bid + 1 < state.best_bid:
                    orders.append(Order(state.product, state.second_bid+1, state.possible_buy_amt))
                elif state.maxamt_bidprc + 1 < state.best_bid:
                    orders.append(Order(state.product, state.maxamt_bidprc+1, state.possible_buy_amt))
        
        # arbitrage opportunity on the asks so buy here
        elif state.best_ask < 10001:
            if state.second_ask is not None and state.second_ask < 10001 and state.rt_position < 0:
                orders.append(Order(state.product, state.second_ask, state.possible_buy_amt))
                if state.maxamt_askprc - 1 > state.second_ask:
                    orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))
                else:
                    orders.append(Order(state.product, state.second_ask + 1, -state.possible_sell_amt))
            else:
                orders.append(Order(state.product, state.best_ask, abs(state.best_ask_amount)))
                remaining_order = state.possible_buy_amt - abs(state.best_ask_amount)
                if remaining_order > 0:
                    orders.append(Order(state.product, state.best_bid + 1, remaining_order))
                if state.second_ask is not None and state.second_ask - 1 > state.best_ask:
                    orders.append(Order(state.product, state.second_ask -1, -state.possible_sell_amt))
                elif state.maxamt_askprc - 1 > state.best_ask:
                    orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))

        # No arbitrage opportunity, simple market making
        else:
            if state.bid_ask_spread > 1:
                orders.append(Order(state.product, state.best_bid+1, state.possible_buy_amt))
                orders.append(Order(state.product, state.best_ask-1, -state.possible_sell_amt))
            else:
                orders.append(Order(state.product, state.best_bid, state.possible_buy_amt))
                orders.append(Order(state.product, state.best_ask, -state.possible_sell_amt))

        return orders
