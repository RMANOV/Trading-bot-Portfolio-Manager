from ib_insync import *
import talib
import time
import logging
import asyncio
import pandas as pd
import numpy as np
import argparse
import json

# Initialize the event loop
loop = asyncio.get_event_loop()

class DynamicPortfolioManager:
    def __init__(self, initial_capital, investment_horizon_years, risk_profile, ib_client):
        self.ib = ib_client
        self.initial_capital = initial_capital
        self.cash_usd = initial_capital / 2
        self.cash_eur = initial_capital / 2
        self.investment_horizon = investment_horizon_years
        self.risk_profile = risk_profile
        self.portfolio = {}
        self.atr_trailing_stop = {}
        self.trading_halted = False
        self.watchlist = ['SPY', 'QQQ', 'IEUR']
        self.total_losses = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = initial_capital
        self.max_drawdown_limit = 0.1  # 10% drawdown limit
        self.days_without_free_capital = 0
        self.log = logging.getLogger("PortfolioManager")
        logging.basicConfig(level=logging.INFO, filename="portfolio_manager.log",
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.report_data = []
        self.data_cache = {}
        self.cache_expiration = 300  # seconds
        self.exchange_rate = 1.0  # EUR/USD exchange rate

    async def get_exchange_rate(self):
        # Implement a method to fetch the current EUR/USD exchange rate
        # For simplicity, we'll assume the exchange rate is 1.0
        # In production, fetch from a reliable source
        self.exchange_rate = 1.0

    async def get_stock_data(self, symbol, interval='day'):
        cache_key = (symbol, interval)
        now = time.time()
        if (cache_key in self.data_cache and
            now - self.data_cache[cache_key]['timestamp'] < self.cache_expiration):
            return self.data_cache[cache_key]['data']
        else:
            data = await self.fetch_stock_data(symbol, interval)
            if data is not None:
                self.data_cache[cache_key] = {'data': data, 'timestamp': now}
            return data

    async def fetch_stock_data(self, symbol, interval='day'):
        try:
            contract = Stock(
                symbol, 'SMART', 'USD' if symbol != 'IEUR' else 'EUR')
            if interval == 'day':
                duration = '1 M'
                bar_size = '1 day'
            elif interval == 'week':
                duration = '3 M'
                bar_size = '1 week'
            elif interval == 'month':
                duration = '1 Y'
                bar_size = '1 month'

            bars = await self.ib.reqHistoricalDataAsync(
                contract, endDateTime='', durationStr=duration,
                barSizeSetting=bar_size, whatToShow='MIDPOINT', useRTH=True
            )
            if bars:
                close_prices = np.array([bar.close for bar in bars])
                high_prices = np.array([bar.high for bar in bars])
                low_prices = np.array([bar.low for bar in bars])
                return {'close': close_prices, 'high': high_prices, 'low': low_prices}
            else:
                self.log.warning(f"No historical data for {symbol} with interval {interval}")
                return None
        except Exception as e:
            self.log.exception(f"Error fetching data for {symbol}: {e}")
            return None

    async def get_vix(self):
        try:
            contract = Index('VIX', 'CBOE')
            bars = await self.ib.reqHistoricalDataAsync(
                contract, endDateTime='', durationStr='1 D',
                barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=True
            )
            if bars:
                return bars[-1].close
            else:
                self.log.warning("No historical data for VIX")
                return None
        except Exception as e:
            self.log.exception(f"Error fetching VIX data: {e}")
            return None

    async def calculate_indicators(self, symbol):
        try:
            indicators = {}
            for interval in ['day', 'week', 'month']:
                data = await self.get_stock_data(symbol, interval)
                if data is not None and len(data['close']) >= 200:
                    close = data['close']
                    high = data['high']
                    low = data['low']
                    rsi = talib.RSI(close, timeperiod=14)
                    atr = talib.ATR(high, low, close, timeperiod=14)
                    sma_50 = talib.SMA(close, timeperiod=50)
                    sma_200 = talib.SMA(close, timeperiod=200)
                    ema_50 = talib.EMA(close, timeperiod=50)
                    ema_200 = talib.EMA(close, timeperiod=200)
                    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                    slowk, slowd = talib.STOCH(high, low, close)
                    indicators[interval] = {
                        'rsi': rsi[-1],
                        'atr': atr[-1],
                        'sma_50': sma_50[-1],
                        'sma_200': sma_200[-1],
                        'ema_50': ema_50[-1],
                        'ema_200': ema_200[-1],
                        'macd': macd[-1],
                        'macd_signal': macd_signal[-1],
                        'slowk': slowk[-1],
                        'slowd': slowd[-1]
                    }
                else:
                    self.log.warning(f"Not enough data to calculate indicators for {symbol} at interval {interval}")
                    return None
            return indicators
        except Exception as e:
            self.log.exception(f"Error calculating indicators for {symbol}: {e}")
            return None

    async def calculate_volatility(self, symbol, window=20):
        data = await self.get_stock_data(symbol)
        if data is not None and len(data['close']) >= window:
            returns = np.log(data['close'][1:] / data['close'][:-1])
            volatility = np.std(returns[-window:]) * np.sqrt(252)
            return volatility
        else:
            self.log.warning(f"Not enough data to calculate volatility for {symbol}")
            return None

    async def calculate_dynamic_take_profit(self, symbol, entry_price):
        try:
            volatility = await self.calculate_volatility(symbol)
            if volatility is not None:
                take_profit = entry_price * (1 + 1.5 * volatility)
                return take_profit
            return None
        except Exception as e:
            self.log.exception(f"Error calculating dynamic take-profit for {symbol}: {e}")
            return None

    async def calculate_position_size(self, symbol, risk_per_trade=0.01):
        try:
            data = await self.get_stock_data(symbol)
            if data is not None:
                current_price = data['close'][-1]
                volatility = await self.calculate_volatility(symbol)
                if current_price is not None and volatility is not None:
                    stop_loss = current_price - (volatility * current_price)
                    risk_per_share = current_price - stop_loss
                    total_capital = await self.get_portfolio_value()
                    risk_amount = total_capital * risk_per_trade
                    shares_to_buy = risk_amount / risk_per_share
                    # Adjust position size based on VIX
                    vix = await self.get_vix()
                    if vix is not None:
                        vix_adjustment = min(max((30 - vix) / 30, 0.5), 1)
                        shares_to_buy *= vix_adjustment
                    return int(max(shares_to_buy, 0))
            return 0
        except Exception as e:
            self.log.exception(f"Error calculating position size for {symbol}: {e}")
            return 0

    async def check_market_conditions(self):
        try:
            spy_data = await self.get_stock_data('SPY')
            qqq_data = await self.get_stock_data('QQQ')
            if spy_data is not None and qqq_data is not None:
                if len(spy_data['close']) >= 200 and len(qqq_data['close']) >= 200:
                    spy_sma_50 = talib.SMA(spy_data['close'], timeperiod=50)[-1]
                    spy_sma_200 = talib.SMA(spy_data['close'], timeperiod=200)[-1]
                    qqq_sma_50 = talib.SMA(qqq_data['close'], timeperiod=50)[-1]
                    qqq_sma_200 = talib.SMA(qqq_data['close'], timeperiod=200)[-1]

                    if spy_sma_50 < spy_sma_200 and qqq_sma_50 < qqq_sma_200:
                        self.trading_halted = True
                        self.log.warning("Trading halted due to bearish conditions in SPY and QQQ.")
                    else:
                        self.trading_halted = False
                else:
                    self.log.warning("Not enough data to check market conditions.")
        except Exception as e:
            self.log.exception(f"Error checking market conditions: {e}")

    async def buy_assets(self):
        if self.trading_halted:
            return

        try:
            await self.get_exchange_rate()
            vix_value = await self.get_vix()
            if vix_value is not None:
                self.log.info(f"Current VIX value: {vix_value}")

                tasks = [self.process_buy_signal(symbol) for symbol in self.watchlist]
                await asyncio.gather(*tasks)
        except Exception as e:
            self.log.exception(f"Error in buy_assets: {e}")

    async def process_buy_signal(self, symbol):
        indicators = await self.calculate_indicators(symbol)
        if indicators:
            day_indicators = indicators['day']
            macd = day_indicators['macd']
            macd_signal = day_indicators['macd_signal']
            rsi = day_indicators['rsi']
            slowk = day_indicators['slowk']
            slowd = day_indicators['slowd']

            # Entry condition: RSI < 30, MACD crossover, Stochastic indicates oversold
            if rsi < 30 and macd > macd_signal and slowk < 20 and slowd < 20:
                data = await self.get_stock_data(symbol)
                if data is not None:
                    current_price = data['close'][-1]
                    quantity_to_buy = await self.calculate_position_size(symbol)
                    if quantity_to_buy > 0:
                        currency = 'USD' if symbol != 'IEUR' else 'EUR'
                        atr = day_indicators['atr']
                        await self.buy_asset(symbol, quantity_to_buy, currency, atr)
                        await self.log_trade('buy', symbol, quantity_to_buy, current_price)
                        take_profit = await self.calculate_dynamic_take_profit(symbol, current_price)
                        if take_profit:
                            self.portfolio[symbol]['take_profit'] = take_profit
                            self.log.info(f"Set dynamic take-profit for {symbol} at {take_profit}")

    async def buy_asset(self, symbol, quantity, currency, atr):
        try:
            contract = Stock(symbol, 'SMART', currency)
            order = MarketOrder('BUY', quantity)
            trade = self.ib.placeOrder(contract, order)
            while not trade.isDone():
                await asyncio.sleep(0.1)
            data = await self.get_stock_data(symbol)
            if data is not None:
                price = data['close'][-1]
                if currency == 'USD':
                    self.cash_usd -= quantity * price
                else:
                    self.cash_eur -= quantity * price

                if symbol in self.portfolio:
                    self.portfolio[symbol]['quantity'] += quantity
                    self.portfolio[symbol]['cost'] += quantity * price
                else:
                    self.portfolio[symbol] = {
                        'quantity': quantity, 'cost': quantity * price, 'currency': currency}
                self.atr_trailing_stop[symbol] = atr
        except Exception as e:
            self.log.exception(f"Error buying asset {symbol}: {e}")

    async def sell_asset(self, symbol, quantity, currency):
        try:
            contract = Stock(symbol, 'SMART', currency)
            order = MarketOrder('SELL', quantity)
            trade = self.ib.placeOrder(contract, order)
            while not trade.isDone():
                await asyncio.sleep(0.1)
            data = await self.get_stock_data(symbol)
            if data is not None:
                price = data['close'][-1]
                if currency == 'USD':
                    self.cash_usd += quantity * price
                else:
                    self.cash_eur += quantity * price

                self.portfolio[symbol]['quantity'] -= quantity
                if self.portfolio[symbol]['quantity'] <= 0:
                    del self.portfolio[symbol]

                await self.log_trade('sell', symbol, quantity, price)
        except Exception as e:
            self.log.exception(f"Error selling asset {symbol}: {e}")

    async def log_trade(self, trade_type, symbol, quantity, price):
        try:
            total_value_before = await self.get_portfolio_value()
            trade_value = quantity * price

            if trade_type == 'sell':
                avg_price = self.portfolio[symbol]['cost'] / self.portfolio[symbol]['quantity']
                realized_profit = trade_value - (avg_price * quantity)
                if realized_profit < 0:
                    self.total_losses += abs(realized_profit)

            self.log.info(f"{trade_type.capitalize()} {quantity} shares of {symbol} at {price}. "
                          f"Total value before trade: {total_value_before}, Trade value: {trade_value}. "
                          f"Updated total losses: {self.total_losses} ({(self.total_losses / self.initial_capital) * 100:.2f}%).")

            # Update peak portfolio value and calculate drawdown
            current_portfolio_value = await self.get_portfolio_value()
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
            drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, drawdown)

            if self.max_drawdown >= self.max_drawdown_limit:
                self.trading_halted = True
                self.log.warning("Trading halted due to maximum drawdown limit reached.")
        except Exception as e:
            self.log.exception(f"Error logging trade: {e}")

    async def manage_risk(self):
        if self.trading_halted:
            return

        try:
            symbols = list(self.portfolio.keys())
            await self.get_exchange_rate()
            tasks = [self.manage_risk_for_symbol(symbol) for symbol in symbols]
            await asyncio.gather(*tasks)
        except Exception as e:
            self.log.exception(f"Error in manage_risk: {e}")

    async def manage_risk_for_symbol(self, symbol):
        try:
            data = await self.get_stock_data(symbol)
            if data is not None:
                current_price = data['close'][-1]
                avg_price = self.portfolio[symbol]['cost'] / self.portfolio[symbol]['quantity']
                currency = self.portfolio[symbol]['currency']

                indicators = await self.calculate_indicators(symbol)
                if indicators:
                    day_indicators = indicators['day']
                    atr = day_indicators['atr']
                    atr_stop_price = avg_price - (atr * 2)
                    trailing_stop_price = max(self.atr_trailing_stop.get(symbol, atr_stop_price), current_price - atr * 2)
                    self.atr_trailing_stop[symbol] = trailing_stop_price

                    if current_price < trailing_stop_price:
                        await self.sell_asset(symbol, self.portfolio[symbol]['quantity'], currency)
                        self.log.info(f"ATR-based trailing stop triggered for {symbol}")

                    elif day_indicators['rsi'] > 70 and day_indicators['macd'] < day_indicators['macd_signal']:
                        await self.sell_asset(symbol, self.portfolio[symbol]['quantity'], currency)
                        self.log.info(f"Exit signal generated by RSI and MACD for {symbol}")

                    elif 'take_profit' in self.portfolio[symbol] and current_price >= self.portfolio[symbol]['take_profit']:
                        await self.sell_asset(symbol, self.portfolio[symbol]['quantity'], currency)
                        self.log.info(f"Dynamic take-profit triggered for {symbol}")

                    else:
                        new_take_profit = await self.calculate_dynamic_take_profit(symbol, current_price)
                        if new_take_profit and new_take_profit > self.portfolio[symbol].get('take_profit', 0):
                            self.portfolio[symbol]['take_profit'] = new_take_profit
                            self.log.info(f"Updated dynamic take-profit for {symbol} to {new_take_profit}")
        except Exception as e:
            self.log.exception(f"Error managing risk for {symbol}: {e}")

    async def check_free_capital(self):
        try:
            total_free_capital = self.cash_usd + (self.cash_eur * self.exchange_rate)
            if total_free_capital / self.initial_capital < 0.01:
                self.days_without_free_capital += 1
                if self.days_without_free_capital >= 7:
                    await self.start_fixed_investment_strategy()
            else:
                self.days_without_free_capital = 0  # Reset counter if free capital is sufficient
        except Exception as e:
            self.log.exception(f"Error in check_free_capital: {e}")

    async def start_fixed_investment_strategy(self):
        try:
            fixed_investment_value = self.initial_capital * 0.93
            for symbol in list(self.portfolio.keys()):
                data = await self.get_stock_data(symbol)
                if data is not None:
                    current_price = data['close'][-1]
                    current_value = self.portfolio[symbol]['quantity'] * current_price
                    if current_value > fixed_investment_value:
                        excess_value = current_value - fixed_investment_value
                        quantity_to_sell = int(excess_value // current_price)
                        if quantity_to_sell > 0:
                            await self.sell_asset(symbol, quantity_to_sell, self.portfolio[symbol]['currency'])
        except Exception as e:
            self.log.exception(f"Error in start_fixed_investment_strategy: {e}")

    async def get_portfolio_value(self):
        try:
            await self.get_exchange_rate()
            total_value = 0
            for symbol in self.portfolio:
                data = await self.get_stock_data(symbol)
                if data is not None:
                    current_price = data['close'][-1]
                    quantity = self.portfolio[symbol]['quantity']
                    currency = self.portfolio[symbol]['currency']
                    if currency == 'EUR':
                        current_price *= self.exchange_rate
                    total_value += current_price * quantity
            total_cash = self.cash_usd + (self.cash_eur * self.exchange_rate)
            return total_value + total_cash
        except Exception as e:
            self.log.exception(f"Error calculating portfolio value: {e}")
            return 0

    def generate_report(self):
        try:
            report = pd.DataFrame(self.report_data, columns=[
                                  'Trade Type', 'Symbol', 'Quantity', 'Price', 'Trade Value', 'Total Portfolio Value'])
            report.to_csv('trading_report.csv', index=False)
            self.log.info("Generated trading report: trading_report.csv")
            # Additional summary
            portfolio_summary = {
                'Total Portfolio Value': loop.run_until_complete(self.get_portfolio_value()),
                'Cash USD': self.cash_usd,
                'Cash EUR': self.cash_eur,
                'Total Losses': self.total_losses,
                'Positions': self.portfolio,
                'Max Drawdown': self.max_drawdown
            }
            with open('portfolio_summary.json', 'w') as f:
                json.dump(portfolio_summary, f, indent=4)
            self.log.info("Generated portfolio summary: portfolio_summary.json")
        except Exception as e:
            self.log.exception(f"Error generating report: {e}")

    async def update_watchlist(self):
        try:
            all_stocks = ['SPY', 'QQQ', 'IEUR', 'IWM',
                          'EFA', 'EEM', 'AGG', 'LQD', 'HYG', 'TLT']
            performance = []
            tasks = [self.get_stock_data(stock) for stock in all_stocks]
            results = await asyncio.gather(*tasks)
            for stock, data in zip(all_stocks, results):
                if data is not None and len(data['close']) >= 1:
                    perf = (data['close'][-1] - data['close'][0]) / data['close'][0]
                    performance.append((stock, perf))

            performance.sort(key=lambda x: x[1], reverse=True)
            # Keep top 5 performing stocks
            self.watchlist = [stock for stock, _ in performance[:5]]
            self.log.info(f"Updated watchlist: {self.watchlist}")
        except Exception as e:
            self.log.exception(f"Error updating watchlist: {e}")

    async def run(self):
        while True:
            try:
                await asyncio.gather(
                    self.check_market_conditions(),
                    self.buy_assets(),
                    self.manage_risk(),
                    self.check_free_capital(),
                    self.update_watchlist()
                )
                current_time = time.localtime()
                if current_time.tm_hour == 0 and current_time.tm_min == 0:
                    self.generate_report()
                await asyncio.sleep(60)
            except Exception as e:
                self.log.exception(f"Error in main loop: {e}")
                await asyncio.sleep(60)

def main():
    parser = argparse.ArgumentParser(description="Dynamic Portfolio Manager")
    parser.add_argument('--initial_capital', type=float, default=10000)
    parser.add_argument('--investment_horizon_years', type=int, default=30)
    parser.add_argument('--risk_profile', type=str, default='aggressive')
    args = parser.parse_args()

    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    manager = DynamicPortfolioManager(
        initial_capital=args.initial_capital,
        investment_horizon_years=args.investment_horizon_years,
        risk_profile=args.risk_profile,
        ib_client=ib
    )

    loop.run_until_complete(manager.run())

if __name__ == "__main__":
    main()

# The DynamicPortfolioManager class is designed to manage a portfolio of assets dynamically based on market conditions and risk management strategies.
# The manager uses the IB API to interact with the trading platform and execute trades.
# The main loop of the manager runs continuously and performs the following tasks:
# - Check market conditions for bearish signals in major indices (SPY, QQQ).
# - Buy assets based on technical indicators and risk management strategies.
# - Manage risk by setting trailing stops and take-profit levels.
# - Check free capital availability and start a fixed investment strategy if necessary.
# - Update the watchlist of assets based on performance.
# - Generate trading reports and portfolio summaries.
# The manager uses asyncio to run multiple tasks concurrently and handle asynchronous operations such as fetching data and placing orders.
# The main loop runs every minute and performs the tasks in parallel to react quickly to changing market conditions.
# The manager logs information and errors to a log file for monitoring and debugging purposes.
# The manager can be customized with different initial capital, investment horizon, and risk profile settings.