"""
Quant AI Trading Model - Comprehensive Backtester
Tests the strategy on historical data with realistic parameters
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    bb_position = (data - lower_band) / (upper_band - lower_band)
    return sma, upper_band, lower_band, bb_position

def calculate_volume_surge(volume, period=20, threshold=1.5):
    """Calculate volume surge"""
    vol_ma = volume.rolling(window=period).mean()
    surge = (volume > vol_ma * threshold).astype(int)
    return surge

def calculate_momentum(data, period=9):
    """Calculate momentum (ROC)"""
    price_change = (data - data.shift(period)) / data.shift(period)
    return price_change

def calculate_market_structure(high, low, lookback=5):
    """Calculate market structure (HH/LL)"""
    highest_recent = high.rolling(window=lookback).max()
    lowest_recent = low.rolling(window=lookback).min()
    
    higher_high = high > highest_recent.shift(1)
    lower_low = low < lowest_recent.shift(1)
    
    return higher_high, lower_low

def generate_signals(df, rsi_overbought=70, rsi_oversold=30, confluence_threshold=3):
    """Generate trading signals based on quant indicators"""
    
    # Calculate all indicators
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['Close'], fast=12, slow=26, signal=9)
    df['SMA_Middle'], df['BB_Upper'], df['BB_Lower'], df['BB_Position'] = calculate_bollinger_bands(df['Close'], period=20, std_dev=2.0)
    df['Volume_Surge'] = calculate_volume_surge(df['Volume'], period=20, threshold=1.5)
    df['Momentum'] = calculate_momentum(df['Close'], period=9)
    df['Higher_High'], df['Lower_Low'] = calculate_market_structure(df['High'], df['Low'], lookback=5)
    
    # Calculate component scores
    df['RSI_Extreme'] = 0
    df.loc[df['RSI'] < rsi_oversold, 'RSI_Extreme'] = 1
    df.loc[df['RSI'] > rsi_overbought, 'RSI_Extreme'] = -1
    
    df['MACD_Bullish'] = ((df['MACD'] > df['Signal_Line']) & (df['MACD_Histogram'] > 0)).astype(int)
    df['MACD_Bearish'] = ((df['MACD'] < df['Signal_Line']) & (df['MACD_Histogram'] < 0)).astype(int)
    
    df['BB_Extreme'] = 0
    df.loc[df['BB_Position'] < 0.2, 'BB_Extreme'] = 1
    df.loc[df['BB_Position'] > 0.8, 'BB_Extreme'] = -1
    
    df['Momentum_Bullish'] = (df['Momentum'] > 0.01).astype(int)
    df['Momentum_Bearish'] = (df['Momentum'] < -0.01).astype(int)
    
    df['Structure_Bullish'] = (df['Higher_High'] & (df['Close'] > df['Close'].shift(1))).astype(int)
    df['Structure_Bearish'] = (df['Lower_Low'] & (df['Close'] < df['Close'].shift(1))).astype(int)
    
    # Calculate ML Score (0-100)
    rsi_score = np.abs(df['RSI_Extreme']) * 25
    macd_score = np.abs((df['MACD_Bullish'].astype(int)) + (df['MACD_Bearish'].astype(int)) * (-1)) * 20
    bb_score = np.abs(df['BB_Extreme']) * 20
    volume_score = df['Volume_Surge'] * 15
    momentum_score = np.abs((df['Momentum_Bullish'].astype(int)) + (df['Momentum_Bearish'].astype(int)) * (-1)) * 20
    
    df['ML_Score'] = np.minimum(100, rsi_score + macd_score + bb_score + volume_score + momentum_score)
    
    # Calculate Confluence
    df['Confluence_Long'] = (
        (df['RSI_Extreme'] == 1).astype(int) +
        (df['MACD_Bullish'] == 1).astype(int) +
        (df['BB_Extreme'] == 1).astype(int) +
        (df['Structure_Bullish'] == 1).astype(int) +
        (df['Momentum_Bullish'] == 1).astype(int)
    )
    
    df['Confluence_Short'] = (
        (df['RSI_Extreme'] == -1).astype(int) +
        (df['MACD_Bearish'] == 1).astype(int) +
        (df['BB_Extreme'] == -1).astype(int) +
        (df['Structure_Bearish'] == 1).astype(int) +
        (df['Momentum_Bearish'] == 1).astype(int)
    )
    
    df['Confluence'] = np.maximum(df['Confluence_Long'], df['Confluence_Short'])
    
    # Generate final signals
    df['Buy_Signal'] = (df['Confluence_Long'] >= confluence_threshold) & (df['RSI_Extreme'] == 1)
    df['Sell_Signal'] = (df['Confluence_Short'] >= confluence_threshold) & (df['RSI_Extreme'] == -1)
    
    return df

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class QuantBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000, 
                 risk_per_trade=0.02, stop_loss_atr=2.0, take_profit_ratio=2.0,
                 commission=0.001):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_ratio = take_profit_ratio
        self.commission = commission
        
        self.data = None
        self.trades = []
        self.equity_curve = []
        
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance"""
        print(f"Fetching data for {self.symbol}...")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)
        self.data['TR'] = self.calculate_true_range()
        self.data['ATR'] = self.data['TR'].rolling(window=14).mean()
        return self.data
    
    def calculate_true_range(self):
        """Calculate True Range for ATR"""
        tr1 = self.data['High'] - self.data['Low']
        tr2 = np.abs(self.data['High'] - self.data['Close'].shift())
        tr3 = np.abs(self.data['Low'] - self.data['Close'].shift())
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def backtest(self):
        """Run backtest"""
        print(f"Running backtest from {self.start_date} to {self.end_date}...")
        
        # Generate signals
        self.data = generate_signals(self.data.copy())
        
        cash = self.initial_capital
        position = None
        entry_price = 0
        entry_date = None
        entry_confluence = 0
        
        for idx in range(len(self.data)):
            date = self.data.index[idx]
            close = self.data['Close'].iloc[idx]
            high = self.data['High'].iloc[idx]
            low = self.data['Low'].iloc[idx]
            atr = self.data['ATR'].iloc[idx]
            ml_score = self.data['ML_Score'].iloc[idx]
            confluence = self.data['Confluence'].iloc[idx]
            
            buy_signal = self.data['Buy_Signal'].iloc[idx]
            sell_signal = self.data['Sell_Signal'].iloc[idx]
            
            # Close existing position if take profit or stop loss hit
            if position is not None:
                if position == 'LONG':
                    stop_loss = entry_price - (atr * self.stop_loss_atr)
                    take_profit = entry_price + (atr * self.stop_loss_atr * self.take_profit_ratio)
                    
                    if high >= take_profit or low <= stop_loss:
                        exit_price = take_profit if high >= take_profit else stop_loss
                        profit = (exit_price - entry_price) * (cash / entry_price)
                        cash += profit - (abs(profit) * self.commission)
                        
                        self.trades.append({
                            'Entry_Date': entry_date,
                            'Exit_Date': date,
                            'Entry_Price': entry_price,
                            'Exit_Price': exit_price,
                            'Type': 'LONG',
                            'Profit_Loss': profit,
                            'Profit_Pct': (exit_price - entry_price) / entry_price,
                            'Confluence': entry_confluence,
                            'ML_Score': ml_score
                        })
                        position = None
                
                elif position == 'SHORT':
                    stop_loss = entry_price + (atr * self.stop_loss_atr)
                    take_profit = entry_price - (atr * self.stop_loss_atr * self.take_profit_ratio)
                    
                    if low <= take_profit or high >= stop_loss:
                        exit_price = take_profit if low <= take_profit else stop_loss
                        profit = (entry_price - exit_price) * (cash / entry_price)
                        cash += profit - (abs(profit) * self.commission)
                        
                        self.trades.append({
                            'Entry_Date': entry_date,
                            'Exit_Date': date,
                            'Entry_Price': entry_price,
                            'Exit_Price': exit_price,
                            'Type': 'SHORT',
                            'Profit_Loss': profit,
                            'Profit_Pct': (entry_price - exit_price) / entry_price,
                            'Confluence': entry_confluence,
                            'ML_Score': ml_score
                        })
                        position = None
            
            # Open new position if signal
            if position is None:
                if buy_signal and not np.isnan(atr):
                    position = 'LONG'
                    entry_price = close
                    entry_date = date
                    entry_confluence = int(confluence)
                    position_size = (cash * self.risk_per_trade) / (atr * self.stop_loss_atr)
                
                elif sell_signal and not np.isnan(atr):
                    position = 'SHORT'
                    entry_price = close
                    entry_date = date
                    entry_confluence = int(confluence)
                    position_size = (cash * self.risk_per_trade) / (atr * self.stop_loss_atr)
            
            # Record equity
            equity = cash
            if position == 'LONG':
                equity = cash + ((close - entry_price) * (cash / entry_price))
            elif position == 'SHORT':
                equity = cash + ((entry_price - close) * (cash / entry_price))
            
            self.equity_curve.append({
                'Date': date,
                'Equity': equity,
                'Cash': cash,
                'Position': position
            })
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze backtest results"""
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        if len(trades_df) == 0:
            print("❌ No trades generated!")
            return None
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['Profit_Loss'] > 0])
        losing_trades = len(trades_df[trades_df['Profit_Loss'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = trades_df['Profit_Loss'].sum()
        avg_win = trades_df[trades_df['Profit_Loss'] > 0]['Profit_Loss'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['Profit_Loss'] < 0]['Profit_Loss'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['Profit_Loss'] > 0]['Profit_Loss'].sum() / 
                          trades_df[trades_df['Profit_Loss'] < 0]['Profit_Loss'].sum()) if losing_trades > 0 else 0
        
        final_equity = equity_df['Equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe Ratio
        equity_returns = equity_df['Equity'].pct_change()
        sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if equity_returns.std() > 0 else 0
        
        # Calculate Max Drawdown
        cummax = equity_df['Equity'].cummax()
        drawdown = (equity_df['Equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Risk/Reward Ratio
        if avg_loss != 0:
            risk_reward_ratio = abs(avg_win / avg_loss)
        else:
            risk_reward_ratio = 0
        
        results = {
            'Symbol': self.symbol,
            'Period': f"{self.start_date} to {self.end_date}",
            'Initial_Capital': self.initial_capital,
            'Final_Equity': final_equity,
            'Total_Return_%': total_return * 100,
            'Total_Trades': total_trades,
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Rate_%': win_rate * 100,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Risk_Reward_Ratio': risk_reward_ratio,
            'Profit_Factor': profit_factor,
            'Total_Profit': total_profit,
            'Max_Drawdown_%': max_drawdown * 100,
            'Sharpe_Ratio': sharpe_ratio,
            'Annual_Return_%': (total_return / ((self.end_date - self.start_date).days / 365)) * 100
        }
        
        return results, trades_df, equity_df
    
    def print_results(self, results):
        """Pretty print results"""
        if results is None:
            return
        
        results_dict, trades_df, equity_df = results
        
        print("\n" + "="*70)
        print(f"BACKTEST RESULTS: {results_dict['Symbol']}")
        print("="*70)
        print(f"Period: {results_dict['Period']}")
        print(f"Initial Capital: €{results_dict['Initial_Capital']:,.2f}")
        print(f"Final Equity: €{results_dict['Final_Equity']:,.2f}")
        print(f"Total Return: {results_dict['Total_Return_%']:.2f}%")
        print(f"Annual Return: {results_dict['Annual_Return_%']:.2f}%")
        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades: {results_dict['Total_Trades']}")
        print(f"Winning Trades: {results_dict['Winning_Trades']}")
        print(f"Losing Trades: {results_dict['Losing_Trades']}")
        print(f"Win Rate: {results_dict['Win_Rate_%']:.2f}%")
        print(f"Avg Win: €{results_dict['Avg_Win']:,.2f}")
        print(f"Avg Loss: €{results_dict['Avg_Loss']:,.2f}")
        print(f"Risk/Reward Ratio: {results_dict['Risk_Reward_Ratio']:.2f}")
        print(f"Profit Factor: {results_dict['Profit_Factor']:.2f}")
        print(f"Total Profit: €{results_dict['Total_Profit']:,.2f}")
        print("\n--- RISK METRICS ---")
        print(f"Max Drawdown: {results_dict['Max_Drawdown_%']:.2f}%")
        print(f"Sharpe Ratio: {results_dict['Sharpe_Ratio']:.2f}")
        print("="*70 + "\n")

# ============================================================================
# RUN BACKTESTS ON MULTIPLE INSTRUMENTS
# ============================================================================

def main():
    """Run comprehensive backtests"""
    
    # Test parameters
    end_date = datetime.now()
    
    # Test on different instruments and timeframes
    test_cases = [
        # (Symbol, Start_Date, Description)
        ('BTC-USD', end_date - timedelta(days=365*2), "Bitcoin - 2 Years"),
        ('ETH-USD', end_date - timedelta(days=365*2), "Ethereum - 2 Years"),
        ('AAPL', end_date - timedelta(days=365*2), "Apple - 2 Years"),
        ('MSFT', end_date - timedelta(days=365*2), "Microsoft - 2 Years"),
        ('EURUSD=X', end_date - timedelta(days=365*2), "EUR/USD - 2 Years"),
        ('SPY', end_date - timedelta(days=365*2), "S&P 500 ETF - 2 Years"),
    ]
    
    all_results = []
    
    for symbol, start_date, description in test_cases:
        try:
            print(f"\n🔄 Testing: {description}")
            
            backtester = QuantBacktester(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000,
                risk_per_trade=0.02,
                stop_loss_atr=2.0,
                take_profit_ratio=2.0,
                commission=0.001
            )
            
            backtester.fetch_data()
            results = backtester.backtest()
            
            if results is not None:
                backtester.print_results(results)
                results_dict, _, _ = results
                all_results.append(results_dict)
        
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
    
    # Summary table
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        print(summary_df[['Symbol', 'Total_Return_%', 'Win_Rate_%', 'Sharpe_Ratio', 'Max_Drawdown_%']].to_string(index=False))
        print("="*70)
        
        # Save detailed results
        summary_df.to_csv('backtest_results.csv', index=False)
        print("\n✅ Detailed results saved to 'backtest_results.csv'")

if __name__ == "__main__":
    main()
