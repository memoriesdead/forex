"""
Forex Backtesting Engine - Production-Ready
Backtest strategies against historical data with realistic execution

Usage:
    python backtest.py --strategy momentum --start 2026-01-01 --end 2026-01-08
    python backtest.py --strategy ml_ensemble --model timeseries --pairs EUR_USD,GBP_USD
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class BacktestEngine:
    def __init__(self, initial_capital=10000, commission_pips=0.5):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_pips = commission_pips

        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0

    def load_historical_data(self, pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        # Try multiple data sources
        data_sources = [
            project_root / "data" / "live",
            project_root / "data" / "dukascopy_local",
            project_root / "data_cleaned" / "training"
        ]

        for source in data_sources:
            if not source.exists():
                continue

            # Look for date range files
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')

            all_data = []
            current = start

            while current <= end:
                date_str = current.strftime('%Y-%m-%d')
                file_pattern = f"{pair.replace('_', '')}_{date_str}.csv"

                # Check in date subdirectory
                date_dir = source / date_str
                if date_dir.exists():
                    file_path = date_dir / f"{pair.replace('_', '')}_{date_str}.csv"
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        all_data.append(df)

                current += timedelta(days=1)

            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                combined = combined.sort_values('timestamp')
                return combined

        print(f"[WARNING] No data found for {pair} from {start_date} to {end_date}")
        return pd.DataFrame()

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for strategy"""
        df = data.copy()

        # Use mid price
        df['price'] = (df['bid'] + df['ask']) / 2

        # Returns
        df['returns'] = df['price'].pct_change()
        df['returns_5'] = df['price'].pct_change(5)
        df['returns_20'] = df['price'].pct_change(20)

        # Moving averages
        df['ma_20'] = df['price'].rolling(20).mean()
        df['ma_50'] = df['price'].rolling(50).mean()
        df['ma_200'] = df['price'].rolling(200).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['atr'] = df['price'].diff().abs().rolling(14).mean()

        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_upper'] = df['ma_20'] + 2 * df['price'].rolling(20).std()
        df['bb_lower'] = df['ma_20'] - 2 * df['price'].rolling(20).std()

        # Momentum
        df['momentum'] = df['price'] - df['price'].shift(10)

        # Trend strength
        df['trend'] = (df['ma_20'] - df['ma_50']) / df['ma_50']

        return df

    def momentum_strategy(self, data: pd.DataFrame, session: str = 'morning') -> pd.DataFrame:
        """
        Momentum strategy (baseline)
        - Buy when price > MA20 and MA20 > MA50 (uptrend)
        - Sell when price < MA20 and MA20 < MA50 (downtrend)
        """
        df = data.copy()

        # Session-specific parameters
        if session == 'morning':
            confidence_threshold = 0.52
            risk_multiplier = 1.5
        else:  # evening
            confidence_threshold = 0.60
            risk_multiplier = 1.0

        # Generate signals
        df['signal'] = 0

        # Bullish: price > MA20, MA20 > MA50, positive momentum
        bullish = (df['price'] > df['ma_20']) & (df['ma_20'] > df['ma_50']) & (df['momentum'] > 0)
        df.loc[bullish, 'signal'] = 1

        # Bearish: price < MA20, MA20 < MA50, negative momentum
        bearish = (df['price'] < df['ma_20']) & (df['ma_20'] < df['ma_50']) & (df['momentum'] < 0)
        df.loc[bearish, 'signal'] = -1

        # Calculate confidence (0-1)
        # Based on trend strength, RSI, and volatility
        trend_conf = np.abs(df['trend']).clip(0, 0.1) / 0.1
        rsi_conf = np.where(
            df['signal'] == 1,
            (df['rsi'] - 50).clip(0, 50) / 50,  # Bullish: RSI > 50
            (50 - df['rsi']).clip(0, 50) / 50   # Bearish: RSI < 50
        )
        vol_conf = 1 - df['volatility'].clip(0, 0.01) / 0.01  # Lower vol = higher conf

        df['confidence'] = (trend_conf + rsi_conf + vol_conf) / 3

        # Filter by confidence threshold
        df.loc[df['confidence'] < confidence_threshold, 'signal'] = 0

        # Position size (risk multiplier applied)
        df['position_size'] = df['confidence'] * risk_multiplier

        return df

    def ml_ensemble_strategy(self, data: pd.DataFrame, model_type: str = 'timeseries') -> pd.DataFrame:
        """
        ML ensemble strategy (placeholder - integrate your models)
        Combines predictions from multiple models
        """
        df = data.copy()

        # Placeholder: Use technical indicators as proxy for ML predictions
        # In production, replace with actual model predictions

        print(f"[INFO] ML strategy with {model_type} - using technical indicators as proxy")
        print("[INFO] Replace with actual model predictions in production")

        # Generate signals from multiple "models" (using different indicators)
        # Model 1: Trend following
        trend_signal = np.where(df['ma_20'] > df['ma_50'], 1, -1)

        # Model 2: Mean reversion
        reversion_signal = np.where(df['price'] < df['bb_lower'], 1,
                                   np.where(df['price'] > df['bb_upper'], -1, 0))

        # Model 3: Momentum
        momentum_signal = np.where(df['momentum'] > 0, 1, -1)

        # Model 4: RSI
        rsi_signal = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))

        # Ensemble: Average signals
        df['signal'] = (trend_signal + reversion_signal + momentum_signal + rsi_signal) / 4

        # Convert to discrete signals
        df['signal'] = np.where(df['signal'] > 0.3, 1, np.where(df['signal'] < -0.3, -1, 0))

        # Confidence: Agreement among models
        signals = np.column_stack([trend_signal, reversion_signal, momentum_signal, rsi_signal])
        agreement = np.abs(signals.sum(axis=1)) / 4
        df['confidence'] = agreement

        # Position size
        df['position_size'] = df['confidence']

        return df

    def execute_backtest(self, data: pd.DataFrame, pair: str, stop_loss_pips: int = 20,
                        take_profit_pips: int = 40) -> Dict:
        """Execute backtest with realistic execution"""
        if data.empty:
            return {}

        position = None
        entry_price = None
        entry_time = None

        for idx, row in data.iterrows():
            if idx < 200:  # Skip initial rows (MA200 warmup)
                continue

            timestamp = row['timestamp']
            price = row['price']
            signal = row.get('signal', 0)
            position_size = row.get('position_size', 1.0)

            # Check for position exit (SL/TP)
            if position is not None:
                if position == 1:  # Long position
                    # Stop loss hit
                    if price <= entry_price - (stop_loss_pips * 0.0001):
                        pnl_pips = -stop_loss_pips - self.commission_pips
                        self.close_position(pair, timestamp, price, pnl_pips, 'SL')
                        position = None
                    # Take profit hit
                    elif price >= entry_price + (take_profit_pips * 0.0001):
                        pnl_pips = take_profit_pips - self.commission_pips
                        self.close_position(pair, timestamp, price, pnl_pips, 'TP')
                        position = None

                elif position == -1:  # Short position
                    # Stop loss hit
                    if price >= entry_price + (stop_loss_pips * 0.0001):
                        pnl_pips = -stop_loss_pips - self.commission_pips
                        self.close_position(pair, timestamp, price, pnl_pips, 'SL')
                        position = None
                    # Take profit hit
                    elif price <= entry_price - (take_profit_pips * 0.0001):
                        pnl_pips = take_profit_pips - self.commission_pips
                        self.close_position(pair, timestamp, price, pnl_pips, 'TP')
                        position = None

            # Check for new position entry
            if position is None and signal != 0:
                position = signal
                entry_price = price
                entry_time = timestamp

                self.open_position(pair, timestamp, price, signal, position_size)

        # Close any open position at end
        if position is not None:
            final_price = data.iloc[-1]['price']
            pnl_pips = (final_price - entry_price) * position * 10000 - self.commission_pips
            self.close_position(pair, data.iloc[-1]['timestamp'], final_price, pnl_pips, 'EOD')

        return self.calculate_performance()

    def open_position(self, pair: str, timestamp, price: float, direction: int, size: float):
        """Open a position"""
        self.positions[pair] = {
            'direction': direction,
            'entry_price': price,
            'entry_time': timestamp,
            'size': size
        }

    def close_position(self, pair: str, timestamp, price: float, pnl_pips: float, reason: str):
        """Close a position and record trade"""
        if pair not in self.positions:
            return

        pos = self.positions[pair]

        # Calculate P&L
        pip_value = 10  # $10 per pip for standard lot
        pnl = pnl_pips * pip_value * pos['size']

        self.capital += pnl
        self.total_pnl += pnl

        # Record trade
        trade = {
            'pair': pair,
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl,
            'size': pos['size'],
            'reason': reason
        }
        self.trades.append(trade)

        # Update stats
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.capital
        })

        # Update max drawdown
        peak = max([self.initial_capital] + [e['equity'] for e in self.equity_curve])
        drawdown = (peak - self.capital) / peak
        self.max_drawdown = max(self.max_drawdown, drawdown)

        del self.positions[pair]

    def calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'error': 'No trades executed'
            }

        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # Average win/loss
        wins = [t['pnl_usd'] for t in self.trades if t['pnl_usd'] > 0]
        losses = [t['pnl_usd'] for t in self.trades if t['pnl_usd'] < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Returns
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Sharpe ratio (simplified - using daily returns if available)
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change()
            sharpe = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if equity_df['returns'].std() > 0 else 0
        else:
            sharpe = 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return_pct': total_return * 100,
            'total_return_usd': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate * 100,
            'avg_win_usd': avg_win,
            'avg_loss_usd': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': self.max_drawdown * 100,
            'sharpe_ratio': sharpe
        }

    def print_results(self, results: Dict):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)

        if 'error' in results:
            print(f"[ERROR] {results['error']}")
            return

        print(f"\nCapital:")
        print(f"  Initial:  ${results['initial_capital']:,.2f}")
        print(f"  Final:    ${results['final_capital']:,.2f}")
        print(f"  Return:   {results['total_return_pct']:.2f}% (${results['total_return_usd']:,.2f})")

        print(f"\nTrades:")
        print(f"  Total:    {results['total_trades']}")
        print(f"  Winners:  {results['winning_trades']} ({results['win_rate_pct']:.1f}%)")
        print(f"  Losers:   {results['losing_trades']}")

        print(f"\nPerformance:")
        print(f"  Avg Win:  ${results['avg_win_usd']:.2f}")
        print(f"  Avg Loss: ${results['avg_loss_usd']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Max Drawdown:  {results['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:  {results['sharpe_ratio']:.2f}")

        print("\n" + "="*70)

        # Recent trades
        if self.trades:
            print("\nRecent Trades (Last 10):")
            print("-"*70)
            for trade in self.trades[-10:]:
                print(f"{trade['exit_time']} | {trade['direction']:5s} | {trade['pair']:8s} | "
                      f"P&L: {trade['pnl_pips']:6.1f} pips (${trade['pnl_usd']:7.2f}) | {trade['reason']}")

        print()


def main():
    parser = argparse.ArgumentParser(description='Forex Backtesting Engine')

    parser.add_argument('--strategy', choices=['momentum', 'ml_ensemble'], default='momentum',
                       help='Strategy to backtest')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--pairs', default='EUR_USD,GBP_USD,USD_JPY',
                       help='Comma-separated pairs to trade')
    parser.add_argument('--session', choices=['morning', 'evening'], default='morning',
                       help='Trading session')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital')
    parser.add_argument('--sl-pips', type=int, default=20,
                       help='Stop loss in pips')
    parser.add_argument('--tp-pips', type=int, default=40,
                       help='Take profit in pips')
    parser.add_argument('--model', default='timeseries',
                       help='ML model type (for ml_ensemble strategy)')

    args = parser.parse_args()

    print("="*70)
    print("FOREX BACKTESTING ENGINE")
    print("="*70)
    print(f"Strategy:     {args.strategy}")
    print(f"Period:       {args.start} to {args.end}")
    print(f"Pairs:        {args.pairs}")
    print(f"Session:      {args.session}")
    print(f"Capital:      ${args.capital:,.2f}")
    print(f"Stop Loss:    {args.sl_pips} pips")
    print(f"Take Profit:  {args.tp_pips} pips")
    print("="*70)

    pairs = args.pairs.split(',')

    # Run backtest for each pair
    all_results = []

    for pair in pairs:
        print(f"\n[BACKTEST] {pair}...")

        engine = BacktestEngine(initial_capital=args.capital / len(pairs))

        # Load data
        data = engine.load_historical_data(pair, args.start, args.end)

        if data.empty:
            print(f"[SKIP] No data for {pair}")
            continue

        print(f"[DATA] Loaded {len(data)} ticks")

        # Calculate features
        data = engine.calculate_features(data)

        # Apply strategy
        if args.strategy == 'momentum':
            data = engine.momentum_strategy(data, session=args.session)
        elif args.strategy == 'ml_ensemble':
            data = engine.ml_ensemble_strategy(data, model_type=args.model)

        # Execute backtest
        results = engine.execute_backtest(data, pair,
                                         stop_loss_pips=args.sl_pips,
                                         take_profit_pips=args.tp_pips)

        results['pair'] = pair
        all_results.append(results)

        # Print individual pair results
        engine.print_results(results)

    # Combined results
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("COMBINED RESULTS (All Pairs)")
        print("="*70)

        total_return = sum(r.get('total_return_usd', 0) for r in all_results)
        total_trades = sum(r.get('total_trades', 0) for r in all_results)
        total_wins = sum(r.get('winning_trades', 0) for r in all_results)

        combined_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        combined_return_pct = (total_return / args.capital) * 100

        print(f"Total Return: ${total_return:,.2f} ({combined_return_pct:.2f}%)")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate:     {combined_win_rate:.1f}%")
        print()


if __name__ == "__main__":
    main()
