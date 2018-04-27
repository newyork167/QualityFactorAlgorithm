import pandas as pd
import pandas_get_stock_data as pgsd
import pandas_technical_indicators as pti
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

output_dir = 'figures'
last_ao = None

start_date = datetime(2017, 1, 1)
end_date = datetime(2018, 4, 26)

output_file = open('output.txt', 'w+')
open_positions_file = open('open_positions.txt', 'w+')


def output_to_file(s):
    print(s)
    output_file.write(s + '\n')
    output_file.flush()


def output_to_open_positions_file(s):
    open_positions_file.write(s.strip() + '\n')
    open_positions_file.flush()


def calculate_delta(df):
    return df['Close'] - df['Open']


def calculate_open_rolling_mean(x, window=20):
    return pd.rolling_mean(x, window)


def calculate_rolling_functions(df):
    df['date'] = df.index

    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    df['20d_ma'] = df['Adj Close'].rolling(center=False, window=20).mean()
    df['50d_ma'] = df['Adj Close'].rolling(center=False, window=50).mean()
    df['Bol_upper'] = df['Adj Close'].rolling(center=False, window=20).mean() + 2 * df['Adj Close'].rolling(center=False, window=20, min_periods=20).std()
    df['Bol_lower'] = df['Adj Close'].rolling(center=False, window=20).mean() - 2 * df['Adj Close'].rolling(center=False, window=20, min_periods=20).std()
    df['Bol_BW'] = ((df['Bol_upper'] - df['Bol_lower']) / df['20d_ma']) * 100
    df['Bol_BW_200MA'] = df['Bol_BW'].rolling(center=False, window=50).mean()
    df['Bol_BW_200MA'] = df['Bol_BW_200MA'].fillna(method='backfill')
    df['20d_exma'] = df['Adj Close'].ewm(span=20, adjust=True, min_periods=0, ignore_na=False).mean()
    df['50d_exma'] = df['Adj Close'].ewm(span=50, adjust=True, min_periods=0, ignore_na=False).mean()

    return df


def colourize_ao(x):
    global last_ao
    colour = 0
    if last_ao is None or x['AO'] > last_ao:
        colour = 1
    last_ao = x['AO']
    return colour


def plot_stock_data(ticker):
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2018, 4, 25)

    print('Getting stock data for {ticker} between {start_date} - {end_date}'.format(ticker=ticker, start_date=start_date, end_date=end_date))

    df = pgsd.get_stock_data_iex(ticker=ticker, start_date=start_date, end_date=end_date)
    df = pti.momentum(df, 10)
    df = pti.bollinger_bands(df, 20)
    df = pti.ICHIMOKU(df)
    print(df.head(15))

    dates = df.index.values.tolist()
    df.plot(y=['Open', 'Ichimoku Conversion Line', 'Ichimoku Base Line', 'Ichimoku Leading Span A', 'Ichimoku Leading Span B', 'Ichimoku Lagging Span'], title=ticker)
    plt.fill_between(dates, df['Ichimoku Leading Span A'], df['Ichimoku Leading Span B'],
                     where=df['Ichimoku Leading Span A'] >= df['Ichimoku Leading Span B'],
                     facecolor='green', alpha=0.2, interpolate=True)
    plt.fill_between(dates, df['Ichimoku Leading Span A'], df['Ichimoku Leading Span B'],
                     where=df['Ichimoku Leading Span A'] < df['Ichimoku Leading Span B'],
                     facecolor='red', alpha=0.2, interpolate=True)
    # plt.show()
    plt.savefig("{output_dir}/{ticker}.png".format(output_dir=output_dir, ticker=ticker))


def determine_cross_color(df):
    df['difference'] = df['AO'] - df['macd']
    sub_df = df[pd.notnull(df['AO'])]
    sub_df = sub_df[pd.notnull(sub_df['macd'])]
    sub_df['cross'] = np.sign(sub_df.difference.shift(1)) != np.sign(sub_df.difference)

    cross_color = 'r'
    skip_first = True
    for i, row in sub_df.iterrows():
        if row['cross']:
            if skip_first:
                skip_first = False
                sub_df.set_value(i, 'cross_color', 'b')
                continue
            cross_color = 'r' if cross_color != 'r' else 'g'
            sub_df.set_value(i, 'cross_color', cross_color)

    return sub_df


def get_profit(ticker, sub_df, starting_capital):
    total_ending_profit = 0

    cross_color = 'r'
    skip_first = True
    current_capital = starting_capital
    shares_owned = 0

    output_to_file("Getting profit data for {ticker}".format(ticker=ticker))
    output_to_file("Starting capital: ${starting_capital}".format(starting_capital=starting_capital))

    for i, row in sub_df.iterrows():
        if row['cross']:
            print(i)
            if skip_first:
                skip_first = False
                sub_df.set_value(i, 'cross_color', 'b')
                cross_color = 'b'
                continue
            share_price = row['Open']
            if cross_color != 'r':
                shares_owned = int(float(current_capital) / share_price)
                output_to_file("Buying {shares_owned} shares for ${share_price:.2f} at {sell_date}".format(shares_owned=shares_owned, share_price=share_price, sell_date=row['Date']))
            else:
                current_capital = shares_owned * share_price
                output_to_file("Selling {shares_owned} shares for ${share_price:.2f} at {sell_date}".format(shares_owned=shares_owned, share_price=share_price, sell_date=row['Date']))
                shares_owned = 0
            cross_color = 'r' if cross_color != 'r' else 'g'
            sub_df.set_value(i, 'cross_color', cross_color)

    # Close out this position
    if shares_owned > 0:
        current_capital = shares_owned * share_price
        output_to_file("Closing - Selling {shares_owned} shares for ${share_price:.2f}".format(shares_owned=shares_owned, share_price=share_price))
        output_to_open_positions_file(ticker)
    else:
        output_to_file("Position already closed")

    output_to_file("Ending capital: ${ending_capital:.2f}".format(ending_capital=current_capital))
    output_to_file("Net Gain: ${net_gain:.2f}".format(net_gain=(current_capital - starting_capital)))
    output_to_file("Percent increase: {percent_increase:.2f}%".format(percent_increase=100.0 * ((current_capital - starting_capital) / starting_capital)))

    total_ending_profit += (current_capital - starting_capital)
    output_to_file("\n\n" + '-' * 50)

    return total_ending_profit


def get_baseline_profit(total_capital):
    global start_date, end_date

    # Calculate profit of just buying and holding SPY in same timeframe
    baseline_df = pgsd.get_stock_data_morningstar(ticker='SPY', start_date=start_date, end_date=end_date)
    baseline_initial_price = baseline_df['Open'].iloc[0]
    baseline_shares_to_buy = total_capital / baseline_initial_price
    baseline_final_price = baseline_df['Open'].iloc[-1]
    baseline_excess = (baseline_shares_to_buy * baseline_final_price) - total_capital

    output_to_file("Profit holding baseline: ${0}".format(round(baseline_excess, 2)))
    output_to_file("Percent profit: {0}%".format(round(100 * baseline_excess / total_capital, 2)))
    output_to_file("-" * 50)

    return baseline_excess


def calculate_indicators(df):
    global last_ao
    df['Open Rolling Mean'] = df['Open'].rolling(10).mean()
    df['Close Rolling Mean'] = df['Close'].rolling(10).mean()

    df = calculate_rolling_functions(df)

    df = pti.MACD(df)

    last_ao = None
    df['AO'] = pti.AO(df)
    df['AO_Colour'] = df.apply(lambda row: colourize_ao(row), axis=1)

    return df


def main():
    global last_ao, start_date, end_date
    tech_sector_stock_tickers = ['CVLT', 'ACIW', 'GOOGL', 'GPN', 'GDDY', 'CTSH', 'CRTO', 'BOX', 'ADSK', 'WIX']

    total_capital = 100000
    starting_capital = round(float(total_capital) / len(tech_sector_stock_tickers), 2)
    total_ending_profit = 0

    for ticker in tech_sector_stock_tickers:
        df = pgsd.get_stock_data_morningstar(ticker=ticker, start_date=start_date, end_date=end_date)

        df = df.reset_index()

        try:
            df['Delta'] = df.apply(lambda row: calculate_delta(row), axis=1)
        except Exception as ex:
            print(ticker, ex)
            tech_sector_stock_tickers.remove(ticker)

        df = calculate_indicators(df)

        sub_df = determine_cross_color(df)

        total_ending_profit += get_profit(ticker=ticker, sub_df=sub_df, starting_capital=starting_capital)

    output_to_file("Model Net Gain: {}".format(round(total_ending_profit, 2)))

    baseline_profit = get_baseline_profit(total_capital=total_capital)

    winner_total = round(total_ending_profit - baseline_profit, 2)
    if winner_total >= 0:
        winner_percentage = round(100 * total_ending_profit / total_capital, 2)
        output_to_file("Model Wins by ${0} @ {1}% with a gain of ${2:.2f} :)".format(winner_total, winner_percentage, total_ending_profit))
    else:
        winner_percentage = round(100 * baseline_profit / total_capital, 2)
        output_to_file("Baseline Wins by ${0} @ {1}% with a gain of ${2:.2f} :(".format(-winner_total, winner_percentage, baseline_profit))


if __name__ == '__main__':
    main()
