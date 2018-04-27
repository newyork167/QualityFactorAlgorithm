from datetime import datetime
import pandas_datareader.data as web
from pandas_datareader import wb


current_year = datetime.now().year


def get_stock_data(ticker, exchange, start_date=None, end_date=None):
    if start_date is not None:
        df = web.DataReader(ticker, exchange, start_date, end_date).sort_index(axis=1)
    else:
        df = web.DataReader(ticker, exchange).sort_index(axis=1)

    # Title case the headers
    df.columns = map(str.title, df.columns)

    return df


def get_stock_data_iex(ticker, start_date, end_date):
    return get_stock_data(ticker=ticker, exchange='iex', start_date=start_date, end_date=end_date)


def get_stock_data_morningstar(ticker, start_date=None, end_date=None):
    return get_stock_data(ticker=ticker, exchange='morningstar', start_date=start_date, end_date=end_date)


def get_stock_data_quandl(ticker, start_date, end_date):
    # Quandl sends data back in reverse time
    return get_stock_data(ticker=ticker, exchange='quandl', start_date=start_date, end_date=end_date).iloc[::-1]


def get_stock_data_robinhood(ticker, start_date, end_date):
    df = get_stock_data(ticker=ticker, exchange='robinhood', start_date=start_date, end_date=end_date)

    # Normalize robinhoods column data
    df = df.rename(index=str, columns={
        'Close_Price': 'Close',
        'High_Price': 'High',
        'Low_Price': 'Low',
        'Open_Price': 'Open'
    })

    # Title case the headers
    df.columns = map(str.title, df.columns)

    df['Close'] = df.apply(lambda row: float(row['Close']), axis=1)
    df['High'] = df.apply(lambda row: float(row['High']), axis=1)
    df['Low'] = df.apply(lambda row: float(row['Low']), axis=1)
    df['Open'] = df.apply(lambda row: float(row['Open']), axis=1)

    return df


def get_world_bank_indicators(indicator='NY.GDP.PCAP.KD', country=None, start=current_year, end=current_year):
    if country is None:
        country = ['US', 'CA', 'MX']

    matches = wb.search('gdp.*capita.*const')
    dat = wb.download(indicator=indicator, country=country, start=start, end=end)

    return dat
