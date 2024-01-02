import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt 
from black_scholes import call_imp_vol, put_imp_vol

'''using yfinance'''

def get_current_price(tikr):
    todays_data = tikr.history(period='1d')
    return todays_data['Close'].iloc[0]

def closest_to_30_days(datetimes):
        thirty_days_from_now = datetime.now() + timedelta(days=30)
        closest_datetime = None
        smallest_diff = float('inf')

        for dt in datetimes:
            diff = abs((dt - thirty_days_from_now).days)
            current_diff = abs((datetime.now() - dt).days)

            # Check if the date is closest to 30 days in the future
            # and within 25 to 35 days from now
            if diff < smallest_diff and (25 <= current_diff <= 35):
                smallest_diff = diff
                DTE = current_diff
                closest_datetime = dt

        return closest_datetime, DTE

def main(tikr):
    ticker = yf.Ticker(tikr)

    print(ticker)

    #  ticker = yf.Ticker('AAPL')
    rate = yf.Ticker('^TNX')
    expiration_dates = pd.to_datetime(ticker.options)

    tR= rate.history()['Close'].iloc[-1] / 100



    tS = get_current_price(ticker)





    nDate, tT = closest_to_30_days(expiration_dates)
    nDate = nDate.strftime('%Y-%m-%d')

    opt = ticker.option_chain(nDate)
    calls = opt.calls
    puts = opt.puts

    puts['abs_diff'] = abs(puts['strike'] - tS)
    put_atm= puts.loc[puts['abs_diff'].idxmin()]
    puts.drop(columns='abs_diff', inplace=True)


    calls['abs_diff'] = abs(calls['strike'] - tS)
    call_atm= calls.loc[calls['abs_diff'].idxmin()]
    calls.drop(columns='abs_diff', inplace=True)

    tKC = call_atm['strike']
    tKP = put_atm['strike']
    tC = call_atm['lastPrice']
    tP = put_atm['lastPrice']

    #historical volatility
    historical60 = ticker.history(start=datetime.now() - timedelta(60), end=datetime.now(), interval="1d")

    log_returns = np.log(historical60['Close'] / historical60['Close'].shift(1))
    window_size = 30  
    rolling_std_dev = log_returns.rolling(window=window_size).std(ddof=1)
    historical60['HV'] = rolling_std_dev * np.sqrt(252)

    HV = historical60['HV'].iloc[-1]

    cIV = call_imp_vol(tS,tKC,tR,tT,tC,HV)
    pIV = put_imp_vol(tS,tKP,tR,tT,tP,HV)
    avg_imp_vol = (cIV + pIV) / 2

    print("call implied volatility is: ", cIV,
        "\nput implied volatility is: ", pIV,
        "\naverage implied volatility is: ", avg_imp_vol,
        "\nrealized volatility is: ", HV)
    
    x = np.array(["Call IV", "Put IV", "Avg IV", "Realized V"])
    y = np.array([cIV, pIV, avg_imp_vol, HV])
    fig, ax = plt.subplots()
    bar_container = ax.bar(x, y)

    ax.set_ylabel('Volatility')  
    ax.set_title('Volatility Types')
    ax.set_ylim(0, 1)
    ax.bar_label(bar_container, fmt='%.5f') 
    plt.show()



if __name__ == "__main__":
    tikr = str(sys.argv[1])
    main(tikr)
    
