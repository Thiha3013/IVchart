import numpy as np
import pandas as pd
import scipy.stats as si
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt 
from black_scholes import call_imp_vol, put_imp_vol

df = pd.read_csv('aapl_2021_2023.csv', low_memory=False)
rdf = pd.read_csv('DGS10.csv', low_memory=False)
df.columns = df.columns.str.strip()


#print(df.columns)

"""refining data"""

df = df.dropna()

#min strike dist

df = df[df['[STRIKE_DISTANCE]'] == df.groupby('[QUOTE_DATE]')['[STRIKE_DISTANCE]'].transform('min')]

#DTE fixed between 25 to 35 and taking closest to 30 at all times

df = df.loc[(df['[DTE]'] >= 25) & (df['[DTE]'] <= 35)]

df['Diff'] = abs(df['[DTE]'] - 30)
df_sorted = df.sort_values(by=['[QUOTE_DATE]', 'Diff'])
df_result = df_sorted.drop_duplicates(subset='[QUOTE_DATE]', keep='first')
df = df_result.drop(columns=['Diff'])

#fixing the format of both dataframes' date time

df['[QUOTE_DATE]'] = pd.to_datetime(df['[QUOTE_DATE]'])
rdf['DATE'] = pd.to_datetime(rdf['DATE'])

#fixing empty cells in rdf
rdf['DGS10'] = rdf['DGS10'].replace('.', pd.NA)

# Forward fill NaN values
rdf['DGS10'] = rdf['DGS10'].ffill()

# Optionally, convert the column to float if it's not already
rdf['DGS10'] = rdf['DGS10'].astype(float)

# historical volatility 

# Calculate log returns
log_returns = np.log(df['[UNDERLYING_LAST]'] / df['[UNDERLYING_LAST]'].shift(1))
window_size = 30  
rolling_std_dev = log_returns.rolling(window=window_size).std(ddof=1)

# Annualize the volatility
# Assuming there are 252 trading days in a year
df['SIGMA30'] = rolling_std_dev * np.sqrt(252)

c_implied_vols = []
p_implied_vols = []

def main():
    for index, row in df.iterrows():

        tDate = row['[QUOTE_DATE]']
        tS = row['[UNDERLYING_LAST]']
        tK = row['[STRIKE]']
        tT = float(row['[DTE]'])
        tR = float(rdf.loc[rdf['DATE'] == tDate, 'DGS10'].iloc[0])/100
        tSig = row['SIGMA30']
        try:
            tC = float(row['[C_LAST]'])
        except ValueError:
            tC = float(0)  # or another default value
        try:
            tP = float(row['[P_LAST]'])
        except ValueError:
            tP = float(0)

            #print(tDate ,tSig , call_imp_vol(tS, tK, tR, tT, tC,tSig),tS)
            #print(tDate ,tS)
            
        c_imp_vol = call_imp_vol(tS, tK, tR, tT, tC, tSig)
        p_imp_vol = put_imp_vol(tS, tK, tR, tT, tP, tSig)

        # Append the result to the list
        c_implied_vols.append(c_imp_vol)
        p_implied_vols.append(p_imp_vol)

    # Create a new column in df for the implied volatilities
    df['CImplied_Vol'] = c_implied_vols
    df['PImplied_Vol'] = p_implied_vols

    #print(df['Implied_Vol'])




    '''plotting the actual graph'''

    plt.plot(df['[QUOTE_DATE]'],df['PImplied_Vol'], label = "PIV30") 
    plt.plot(df['[QUOTE_DATE]'],df['CImplied_Vol'], label = "CIV30") 
    plt.plot(df['[QUOTE_DATE]'],df['SIGMA30'] , label = "HV30") 
    plt.ylabel('IV vs HV') 
    plt.title('Date') 
    plt.legend() 
    plt.show() 

if __name__ == "__main__":
    main()
