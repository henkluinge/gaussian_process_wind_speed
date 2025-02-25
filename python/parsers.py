import numpy as np
import pandas as pd
from datetime import datetime


def get_eelde_weather_data(fname = 'eelde_2021-2030/uurgeg_280_2021-2030.txt'):
    """
    Data from https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens

    To pandas dataframe.
    """
    # Read header
    with open(fname, 'r') as f:
        for line in f:
            # Print each line
            s = line.strip()
            
            if s.startswith('#'):
                break
        cols = [si.strip() for si in s[1:].split(',')]

        # Read data
        df = pd.read_csv(f, low_memory=False, names=cols)

    # Groom data.
    df['HH'] = df['HH'].apply(lambda hr: 0 if hr==24 else hr)
    df['datetime'] = df.apply(lambda d: datetime.strptime(str(d.YYYYMMDD)+str(d.HH).zfill(2), r'%Y%m%d%H') , axis=1)
    col_names = {'DD': 'windrichting', 'FF': 'windsnelheid', 'FX':'max_wind', 'T': 'temperatuur', 'SQ': 'zonneschijn', 'RH': 'neerslag', 'U': 'rel_luchtvochtigheid'}
    df = df.set_index('datetime')[col_names.keys()].rename(columns=col_names)

    # Temperature is logged in 0.1 C
    df['temperatuur'] = 0.1*df['temperatuur']

    return df


def get_wind_power_per_direction(fname = r'C:\Users\HenkColleenNiamh\Code\cad\nieuw_beerta_weather_export.csv'):
    """Dataframe with wind direction index (0 TO 2pi). """
    df = pd.read_csv(fname)
    df['date'] = df['date'].apply(lambda s: datetime.strptime(s, '%Y-%m-%d'))
    df.drop(columns=['tsun',])
    df['month'] = df.date.apply(lambda t:t.month)

    df = df.set_index('date')

    df = df[['wdir', 'wspd']]
    df.sort_values(by='wdir', inplace=True)
    df['wdir'] = (np.pi/180)*df['wdir'] #- np.pi
    df.set_index('wdir', inplace=True)
    df = df.groupby('wdir').sum()

    df['wspd_rolling_mean'] = df['wspd'].rolling(window=15, center=True).mean()
    return df