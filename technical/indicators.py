"""
This file contains a collection of common indicators, which are based on third party or custom libraries

"""
from numpy.core.records import ndarray
from pandas import Series, DataFrame


def aroon(dataframe, period=25, field='close', colum_prefix="bb") -> DataFrame:
    from pyti.aroon import aroon_up as up
    from pyti.aroon import aroon_down as down
    dataframe["{}_aroon_up".format(colum_prefix)] = up(dataframe[field], period)
    dataframe["{}_aroon_down".format(colum_prefix)] = down(dataframe[field], period)
    return dataframe


def atr(dataframe, period, field='close') -> ndarray:
    from pyti.average_true_range import average_true_range
    return average_true_range(dataframe[field], period)


def atr_percent(dataframe, period, field='close') -> ndarray:
    from pyti.average_true_range_percent import average_true_range_percent
    return average_true_range_percent(dataframe[field], period)


def bollinger_bands(dataframe, period=21, stdv=2, field='close', colum_prefix="bb") -> DataFrame:
    from pyti.bollinger_bands import lower_bollinger_band, middle_bollinger_band, upper_bollinger_band
    dataframe["{}_lower".format(colum_prefix)] = lower_bollinger_band(dataframe[field], period, stdv)
    dataframe["{}_middle".format(colum_prefix)] = middle_bollinger_band(dataframe[field], period, stdv)
    dataframe["{}_upper".format(colum_prefix)] = upper_bollinger_band(dataframe[field], period, stdv)

    return dataframe


def cmf(dataframe, period=14) -> ndarray:
    from pyti.chaikin_money_flow import chaikin_money_flow

    return chaikin_money_flow(dataframe['close'], dataframe['high'], dataframe['low'], dataframe['volume'], period)


def accumulation_distribution(dataframe) -> ndarray:
    from pyti.accumulation_distribution import accumulation_distribution as acd

    return acd(dataframe['close'], dataframe['high'], dataframe['low'], dataframe['volume'])


def osc(dataframe, periods=14) -> ndarray:
    """
    1. Calculating DM (i).
        If HIGH (i) > HIGH (i - 1), DM (i) = HIGH (i) - HIGH (i - 1), otherwise DM (i) = 0.
    2. Calculating DMn (i).
        If LOW (i) < LOW (i - 1), DMn (i) = LOW (i - 1) - LOW (i), otherwise DMn (i) = 0.
    3. Calculating value of OSC:
        OSC (i) = SMA (DM, N) / (SMA (DM, N) + SMA (DMn, N)).

    :param dataframe:
    :param periods:
    :return:
    """
    df = dataframe
    df['DM'] = (df['high'] - df['high'].shift()).apply(lambda x: max(x, 0))
    df['DMn'] = (df['low'].shift() - df['low']).apply(lambda x: max(x, 0))
    return Series.rolling_mean(df.DM, periods) / (
            Series.rolling_mean(df.DM, periods) + Series.rolling_mean(df.DMn, periods))


def cmo(dataframe, period, field='close') -> ndarray:
    from pyti.chande_momentum_oscillator import chande_momentum_oscillator
    return chande_momentum_oscillator(dataframe[field], period)


def cci(dataframe, period) -> ndarray:
    from pyti.commodity_channel_index import commodity_channel_index

    return commodity_channel_index(dataframe['close'], dataframe['high'], dataframe['low'], period)


def laguerre(dataframe, gamma=0.75, smooth=1, debug=bool):
    """
    laguerre RSI
    Author Creslin
    Original Author: John Ehlers 1979


    :param dataframe: df
    :param gamma: Between 0 and 1, default 0.75
    :param smooth: 1 is off. Valid values over 1 are alook back smooth for an ema
    :param debug: Bool, prints to console
    :return: Laguerre RSI:values 0 to +1
    """
    """
    Laguerra RSI 
    How to trade lrsi:  (TL, DR) buy on the flat 0, sell on the drop from top,
    not when touch the top
    http://systemtradersuccess.com/testing-laguerre-rsi/

    http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/Laguerre_RSI.html
    """
    import talib as ta
    import pandas as pd
    ema = ta.EMA

    df = dataframe
    g = gamma
    smooth = smooth
    debug = debug
    if debug:
        from pandas import set_option
        set_option('display.max_rows', 2000)
        set_option('display.max_columns', 8)

    """
    Vectorised pandas or numpy calculations are not used
    in Laguerre as L0 is self referencing.
    Therefore we use an intertuples loop as next best option. 
    """
    lrsi_l = []
    L0, L1, L2, L3 = 0.0, 0.0, 0.0, 0.0
    for row in df.itertuples(index=True, name='lrsi'):
        """ Original Pine Logic  Block1
        p = close
        L0 = ((1 - g)*p)+(g*nz(L0[1]))
        L1 = (-g*L0)+nz(L0[1])+(g*nz(L1[1]))
        L2 = (-g*L1)+nz(L1[1])+(g*nz(L2[1]))
        L3 = (-g*L2)+nz(L2[1])+(g*nz(L3[1])) 
        """
        # Feed back loop
        L0_1, L1_1, L2_1, L3_1 = L0, L1, L2, L3

        L0 = (1 - g) * row.close + g * L0_1
        L1 = -g * L0 + L0_1 + g * L1_1
        L2 = -g * L1 + L1_1 + g * L2_1
        L3 = -g * L2 + L2_1 + g * L3_1

        """ Original Pinescript Block 2 
        cu=(L0 > L1? L0 - L1: 0) + (L1 > L2? L1 - L2: 0) + (L2 > L3? L2 - L3: 0)
        cd=(L0 < L1? L1 - L0: 0) + (L1 < L2? L2 - L1: 0) + (L2 < L3? L3 - L2: 0)
        """
        cu = 0.0
        cd = 0.0
        if L0 >= L1:
            cu = L0 - L1
        else:
            cd = L1 - L0

        if L1 >= L2:
            cu = cu + L1 - L2
        else:
            cd = cd + L2 - L1

        if L2 >= L3:
            cu = cu + L2 - L3
        else:
            cd = cd + L3 - L2

        """Original Pinescript  Block 3 
        lrsi=ema((cu+cd==0? -1: cu+cd)==-1? 0: (cu/(cu+cd==0? -1: cu+cd)), smooth)
        """
        if (cu + cd) != 0:
            lrsi_l.append(cu / (cu + cd))
        else:
            lrsi_l.append(0)
