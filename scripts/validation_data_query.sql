-- OptionMetrics: Comprehensive Option, Underlying, and Rate Data for TSLA (2020-2023)
-- This query pulls all available option data including full Greeks, bid/ask spreads, and market data
-- Uses UNION ALL to combine data from multiple yearly tables to exhaust all options
-- Added Fama-French 3 Factors Plus Momentum (Mkt-RF, SMB, HML, UMD) and Risk-Free Rate (RF) from ff.factors_daily

-- 2024 Data
SELECT
    o.date,
    o.secid,
    o.optionid,
    o.symbol,
    o.cp_flag,
    o.strike_price / 1000.0 AS strike_price,  -- Convert from cents to dollars
    o.exdate AS expiration,
    o.best_bid,
    o.best_offer,
    (o.best_bid + o.best_offer) / 2.0 AS mid_quote,
    o.volume,
    o.open_interest,
    o.impl_volatility AS implied_volatility,
    o.delta,
    o.gamma,
    o.theta,
    o.vega,
    o.cfadj,
    o.am_settlement,
    o.contract_size,
    u.close AS underlying_close,
    u.open AS underlying_open,
    u.high AS underlying_high,
    u.low AS underlying_low,
    u.volume AS underlying_volume,
    -- Calculate time to expiration in years
    (o.exdate::date - o.date::date) / 365.25 AS time_to_expiration,
    -- Calculate moneyness (strike/spot)
    (o.strike_price / 1000.0) / u.close AS moneyness,
    -- Flag for options with valid pricing data
    CASE 
        WHEN o.best_bid > 0 AND o.best_offer > 0 AND o.best_offer > o.best_bid 
        THEN 1 
        ELSE 0 
    END AS valid_pricing,
    f.mktrf AS market_excess_return,
    f.smb AS smb,
    f.hml AS hml,
    f.umd AS momentum,
    f.rf AS risk_free_rate
FROM
    optionm.opprcd2023 o
LEFT JOIN
    optionm.secprd2023 u
    ON o.secid = u.secid AND o.date = u.date
LEFT JOIN
    ff.factors_daily f
    ON o.date = f.date
WHERE
    o.secid = 143439  -- TSLA secid
    AND o.date >= '2023-01-01'
    AND o.best_bid IS NOT NULL
    AND o.best_offer IS NOT NULL
    AND o.impl_volatility IS NOT NULL
    AND o.impl_volatility > 0
    AND o.strike_price > 0

ORDER BY
    date, expiration, cp_flag, strike_price;