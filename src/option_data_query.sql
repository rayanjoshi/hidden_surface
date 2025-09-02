-- OptionMetrics: Comprehensive Option, Underlying, and Rate Data for TSLA (2020-2023)
-- This query pulls all available option data including full Greeks, bid/ask spreads, and market data
-- Uses UNION ALL to combine data from multiple yearly tables to exhaust all options

-- 2020 Data
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
    END AS valid_pricing
FROM
    optionm.opprcd2020 o
LEFT JOIN
    optionm.secprd2020 u
    ON o.secid = u.secid AND o.date = u.date
WHERE
    o.secid = 143439  -- TSLA secid
    AND o.date >= '2020-01-01'
    AND o.best_bid IS NOT NULL
    AND o.best_offer IS NOT NULL
    AND o.impl_volatility IS NOT NULL
    AND o.impl_volatility > 0
    AND o.strike_price > 0

UNION ALL

-- 2021 Data
SELECT
    o.date,
    o.secid,
    o.optionid,
    o.symbol,
    o.cp_flag,
    o.strike_price / 1000.0 AS strike_price,
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
    (o.exdate::date - o.date::date) / 365.25 AS time_to_expiration,
    (o.strike_price / 1000.0) / u.close AS moneyness,
    CASE 
        WHEN o.best_bid > 0 AND o.best_offer > 0 AND o.best_offer > o.best_bid 
        THEN 1 
        ELSE 0 
    END AS valid_pricing
FROM
    optionm.opprcd2021 o
LEFT JOIN
    optionm.secprd2021 u
    ON o.secid = u.secid AND o.date = u.date
WHERE
    o.secid = 143439
    AND o.best_bid IS NOT NULL
    AND o.best_offer IS NOT NULL
    AND o.impl_volatility IS NOT NULL
    AND o.impl_volatility > 0
    AND o.strike_price > 0

UNION ALL

-- 2022 Data
SELECT
    o.date,
    o.secid,
    o.optionid,
    o.symbol,
    o.cp_flag,
    o.strike_price / 1000.0 AS strike_price,
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
    (o.exdate::date - o.date::date) / 365.25 AS time_to_expiration,
    (o.strike_price / 1000.0) / u.close AS moneyness,
    CASE 
        WHEN o.best_bid > 0 AND o.best_offer > 0 AND o.best_offer > o.best_bid 
        THEN 1 
        ELSE 0 
    END AS valid_pricing
FROM
    optionm.opprcd2022 o
LEFT JOIN
    optionm.secprd2022 u
    ON o.secid = u.secid AND o.date = u.date
WHERE
    o.secid = 143439
    AND o.best_bid IS NOT NULL
    AND o.best_offer IS NOT NULL
    AND o.impl_volatility IS NOT NULL
    AND o.impl_volatility > 0
    AND o.strike_price > 0

UNION ALL

-- 2023 Data
SELECT
    o.date,
    o.secid,
    o.optionid,
    o.symbol,
    o.cp_flag,
    o.strike_price / 1000.0 AS strike_price,
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
    (o.exdate::date - o.date::date) / 365.25 AS time_to_expiration,
    (o.strike_price / 1000.0) / u.close AS moneyness,
    CASE 
        WHEN o.best_bid > 0 AND o.best_offer > 0 AND o.best_offer > o.best_bid 
        THEN 1 
        ELSE 0 
    END AS valid_pricing
FROM
    optionm.opprcd2023 o
LEFT JOIN
    optionm.secprd2023 u
    ON o.secid = u.secid AND o.date = u.date
WHERE
    o.secid = 143439
    AND o.date <= '2023-12-31'
    AND o.best_bid IS NOT NULL
    AND o.best_offer IS NOT NULL
    AND o.impl_volatility IS NOT NULL
    AND o.impl_volatility > 0
    AND o.strike_price > 0

ORDER BY
    date, expiration, cp_flag, strike_price;