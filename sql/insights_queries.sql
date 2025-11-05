USE stocks_analysis;
SELECT * FROM stocks_data LIMIT 5;

-- ============================================
-- Query 1: Stock Performance Summary
-- ============================================
-- Purpose: Calculate overall performance metrics for each stock
-- Metrics: Latest price, earliest price, total return %, average trading volume
-- Business Use: Quick portfolio performance overview, identify best/worst performers
-- ============================================

WITH price_endpoints AS (
    SELECT 
        s1.Ticker,
        (SELECT Close FROM stocks_data s2 WHERE s2.Ticker = s1.Ticker ORDER BY Date DESC LIMIT 1) AS Latest_Price,
        (SELECT Close FROM stocks_data s2 WHERE s2.Ticker = s1.Ticker ORDER BY Date ASC LIMIT 1) AS Earliest_Price,
        AVG(s1.Volume) as AVG_Volume 
    FROM stocks_data s1
    GROUP BY s1.Ticker
)
SELECT 
    Ticker,
    Latest_Price,
    Earliest_Price,        
    CONCAT((((Latest_Price - Earliest_Price) / Earliest_Price) * 100), "%") AS Total_pct,
    AVG_Volume  
FROM price_endpoints;

-- ============================================
-- FINDINGS & INSIGHTS:
-- ============================================
-- Performance Period: Sep 30, 2022 to Sep 29, 2025 (~3 years)
--
-- Results:
--   AAPL:   $143.41 → $249.53  |  +74.0%  |  Avg Vol: 60.1M
--   GOOGL:  $99.74  → $188.68  |  +89.2%  |  Avg Vol: 31.8M
--   MSFT:   $247.31 → $419.20  |  +69.5%  |  Avg Vol: 24.1M
--
-- Key observations:
-- 1. GOOGL is the top performer (+89.2%)
--    - Nearly doubled in 3 years
--    - Despite higher volatility (from Query 2)
--    - Strong recovery from 2022 lows
--
-- 2. AAPL solid second place (+74.0%)
--    - Highest trading volume (60M avg)
--    - Most liquid stock in the portfolio
--    - Steady growth trajectory
--
-- 3. MSFT strong absolute returns (+69.5%)
--    - Highest absolute price ($419)
--    - Lowest trading volume (24M avg)
--    - Most stable (lowest vol days from Query 2)
--
-- 4. Volume patterns:
--    - AAPL: 2x MSFT volume (retail favorite)
--    - GOOGL: Mid-range liquidity
--    - Lower volume = potentially lower volatility
--
-- Risk-return trade-off:
-- - GOOGL: Highest return BUT highest volatility (55 high-vol days)
-- - MSFT: Lower return BUT most stable (32 high-vol days)
-- - AAPL: Balanced risk-return profile
--
-- Portfolio implications:
-- - All three significantly outperformed broader market
-- - GOOGL best for aggressive growth seekers
-- - MSFT best for stability-focused investors
-- - AAPL good core holding (balance of growth + liquidity)
-- ============================================

-- ============================================
-- Query 2: High Volatility & Volume Spike Detection
-- ============================================
-- Purpose: Identify trading days with unusually high volatility and volume
-- Logic: Find days where realized volatility > 2.5 AND volume exceeds stock's average
-- Business Use: Detect significant market events, news impact, potential trading opportunities
-- ============================================

SELECT 
	Date,
    Ticker,
    Volume,
    Realized_vol_20d,
    (SELECT AVG(Volume) FROM stocks_data s2 where s2.Ticker = s1.Ticker) AS AVG_Vol
FROM stocks_data s1
WHERE Realized_vol_20d>2.5
	AND volume > (SELECT AVG(Volume) from stocks_data s2 where s2.Ticker = s1.Ticker);

SELECT 
    Ticker,
    COUNT(*) as High_Vol_Days
FROM stocks_data s1
WHERE Realized_vol_20d > 2.5
  AND Volume > (SELECT AVG(Volume) FROM stocks_data s2 WHERE s2.Ticker = s1.Ticker)
GROUP BY Ticker;   

-- ============================================
-- FINDINGS & INSIGHTS:
-- ============================================
-- Total high volatility days: 122 rows
-- 
-- Breakdown by ticker:
--   - GOOGL: 55 days (45% of total)
--   - AAPL: 36 days (30%)
--   - MSFT: 32 days (26%)
--
-- Key observations:
-- 1. Peak volatility period: Oct-Dec 2022
--    - Coincides with aggressive Fed rate hikes
--    - All three stocks showed elevated vol/volume
--
-- 2. GOOGL most volatile overall
--    - Dominated Feb 2023 (consecutive high-vol days)
--    - Possible earnings or product announcements
--
-- 3. AAPL recent spike: April-May 2025
--    - Realized vol reached 4.9+ (extremely high)
--    - Volume 2-3x above average
--    - Needs investigation: product launch? regulatory news?
--
-- 4. MSFT relatively stable
--    - Fewest high-vol days
--    - Lower debt-to-equity suggests more stability
--
-- Business implications:
-- - Use for risk management: avoid or hedge during high-vol periods
-- - Trading opportunities: momentum strategies during volume spikes
-- - Portfolio rebalancing triggers
-- ============================================

-- ============================================
-- Query 3: Value vs Growth Stock Categorization
-- ============================================
-- Purpose: Categorize stocks based on fundamental metrics
-- Logic: Classify as Value (low PE, low debt), Growth (high PE), or Balanced
-- Criteria:
--   Value: PE_Ratio < 30 AND Debt_to_Equity < 1
--   Growth: PE_Ratio >= 30
--   Balanced: PE_Ratio < 30 AND Debt_to_Equity >= 1
-- Business Use: Portfolio construction, investment strategy alignment, risk profiling
-- ============================================

WITH latest_fundamentals AS (
    SELECT 
        s1.Ticker,
        (SELECT PE_Ratio FROM stocks_data s2 WHERE s2.Ticker = s1.Ticker ORDER BY Date DESC LIMIT 1) AS PE_Ratio,
        (SELECT debt_to_Equity FROM stocks_data s2 WHERE s2.Ticker = s1.Ticker ORDER BY Date DESC LIMIT 1) AS debt_to_Equity
    FROM stocks_data s1
    GROUP BY Ticker
)
SELECT
    Ticker,
    PE_Ratio,
    debt_to_Equity,
    CASE 
        WHEN PE_Ratio < 30 AND Debt_to_Equity < 1 THEN 'Value'
        WHEN PE_Ratio >= 30 THEN 'Growth'
        ELSE 'Balanced'
    END AS Stock_Category
FROM latest_fundamentals;

-- ============================================
-- FINDINGS & INSIGHTS:
-- ============================================
-- Results (as of latest data):
--   AAPL:   PE 38.76  |  D/E 1.45   |  Growth
--   GOOGL:  PE 26.31  |  D/E 0.042  |  Value
--   MSFT:   PE 37.44  |  D/E 0.15   |  Growth
--
-- Portfolio composition:
-- - 2 Growth stocks (67%)
-- - 1 Value stock (33%)
-- - 0 Balanced stocks
--
-- Key observations:
-- 1. GOOGL is the only Value stock
--    - Lowest PE ratio (26.31) - trading at reasonable valuation
--    - Extremely low debt (0.042 D/E) - very healthy balance sheet
--    - Despite being highest performer (+89% from Query 1)
--    - Classic "quality at reasonable price"
--
-- 2. AAPL classified as Growth
--    - High PE (38.76) reflects market's growth expectations
--    - Higher debt load (1.45 D/E) than GOOGL/MSFT
--    - Market pricing in iPhone innovation, services growth
--
-- 3. MSFT also Growth despite stability
--    - PE 37.44 indicates premium valuation
--    - Low debt (0.15 D/E) - financially conservative
--    - Cloud/AI growth story priced in
--
-- 4. Debt management comparison:
--    - GOOGL: 0.042 (virtually debt-free)
--    - MSFT: 0.15 (conservative leverage)
--    - AAPL: 1.45 (uses debt strategically for buybacks)
--
-- Investment implications:
-- - Growth-heavy portfolio = higher risk/reward
-- - GOOGL offers value opportunity despite strong returns
-- - Both growth stocks (AAPL/MSFT) trade at similar premiums
-- - Consider rebalancing if seeking more value exposure
--
-- Risk considerations:
-- - Growth stocks vulnerable to multiple compression if rates rise
-- - GOOGL's value status + strong fundamentals = defensive pick
-- - AAPL's debt level warrants monitoring in rising rate environment
-- ============================================

-- ============================================
-- Query 4: Market Regime Analysis (VIX-Based)
-- ============================================
-- Purpose: Analyze stock returns during different market volatility regimes
-- Logic: Compare average returns when VIX is high (>25) vs low (<20) vs medium
-- Business Use: Understand how stocks perform in fear vs calm markets, risk management
-- ============================================
SELECT 
    CASE 
        WHEN VIX > 25 THEN 'High'
        WHEN VIX < 20 THEN 'Low'
        ELSE 'Medium'
    END AS VIX_Regime,
    ROUND(AVG(CASE WHEN Ticker = 'AAPL' THEN Return_Pct END), 3) AS Avg_Return_AAPL,
    ROUND(AVG(CASE WHEN Ticker = 'GOOGL' THEN Return_Pct END), 3) AS Avg_Return_GOOGL,
    ROUND(AVG(CASE WHEN Ticker = 'MSFT' THEN Return_Pct END), 3) AS Avg_Return_MSFT,
    COUNT(*)/3 AS Days_Per_Stock
FROM stocks_data
WHERE Return_Pct IS NOT NULL
GROUP BY VIX_Regime
ORDER BY VIX_Regime;

-- ============================================
-- FINDINGS & INSIGHTS:
-- ============================================
-- Analysis Period: 950 total trading days across all stocks
--   - High VIX days (>25): 251 days (26.4%)
--   - Low VIX days (<20): 571 days (60.1%)
--   - Medium VIX (20-25): 128 days (13.5%)
--
-- Performance by regime:
--
-- HIGH VIX (Fear/Panic) - 251 days:
--   AAPL:  -0.378% avg daily return
--   GOOGL: -0.433% avg daily return (worst)
--   MSFT:  -0.218% avg daily return (most defensive)
--
-- LOW VIX (Calm Markets) - 571 days:
--   AAPL:  +0.215% avg daily return
--   GOOGL: +0.287% avg daily return (best performer)
--   MSFT:  +0.175% avg daily return
--
-- MEDIUM VIX (Normal) - 128 days:
--   AAPL:  -0.231% avg daily return
--   GOOGL: -0.239% avg daily return
--   MSFT:  +0.015% avg daily return (only positive)
--
-- Key observations:
--
-- 1. MSFT is the defensive winner
--    - Best performance during high VIX (-0.218% vs -0.38%/-0.43%)
--    - Only stock positive during medium VIX
--    - Loses least during market turmoil
--    - Trade-off: Lowest gains in calm markets (+0.175%)
--
-- 2. GOOGL shows highest volatility sensitivity
--    - Worst performance in high VIX (-0.433%)
--    - Best performance in low VIX (+0.287%)
--    - Classic "beta play" - amplifies market moves
--    - Explains why it had most high-vol days in Query 2
--
-- 3. AAPL sits in the middle
--    - Mid-range performance across all regimes
--    - Balanced risk profile
--    - Consistent with "core holding" status from Query 1
--
-- 4. Market regime distribution insight:
--    - 60% of days are low VIX (calm markets dominated period)
--    - Only 26% high VIX days
--    - Most gains made during calm periods
--
-- Portfolio implications:
-- - MSFT best for risk-off/defensive positioning
-- - GOOGL best for aggressive growth in stable markets
-- - AAPL provides balanced exposure
-- - Consider overweighting MSFT before expected volatility spikes
-- - GOOGL vulnerable during market corrections despite strong overall returns
--
-- Risk management takeaway:
-- - In bear markets/crises: Expect GOOGL to drop ~0.43% daily, MSFT ~0.22%
-- - In bull markets: GOOGL outperforms, gaining ~0.29% daily
-- - Portfolio should tilt defensive (more MSFT) if VIX trending up
-- ============================================

-- ============================================
-- Query 5: Advanced Multi-Factor Performance Attribution
-- ============================================
-- Purpose: Rank stocks by risk-adjusted returns across different Fed rate environments
-- Logic: Calculate Sharpe-like ratios (return/volatility) per ticker per rate regime, 
--        analyze volume patterns, and rank performance
-- Business Use: Portfolio optimization, regime-based asset allocation, risk management
-- ============================================

-- ============================================
-- Query 5: Advanced Multi-Factor Performance Attribution
-- ============================================
-- Purpose: Rank stocks by risk-adjusted returns across different Fed rate environments
-- Logic: Calculate Sharpe-like ratios (return/volatility) per ticker per rate regime, 
--        analyze volume patterns, and rank performance
-- Business Use: Portfolio optimization, regime-based asset allocation, risk management
-- ============================================

WITH fed_regime AS (
    SELECT 
        *,
        CASE 
            WHEN Fed_Funds_Rate < 3 THEN 'Low_Rate'
            WHEN Fed_Funds_Rate <= 4 THEN 'Medium_Rate'
            ELSE 'High_Rate'
        END AS Rate_Regime
    FROM stocks_data
    WHERE Return_Pct IS NOT NULL
),
performance_metrics AS (
    SELECT 
        Rate_Regime,
        Ticker,
        ROUND(AVG(Return_Pct), 3) AS Avg_Return,
        ROUND(STDDEV(Return_Pct), 3) AS Volatility,
        COUNT(*) AS Days_Traded,
        ROUND(AVG(Volume), 0) AS Avg_Volume_In_Regime
    FROM fed_regime
    GROUP BY Rate_Regime, Ticker
),
overall_volumes AS (
    SELECT 
        Ticker,
        ROUND(AVG(Volume), 0) AS Overall_Avg_Volume
    FROM stocks_data
    GROUP BY Ticker
),
risk_adjusted AS (
    SELECT 
        p.*,
        o.Overall_Avg_Volume,
        ROUND(p.Avg_Return / NULLIF(p.Volatility, 0), 4) AS Sharpe_Ratio,
        ROUND(p.Avg_Volume_In_Regime / o.Overall_Avg_Volume, 2) AS Volume_Ratio
    FROM performance_metrics p
    JOIN overall_volumes o ON p.Ticker = o.Ticker
)
SELECT 
    Rate_Regime,
    Ticker,
    Avg_Return,
    Volatility,
    Sharpe_Ratio,
    Volume_Ratio,
    Days_Traded,
    RANK() OVER (PARTITION BY Rate_Regime ORDER BY Sharpe_Ratio DESC) AS Performance_Rank
FROM risk_adjusted
ORDER BY Rate_Regime, Performance_Rank;

-- ============================================
-- FINDINGS & INSIGHTS:
-- ============================================
-- Analysis covers 2 Fed rate regimes (no Low_Rate data in dataset):
--   - High Rate (>4%): 69 days per stock
--   - Medium Rate (3-4%): 52 days per stock
--
-- HIGH RATE ENVIRONMENT (Fed Funds > 4%):
-- Rank 1: GOOGL - Sharpe 0.082
--   - Best risk-adjusted returns (+0.157% avg, 1.912% vol)
--   - Highest raw returns in high-rate environment
--   - Normal volume (1.01x baseline)
--   - Winner in restrictive monetary policy
--
-- Rank 2: MSFT - Sharpe 0.076
--   - Solid returns (+0.113%) with lower volatility (1.488%)
--   - Most stable performer (lowest vol of all)
--   - Slightly below-normal volume (0.99x)
--   - Defensive characteristics shine through
--
-- Rank 3: AAPL - Sharpe 0.059
--   - Lowest risk-adjusted returns
--   - Moderate return (+0.098%) but higher volatility (1.671%)
--   - Slightly suppressed volume (0.98x)
--   - Struggles most in tight monetary conditions
--
-- MEDIUM RATE ENVIRONMENT (Fed Funds 3-4%):
-- Rank 1: MSFT - Sharpe 0.087
--   - Best risk-adjusted performance
--   - Highest returns across ALL scenarios (+0.233%)
--   - Volume spike to 1.25x (increased trading interest)
--   - Sweet spot for MSFT performance
--
-- Rank 2: AAPL - Sharpe 0.040
--   - Modest returns (+0.104%) with high volatility (2.572%)
--   - Massive volume increase (1.39x) - most traded
--   - High activity but poor risk-adjusted performance
--   - Retail/institutional rotation?
--
-- Rank 3: GOOGL - Sharpe 0.010
--   - Terrible risk-adjusted returns
--   - Near-zero returns (+0.026%) with highest vol (2.737%)
--   - Worst performance of entire analysis
--   - Medium rates are GOOGL's kryptonite
--
-- CRITICAL INSIGHTS:
--
-- 1. GOOGL is regime-dependent:
--    - Dominant in high-rate environments (Rank #1)
--    - Collapses in medium-rate environments (Rank #3)
--    - Requires strong conviction on Fed trajectory
--    - NOT a all-weather stock
--
-- 2. MSFT is the all-weather winner:
--    - Ranks #1 or #2 in both regimes
--    - Only stock with consistent strong Sharpe ratios
--    - Medium rates = optimal environment (+0.233% returns)
--    - Portfolio anchor for any Fed scenario
--
-- 3. AAPL underperforms on risk-adjusted basis:
--    - Always ranks #2 or #3
--    - High volatility drags down Sharpe ratios
--    - Volume spikes don't translate to better returns
--    - Question: Is retail hype masking poor fundamentals?
--
-- 4. Volatility patterns:
--    - All stocks more volatile in medium-rate regime
--    - High rates compress volatility (clearer direction)
--    - Medium rates = maximum uncertainty/choppiness
--
-- 5. Volume insights:
--    - AAPL sees biggest volume surge in medium rates (+39%)
--    - MSFT moderate increase (+25%)
--    - GOOGL barely changes (+3%)
--    - High volume ≠ good returns (AAPL case study)
--
-- PORTFOLIO STRATEGY RECOMMENDATIONS:
--
-- Current environment (Fed Funds ~4%+):
--   1. Overweight: GOOGL (proven high-rate winner)
--   2. Core holding: MSFT (consistent performer)
--   3. Underweight: AAPL (weakest risk-adjusted)
--
-- If Fed cuts to 3-4% range:
--   1. Rotate to MSFT heavily (sweet spot)
--   2. Reduce GOOGL exposure (disaster zone)
--   3. AAPL neutral (volume but no alpha)
--
-- Risk management:
--   - MSFT provides stability across regimes
--   - GOOGL requires active tactical allocation
--   - AAPL offers liquidity but not efficiency
--   - Sharpe ratios suggest MSFT deserves 40-50% allocation
--
-- KEY TAKEAWAY:
-- Traditional performance rankings (Query 1) showed GOOGL as best (+89%).
-- But risk-adjusted analysis reveals MSFT as superior - better returns 
-- per unit of risk, regime-resilient, and optimal in most Fed scenarios.
-- This is why Sharpe ratios matter more than raw returns.
-- ============================================