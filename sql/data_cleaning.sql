USE stocks_analysis;

#1. Total row count
SELECT COUNT(*) as Total_count FROM stocks_data;

#2. Sample data
SELECT * FROM stocks_data LIMIT 5;

#3. Check date range
SELECT
	MIN(Date) AS Earliest_date,
    MAX(Date) AS Latest_date
FROM stocks_data;

#4. How many unique tickers?
SELECT COUNT(DISTINCT ticker) as unique_tickers
from stocks_data;

#5. Any NULL values in critical columns?
SELECT *
FROM stocks_data
WHERE Close IS NULL 
   OR Volume IS NULL 
   OR Return_Pct IS NULL
   OR Realized_Vol_20d IS NULL
   OR EPS IS NULL
   OR PE_Ratio IS NULL
   OR Debt_to_Equity IS NULL
   OR Revenue IS NULL
   OR Fed_Funds_Rate IS NULL
   OR CPI IS NULL
   OR VIX IS NULL
   OR GoogleTrends IS NULL
   OR Target_Price IS NULL
   OR Target_Vol IS NULL;

#6.  Check for duplicate records
SELECT 
    Date, 
    Ticker,
    COUNT(*) AS duplicate_count
FROM stocks_data
GROUP BY Date, Ticker
HAVING COUNT(*) > 1;
