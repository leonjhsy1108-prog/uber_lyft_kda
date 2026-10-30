CREATE OR REPLACE TEMP TABLE cleaned_rides AS
WITH typed AS (
    SELECT
        CAST(id AS VARCHAR) AS id,
        TRY_CAST("timestamp" AS DOUBLE) AS quote_timestamp,
        TRY_CAST(price AS DOUBLE) AS price,
        TRY_CAST(distance AS DOUBLE) AS distance,
        TRY_CAST(surge_multiplier AS DOUBLE) AS surge_multiplier,
        NULLIF(TRIM(CAST(cab_type AS VARCHAR)), '') AS cab_type,
        NULLIF(TRIM(CAST(name AS VARCHAR)), '') AS name,
        NULLIF(TRIM(CAST(source AS VARCHAR)), '') AS source,
        NULLIF(TRIM(CAST(destination AS VARCHAR)), '') AS destination,
        TRY_CAST(temperature AS DOUBLE) AS temperature,
        TRY_CAST(precipIntensity AS DOUBLE) AS precipIntensity,
        TRY_CAST(windSpeed AS DOUBLE) AS windSpeed
    FROM read_csv_auto('data/rideshare_kaggle.csv')
),
valid_rows AS (
    SELECT *
    FROM typed
    WHERE id IS NOT NULL
      AND quote_timestamp IS NOT NULL
      AND price IS NOT NULL AND price > 0
      AND distance IS NOT NULL AND distance > 0
      AND surge_multiplier IS NOT NULL AND surge_multiplier >= 1
      AND cab_type IS NOT NULL AND name IS NOT NULL
      AND source IS NOT NULL AND destination IS NOT NULL
      AND temperature IS NOT NULL
      AND precipIntensity IS NOT NULL
      AND windSpeed IS NOT NULL
),
deduplicated AS (
    SELECT *
    FROM valid_rows
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY id ORDER BY quote_timestamp DESC
    ) = 1
),
with_local_time AS (
    SELECT *,
        timezone('America/New_York', to_timestamp(quote_timestamp))
            AS local_datetime
    FROM deduplicated
)
SELECT
    id,
    quote_timestamp,
    local_datetime,
    CAST(local_datetime AS DATE) AS local_date,
    CAST(EXTRACT(hour FROM local_datetime) AS INTEGER) AS local_hour,
    strftime(local_datetime, '%A') AS weekday,
    price,
    distance,
    cab_type,
    name,
    surge_multiplier,
    source,
    destination,
    temperature,
    precipIntensity,
    windSpeed
FROM with_local_time;

SELECT COUNT(*) AS cleaned_rows,
       MIN(price) AS minimum_price,
       MAX(price) AS maximum_price,
       MIN(local_date) AS first_day,
       MAX(local_date) AS last_day
FROM cleaned_rides;

COPY (
    SELECT * FROM cleaned_rides ORDER BY quote_timestamp, id
) TO 'data/rideshare_model.csv' (HEADER, DELIMITER ',');
