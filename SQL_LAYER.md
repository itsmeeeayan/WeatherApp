# SQL Data Layer

The app originally read `kolkataweather.csv` directly into pandas. It now loads
that CSV into a normalized SQLite database and serves its analytics and its
model-training frame from SQL.

## Why

The raw CSV is one wide table. The city name and the weather condition are
repeated on every single row, which wastes space and — more importantly — lets
the same value be spelled two ways (`kolkata` / `Kolkata`) with nothing to stop
it. Splitting the repeated values into their own tables fixes that at the schema
level rather than in application code.

## Schema

Three tables in third normal form:

```
cities                    conditions
  city_id      PK           condition_id    PK
  city_name    UNIQUE       condition_name  UNIQUE
      |                            |
      |                            |
      +---------+   +--------------+
                |   |
            observations
              obs_id            PK
              city_id           FK -> cities
              condition_id      FK -> conditions
              obs_date          ISO-8601 'YYYY-MM-DD'
              temp, tempmax, tempmin, humidity,
              precip, precip_flag, windspeed,
              sealevelpressure, cloudcover
              UNIQUE (city_id, obs_date)
```

`observations` is the fact table: one row per city per day. Every non-key
column describes that specific reading and nothing else, which is what 3NF
requires. The names live once in the lookup tables and are referenced by
integer foreign keys, so a rename happens in exactly one place.

The `UNIQUE (city_id, obs_date)` constraint makes re-running the loader
idempotent — a second run of the same file updates rows instead of duplicating
them.

Dates are stored as ISO-8601 text because SQLite has no `DATE` type. In that
format lexicographic ordering equals chronological ordering, so `ORDER BY`,
`BETWEEN` and index range scans all behave correctly.

## ETL

`weather_db.py` implements extract / transform / load as three separate
functions:

- **extract** — read the CSV, keep the 11 columns the model uses, fail loudly
  if any are missing
- **transform** — parse dates (day-first) to ISO-8601, trim and title-case city
  names so lookup keys collapse correctly, fill null precipitation with zero,
  derive the binary `precip_flag`, drop rows with no date or no temperature,
  de-duplicate on (city, date)
- **load** — insert distinct values into the two lookup tables with
  `INSERT OR IGNORE`, resolve the foreign keys, bulk-insert the fact rows,
  then build the indexes

Indexes are created **after** the bulk load, not before. Maintaining an index
during an insert is slower than building it once at the end.

## Queries

Four analytical queries replace what pandas was doing in memory:

| Query | Demonstrates |
|---|---|
| `monthly_summary` | `JOIN`, `GROUP BY`, `HAVING`, aggregate functions |
| `condition_breakdown` | three-table join, scalar subquery for a percentage |
| `above_average_days` | correlated subquery against each city's own mean |
| `rolling_trend` | CTE plus `AVG() OVER` and `LAG()` window functions |

`load_training_frame()` returns the model-ready frame straight from SQL, doing
the join and the `strftime` date-part extraction in the database so the
application never has to know how the raw file was shaped.

## Query optimization

`EXPLAIN QUERY PLAN` on a city-and-date-range filter, before and after adding
the composite index:

```
without idx_obs_city_date:
  SEARCH observations USING INDEX sqlite_autoindex_observations_1 ...

with idx_obs_city_date:
  SEARCH observations USING INDEX idx_obs_city_date (city_id=? AND obs_date>? AND obs_date<?)
```

The column order in `(city_id, obs_date)` matters: equality predicate first,
range predicate second, so the range scan runs inside a single city's block
rather than across the whole table.

## Running it

```bash
python weather_db.py --csv kolkataweather.csv --db weather.db --rebuild
python weather_db.py --db weather.db --demo      # run every query and print results
```

## Using it from the app

Replace the CSV read in `weather.py` with:

```python
from weather_db import load_training_frame, query

data = load_training_frame("weather.db")          # model-ready, already joined
monthly = query("weather.db", "monthly_summary")  # for the dashboard section
```
