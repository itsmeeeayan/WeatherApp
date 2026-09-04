"""
weather_db.py
=============
ETL layer for the Weather Forecasting app.

Takes the flat Visual Crossing weather CSV and loads it into a normalized
SQLite database, then serves the app's analytics from SQL instead of from
in-memory pandas operations.

Why this exists
---------------
The raw CSV is a single wide table with the city name and the condition
description repeated on every row. That is fine for a 15-row demo and wrong
for anything larger: the repetition wastes space and lets the same city be
spelled two different ways on two different rows.

The schema below splits it into three tables in third normal form:

    cities(city_id, city_name)                    -- one row per city
    conditions(condition_id, condition_name)      -- one row per weather type
    observations(obs_id, city_id, obs_date, ...,  -- one row per city per day
                 condition_id)

Every non-key column in `observations` depends on the full primary key and
on nothing else, which is what 3NF asks for. `city_name` and
`condition_name` live in their own tables and are referenced by integer
foreign keys, so a rename happens in exactly one place.

Usage
-----
    python weather_db.py --csv kolkataweather.csv --db weather.db --rebuild
    python weather_db.py --db weather.db --demo
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cities (
    city_id    INTEGER PRIMARY KEY,
    city_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS conditions (
    condition_id    INTEGER PRIMARY KEY,
    condition_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS observations (
    obs_id            INTEGER PRIMARY KEY,
    city_id           INTEGER NOT NULL,
    obs_date          TEXT    NOT NULL,          -- ISO-8601 'YYYY-MM-DD'
    temp              REAL,
    tempmax           REAL,
    tempmin           REAL,
    humidity          REAL,
    precip            REAL,
    precip_flag       INTEGER NOT NULL DEFAULT 0,-- derived: 1 if precip > 0
    windspeed         REAL,
    sealevelpressure  REAL,
    cloudcover        REAL,
    condition_id      INTEGER,
    FOREIGN KEY (city_id)      REFERENCES cities(city_id),
    FOREIGN KEY (condition_id) REFERENCES conditions(condition_id),
    UNIQUE (city_id, obs_date)                   -- one reading per city per day
);
"""

# Created AFTER the bulk insert: maintaining an index during a load is slower
# than building it once at the end.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_obs_city_date  ON observations(city_id, obs_date);
CREATE INDEX IF NOT EXISTS idx_obs_condition  ON observations(condition_id);
"""

USED_COLUMNS = [
    "City", "Date", "temp", "tempmax", "tempmin", "humidity",
    "precip", "windspeed", "sealevelpressure", "cloudcover", "conditions",
]


# --------------------------------------------------------------------------
# Extract + Transform
# --------------------------------------------------------------------------

def extract(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and keep only the columns the model actually uses."""
    df = pd.read_csv(csv_path)
    missing = [c for c in USED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    return df[USED_COLUMNS].copy()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise before loading.

    - dates parsed to a real date and stored ISO-8601 so string ordering
      equals chronological ordering (SQLite has no DATE type)
    - city and condition names trimmed and case-normalised so 'kolkata'
      and 'Kolkata' collapse to one row in the lookup table
    - precip nulls filled with 0 and a binary precip_flag derived
    - rows with no date or no temperature are unusable; drop them
    """
    df = df.rename(columns={"City": "city_name", "conditions": "condition_name"})

    df["obs_date"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["Date"])

    df["city_name"] = df["city_name"].astype(str).str.strip().str.title()
    df["condition_name"] = (
        df["condition_name"].astype(str).str.strip().replace({"nan": None})
    )

    df["precip"] = pd.to_numeric(df["precip"], errors="coerce").fillna(0.0)
    df["precip_flag"] = (df["precip"] > 0).astype(int)

    before = len(df)
    df = df.dropna(subset=["obs_date", "temp"])
    df = df.drop_duplicates(subset=["city_name", "obs_date"], keep="last")
    print(f"  transform: {before} rows in, {len(df)} rows out "
          f"({before - len(df)} dropped as null or duplicate)")
    return df


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load(df: pd.DataFrame, db_path: str | Path, rebuild: bool = False) -> None:
    """Create the schema and load the frame into the normalized tables."""
    if rebuild and Path(db_path).exists():
        Path(db_path).unlink()

    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)

        # --- dimension tables: insert distinct values, ignore repeats -------
        con.executemany(
            "INSERT OR IGNORE INTO cities (city_name) VALUES (?)",
            [(c,) for c in sorted(df["city_name"].unique())],
        )
        con.executemany(
            "INSERT OR IGNORE INTO conditions (condition_name) VALUES (?)",
            [(c,) for c in sorted(df["condition_name"].dropna().unique())],
        )

        city_map = dict(con.execute("SELECT city_name, city_id FROM cities"))
        cond_map = dict(con.execute(
            "SELECT condition_name, condition_id FROM conditions"))

        # --- fact table ----------------------------------------------------
        rows = [
            (
                city_map[r.city_name],
                r.obs_date,
                r.temp, r.tempmax, r.tempmin, r.humidity,
                r.precip, int(r.precip_flag), r.windspeed,
                r.sealevelpressure, r.cloudcover,
                cond_map.get(r.condition_name),
            )
            for r in df.itertuples(index=False)
        ]
        con.executemany(
            """INSERT OR REPLACE INTO observations
               (city_id, obs_date, temp, tempmax, tempmin, humidity,
                precip, precip_flag, windspeed, sealevelpressure,
                cloudcover, condition_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

        con.executescript(INDEXES)   # build indexes after the load, not during
        con.commit()

        counts = {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("cities", "conditions", "observations")
        }
        print(f"  load: {counts}")
    finally:
        con.close()


def run_etl(csv_path: str | Path, db_path: str | Path, rebuild: bool = True):
    print(f"ETL: {csv_path} -> {db_path}")
    load(transform(extract(csv_path)), db_path, rebuild=rebuild)
    print("ETL complete.")


# --------------------------------------------------------------------------
# Analytical queries — these replace the app's pandas aggregations
# --------------------------------------------------------------------------

Q_MONTHLY_SUMMARY = """
-- Monthly aggregate per city. JOIN + GROUP BY + HAVING.
SELECT  c.city_name,
        strftime('%Y-%m', o.obs_date)      AS month,
        COUNT(*)                           AS days_recorded,
        ROUND(AVG(o.temp), 2)              AS avg_temp,
        ROUND(MAX(o.tempmax), 2)           AS peak_temp,
        ROUND(MIN(o.tempmin), 2)           AS lowest_temp,
        SUM(o.precip_flag)                 AS wet_days
FROM        observations o
JOIN        cities       c ON c.city_id = o.city_id
GROUP BY    c.city_name, month
HAVING      COUNT(*) >= 1
ORDER BY    month;
"""

Q_CONDITION_BREAKDOWN = """
-- Three-table join with a percentage computed against a scalar subquery.
SELECT  ct.condition_name,
        COUNT(*)                                       AS n_days,
        ROUND(AVG(o.temp), 2)                          AS avg_temp,
        ROUND(AVG(o.humidity), 2)                      AS avg_humidity,
        ROUND(100.0 * COUNT(*) /
              (SELECT COUNT(*) FROM observations), 1)  AS pct_of_days
FROM        observations o
JOIN        conditions   ct ON ct.condition_id = o.condition_id
JOIN        cities       c  ON c.city_id       = o.city_id
GROUP BY    ct.condition_name
ORDER BY    n_days DESC;
"""

Q_ABOVE_AVERAGE_DAYS = """
-- Correlated subquery: days hotter than that city's own mean.
SELECT  c.city_name,
        o.obs_date,
        o.temp,
        ROUND(o.temp - (SELECT AVG(o2.temp)
                        FROM   observations o2
                        WHERE  o2.city_id = o.city_id), 2) AS deg_above_mean
FROM        observations o
JOIN        cities       c ON c.city_id = o.city_id
WHERE       o.temp > (SELECT AVG(o2.temp)
                      FROM   observations o2
                      WHERE  o2.city_id = o.city_id)
ORDER BY    deg_above_mean DESC;
"""

Q_ROLLING_TREND = """
-- Window functions: 3-day moving average and day-on-day change.
WITH ordered AS (
    SELECT  c.city_name,
            o.obs_date,
            o.temp
    FROM    observations o
    JOIN    cities       c ON c.city_id = o.city_id
)
SELECT  city_name,
        obs_date,
        temp,
        ROUND(AVG(temp) OVER (PARTITION BY city_name
                              ORDER BY obs_date
                              ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
            AS temp_3day_avg,
        ROUND(temp - LAG(temp) OVER (PARTITION BY city_name
                                     ORDER BY obs_date), 2)
            AS change_from_prev_day
FROM        ordered
ORDER BY    city_name, obs_date;
"""

QUERIES = {
    "monthly_summary":     Q_MONTHLY_SUMMARY,
    "condition_breakdown": Q_CONDITION_BREAKDOWN,
    "above_average_days":  Q_ABOVE_AVERAGE_DAYS,
    "rolling_trend":       Q_ROLLING_TREND,
}


def query(db_path: str | Path, name: str) -> pd.DataFrame:
    """Run one named analytical query and return it as a DataFrame."""
    if name not in QUERIES:
        raise KeyError(f"unknown query {name!r}; choose from {list(QUERIES)}")
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(QUERIES[name], con)
    finally:
        con.close()


def load_training_frame(db_path: str | Path) -> pd.DataFrame:
    """Feed the model from SQL rather than from the raw CSV.

    Doing the join and the date-part extraction in the database means the
    app receives a clean, model-ready frame and never has to know how the
    raw file was shaped.
    """
    sql = """
    SELECT  o.temp,
            o.humidity,
            o.sealevelpressure,
            o.precip,
            o.windspeed,
            o.cloudcover,
            o.precip_flag,
            CAST(strftime('%j', o.obs_date) AS INTEGER) AS day_of_year,
            CAST(strftime('%m', o.obs_date) AS INTEGER) AS month,
            COALESCE(ct.condition_name, 'Unknown')      AS conditions
    FROM        observations o
    LEFT JOIN   conditions   ct ON ct.condition_id = o.condition_id
    ORDER BY    o.obs_date;
    """
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def explain(db_path: str | Path, sql: str) -> list[str]:
    """Return SQLite's query plan — used to show the index is being hit."""
    con = sqlite3.connect(db_path)
    try:
        return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + sql)]
    finally:
        con.close()


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="kolkataweather.csv")
    p.add_argument("--db", default="weather.db")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="run every analytical query and print the results")
    args = p.parse_args()

    if not args.demo or args.rebuild or not Path(args.db).exists():
        run_etl(args.csv, args.db, rebuild=args.rebuild)

    if args.demo:
        for name in QUERIES:
            print(f"\n=== {name} " + "=" * (60 - len(name)))
            print(query(args.db, name).to_string(index=False))

        print("\n=== query plan (index check) " + "=" * 36)
        probe = ("SELECT * FROM observations "
                 "WHERE city_id = 1 AND obs_date BETWEEN '2025-01-01' AND '2025-12-31'")
        for line in explain(args.db, probe):
            print(" ", line)


if __name__ == "__main__":
    main()
