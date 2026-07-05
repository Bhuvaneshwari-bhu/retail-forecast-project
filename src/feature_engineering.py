"""
Phase 2, Step 1: Feature engineering with DuckDB - safe on memory-constrained
laptops, and produces a Colab-ready training sample.

WHY DUCKDB INSTEAD OF PANDAS HERE:
DuckDB processes parquet files out-of-core (streaming from disk, spilling to
disk when needed) instead of loading everything into RAM like pandas does.
It can run window functions (lags, rolling averages) over the full 58M-row
table without crashing your machine.

WHAT THIS SCRIPT DOES:
1. Finds the top N highest-volume items across the full dataset (a
   representative, not-cherry-picked sample - ranked by real sales volume)
2. Filters the 58M-row table down to just those items (still all 10 stores,
   full date range, real + simulated data)
3. Engineers time-series features via SQL window functions:
   - lag_1, lag_7, lag_28       (units sold N days ago)
   - rolling_mean_7/28          (avg over past week/month, EXCLUDING today -
                                  this avoids data leakage)
   - rolling_std_7              (recent volatility)
   - price_change               (day-over-day price delta)
   - weekday, month, year, is_weekend (calendar features)
4. Drops the first 28 days per item/store (they can't have a valid lag_28 -
   keeping them would introduce nulls that break most ML libraries)
5. Writes ONE small, Colab-ready parquet file

RUN (default: top 300 items, ~3GB memory ceiling):
    python src/feature_engineering.py

RUN with custom sample size or memory limit:
    python src/feature_engineering.py --n_items 500 --memory_limit 2GB
"""

import os
import argparse
import duckdb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELING_TABLE = os.path.join(PROCESSED_DIR, "modeling_table.parquet")
OUT_PATH = os.path.join(PROCESSED_DIR, "features_sample.parquet")
TMP_DIR = os.path.join(os.path.dirname(__file__), "..", "duckdb_tmp")


def main(n_items, memory_limit):
    if not os.path.exists(MODELING_TABLE):
        raise FileNotFoundError(
            f"Missing {MODELING_TABLE}. Run src/merge_datasets.py first (Phase 1)."
        )

    os.makedirs(TMP_DIR, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{TMP_DIR}'")

    print(f"Finding top {n_items} items by total sales volume...")
    top_items = con.execute(f"""
        SELECT item_id
        FROM read_parquet('{MODELING_TABLE}')
        GROUP BY item_id
        ORDER BY SUM(units_sold) DESC
        LIMIT {n_items}
    """).fetchall()
    top_items = [row[0] for row in top_items]
    print(f"  selected {len(top_items)} items")

    items_list_sql = ",".join(f"'{i}'" for i in top_items)

    print("Engineering features via DuckDB (lags, rolling stats, calendar)...")
    print("(this streams from disk - may take a few minutes, won't crash your RAM)")

    query = f"""
    COPY (
        WITH base AS (
            SELECT *
            FROM read_parquet('{MODELING_TABLE}')
            WHERE item_id IN ({items_list_sql})
        ),
        feat AS (
            SELECT
                *,
                EXTRACT(dow FROM date)::INTEGER AS weekday,
                EXTRACT(month FROM date)::INTEGER AS month,
                EXTRACT(year FROM date)::INTEGER AS year,
                CASE WHEN EXTRACT(dow FROM date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
                LAG(units_sold, 1) OVER w AS lag_1,
                LAG(units_sold, 7) OVER w AS lag_7,
                LAG(units_sold, 28) OVER w AS lag_28,
                AVG(units_sold) OVER (
                    PARTITION BY item_id, store_id ORDER BY date
                    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                ) AS rolling_mean_7,
                AVG(units_sold) OVER (
                    PARTITION BY item_id, store_id ORDER BY date
                    ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
                ) AS rolling_mean_28,
                STDDEV(units_sold) OVER (
                    PARTITION BY item_id, store_id ORDER BY date
                    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                ) AS rolling_std_7,
                sell_price - LAG(sell_price, 1) OVER w AS price_change
            FROM base
            WINDOW w AS (PARTITION BY item_id, store_id ORDER BY date)
        )
        SELECT *
        FROM feat
        WHERE lag_28 IS NOT NULL
        ORDER BY item_id, store_id, date
    ) TO '{OUT_PATH}' (FORMAT PARQUET);
    """
    con.execute(query)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OUT_PATH}')"
    ).fetchone()[0]
    print(f"\nDone. Wrote {n_rows:,} feature rows to {OUT_PATH}")

    preview = con.execute(f"SELECT * FROM read_parquet('{OUT_PATH}') LIMIT 5").df()
    print("\nPreview:")
    print(preview)

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"\nFile size: {size_mb:.1f} MB (this is what you'll upload to Colab)")

    con.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_items", type=int, default=300,
                    help="number of top-selling items to include (default 300)")
    p.add_argument("--memory_limit", type=str, default="3GB",
                    help="DuckDB memory ceiling before it spills to disk (default 3GB)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.n_items, args.memory_limit)
