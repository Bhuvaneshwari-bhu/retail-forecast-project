"""
Verify data/processed/modeling_table.parquet WITHOUT loading the whole
58M-row file into memory at once (that's what killed the one-liner check).

Strategy:
  - Row count + schema: read from parquet metadata only (instant, no data read)
  - Per-column stats (missing %, min/max, value counts): read ONE column at a
    time, since a single column of 58M rows is small; the whole table with
    12+ columns at once is what caused the OOM kill.
  - Head preview: read just the first small batch, not the whole file.

RUN:
    python src/verify_data.py
"""

import os
import pandas as pd
import pyarrow.parquet as pq

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PATH = os.path.join(PROCESSED_DIR, "modeling_table.parquet")


def main():
    pf = pq.ParquetFile(PATH)
    schema = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    print(f"File: {PATH}")
    print(f"Total rows: {total_rows:,}")
    print(f"Columns: {schema.names}")
    print()

    # --- is_simulated split (read only that one column) ---
    col = pd.read_parquet(PATH, columns=["is_simulated"])["is_simulated"]
    print("Real vs simulated row counts:")
    print(col.value_counts())
    del col
    print()

    # --- date range (read only date column) ---
    dates = pd.read_parquet(PATH, columns=["date"])["date"]
    print(f"Date range: {dates.min()} to {dates.max()}")
    del dates
    print()

    # --- missing % per column, one column at a time ---
    print("Missing values (%) by column:")
    for col_name in schema.names:
        series = pd.read_parquet(PATH, columns=[col_name])[col_name]
        pct_missing = series.isna().mean() * 100
        print(f"  {col_name:25s} {pct_missing:6.2f}%")
        del series
    print()

    # --- head preview: just first batch, not full file ---
    print("First 5 rows:")
    batch = next(pf.iter_batches(batch_size=5))
    print(batch.to_pandas())


if __name__ == "__main__":
    main()
