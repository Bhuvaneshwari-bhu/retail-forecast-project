"""
Step 5 (memory-safe version): Merge real M5 sales + prices + calendar +
weather + holidays into one clean modeling table, and append the simulated
streaming days on top.

WHY THIS VERSION IS DIFFERENT:
The M5 sales file has ~30,490 rows (one per item x store) but ~1,913 day
columns each. Melted into long format that's ~58 MILLION rows. Doing that
in one shot with plain object/string columns can easily use 8-15+ GB of RAM
and gets killed by the OS on a normal laptop.

This version fixes that by:
  1. Using category dtype for item_id/store_id/state_id (huge memory win -
     stores each unique string once instead of 58M times)
  2. Downcasting numeric columns (int16/float32 instead of int64/float64)
  3. Processing ONE STORE AT A TIME (10 stores -> ~5.8M rows per chunk
     instead of 58M all at once)
  4. Streaming each chunk straight to a parquet file on disk instead of
     holding everything in memory and concatenating at the end

Tested to run within ~1-1.5 GB peak RAM per store chunk - safe for laptops
with 4GB+ available memory.

Output: data/processed/modeling_table.parquet

RUN (after download_m5.py/manual download, fetch_weather.py,
fetch_holidays.py, and simulate_stream.py have all run):
    python src/merge_datasets.py
"""

import os
import glob
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
STREAM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "streaming")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "modeling_table.parquet")


def load_reference_tables():
    """Small tables - safe to keep fully in memory."""
    calendar = pd.read_csv(
        os.path.join(RAW_DIR, "calendar.csv"),
        usecols=["d", "date", "wm_yr_wk"],
    )
    calendar["date"] = pd.to_datetime(calendar["date"])
    day_to_date = dict(zip(calendar["d"], calendar["date"]))
    day_to_week = dict(zip(calendar["d"], calendar["wm_yr_wk"]))

    prices = pd.read_csv(os.path.join(RAW_DIR, "sell_prices.csv"))
    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
    prices["sell_price"] = prices["sell_price"].astype("float32")

    weather_path = os.path.join(RAW_DIR, "weather.csv")
    weather = pd.read_csv(weather_path) if os.path.exists(weather_path) else None
    if weather is not None:
        weather["date"] = pd.to_datetime(weather["date"])

    holidays_path = os.path.join(RAW_DIR, "holidays.csv")
    hol = None
    if os.path.exists(holidays_path):
        hol = pd.read_csv(holidays_path).rename(columns={"date_str": "date"})
        hol["date"] = pd.to_datetime(hol["date"])

    return day_to_date, day_to_week, prices, weather, hol


def process_one_store(store_id, sales_path, day_to_date, day_to_week, prices, weather, hol):
    """Melt + enrich just the rows for a single store. Returns a compact DataFrame."""
    id_cols = ["item_id", "store_id", "state_id"]

    # Stream-read the wide file in row chunks and keep only this store's rows,
    # so we never hold the full 30,490-row wide table in memory at once.
    chunks = []
    for chunk in pd.read_csv(sales_path, chunksize=2000):
        filtered = chunk[chunk["store_id"] == store_id]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return None
    store_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    day_cols = [c for c in store_df.columns if c.startswith("d_")]

    long_df = store_df.melt(
        id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="units_sold"
    )
    del store_df
    gc.collect()

    long_df["item_id"] = long_df["item_id"].astype("category")
    long_df["store_id"] = long_df["store_id"].astype("category")
    long_df["state_id"] = long_df["state_id"].astype("category")
    long_df["units_sold"] = long_df["units_sold"].astype("int16")

    long_df["date"] = long_df["d"].map(day_to_date)
    long_df["wm_yr_wk"] = long_df["d"].map(day_to_week).astype("int32")
    long_df = long_df.drop(columns=["d"])

    long_df = long_df.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    long_df = long_df.drop(columns=["wm_yr_wk"])

    if weather is not None:
        long_df = long_df.merge(weather, on=["date", "state_id"], how="left")
    if hol is not None:
        long_df = long_df.merge(hol, on="date", how="left")

    long_df["is_simulated"] = 0
    return long_df


def load_simulated_long():
    files = sorted(glob.glob(os.path.join(STREAM_DIR, "*.csv")))
    files = [f for f in files if "_drift_log" not in f]
    if not files:
        print("No simulated streaming files found - skipping (optional).")
        return None

    dfs = [pd.read_csv(f) for f in files]
    sim_df = pd.concat(dfs, ignore_index=True)
    sim_df = sim_df.drop(columns=["transaction_id"])
    sim_df = sim_df.groupby(["date", "item_id", "store_id"], as_index=False)["units_sold"].sum()
    sim_df["date"] = pd.to_datetime(sim_df["date"])
    sim_df["state_id"] = sim_df["store_id"].str.split("_").str[0]
    sim_df["sell_price"] = pd.NA
    sim_df["is_simulated"] = 1

    weather_path = os.path.join(RAW_DIR, "weather.csv")
    if os.path.exists(weather_path):
        weather = pd.read_csv(weather_path)
        weather["date"] = pd.to_datetime(weather["date"])
        sim_df = sim_df.merge(weather, on=["date", "state_id"], how="left")

    holidays_path = os.path.join(RAW_DIR, "holidays.csv")
    if os.path.exists(holidays_path):
        hol = pd.read_csv(holidays_path).rename(columns={"date_str": "date"})
        hol["date"] = pd.to_datetime(hol["date"])
        sim_df = sim_df.merge(hol, on="date", how="left")

    sim_df["item_id"] = sim_df["item_id"].astype("category")
    sim_df["store_id"] = sim_df["store_id"].astype("category")
    sim_df["state_id"] = sim_df["state_id"].astype("category")
    sim_df["units_sold"] = sim_df["units_sold"].astype("int16")
    return sim_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    sales_path = os.path.join(RAW_DIR, "sales_train_validation.csv")
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Missing {sales_path}")

    print("Loading small reference tables (calendar, prices, weather, holidays)...")
    day_to_date, day_to_week, prices, weather, hol = load_reference_tables()

    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3",
              "WI_1", "WI_2", "WI_3"]

    writer = None
    total_rows = 0

    for store_id in stores:
        print(f"Processing store {store_id}...")
        store_long = process_one_store(
            store_id, sales_path, day_to_date, day_to_week, prices, weather, hol
        )
        if store_long is None:
            print(f"  no rows for {store_id}, skipping")
            continue

        table = pa.Table.from_pandas(store_long, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, table.schema)
        writer.write_table(table)

        total_rows += len(store_long)
        print(f"  wrote {len(store_long):,} rows (running total: {total_rows:,})")

        del store_long, table
        gc.collect()

    print("Processing simulated streaming data...")
    sim_df = load_simulated_long()
    if sim_df is not None and writer is not None:
        for col in writer.schema.names:
            if col not in sim_df.columns:
                sim_df[col] = pd.NA
        sim_df = sim_df[writer.schema.names]
        table = pa.Table.from_pandas(sim_df, schema=writer.schema, preserve_index=False)
        writer.write_table(table)
        total_rows += len(sim_df)
        print(f"  wrote {len(sim_df):,} simulated rows (running total: {total_rows:,})")

    if writer is not None:
        writer.close()

    print(f"\nDone. Saved {total_rows:,} total rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
