"""
Step 3: Generate a clean US holiday calendar for the modeling date range.

WHY: M5's calendar.csv already has some event flags, but they're inconsistent
and incomplete. A clean, explicit holiday feature (is_holiday, holiday_name,
days_to_next_holiday) is a much stronger signal for the model and is easy to
explain in an interview: "I engineered holiday-proximity features because
retail demand spikes before holidays, not just on them."

RUN:
    python src/fetch_holidays.py
"""

import os
import pandas as pd
import holidays

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

START_YEAR = 2011
END_YEAR = 2017  # a little past M5's end date for safety


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    us_holidays = holidays.US(years=range(START_YEAR, END_YEAR + 1))

    date_range = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D")
    df = pd.DataFrame({"date": date_range})
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = df["date"].apply(lambda d: d in us_holidays).astype(int)
    df["holiday_name"] = df["date"].apply(lambda d: us_holidays.get(d, ""))

    # days until the next holiday (useful for "pre-holiday demand ramp" signal)
    holiday_dates = df.loc[df["is_holiday"] == 1, "date"].tolist()

    def days_to_next(d):
        future = [h for h in holiday_dates if h >= d]
        return (min(future) - d).days if future else None

    df["days_to_next_holiday"] = df["date"].apply(days_to_next)

    out_path = os.path.join(RAW_DIR, "holidays.csv")
    df.drop(columns=["date"]).to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df[df["is_holiday"] == 1].head(10)[["date_str", "holiday_name"]])


if __name__ == "__main__":
    main()
