import pandas as pd

def filter_data(df, col_name = None, sector_filter = None):
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    if sector_filter:
        sector_filter = sector_filter.strip()
        filtered_df = df[df[col_name].str.strip().str.lower() == sector_filter.lower()]
    else:
        filtered_df = df.copy()
    unique_rows = filtered_df.drop_duplicates()

    return unique_rows
