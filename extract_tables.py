import pandas as pd

def extract_tables(df):
    tables = []
    current_table = []
    for i, row in df.iterrows():
        if row.isnull().all():
            if current_table:
                tables.append(pd.DataFrame(current_table).reset_index(drop=True))
                current_table = []
        else:
            current_table.append(row)
    if current_table:
        tables.append(pd.DataFrame(current_table).reset_index(drop=True))
    if len(tables) > 1:
        merged_table = pd.concat([tables[0], tables[1]], ignore_index=True)
        tables = [merged_table] + tables[2:]
    return tables
    # return tables
