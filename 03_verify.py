import numpy as np
import pandas as pd


filename = 'out/data_withgroups_final.csv'
threshold = 22


def verify(df, threshold, col_list):
    # Group data by all columns (except for index), then determine the "length"
    # of each group (that is, the number of rows that belong to each group).
    group_lengths = df.groupby(col_list).agg(
        {col_list[0]: 'count'}).values
    # Determine the minimum group size among all groups
    v = group_lengths.min()
    # Determine whether the smallest group size is at least as large as the
    # threshold, in which case the data is compliant.
    s = 'PASSED' if v >= threshold else 'FAILED'
    print(f'Minimum number of matching rows found: {v} ({s})')


if __name__ == "__main__":
    df = pd.read_csv(filename, index_col=0)  # , dtype=np.int64
    all_columns = list(df.columns)
    print('Columns:', all_columns)
    verify(df, threshold=threshold, col_list=all_columns)
