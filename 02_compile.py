import sys
import numpy as np
import pandas as pd
import numba
import time

# from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from k_means_constrained import KMeansConstrained
# import minsize_kmeans

import setup


ALG_MODE = ['custom', 'nn', 'minsize_kmeans'][0]
CONST_USE_WEIGHTS = True
CONST_N = 22  # The threshold which needs to be met
CONST_N_ROWS = -1
CONST_KMEANS_RATIO = [CONST_N, 100, 15, 30][0]

mode_idx = 1
LOSS_FN = ['abs', 'square'][mode_idx]
# Note: if LOSS_FN == 'abs', np.median will do better. If LOSS_FN == 'square', np.average will do better
HOMOG_FN = [np.median, np.average][mode_idx]


setup = setup.get()


def print_correlations(df):
    """Optional: print correlation coefficient for each column pair."""

    q = list(range(len(df.columns)))
    for i in q:
        for j in q[i + 1:]:
            col1 = df.columns[i]
            col2 = df.columns[j]
            c = np.corrcoef(df[col1], df[col2])[0, 1]
            print(f'{col1} <> {col2}: {c:.2f}')


def logify(df):
    for col, config in setup.items():
        if config['log'] == 1:
            if np.isclose(df[col].min(), 0) or df[col].min() < 0:
                df[col] = np.round(np.log(df[col] + 1), 4)
            else:
                df[col] = np.round(np.log(df[col]), 4)
    return df


def convert_categorical_to_dummies(df, cols, col_weights):
    """Step 1: convert categorical variables into dummy columns."""

    for col, config in setup.items():
        if config['type'] == 2:
            df = df.join(pd.get_dummies(df[col], prefix=col))
            df.drop(columns=col, inplace=True)
            new_cols = [s for s in df.columns if s.startswith(f'{col}_')]
            cols.extend(new_cols)
            col_weights.extend([config['w'], ] * len(new_cols))
        else:
            cols.append(col)
            col_weights.append(config['w'])

    return df, cols, col_weights


# @numba.jit(nopython=True)
def make_values_equal_by_col(v, idx):
    w = v[idx, :]
    # before = np.copy(w)
    for i in range(w.shape[1]):
        # Set all values to the value closest to the median (if HOMOG_FN = np.median) among all values
        w[:, i] = w[np.abs(w[:, i] - HOMOG_FN(w[:, i])).argmin(), i]

    v[idx, :] = w
    return v


# @numba.jit(nopython=True)
def algorithm(v, vnorm, col_weights):
    """Part of Step 3 (used in function apply_algorithm)

    Input:
        v       numpy array containing values of all rows
        vnorm   numpy array containing same values, but normalized so that min
                of each column is 0 and max is 1. Should be float.
        col_weights numpy array containing column weights

    Returns:
        v       modified array of values that now all match threshold
    """

    def get_delta(v_reference, v_available):
        # Note: np.power() here always seems better even if LOSS_FN is 'abs', interestingly
        return np.sum(np.power(v_available - v_reference, 2), axis=1)

    def verify_identical_values(v):
        if np.unique(v, axis=0).shape[0] != 1:
            raise ValueError('Columns of rows that are supposed to be identical have'
                             'non-unique values')

    def verify_idx_length_all(idx, n):
        if len(idx) < n:
            raise ValueError(
                f'Number of matched rows ({len(idx)}) is smaller than threshold')

    i = 0
    print('Starting algorithm...')
    idx = np.arange(v.shape[0])
    completed_rows = (idx < 0).astype(np.int32)

    print('Calculating average difference of each row...')
    t1 = time.perf_counter()

    delta_init = np.absolute(vnorm - vnorm.mean(axis=0)).mean(axis=1)
    group_idx = np.zeros(vnorm.shape[0])

    t2 = time.perf_counter()
    print(f'  -> {t2-t1:.1f}s\n')
    # delta_init_sorted = np.argsort(delta_init)
    print('Starting to process groups...')

    while np.any(completed_rows == 0):

        available = (completed_rows == 0)
        # ref_row = np.random.choice(idx[available])
        ref_row = idx[available][0]
        # As the next row, among those still available, select the one that's the most different
        # from the average row
        max_difference = np.argmax(delta_init[available])
        ref_row = idx[available][max_difference]
        # Get difference in values between each row that is a potential match
        delta_norm = get_delta(vnorm[ref_row, :], vnorm[available, :])

        if available.sum() >= CONST_N * 2:
            idx_min = np.argpartition(delta_norm, CONST_N)[:CONST_N]
            # idx_min = np.argsort(delta_norm)[:CONST_N]
            idx_all = idx[available][idx_min]
        else:
            print('Matching would fail because number of available rows is smaller '
                  'than number of rows that need to matched to reach threshold. '
                  f'Adding all remaining {available.sum()} rows to last bracket.')
            idx_all = idx[available]

        v = make_values_equal_by_col(v, idx_all)
        completed_rows[idx_all] = 1
        group_idx[idx_all] = i

        # verify_idx_length_all(idx_all, CONST_N)
        # verify_identical_values(v[idx_all, :])

        i += 1
        if i % 100 == 0:
            print(f'{i} groups processed')

    return v, group_idx


# def algorithm_nn(v, vnorm):

#     n_groups = 5000
#     kmeans = NearestNeighbors(n_neighbors=CONST_N, algorithm='ball_tree').fit(vnorm)
#     unique, counts = np.unique(kmeans.labels_, return_counts=True)
#     print(f'-> {len(unique)} groups (min {counts.min()} and '
#           f'max {counts.max():,} values per group)')
#     for i in unique:
#         idx_all = (kmeans.labels_ == i)
#         make_values_equal_by_col(v, idx_all)
#     return v


def algorithm_km_minsize(v, vnorm):

    if CONST_N_ROWS > 0:
        n_groups = CONST_N_ROWS // CONST_KMEANS_RATIO
    else:
        n_groups = v.shape[0] // CONST_KMEANS_RATIO

    print('Clustering...')
    kmeans = KMeansConstrained(n_clusters=n_groups, size_min=CONST_N, random_state=0).fit(vnorm)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    print(f'-> {len(unique)} groups (min {counts.min()} and '
          f'max {counts.max():,} values per group)')
    for i in unique:
        idx_all = (kmeans.labels_ == i)
        make_values_equal_by_col(v, idx_all)
    return v


def get_vnorm(v, col_weights):
    vnorm = (v - v.min(axis=0)) / (v.max(axis=0) - v.min(axis=0))
    if CONST_USE_WEIGHTS:
        vnorm = vnorm * col_weights[np.newaxis, :]
    return vnorm


def apply_algorithm(df, col_weights):
    """Step 3: apply algorithm."""

    v = df.values
    vnorm = get_vnorm(v, col_weights)
    # cov = np.std(vnorm, axis=0) / np.mean(vnorm, axis=0)
    # for i in range(vnorm.shape[1]):
    #     print(f'COV | {cols[i]}: {cov[i]:.3f}')
    if ALG_MODE == 'nn':
        v = algorithm_nn(v, vnorm)
    elif ALG_MODE == 'minsize_kmeans':
        v = algorithm_km_minsize(v, vnorm)
    else:
        v, group_idx = algorithm(v, vnorm, col_weights)
    df = pd.DataFrame(v, index=df.index, columns=df.columns)
    return df, group_idx


def get_metrics(res_all, df_before, df_after, scale, title, cols, col_weights, weighted=False):

    df_before = df_before[[col for col in df_after.columns if not col.startswith('counts')]]
    df_after = df_after[[col for col in df_after.columns if not col.startswith('counts')]]
    col_weights = col_weights / col_weights.sum() * len(col_weights)

    if list(df_before.columns) != list(df_after.columns):
        raise ValueError('DataFrame columns must be identical')
    if len(df_before) != len(df_after):
        raise ValueError('DataFrame lengths must be identical')

    min_not_zero = (df_before.values.min(axis=0) > 0)

    v_before = df_before.values.astype(float)
    v_after = df_after.values.astype(float)

    df = df_after
    different_values = ~np.isclose(v_after, v_before)
    df['n_changed'] = np.sum(different_values, axis=1)

    if LOSS_FN == 'square':
        loss = (np.power(v_after - v_before, 2)).mean(axis=0)
    else:
        loss = np.average(np.absolute(v_after - v_before), axis=0)
    if not weighted:
        loss = loss / scale
    else:
        loss = loss * col_weights / scale
        # loss = info_loss_fn(v_after, v_before) * col_weights[np.newaxis, :] / scale[np.newaxis, :]

    # df['loss'] = np.average(loss, axis=1)

    res = {}
    print(f'Starting metrics: {title}')

    v = (df['n_changed'] > 0).sum() / len(df) * 100
    res['Fraction of affected rows'] = np.round(v, 2)

    v = different_values.sum() / (len(df) * len(df.columns)) * 100
    res['Fraction of affected values'] = np.round(v, 2)

    v = np.average(df['n_changed'])
    res['Average number of changed values per row'] = np.round(v, 2)

    # v = np.average(df.loc[df.n_changed > 0, 'n_changed'])
    # res['Average number of changed values per row (of affected rows)'] = np.round(v, 2)

    v = np.average(loss) * 100
    res['Average change relative to scale'] = np.round(v, 2)

    # v = np.average(loss_w) * 100
    # res['Average change relative to scale (weighted)'] = np.round(v, 2)

    # v = np.average(change_rel) * 100
    # res['Average percentage change (where possible)'] = np.round(v, 2)

    # v = np.average(loss[loss > 0]) * 100
    # res['Average absolute change (vs scale) of changed values'] = np.round(v, 2)

    # v = np.average(change_rel[change_rel > 0]) * 100
    # res['Average relative change of changed values'] = np.round(v, 2)

    for i, col in enumerate(cols):
        # if not weighted:
        #     loss = info_loss_fn(v_after[:, i], v_before[:, i]) / scale[i]
        # else:
        #     loss = info_loss_fn(v_after[:, i], v_before[:, i]) * col_weights[i] / scale[i]
        v = loss[i] * 100
        res[f'  {col}'] = np.round(v, 2)

    res_all[title] = res
    return df


def reapply_bin_values(df, bin_values):

    # bin_values = pd.read_csv('bins/bin_median_values.csv', index_col=0)
    # bin_map = pd.read_csv('data/stanford_unique_bins.csv', index_col=0)
    # x = list(bin_map.columns)

    for col in df.columns:
        if col in bin_values.columns:
            df[col] = df[col].apply(lambda idx: np.round(bin_values[col].loc[idx], 1))
        if pd.isnull(df[col]).sum() > 0:
            raise ValueError(f'Null value detected in column {col}')

    return df


if __name__ == "__main__":

    # Processing
    df = pd.read_csv('out/intermediate.csv', index_col=0)
    # if CONST_N_ROWS > 0:
    #     df = df.sample(CONST_N_ROWS, random_state=0)
    if CONST_N_ROWS > 0:
        df = df.iloc[0:CONST_N_ROWS]
    cols = []
    col_weights = []
    df, cols, col_weights = convert_categorical_to_dummies(df, cols, col_weights)
    df = logify(df)
    # Reorder columns
    df_original = df[cols]
    col_weights = np.array(col_weights)
    t1 = time.perf_counter()
    df_processed, group_idx = apply_algorithm(df_original, col_weights)
    t2 = time.perf_counter()

    print(f'\nDONE. Data contains {len(df_processed)} rows x {len(df_processed.columns)} cols.')
    counts = df_processed.groupby(cols)[cols[0]].count().values
    print(f'-> {len(counts):,} groups (min {counts.min()} and '
          f'max {counts.max():,} values per group).\n'
          f'-> {t2-t1:.1f}s\n---\n')

    # Metrics
    res = {}
    # Need to define scale here so that it's the same across all metrics calculations
    if LOSS_FN == 'square':
        scale = np.power(df_original.values - df_original.values.mean(axis=0), 2).mean(axis=0)
    else:
        scale = np.absolute(df_original.values - df_original.values.mean(axis=0)).mean(axis=0)

    # for i in range(df_original.values.shape[1]):
    #     tmp = np.corrcoef(df_original.values[:, i], df_processed.values[:, i])
    #     print(i, tmp[0, 1])

    get_metrics(res, df_original, df_processed, scale, 'Unweighted', cols, col_weights)
    get_metrics(res, df_original, df_processed, scale, 'Weighted', cols, col_weights, True)
    res = pd.DataFrame.from_dict(res)
    pd.set_option('display.width', 1000)
    print(res)

    # col = 'BUILDING_FLOORSPACE_SQFT'
    # test = df_original[[col]]
    # test = test.join(df_binned[[col]], rsuffix='_BIN')
    # test = test.join(df_processed[[col]], rsuffix='_FINAL')
    # test['D1'] = (test[col] - test[f'{col}_BIN']).abs()
    # test['D2'] = (test[f'{col}_BIN'] - test[f'{col}_FINAL']).abs()
    # test['D3'] = (test[col] - test[f'{col}_FINAL']).abs()
    # test['DIDX'] = (test['D3'] < test['D2']).astype(int)
    # print(test['D1'].sum())
    # print(test['D2'].sum())
    # print(test['D3'].sum())
    # print(test['DIDX'].sum())
    # test.to_csv('test.csv')
    # sys.exit()

    df_processed['Bin_ID'] = group_idx

    # Storing
    df_processed[cols + ['Bin_ID', ]].to_csv('out/data.csv')
    df_processed.to_csv('out/data_withmeta.csv', float_format='%.2f')
