import numpy as np
import pandas as pd
import geopandas as gpd

import setup

# from pyproj import Transformer
from pyproj import Geod

from collections import Counter


setup = setup.get()

included_parcel_types = {
    'FUNCTION_COMMERCIAL_INDUSTRIAL_OTHER': 'COMMERCIAL',
    'FUNCTION_RESIDENTIAL_COMPOSITION_MULTIPLE': 'RESIDENTIAL_MULTIPLE',
    'FUNCTION_RESIDENTIAL_COMPOSITION_SINGLE': 'RESIDENTIAL_SINGLE',
}

# WANT:
# Important: FIRST_FLOOR_AREA for single residential
# If you have it: HEAT_AIR_COND / AIR_COND_FLAG


def load_data():
    """Step 1: Load data and remove unnecessary columns."""

    def force_float(s):
        try:
            return float(s)
        except ValueError:
            # print(f'Not float: {s}')
            return -1

    df = pd.read_csv('in/svce/ITEM16_2018_tax_assessor_addtl_cols_masked_v2.csv',
                     dtype={'TRACTCE10': str}, index_col=0)
    use_codes = pd.read_csv('in/svce/use_codes.csv', index_col=0)['STRING'].to_dict()
    df['PARCELFN'] = df['PARCEL_FUNCTION'] + '_' + df['PARCEL_COMPOSITION'].fillna('OTHER')
    df['VALUE_LAND'] = df['LANDVALUE'] / (df['LAND_ACRES'] * 4046.0)
    df['VALUE_BLDG'] = df['TOTALIMPROVE'] / (df['BUILDING_FLOORSPACE_SQFT'] * 0.093)

    df['FIRST_FLOOR_AREA'] = df['FIRST_FLOOR_AREA'].apply(force_float).fillna(-1)
    df['NUMBER_FLOORS'] = df['NUMBER_FLOORS'].fillna(-1).astype(int)

    # If number of floors not set, but first floor area is available, calculate number of floors
    f = (df.NUMBER_FLOORS < 1) & (df.FIRST_FLOOR_AREA > 0)
    df.loc[f, 'NUMBER_FLOORS'] = np.maximum(
        1, np.round(df.loc[f, 'BUILDING_FLOORSPACE_SQFT'] / df.loc[f, 'FIRST_FLOOR_AREA']).astype(int))
    # Similarly, fill in first_floor_area using number of floors
    f = (df.NUMBER_FLOORS >= 1) & (df.FIRST_FLOOR_AREA <= 0)
    df.loc[f, 'FIRST_FLOOR_AREA'] = (
        df.loc[f, 'BUILDING_FLOORSPACE_SQFT'] / df.loc[f, 'NUMBER_FLOORS'])

    df['BUILDING_COVERAGE_RATIO'] = -1
    # Calculate BCR if number of floors is set
    f = (df.FIRST_FLOOR_AREA > 1) & (df.USABLE_SQ_FEET > 0)
    df.loc[f, 'BUILDING_COVERAGE_RATIO'] = df.loc[f,
                                                  'FIRST_FLOOR_AREA'] / df.loc[f, 'USABLE_SQ_FEET']
    # print((df.NUMBER_FLOORS < 1).sum() / len(df))
    # print((df.FIRST_FLOOR_AREA <= 0).sum() / len(df))
    # print((df.USABLE_SQ_FEET <= 0).sum() / len(df))
    # print((df.BUILDING_COVERAGE_RATIO < 0).sum() / len(df))

    # This fraction should be close to 0 (since the BCR shouldn't generally be above 1)
    print((df.BUILDING_COVERAGE_RATIO > 1).sum() / (df.BUILDING_COVERAGE_RATIO > 0).sum())
    df.loc[df.BUILDING_COVERAGE_RATIO > 1, 'BUILDING_COVERAGE_RATIO'] = 1.0

    df['HAS_AC'] = df.HEAT_AIR_COND.isin(['A', 'B']).astype(int)
    df.loc[df.AIR_COND_FLAG == 'Y', 'HAS_AC'] = 1
    df['HAS_HEAT'] = df.HEAT_AIR_COND.isin(['H', 'B']).astype(int)

    n1 = len(df)
    df = df[df.PARCELFN.isin(list(included_parcel_types.keys()))]
    n2 = len(df)
    print(f'Removed {n1-n2:,} rows ({(n1-n2)/n1*100:.1f}%) whose parcel type '
          f'is not among included types')

    # THIS IS BEING DONE LATER (see setup.py for filters)
    # df = df[(df.NUMBER_FLOORS >= 1) & (df.BUILDING_COVERAGE_RATIO > 0)]
    # n3 = len(df)
    # print(f'Removed {n2-n3:,} rows ({(n2-n3)/n2*100:.1f}%) where N_FLOORS / BCR could not be infered')
    df['PARCELFN'] = df['PARCELFN'].replace(included_parcel_types)
    df['USE'] = df['USE_CODE'].replace(use_codes)
    return df


# def add_density_gradients(cdata):
#     """Part of Step 2 (add change in density (e.g. population density) around each tract)

#     CURRENTLY NOT IN USE SINCE TRACTS ARE QUITE LARGE.
#     """

#     def get_proj():
#         transformer = Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
#         return transformer

#     def get_distance(lon1, lat1, lon2, lat2):
#         g = Geod(ellps='WGS84')
#         n = len(lon1)
#         lon2, lat2 = np.repeat(lon2, n), np.repeat(lat2, n)
#         _, _, distance = g.inv(lon1, lat1, lon2, lat2)
#         return distance

#     def get_gradient(df, field, lon, lat):

#         n_closest = 5
#         d = get_distance(df['INTPTLON'].values, df['INTPTLAT'].values, lon, lat)
#         v = df[field].values
#         q = np.argsort(d)
#         v = v - v[q[0]]
#         d = d[q[0:n_closest], np.newaxis]
#         a, _, _, _ = np.linalg.lstsq(d, v[q[0:n_closest]], rcond=None)
#         return a[0]

#     for s in 'people', 'jobs', 'poi':
#         cdata['GRADIENT_' + s.upper()] = cdata.apply(lambda r: get_gradient(cdata,
#                                                                             'DENSITY_' + s.upper(), r.INTPTLON, r.INTPTLAT), axis=1)

#     return cdata


def add_external_data(df):
    """Step 2: Add density of people, jobs, and POIs to each row based on census tract."""

    def mc(l):
        return Counter(l).most_common(1)[0][0]

    def strip_trct(i):
        # return int(str(i)[4:])
        return i[5:]

    def add_leading_zero(s):
        if len(s) < 12:
            return '0' + s

    xwalk = pd.read_csv('in/census/ca_xwalk.csv', dtype={'bgrp': 'str', 'trct': 'str'})
    xwalk = xwalk.groupby('bgrp').agg({'trct': mc})

    cdata = pd.read_csv('in/census/cbg_data_ca.csv',
                        dtype={'idx': 'str', 'trct': 'str'}).set_index('idx')
    cdata.index = cdata.index.map(add_leading_zero)
    cdata = cdata.join(xwalk)
    cdata['n_residential'] = cdata['n_people']
    # / cdata['n_jobs'].sum() * cdata['n_people'].sum()).fillna(0)
    cdata['n_commercial'] = cdata['n_jobs']
    # cdata['n_poi'] = (cdata['n_poi'] / cdata['n_poi'].sum() * cdata['n_poi'].sum()).fillna(0)
    # cdata['n_commercial'] = ((cdata['n_jobs'] + cdata['n_poi']) / 2).astype(int)

    f = {
        'n_residential': sum,
        'n_commercial': sum,
    }

    n1 = len(cdata)
    cdata = cdata.groupby('trct').agg(f, dropna=False)
    n2 = len(cdata)

    gdf = gpd.read_file('in/tiger/tl_2019_06_tract.shp').set_index('GEOID')
    gdf['INTPTLON'] = gdf['INTPTLON'].astype(float)
    gdf['INTPTLAT'] = gdf['INTPTLAT'].astype(float)
    cdata = cdata.join(gdf[['ALAND', 'INTPTLON', 'INTPTLAT']])

    f = {**f, **{
        'ALAND': sum,
        'INTPTLON': np.average,
        'INTPTLAT': np.average,
    }}

    cdata['ALAND'] = cdata['ALAND'] / 1e6
    cdata['shorttrct'] = cdata.index.map(strip_trct)
    cdata = cdata.groupby('shorttrct').agg(f, dropna=False)
    n3 = len(cdata)

    for s in 'residential', 'commercial':
        cdata['DENSITY_' + s.upper()] = cdata['n_' + s] / cdata['ALAND']

    cdata = cdata[cdata.index.isin(df['TRACTCE10'])]
    n4 = len(cdata)
    print(f"\nProcessed external data: {n1} census block groups > {n2} tracts > {n3} unique tracts > "
          f"{n4} included tracts (of {len(df.groupby('TRACTCE10'))})\n")
    # cdata = add_density_gradients(cdata)
    df = df.join(cdata, on='TRACTCE10')
    return df


def drop_columns(df):
    """Step 3: drop columns not in use."""

    df = df.drop(columns=[col for col in df.columns if col not in setup.keys()])
    return df


def drop_invalid_rows(df):
    """Step 4: Drop rows where at least one column has an invalid value."""

    n = len(df)
    for col, config in setup.items():
        n1 = len(df)
        df.drop(df[config['filter'](df[col])].index, inplace=True)
        n2 = len(df)
        print(
            f'Dropped {n1-n2} rows after filtering invalid values for `{col}`')
    print(f'Dropped {n-len(df)} rows in total ({(n-len(df))/n*100:.1f}%)\n')
    return df


# def logify(df):
#     for col in df.columns:
#         n_unique = len()

if __name__ == "__main__":
    df = load_data()
    df = add_external_data(df)
    df = drop_columns(df)
    df = drop_invalid_rows(df)
    # df = logify(df)
    df.to_csv('out/intermediate.csv')
