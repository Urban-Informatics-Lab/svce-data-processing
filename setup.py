"""
For each column, set:
* 'brackets' to: * an integer for the number of brackets, each containing the
                   same number of values (i.e., split by percentiles) to bin
                   the column values into.
                 * a list to define custom edge bins for the brackets.
                 * None if the column is categorical and should be turned into
                   dummies.
* 'filter', which should contain a lambda function that returns those rows
  that should be filtered out (.e.g, value too small, no value, etc). The
  input of that lambda function is the pandas Series containing the corresponding
  column.
If a column is not listed in setup, it will be dropped from the data.

"""

import pandas as pd
import numpy as np


setup = {
    'BUILDING_FLOORSPACE_SQFT': {'log': 1, 'type': 0, 'w': 3.0, 'filter': lambda v: (v < 1) | pd.isnull(v)},
    'PARCEL_UNITS':             {'log': 0, 'type': 1, 'w': 5.0, 'filter': lambda v: (v < 1) | pd.isnull(v)},
    'YEAR_BUILT':               {'log': 0, 'type': 1, 'w': 1.0, 'filter': lambda v: (v < 1500) | pd.isnull(v)},
    'EFFECTIVE_YEAR':           {'log': 0, 'type': 1, 'w': 1.0, 'filter': lambda v: (v < 1500) | pd.isnull(v)},
    'NUMBER_FLOORS':            {'log': 0, 'type': 0, 'w': 3.0, 'filter': lambda v: (v < 1) | pd.isnull(v)},
    'BUILDING_COVERAGE_RATIO':  {'log': 0, 'type': 0, 'w': 1.0, 'filter': lambda v: (v < 0.0001) | pd.isnull(v) | np.isinf(v)},
    # 'USE':                    {'log': 0, 'type': 2, 'w': 1.0, 'filter': lambda v: pd.isnull(v)},
    'PARCELFN':                 {'log': 0, 'type': 2, 'w': 100.0, 'filter': lambda v: pd.isnull(v)},
    'DENSITY_RESIDENTIAL':      {'log': 1, 'type': 0, 'w': 2.0, 'filter': lambda v: (v < 0) | pd.isnull(v) | np.isinf(v)},
    'DENSITY_COMMERCIAL':       {'log': 1, 'type': 0, 'w': 2.0, 'filter': lambda v: (v < 0) | pd.isnull(v) | np.isinf(v)},
    'VALUE_LAND':               {'log': 1, 'type': 0, 'w': 1.0, 'filter': lambda v: (v < 0) | pd.isnull(v) | np.isinf(v)},
    'VALUE_BLDG':               {'log': 1, 'type': 0, 'w': 1.0, 'filter': lambda v: (v < 0) | pd.isnull(v) | np.isinf(v)},
    'HAS_AC':                   {'log': 0, 'type': 1, 'w': 1.0, 'filter': lambda v: (v < 0) | pd.isnull(v)},
    'HAS_HEAT':                 {'log': 0, 'type': 1, 'w': 1.0, 'filter': lambda v: (v < 0) | pd.isnull(v)},
}


def get():
  return setup
