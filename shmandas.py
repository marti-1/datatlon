# -*- coding: utf-8 -*-
import numpy as np
from functools import reduce
import pandas as pd
from tabulate import tabulate

def cond(xs, fn):
    if type(xs) == list:
        return np.array([True if fn(x) else False for i, x in enumerate(xs)])
    else:
        return fn(xs)


def condh(xs, cmp, value):
    if cmp == '==':
        return cond(xs, lambda x: x == value)
    elif cmp == '>':
        return cond(xs, lambda x: x > value)
    elif cmp == '<':
        return cond(xs, lambda x: x < value)
    elif cmp == '>=':
        return cond(xs, lambda x: x >= value)
    elif cmp == '<=':
        return cond(xs, lambda x: x <= value)
    else:
        raise Exception(f'Unknown compare operation "{cmp}"')


def land(a,b):
    return np.logical_and(a,b)


def lor(a,b):
    return np.logical_or(a,b)


def lnot(a):
    return np.logical_not(a)


def _clone(t):
    t_copy = {}
    for f in t.keys():
        t_copy[f] = t[f].copy()
    return t_copy


def nrows(t):
    return len(t[list(t.keys())[0]])


def get_row(t, idx):
    y = {}
    for c in t.keys():
        y[c] = t[c][idx]
    return y


def rows(t, idx):
    """
    expect idx to be a boolean array/list
    """
    y = {}
    for f in t.keys():
        if type(t[f]) == list:
            if type(idx[0]) in {bool, np.bool_}:
                y[f] = [x for i, x in enumerate(t[f]) if idx[i]]
            elif type(idx[0]) == int:
                y[f] = [t[f][i] for i in idx]
            else:
                raise Exception(f'Index type not "{type(idx[0])}" supported')
        else:
            y[f] = t[f][idx]
    return y
    

def slice_rows(t, _from, _to):
    fields = t.keys()
    s = {}
    for f in fields:
        s[f] = t[f][_from:_to]
    return s


def group_by(t, fields):
    """
    Expects t to be already sorted by fields.
    fields -- (f1, f2, ...)
    """
    if type(fields) == str:
        fields = [fields]
    
    groups = []
    
    def make_key(row, fields):
        key = []
        for f in fields:
            key.append(row[f])
        return tuple(key)
    
    groups = {}
    for i in range(nrows(t)):
        row = get_row(t,i)
        key = make_key(row, fields)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    return [from_dicts(g) for g in groups.values()]


def apply(gs, fn):
    """
    Immutable.
    list of groups made from a table, e.g. with `group_by`
    
    ```
    apply(gs, {'ret': lambda x: pctReturn(x['price'])})
    ```
    """
    gs = list(map(_clone, gs))
    if type(fn) == dict:
        for g in gs:
            for f in fn.keys():
                g[f] = fn[f](g)
                assert(len(g[f]) == nrows(g))
        return gs
    else:
        return list(map(fn, gs))
    
    
def fields(t, start, end=None):
    f = list(t.keys())
    if type(start) is int:
        if end is None:
            end = len(f)
    elif type(start) is str:
        start = f.index(start)
        if end is None:
            end = len(f)
        else:
            end = f.index(end)
    else:
        raise Exception('Unknown range type.')
            
    return f[start:end]

def select(t, fields):
    """
    TODO: add select(t, rangeStart, rangeEnd) syntax, maybe there should be just a helper function to turn range into column names.
    select(t, fields(t, 'school', 'guardian'))
    """
    return {k: t[k] for k in fields}


def ungroup(gs):
    fields = gs[0].keys()

    def concat_lists(xs):
        out = xs[0]
        for x in xs[1:]:
            for xi in x:
                out.append(xi)
        return out

    def concat(cols):
        fn = None
        if type(cols[0]) == list:
            return concat_lists(cols)
        elif type(cols[0]) == np.ndarray:
            try:
                return np.concatenate(cols)
            except Exception:
                print(cols)
            
        else:
            raise Exception('Unknown column type, must be either a list or np.ndarray')
    
    t = {}   
    for f in fields:
        t[f] = concat([g[f] for g in gs])
    return t


def _sorted_index(xs, asc=True):
    if type(xs) == list:
        return [x[0] for x in sorted(enumerate(xs), key=lambda x: x[1], reverse=not asc)]
    elif type(xs) == np.ndarray:
        idx = xs.argsort()
        if not asc:
            return idx[::-1]
        else:
            return idx
    else:
        raise Exception(f"List type \"{type(xs)}\" not supported")


def arrange(t, fields, asc = None):
    """
    Immutable.
    """
    if type(fields) == str:
        fields = [fields]    
    
    if asc is None:
        asc = [True] * len(fields)

        
    if type(asc) == bool:
        asc = [asc]        

    assert(len(fields) == len(asc))
    

    f = fields[0]
    idx = _sorted_index(t[f], asc[0])
    t = rows(t, idx)
    if len(fields) > 1:
        t_grouped = group_by(t, (f,))
        out = ungroup([arrange(g, fields[1:], asc[1:]) for g in t_grouped])
        return out
    else:
        return t

def first(x):
    return x[0]

def last(x):
    return x[-1]

def sf(fn, attr):
    if type(attr) == str:
        attr = [attr]
        
    def _make_fn(g):
        cols = [g[a] for a in attr]
        return fn(*cols)
    
    return lambda g: _make_fn(g)

def summarize(gs, **kwargs):
    acc = []
    for g in gs:
        row = {}
        for f in kwargs.keys():
            row[f] = np.array([kwargs[f](g)])
        acc.append(row)
    return ungroup(acc)


def read_csv(fn):
    return from_df(pd.read_csv(fn))


def to_df(t):
    return pd.DataFrame.from_dict(t)


def from_df(df):
    ds = df.to_dict('list')
    for k in ds.keys():
        ds[k] = np.array(ds[k])
    return ds


def from_dicts(ds):
    fields = ds[0].keys()
    out = {f: [] for f in fields}
    for d in ds:
        for f in fields:
            out[f].append(d[f])
    for f in fields:
        out[f] = np.array(out[f])
    return out


def head(t, n = 5):
    n = min(nrows(t), n)
    s = slice_rows(t, 0, n)
    to_str(s)
    
def to_str(t):
    headers = list(t.keys())
    data = []
    for i in range(nrows(t)):
        row = []
        for h in headers:
            row.append(t[h][i])
        data.append(row)
    print(tabulate(data, headers=headers))    
    
def tail(t, n = 5):
    start_idx = max(nrows(t) - n - 1, 0)
    s = slice_rows(t, start_idx, nrows(t))
    to_str(s)