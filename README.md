# datatlon

## Philosophy

* Less is exponentially more.
* Immutable where matters.

## API

### Data structures

#### `Table : Map[String -> numpy.ndarray]`

#### `Groups: List[Table]`



### Data loading

#### `read_csv(fn: String) -> Table`

### `from_df(df: pandas.DataFrame) -> Table`

### `to_df(t: Table) -> pandas.DataFrame`

### `from_dicts(ds: List[Map]) -> Table`



### Inspection

#### `nrows(t: Table) -> Integer`

#### `head(t: Table, n: Integer = 5)`

#### `tail(t: Table, n: Integer = 5)`




### Select


#### `select(t: Table, fields: List[String])`

```
df = select(df, ['Capital', 'Population'])
```

Helpers:
```
df = select(df, fields(df, 0,10))
```

#### `rows(t: Table, idx: List[Bool]) -> Table`

```
df2 = rows(df, df['Age'] > 30)
```

Helpers:
```
df_slice = rows(df, range_mask(df, 100, 200))
```

#### `row(t: Table, idx: Integer) -> Map[String -> Object]`


### Aggregate

#### `group_by(t: Table, fields: List[String]) -> Groups`

#### `ungroup(gs: Groups) -> Table`

### `summarize(gs: Groups, **kwargs) -> Table`

```
```

```

```
