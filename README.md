# datatlon

## Philosophy

* Immutable where matters.
* Provide lego building blocks for table like data manipulation.

## API

### Data structures

#### `Table : Map[String -> numpy.ndarray]`

#### `Groups: List[Table]`



### Data loading

#### `read_csv(fn: String) -> Table`

#### `from_df(df: pandas.DataFrame) -> Table`

#### `to_df(t: Table) -> pandas.DataFrame`

#### `from_dicts(ds: List[Map]) -> Table`



### Inspection

#### `nrows(t: Table) -> Integer`

#### `head(t: Table, n: Integer = 5)`

#### `tail(t: Table, n: Integer = 5)`




### Select and Filter

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

Present only the Shooting Accuracy from England, Italy and Russia
```
select(
    rows(euro12, np.in1d(euro12['Team'], ['England', 'Italy', 'Russia'])),
    ['Team','Shooting Accuracy']
)
```

Helpers:
```
df_slice = rows(df, range_mask(df, 100, 200))
```

#### `row(t: Table, idx: Integer) -> Map[String -> Object]`


### Aggregate

#### `group_by(t: Table, fields: Union[String, List[String]]) -> Groups`

#### `ungroup(gs: Groups) -> Table`

#### `summarize(gs: Groups, **kwargs) -> Table`

```
gs = group_by(t, 'continent')
summarize(gs, 
  continent = sf(first, 'continent`),
  avg_served = sf(np.mean, 'beer_servings'),
  male_ratio = lambda g: np.sum(g['gender'] == 'M') / nrows(g)
)
```

Helpers:
* `sf(fn: VectorFn, attr: String) -> Object`, where `VectorFn = f(x: numpy.ndarray) -> Object`
* `first`
* `last`

#### `apply(gs: Groups, **kwargs) -> Groups`

Add additional column `ret` for every symbol group:

```
gs = group_by(t, 'symbol')
t = apply(gs, ret = af(pct_change, 'price'))
```

Helpers:
* `af(fn: VectorFn, attr: String) -> numpy.ndarray`, where `VectorFn = f(x: numpy.ndarray) -> numpy.ndarray`

### Order

#### `arrange(t: Table, fields: List[String], asc: List[Bool] = None) -> Table`

```
arrange(discipline, ['Red Cards', 'Yellow Cards'], [False, False])
```

### Combine

#### `join(t1 Table, t2: Table, t1_on: Union[String, List[String]], t2_on: Union[String, List[String]], t1_prefix=None, t2_prefix=None) -> Table`


### Mutate

#### `rename(t: Table, from_names: List[String], to_names: List[String])`

**Note**: mutable.

```
rename(t, ['CLVMEURSCAB1GQEA19','CP0000EZ19M086NEST'], ['GDP','CPI'])
```

#### `cf(fn(x: Object) -> Object) -> fn(x: numpy.ndarray) -> numpy.ndarray`

```
capitalize = lambda x: x.capitalize()
t['Mjob'] = cf(capitalize)(t['Mjob'])
```
