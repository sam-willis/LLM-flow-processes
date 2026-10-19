# Observational data

NOAA Global Monitoring Laboratory source files. The `mauna_loa*` presets in `neural_diffusion_processes/presets.py` are generated from these.

All five are Mauna Loa Observatory, Hawaii (site code `MLO`).

| file | series | source |
|---|---|---|
| `co2_mm_mlo.txt` | CO₂ monthly mean, ppm | [gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt](https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt) |
| `co2_weekly_mlo.txt` | CO₂ weekly mean, ppm | [gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.txt](https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.txt) |
| `ch4_mlo_surface-flask_1_ccgg_month.txt` | CH₄ monthly flask mean, ppb | [gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/txt/](https://gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/txt/) |
| `n2o_mlo_surface-flask_1_ccgg_month.txt` | N₂O monthly flask mean, ppb | [gml.noaa.gov/aftp/data/trace_gases/n2o/flask/surface/txt/](https://gml.noaa.gov/aftp/data/trace_gases/n2o/flask/surface/txt/) |
| `sf6_mlo_surface-flask_1_ccgg_month.txt` | SF₆ monthly flask mean, ppt | [gml.noaa.gov/aftp/data/trace_gases/sf6/flask/surface/txt/](https://gml.noaa.gov/aftp/data/trace_gases/sf6/flask/surface/txt/) |

## Formats

The two trends files are whitespace-separated with a `#` header.

```
co2_mm_mlo.txt       year  month  decimal_date  average  deseasonalized  ...
co2_weekly_mlo.txt   year  month  day  decimal_date  ppm  #days  ...
```

The presets use `average` and the weekly `ppm`. Missing values are `-999.99`.

The three flask files share one layout, four fields, with a 54-line `#` header:

```
site  year  month  value        e.g.  MLO 1983  5  1639.47
```

## Windows used by the presets

Each preset takes a contiguous slice, not the whole record:

| preset family | window | n |
|---|---|---|
| `mauna_loa_co2_weekly*` | weeks beginning 2019-01-06 … 2023-11-26 | 256 |
| `mauna_loa_co2_interp_monthly*` (monthly CO₂ on a weekly grid) | 2019-01 … 2024-04 | 64 |
| `mauna_loa_ch4_2009*` | 2009-03 … 2014-02 | 60 |
| `mauna_loa_n2o_2020*` | 2020-01 … 2024-12 | 60 |
| `mauna_loa_sf6_2020*` | 2020-01 … 2024-12 | 60 |

## A note on revisions

These files are **not static upstream**. NOAA reprocesses flask records and
revises recent provisional values, so re-downloading will not always reproduce
the committed numbers.