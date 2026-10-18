# Data sources

Every dataset the system loads, where it comes from, and how to cite it. The same
information is stored in the `data_source` table, and every measurement row points to its source.

## Research data (paper)

Files from the paper's code and data repository:
<https://github.com/isabgmb/dengueknowledgePaper_pi>

> *Data-Driven Computational Intelligence Applied to Dengue Outbreak Forecasting: a case study at the
> scale of the city of Natal, RN-Brazil.*

**License: CC BY-NC-SA 4.0** — attribution required, no commercial use, derivatives under the same
license. This applies to anything built from these files, including the database and trained models.

| File (in `data/raw/`) | Part used | Loaded into |
|---|---|---|
| `Dados_Modelagem.xlsx` | sheet *Casos de Dengue*: weekly notified dengue cases per neighborhood, epidemiological weeks 2016-01 to 2019-52 | `dengue_case` |
| `Rearranged_Data.mat` | `EggIndice_Agregated`: weekly OPI (`[:,:,1]` in MATLAB) and EDI (`[:,:,2]`) per neighborhood; `ReducedTable`: row order, municipal codes, health districts | `ovitrap_reading`; `data/reference/neighborhoods.csv` |
| `DADOS GERAIS_01.xlsx` | sheet *Renda por Bairro*: population, total and per-capita monthly income, 2010 Census | `socioeconomic` |
| `Models_1DLSTM_2022_02_04_EvalResults.mat` | the paper's LSTM evaluation results | served by `/resultados` |

Not loaded yet (available in the same files): hospitalizations (`InternamentosDengueNatal.xlsx`),
other arboviruses for 2018–2019 and sanitation indicators (`DADOS GERAIS_01.xlsx`).

### Notes on the research data

- Weeks are Brazilian epidemiological weeks (Sunday–Saturday); see `src/core/epiweek.py`.
- `t_Date` inside `Rearranged_Data.mat` is not used: it is 208 evenly spaced points between
  2016-01-01 and 2019-12-31 (about 7.05 days apart), not real week dates.
- The egg-index rows are in a different order from the dengue sheet; `egg_row` in
  `data/reference/neighborhoods.csv` maps each neighborhood to its row.

## Neighborhood coordinates

Centroids of IBGE's 2010 Census neighborhood boundaries (*malha de bairros*), republished by
Ipea's geobr project.

- URL: <https://github.com/ipea/geobr_prep_data/releases/download/v2.0.0/neighborhoods_2010.parquet>
- Retrieved 2026-10-16; SHA-256 and method in `data/reference/README.md`
- Cite as: IBGE — Instituto Brasileiro de Geografia e Estatística, Censo Demográfico 2010
- Stored in `neighborhood.latitude/longitude`, with `coord_source_id` and `coord_retrieved_at`
