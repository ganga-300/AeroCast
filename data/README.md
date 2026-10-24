# Data

The raw dataset is **not** included in this repository.

1. Download the faculty starter dataset from Kaggle: <https://www.kaggle.com/datasets/matinsajadi/flights>
2. Extract `flights.csv` to `data/raw/flights.csv`.
3. Verify the checksum matches the one recorded in `configs/config.yaml`:

```bash
shasum -a 256 data/raw/flights.csv
```

Expected: `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`

The loader refuses to run if the checksum differs. Raw files are never modified; derived data goes
to `data/interim/` or `data/processed/` (both git-ignored).
