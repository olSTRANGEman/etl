import json
import pandas as pd
from pathlib import Path

from sdmetrics.reports.multi_table import QualityReport

# --------------------------------------------------
# 1. Загрузка metadata
# --------------------------------------------------
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --------------------------------------------------
# 2. Пути к данным
# --------------------------------------------------
REAL_DIR = Path("szri_data")
SYNTH_DIR = Path("synth_out")

# имена таблиц ДОЛЖНЫ совпадать с metadata["tables"].keys()
TABLE_NAMES = list(metadata["tables"].keys())

# --------------------------------------------------
# 3. Загрузка multitable данных
# --------------------------------------------------
def load_tables(folder: Path, prefix: str = ""):
    tables = {}
    for table in TABLE_NAMES:
        filename = f"{prefix}{table}.csv"
        path = folder / filename
        tables[table] = pd.read_csv(path)
    return tables

real_tables = load_tables(REAL_DIR, prefix="")
synth_tables = load_tables(SYNTH_DIR, prefix="synth_")

# --------------------------------------------------
# 4. Multi-table QualityReport
# --------------------------------------------------
report = QualityReport()

report.generate(
    real_data=real_tables,
    synthetic_data=synth_tables,
    metadata=metadata
)

# --------------------------------------------------
# 5. Результаты
# --------------------------------------------------
overall_score = report.get_score()
properties = report.get_properties()

print("\n=== MULTI-TABLE QUALITY REPORT ===")
print("Overall score:", overall_score)
print("\nProperties:")
print(properties)

# --------------------------------------------------
# 6. Сохранение summary
# --------------------------------------------------
summary = properties.copy()
summary.loc["OVERALL"] = ["Overall", overall_score]

summary.to_csv("multitable_summary.csv", index=False)

print("\nSaved to multitable_summary.csv")
