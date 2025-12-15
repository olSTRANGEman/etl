"""
chinook_dp_synth.py

Пример генерации дифференциально-приватной (если доступно) синтетики
для Chinook relational database с использованием SDV MultiTable.

Запуск:
    python chinook_dp_synth.py --data-dir /path/to/chinook_csvs --out-dir ./synth_out --epsilon 1.0 --num-samples-mult 1.0

CSV-файлы ожидаются с именами:
 Artist.csv, Album.csv, Track.csv, Genre.csv, MediaType.csv,
 Invoice.csv, InvoiceLine.csv, Customer.csv, Employee.csv,
 Playlist.csv, PlaylistTrack.csv
"""

import os
import argparse
import textwrap
import pandas as pd
import numpy as np
import sqlite3

def load_chinook_tables_from_sqlite(sqlite_file):
    """
    Загружает все таблицы Chinook из SQLite файла в словарь DataFrame.
    Возвращает dict: {table_name: pd.DataFrame}.
    """
    tables = {}
    conn = sqlite3.connect(sqlite_file)

    # Получаем список таблиц из схемы SQLite
    table_names = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()]

    for name in table_names:
        df = pd.read_sql_query(f"SELECT * FROM {name};", conn)
        df.to_csv(os.path.join("./szri_data", f"{name}_data.csv"), index=False)
        tables[name] = df
        print(f"Loaded table {name} ({df.shape})")

    conn.close()
    return tables

def build_metadata_for_chinook(tables):
    """
    Построение метаданных SDV для схемы Chinook (устойчиво к разным версиям SDV).
    Аргумент `tables` — dict: {table_name: pandas.DataFrame}
    Возвращает объект metadata (SDV Metadata / MultiTableMetadata).
    """
    # предполагаемые PK (если хотим явно задать)
    primary_keys = {
        "Artist": "ArtistId",
        "Album": "AlbumId",
        "Track": "TrackId",
        "Genre": "GenreId",
        "MediaType": "MediaTypeId",
        "Invoice": "InvoiceId",
        "InvoiceLine": "InvoiceLineId",
        "Customer": "CustomerId",
        "Employee": "EmployeeId",
        "Playlist": "PlaylistId",
        "PlaylistTrack": None,  # composite PK (PlaylistId, TrackId)
    }

    # 1) Попробуем импортировать современный Metadata API
    metadata = None
    detect_callable = None
    try:
        # SDV >= 1.0
        from sdv.metadata import Metadata
        metadata_cls = Metadata
        detect_callable = getattr(Metadata, "detect_from_dataframes", None)
    except Exception:
        metadata_cls = None
        detect_callable = None

    # 2) fallback: старые пути (MultiTableMetadata)
    if detect_callable is None:
        try:
            from sdv.metadata.multi_table import MultiTableMetadata
            metadata_cls = MultiTableMetadata
            detect_callable = getattr(MultiTableMetadata, "detect_from_dataframes", None)
        except Exception:
            detect_callable = None

    # 3) Если есть detect_from_dataframes — используем его (самый простой и надёжный путь)
    if detect_callable is not None:
        try:
            # detect keys/relationships automatically
            print("Using detect_from_dataframes on tables:", list(tables.keys()))
            metadata = detect_callable(
                data=tables,
                infer_sdtypes=True,
                infer_keys="primary_and_foreign"  # detect pk + fk
            )
            print("Auto-detected metadata OK.")
        except TypeError:
            # некоторые версии могут иметь сигнатуру без именованных аргументов
            try:
                metadata = detect_callable(tables)
                print("Auto-detected metadata OK (fallback signature).")
            except Exception as e:
                print("detect_from_dataframes failed:", e)
                metadata = None
        except Exception as e:
            print("detect_from_dataframes raised:", e)
            metadata = None
    else:
        # нет detect_from_dataframes — пробуем ручное создание через MultiTableMetadata API
        try:
            from sdv.metadata.multi_table import MultiTableMetadata
            metadata = MultiTableMetadata()
            # старые версии: нужно вызвать add_table + detect columns manually
            for tname, df in tables.items():
                try:
                    # try modern signature: table_name, data=..., primary_key=...
                    metadata.add_table(tname, data=df, primary_key=primary_keys.get(tname))
                    print(f"Added table {tname} via add_table(name,data,primary_key).")
                except TypeError:
                    try:
                        # older signature: add_table(table_name) then detect_from_dataframe
                        metadata.add_table(tname)
                        if hasattr(metadata, "detect_table_from_dataframe"):
                            metadata.detect_table_from_dataframe(tname, df)
                        print(f"Added table {tname} via fallback add_table + detect_table_from_dataframe.")
                    except Exception as e:
                        print(f"Failed to add table metadata for {tname}: {e}")
        except Exception as e:
            raise RuntimeError("Невозможно создать metadata: нет detect_from_dataframes и MultiTableMetadata недоступен.") from e

    if metadata is None:
        raise RuntimeError("Не удалось автоматически создать metadata. Проверь версию SDV (рекомендуется sdv>=1.0).")

    # 4) Установим (если возможно) желаемые primary keys из primary_keys (если auto-detect не их выбрал)
    for tbl, pk in primary_keys.items():
        if pk is None or tbl not in tables:
            continue
        try:
            
            # modern API: set_primary_key(table_name=..., column_name=...)
            if hasattr(metadata, "set_primary_key"):
                # some versions expect (column_name, table_name) or kwargs
                try:
                    metadata.set_primary_key(table_name=tbl, column_name=pk)
                    print(f"Set primary key for {tbl} -> {pk} (table_name, column_name).")
                except TypeError:
                    try:
                        metadata.set_primary_key(tbl, pk)
                        print(f"Set primary key for {tbl} -> {pk} (fallback positional).")
                    except Exception as e:
                        print(f"Could not set primary key for {tbl}: {e}")
            else:
                # older API: maybe metadata[table].set_primary_key or SingleTableMetadata.set_primary_key
                if hasattr(metadata, "tables") and tbl in getattr(metadata, "tables", {}):
                    try:
                        stm = metadata.tables[tbl]
                        if hasattr(stm, "set_primary_key"):
                            stm.set_primary_key(pk)
                            print(f"Set primary key for {tbl} via SingleTableMetadata.set_primary_key -> {pk}")
                    except Exception as e:
                        print(f"Warning: couldn't set pk on internal table {tbl}: {e}")
        except Exception as e:
            print(f"Warning: error while setting primary key for {tbl}: {e}")

    # 5) Попробуем добавить недостающие отношения явно (без краха)
    def safe_add_relation(parent, child, fk):
        if parent not in tables or child not in tables:
            print(f"Skipping relation {parent}->{child} because one of tables missing.")
            return
        # Try modern signature with names
        try:
            metadata.add_relationship(
                parent_table_name=parent,
                child_table_name=child,
                parent_primary_key=primary_keys.get(parent),
                child_foreign_key=fk
            )
            print(f"Added relationship: {parent} -> {child} ({fk}) [modern api].")
            return
        except TypeError:
            pass
        except Exception as e:
            # if already detected, SDV may raise — ignore
            print(f"Note: add_relationship (modern) raised: {e}")

        # fallback older signature attempts
        try:
            metadata.add_relationship(parent=parent, child=child, foreign_key=fk)
            print(f"Added relationship: {parent} -> {child} ({fk}) [fallback].")
            return
        except Exception as e:
            print(f"Warning: could not add relationship {parent}->{child}({fk}): {e}")

    # list of standard relationships in Chinook
    rels_to_add = [
        ('Artist', 'Album', 'ArtistId'),
        ('Album', 'Track', 'AlbumId'),
        ('Genre', 'Track', 'GenreId'),
        ('MediaType', 'Track', 'MediaTypeId'),
        ('Track', 'InvoiceLine', 'TrackId'),
        ('Invoice', 'InvoiceLine', 'InvoiceId'),
        ('Playlist', 'PlaylistTrack', 'PlaylistId'),
        ('Track', 'PlaylistTrack', 'TrackId'),
        ('Customer', 'Invoice', 'CustomerId'),
    ]
    for parent, child, fk in rels_to_add:
        safe_add_relation(parent, child, fk)

    # special Employee -> Customer relation if data exists
    if 'Customer' in tables and 'Employee' in tables and 'SupportRepId' in tables['Customer'].columns:
        safe_add_relation('Employee', 'Customer', 'SupportRepId')

    print("Final metadata tables:", getattr(metadata, "get_table_names", lambda: list(getattr(metadata,'tables',{}).keys()))())
    import json

    metadata.save_to_json("metadata.json")

    return metadata

def build_synthesizer(metadata, epsilon=None, default_single_table_synthesizer=None, verbose=True):
    """
    Пытаемся инициализировать multi-table synthesizer.
    Попробуем несколько имен классов SDV (HMA1, HMASynthesizer, HMA).
    """
    # import single-table synthesizer (prefer DP variant if requested)
    single_table_cls = None
    if default_single_table_synthesizer is not None:
        single_table_cls = default_single_table_synthesizer
    else:
        # Попробуем DPCTGANSynthesizer -> CTGANSynthesizer
        try:
            from snsynth import Synthesizer
            from sdv.single_table import CTGANSynthesizer as CTG

            # single_table_cls = CTG
            single_table_cls = Synthesizer.create("dpctgan", epsilon=3.0, verbose=True)

            print("Using DPCTGANSynthesizer for single-table synthesis.")
        except Exception:
            try:
                print("die")
                exit()
                
                from sdv.single_table import CTGANSynthesizer as CTG
                single_table_cls = CTG
                print("DPCTGANSynthesizer not available; falling back to CTGANSynthesizer (no DP).")
            except Exception:
                single_table_cls = None
                print("No CTGAN class available in sdv.single_table — check your SDV installation.")

    # Try multi-table synthesizer classes
    mt_synth = None
    tried = []
    for cls_name in ["HMASynthesizer", "HMA1", "HMA"] :
        try:
            module = __import__("sdv.multi_table", fromlist=[cls_name])
            SynthClass = getattr(module, cls_name)
            # signature may differ; try common kwargs
            try:
                mt_synth = SynthClass(metadata=metadata, default_single_table_synthesizer=single_table_cls, verbose=verbose)
            except TypeError:
                try:
                    mt_synth = SynthClass(metadata=metadata)
                except Exception as e:
                    print(f"Could not init {cls_name} with metadata kw: {e}")
            print(f"Using multi-table synthesizer class: {cls_name}")
            break
        except Exception as e:
            tried.append(cls_name)
            # ignore and continue
    if mt_synth is None:
        raise RuntimeError(f"Could not find a compatible multi-table synthesizer in sdv.multi_table. Tried: {tried}. Ensure sdv>=1.0 is installed.")
    return mt_synth

def train_and_sample(synthesizer, tables, sample_mult=1.0, out_dir="./synth_out"):
    """
    Обучаем synthesizer на tables (dict name->df),
    затем сэмплируем и сохраняем результаты.
    sample_mult — множитель относительно исходной таблицы по числу строк.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("Fitting synthesizer (this may take time)...")
    synthesizer.fit(tables)
    print("Fit complete. Sampling...")

    # number_rows: по умолчанию попробуем сгенерировать такую же величину, умноженную на sample_mult
    desired = { name: max(1, int(df.shape[0] * sample_mult)) for name, df in tables.items() }

    synthetic = synthesizer.sample(scale=1.5)
    # synthetic может быть DataFrame (если num_rows single) или dict name->df
    # SDV обычно возвращает dict{table_name: df}
    if isinstance(synthetic, dict):
        for name, df in synthetic.items():
            path = os.path.join(out_dir, f"synth_{name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved synthetic {name} -> {path} ({df.shape})")
    else:
        # если synth возвращает DataFrame for single-table
        synthetic.to_csv(os.path.join(out_dir, "synth_data.csv"), index=False)
        print("Saved single-table synthetic to synth_data.csv")

def main(args):
    tables = load_chinook_tables_from_sqlite(args.data_dir)
    if len(tables) == 0:
        raise RuntimeError("No tables loaded. Place Chinook CSVs into the data-dir.")
    metadata = build_metadata_for_chinook(tables)

    # choose DP epsilon param if DP single-table available
    dp_epsilon = args.epsilon
    # build synthesizer (tries DP variant for single-table if available)
    synthesizer = build_synthesizer(metadata, epsilon=dp_epsilon)

    # train and sample
    train_and_sample(synthesizer, tables, sample_mult=args.sample_mult, out_dir=args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent("""
Generate relational synthetic data (Chinook) using SDV MultiTable.
The script attempts to use DP single-table synthesizer (DPCTGAN) if available,
otherwise falls back to CTGAN.

Prepare a folder with Chinook CSVs and pass it via --data-dir.
"""))
    p.add_argument("--data-dir", type=str, default="./chinook_csvs", help="Path to directory with Chinook CSV files")
    p.add_argument("--out-dir", type=str, default="./synth_out", help="Where to save synthetic CSVs")
    p.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for DP single-table synthesizers (if supported)")
    p.add_argument("--sample-mult", type=float, default=1.0, help="Multiply number of rows to sample per table (1.0 = same as original)")
    args = p.parse_args()
    main(args)
