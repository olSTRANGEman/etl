import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic





def main(dataset):
    



    for i in range(1,6):
        data = pd.read_csv(f'data/{i}.csv')
        data_df = data.copy()
        imputer = SimpleImputer(strategy="mean")  # или "median"
        # Разделение на train/test
        data = pd.DataFrame(imputer.fit_transform(data_df), columns=data_df.columns)


        # Нормализация — только на train делаем fit!
        scaler = MinMaxScaler()

        data = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
    )



        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        metadata.to_dict()
        print(metadata)

        model = TVAESynthesizer(
            metadata,
            epochs=500,
            verbose=True,
            cuda=True
        )

        model.fit(data)
        synth = model.sample(10000)
        np.savetxt(f"datab{i}.csv", synth, delimiter=",", fmt="%s")



        diagnostic = run_diagnostic(
            real_data=data,
            synthetic_data=synth,
            metadata=metadata
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help = "dataset path",
        type = str
    )
    args = parser.parse_args()
    main(args.dataset)