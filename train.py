import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel




def load_data(file):
    df = pd.read_csv(file)

    # Преобразуем все к числовому типу
    df = df.apply(pd.to_numeric, errors='coerce')

    # Заполнение NaN средним значением по столбцу
    imputer = SimpleImputer(strategy="mean")
    df.iloc[:, :] = imputer.fit_transform(df)

    # Последний столбец — таргет
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model




def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics


def compute_mmd(X, Y, gamma=1.0, max_samples=3000):
    """
    MMD with RBF kernel using random subsampling
    (safe for large datasets)
    """

    # Subsample if necessary
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]

    if len(Y) > max_samples:
        idx = np.random.choice(len(Y), max_samples, replace=False)
        Y = Y[idx]

    # Compute kernels
    KXX = rbf_kernel(X, X, gamma=gamma)
    KYY = rbf_kernel(Y, Y, gamma=gamma)
    KXY = rbf_kernel(X, Y, gamma=gamma)

    return KXX.mean() + KYY.mean() - 2 * KXY.mean()

def compare_datasets(path1, path2, output="datasets_comparison.txt"):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    print("Comparing datasets...")
    # === MMD (only numeric) ===
    X1 = df1.select_dtypes(include=[np.number]).values
    X2 = df2.select_dtypes(include=[np.number]).values
    mmd = float(compute_mmd(X1, X2))

    # === SAVE REPORT ===
    with open(output, "w") as f:


        f.write("### MMD (overall) ###\n")
        f.write(f"{mmd}\n\n")

    print(f"✓ Comparison saved to {output}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="path to TRAIN CSV")
    parser.add_argument("--test_file", required=False, help="path to TEST CSV (optional)")
    parser.add_argument("--test_size", default=0.2, type=float)
    args = parser.parse_args()


    print("\n📌 Loading TRAIN dataset...")


    # Загрузка train
    X_train_raw, y_train = load_data(args.train_file)
    print(np.unique(y_train, return_counts=True))


    if args.test_file:
        X_test_raw, y_test = load_data(args.test_file)

        # Приведение колонок к одному порядку
        missing_cols = set(X_train_raw.columns) - set(X_test_raw.columns)
        for col in missing_cols:
            X_test_raw[col] = 0.0
        X_test_raw = X_test_raw[X_train_raw.columns]

    # 🔹 Нормализация
    scaler = MinMaxScaler()
    X_train_raw = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test_raw = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)



    print("🔧 Scaling...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)




    print("\n🚀 Training model...")
    model = train_model(X_train, y_train)

    print("\n📈 Evaluating...")
    print(np.unique(y_train, return_counts=True))
    metrics = evaluate(model, X_test, y_test)




    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: \n {v}")

    print("\n✓ Done")


if __name__ == "__main__":
    main()
