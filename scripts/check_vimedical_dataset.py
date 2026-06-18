import pandas as pd


def main():
    file_path = "data/raw/ViMedical_Disease.csv"
    df = pd.read_csv(file_path)

    print("=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== NULL COUNTS ===")
    print(df.isnull().sum())

    print("\n=== NUMBER OF UNIQUE DISEASES ===")
    print(df["Disease"].nunique())

    print("\n=== TOP 20 DISEASE COUNTS ===")
    print(df["Disease"].value_counts().head(20))

    print("\n=== BOTTOM 20 DISEASE COUNTS ===")
    print(df["Disease"].value_counts().tail(20))


if __name__ == "__main__":
    main()