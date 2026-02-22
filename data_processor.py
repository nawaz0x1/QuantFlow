import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple
from pathlib import Path
import pickle


def create_directory(dir_path: Path) -> None:
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def generate_datetime_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:

    df = df.copy()
    existing_cols = df.columns
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    if df[date_column].isnull().any():
        df = df.dropna(subset=[date_column])

    dt_ref = df[date_column].dt

    df["year"] = dt_ref.year
    df["month"] = dt_ref.month
    df["day"] = dt_ref.day
    df["hour"] = dt_ref.hour
    df["minute"] = dt_ref.minute
    df["second"] = dt_ref.second
    df["day_of_week"] = dt_ref.dayofweek
    df["day_of_year"] = dt_ref.dayofyear
    df["week_of_year"] = dt_ref.isocalendar().week.astype(int)
    df["quarter"] = dt_ref.quarter

    cyclical_features = {
        "month": 12,
        "day_of_week": 7,
        "day_of_year": 366,
        "hour": 24,
        "minute": 60,
        "second": 60,
    }

    for feat, max_val in cyclical_features.items():
        df[f"{feat}_sin"] = np.sin(2 * np.pi * df[feat] / max_val)
        df[f"{feat}_cos"] = np.cos(2 * np.pi * df[feat] / max_val)

    current_columns = df.columns
    generated_features = [col for col in current_columns if col not in existing_cols]

    return df, generated_features


def prepare_training_data(
    df: pd.DataFrame, features: List[str], main_series: str
) -> pd.DataFrame:
    df.rename(columns={main_series: "target"}, inplace=True)
    df = df[features + ["target"]]
    return df


def prepare_dataset(
    df: pd.DataFrame,
    date_column: str,
    main_series: str,
    save_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    df, generated_features = generate_datetime_features(df, date_column)
    df = prepare_training_data(df, generated_features, main_series)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    if save_dir:
        df.to_csv(save_dir / "raw.csv", index=False)
        scaled_df.to_csv(save_dir / "scaled.csv", index=False)
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    return df, scaled_df, scaler


def plot_series(
    df: pd.DataFrame,
    date_column: str,
    series_column: str,
    title: str,
    column_label: str,
    save_dir: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    df[date_column] = pd.to_datetime(df[date_column])
    plt.figure(figsize=(15, 7))
    plt.plot(df[date_column], df[series_column], label=column_label)
    plt.title(title, fontsize=26)
    plt.xlabel("Date", fontsize=24)
    plt.ylabel(column_label, fontsize=24)
    plt.grid(True)
    # plt.legend(fontsize=22)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    if save_dir:
        plt.savefig(
            save_dir / f"{series_column}_plot.svg", format="svg", bbox_inches="tight"
        )
    if show_plot:
        plt.show()


if __name__ == "__main__":

    file_path = Path("data.csv")
    save_dir = Path("processed_data")

    date_column = "date"
    main_series = "target"

    df = pd.read_csv(file_path)
    create_directory(save_dir)
    raw_df, scaled_df, scaler = prepare_dataset(
        df=df,
        date_column=date_column,
        main_series=main_series,
        save_dir=save_dir,
    )
