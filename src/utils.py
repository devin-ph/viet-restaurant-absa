import os
import pandas as pd

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path(*paths):
    root = get_project_root()
    return os.path.join(root, *paths)


def load_csv(path):
    return pd.read_csv(path)


def save_csv(df, path, index=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved to {path}")


from sklearn.model_selection import train_test_split

def split_train_val_test(df, text_col="review", test_size=0.2, val_size=0.1, random_state=42):

    unique_reviews = df[text_col].unique().tolist()

    train_rev, temp_rev = train_test_split(
        unique_reviews, test_size=test_size + val_size, random_state=random_state
    )
    val_rev, test_rev = train_test_split(
        temp_rev, test_size=test_size / (test_size + val_size), random_state=random_state
    )

    train_df = df[df[text_col].isin(train_rev)].reset_index(drop=True)
    val_df = df[df[text_col].isin(val_rev)].reset_index(drop=True)
    test_df = df[df[text_col].isin(test_rev)].reset_index(drop=True)

    return train_df, val_df, test_df


def get_aspect_columns(df, text_col="review"):
    return [col for col in df.columns if col != text_col]


def encode_labels(df, aspect_cols):
    return df[aspect_cols].values