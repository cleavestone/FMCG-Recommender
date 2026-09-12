"""Item content feature matrix built from the product hierarchy.

Extracted from notebooks/03_content_based_model.ipynb (section 1).

Known limitation, confirmed empirically in that notebook: pack size is not
encoded, so ~83% of candidate items in the Dunnhumby data share their exact
feature vector with at least one other item (different pack sizes of the
same brand/commodity are indistinguishable). This is fine for
ContentBasedRecommender used as one signal inside a hybrid (see models.hybrid)
but means it should never be trusted as a standalone ranker without a
tiebreak from a behavioural signal.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_item_feature_matrix(product_df: pd.DataFrame, candidate_items: set):
    """Returns (sparse TF-IDF matrix, item_ids in row order, fitted vectorizer).

    Rows are L2-normalised by TfidfVectorizer, so cosine similarity between
    two items is just their dot product.
    """
    df = product_df[product_df["product_id"].isin(candidate_items)].reset_index(drop=True)

    def tok(series):
        return series.astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)

    text = (
        "DEPT__" + tok(df["department"]) + " "
        + "COMM__" + tok(df["commodity_desc"]) + " "
        + "SUBCOMM__" + tok(df["sub_commodity_desc"]) + " "
        + "BRAND__" + tok(df["brand"]) + " "
        + "MANUF__" + df["manufacturer"].astype(str)
    )

    vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
    matrix = vectorizer.fit_transform(text)
    item_ids = df["product_id"].tolist()
    return matrix, item_ids, vectorizer
