import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    keys = ["age", "zip3", "gender"]
    anon_counts = anon_df.groupby(keys, dropna=False).size().rename("anon_count")
    aux_counts = aux_df.groupby(keys, dropna=False).size().rename("aux_count")
    merged = anon_df.merge(aux_df, on=keys, how="inner")
    merged = merged.join(anon_counts, on=keys).join(aux_counts, on=keys)
    unique = merged[(merged["anon_count"] == 1) & (merged["aux_count"] == 1)]
    return unique[["anon_id", "name"]].rename(columns={"name": "matched_name"}).reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df) if len(anon_df) else 0.0
