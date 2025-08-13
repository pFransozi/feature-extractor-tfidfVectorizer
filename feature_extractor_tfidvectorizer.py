from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def read_event_col(csv_path: Path) -> list[str]:
    try:
        df = pd.read_csv(csv_path, dtype='str')
    except:
        return []
    
    cols = [c for c in df.columns if c.lower() == 'eventname']

    if not cols:
        return []
    
    col = cols[0]

    data = df[col].dropna().astype(str)
    data = data[data.str.strip() != ""]
    return data.tolist()
    



def gather_csvs_from_dir(root:Path, label_value:int):
    
    csv_events, hashes, labels, paths = [],[],[],[]

    for csv_path in sorted(root.rglob("*.csv")):
        events = read_event_col(csv_path=csv_path)
        if not events:
            continue

        events = "".join(events)
        csv_events.append(events)

        file_hash = csv_path.stem
        hashes.append(file_hash)
        labels.append(label_value)

    return csv_events, hashes, labels


def build_tfidf_dataset(goodware_dir: str, malware_dir:str
                        , save_path: str | None = None
                        , min_df: int | float =1, max_df:int | float = 1.0):
    
    goodware_root = Path(goodware_dir)
    malware_root = Path(malware_dir)

    g_csvs_events, g_hashes, g_labels = gather_csvs_from_dir(goodware_root, 0)
    m_csvs_events, m_hashes, m_labels = gather_csvs_from_dir(malware_root, 1)

    csvs_events = g_csvs_events + m_csvs_events
    hashes = g_hashes + m_hashes
    labels = g_labels + m_labels

    vectorizer = TfidfVectorizer(
        lowercase=False # lowercase=False, “Write” ≠ “write” (são features distintas).
        ,tokenizer=str.split # divide por qualquer whitespace
        ,preprocessor=None #usar o pré-processador padrão do scikit-learn
        ,token_pattern=None
        ,min_df=min_df # mínimo de occurs nos docs
        ,max_df=max_df # max de occurs nos docs
    )

    # TF (Term Frequency) = quão frequente o evento aparece nesse CSV específico.
    # IDF (Inverse Document Frequency) = quão raro esse evento é considerando todos os CSVs lidos (o “global”).
    # O valor na célula é TF × IDF
    X = vectorizer.fit_transform(csvs_events)

    feature_names = [f"{t}" for t in vectorizer.get_feature_names_out()]
    feats_df = pd.DataFrame.sparse.from_spmatrix(X, columns=feature_names)
    meta_df  = pd.DataFrame({"hash": hashes, "label": labels})
    final_df = pd.concat([meta_df, feats_df], axis=1)

    final_df.to_csv(save_path, index=False)


def main():
    goodware_dir = "./apks/group1"
    malware_dir = "./apks/group2"

    build_tfidf_dataset(goodware_dir, malware_dir, './vectorizer.csv', min_df=1, max_df=1.0)

if __name__ == "__main__":
    main()

