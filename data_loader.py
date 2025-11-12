import pandas as pd
from underthesea import sent_tokenize         
import config
from vnnlpcore import mvn_word_tokenize


def load_and_preprocess_data(data_path: str):
    df = pd.read_csv(data_path)
    clusters = []

    for _, row in df.iterrows():
        cluster_id = row['cluster']
        raw_text   = row['text']
        sentences = sent_tokenize(raw_text)
        human_sums = [
            mvn_word_tokenize(value)                    
            for key, value in row.items()
            if key.startswith("human_summary_")
        ]

        clusters.append({
            'cluster_id': cluster_id,
            'sentences'  : sentences,
            'human_sums' : human_sums
        })

    return clusters
