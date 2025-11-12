from vncorenlp import VnCoreNLP
import config

print("Initializing VnCoreNLP") 

mvncorenlp = VnCoreNLP(
    config.VNCORENLP_PATH,
    annotators="wseg",
    max_heap_size='-Xmx2g'
)

def mvn_word_tokenize(text: str) -> str:
    tokenized = mvncorenlp.tokenize(text)
    flat = [token for sent in tokenized for token in sent]
    return " ".join(flat)