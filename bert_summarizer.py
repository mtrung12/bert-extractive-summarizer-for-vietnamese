# summarizer.py
import torch
import numpy as np
from summarizer import Summarizer
from summarizer.text_processors.sentence_handler import SentenceHandler
from transformers import AutoConfig, AutoTokenizer, AutoModel
from underthesea import sent_tokenize
from typing import List
from vnnlpcore import mvn_word_tokenize

class VietnameseSentenceHandler(SentenceHandler):
    def __init__(self, min_length: int = 5, max_length: int = 1000):
        self.min_length = min_length
        self.max_length = max_length

    def process(self, body: str, min_length: int = None, max_length: int = None, **__) -> List[str]:
        min_length = min_length if min_length is not None else self.min_length
        max_length = max_length if max_length is not None else self.max_length
        sents = sent_tokenize(body)
        return [s.strip() for s in sents if min_length <= len(s) <= max_length]


def chunk_sentence(sentence: str, tokenizer, max_length=256, stride=128):
    segmented_sentence = mvn_word_tokenize(sentence)
    tokens = tokenizer.tokenize(segmented_sentence)
    if len(tokens) <= max_length - 2:
        return [segmented_sentence]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_length - 2
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        start += stride
        if end >= len(tokens):
            break
    return chunks


def get_sentence_embedding(model, tokenizer, sentence: str, device, hidden=-2):
    chunks = chunk_sentence(sentence, tokenizer, max_length=256, stride=128)
    chunk_embs = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[hidden][0, 1:-1, :]  # (seq_len-2, dim)
            emb = hidden.mean(dim=0).cpu().numpy()  # (dim,)
            chunk_embs.append(emb)
    return np.mean(chunk_embs, axis=0)


def build_summarizer(model_name: str = "vinai/phobert-base", hidden: int = -2, device=None) -> Summarizer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AutoConfig.from_pretrained(model_name)
    cfg.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=cfg).to(device)

    def custom_embedding_fn(sentences: List[str]) -> np.ndarray:
        return np.stack([get_sentence_embedding(model, tokenizer, s, device, hidden) for s in sentences])

    summarizer = Summarizer(
        custom_model=model,
        custom_tokenizer=tokenizer,
        sentence_handler=VietnameseSentenceHandler(min_length=5, max_length=1000),
        hidden=hidden,
        reduce_option="mean"
    )
    summarizer._get_embeddings = lambda sents: custom_embedding_fn(sents)
    return summarizer