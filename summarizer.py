import torch
from summarizer import Summarizer
from summarizer.text_processors.sentence_handler import SentenceHandler
from transformers import AutoConfig, AutoTokenizer, AutoModel
from underthesea import sent_tokenize
from typing import List


class VietnameseSentenceHandler(SentenceHandler):
    def __init__(self, min_length: int = 20, max_length: int = 500):
        self.min_length = min_length
        self.max_length = max_length

    def process(self, body: str, **__) -> List[str]:
        sents = sent_tokenize(body)
        return [
            s.strip() for s in sents
            if self.min_length <= len(s) <= self.max_length
        ]


def build_summarizer(model_name: str,
                     hidden: int = -2,
                     reduce: str = "mean",
                     device: torch.device = None) -> Summarizer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AutoConfig.from_pretrained(model_name)
    cfg.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=cfg).to(device)

    return Summarizer(
        custom_model=model,
        custom_tokenizer=tokenizer,
        sentence_handler=VietnameseSentenceHandler(),
        hidden=hidden,
        reduce_option=reduce
    )