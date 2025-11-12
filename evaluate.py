from rouge import Rouge
from vnnlpcore import mvn_word_tokenize

def evaluate_rouge(generated_sum, human_sums):
    rouge = Rouge()
    scores = []
    generated_sum_tokenized = mvn_word_tokenize(generated_sum)
    
    for human_sum in human_sums:
        score = rouge.get_scores(generated_sum_tokenized, human_sum)[0]
        scores.append((
            score['rouge-1']['f'],
            score['rouge-2']['f'],
            score['rouge-l']['f'] 
        ))
    
    max_r1 = max([s[0] for s in scores])
    max_r2 = max([s[1] for s in scores])
    max_rl = max([s[2] for s in scores])  
    
    return max_r1, max_r2, max_rl