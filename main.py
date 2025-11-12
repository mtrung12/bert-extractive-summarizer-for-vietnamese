# main.py
import argparse
import pandas as pd
from pathlib import Path
import config
from data_loader import load_and_preprocess_data
from evaluate import evaluate_rouge
from summarizer import build_summarizer
import torch
from vnnlpcore import mvn_word_tokenize
import os


def count_syllables(text: str) -> int:
    return len(mvn_word_tokenize(text))


def generate_summary(summarizer, text: str, ratio: float = 0.2) -> str:
    sentences = summarizer(text, ratio=ratio)
    return " ".join(sentences)


def main():
    parser = argparse.ArgumentParser(description="Vietnamese ETS with bert-extractive-summarizer")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV (text, human_summary_1, human_summary_2, cluster)")
    parser.add_argument("--model", type=str, default="phobert-base",
                        choices=list(config.MODEL_MAP.keys()),
                        help="Model key from config.MODEL_MAP")
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="Summary length ratio (input_compression_rate)")
    args = parser.parse_args()


    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    model_key = args.model
    hf_model_name = config.MODEL_MAP[model_key]
    model_name_safe = model_key.replace("-", "_")

    perf_file = result_dir / f"{model_name_safe}_{args.ratio}.txt"
    model_csv = result_dir / f"{model_name_safe}.csv"  # e.g., phobert_base.csv

    clusters = load_and_preprocess_data(args.data)
    print(f"Loaded {len(clusters)} clusters")


    summarizer = build_summarizer(hf_model_name,
                                  hidden=-2,
                                  reduce="mean",
                                  device=config.DEVICE)
    print(f"Summarizer ready – model: {hf_model_name} on {config.DEVICE}")
    perf_records = []
    summary_records = []

    for i, cluster in enumerate(clusters, 1):
        cid = cluster["cluster_id"]
        raw_text = " ".join(cluster["sentences"])
        human_sums = cluster["human_sums"]

        # Generate
        gen_sum = generate_summary(summarizer, raw_text, ratio=args.ratio)

        # ROUGE
        r1, r2, rl = evaluate_rouge(gen_sum, human_sums)

        # Syllables & compression
        input_syl = count_syllables(raw_text)
        output_syl = count_syllables(gen_sum)
        actual_comp = output_syl / input_syl if input_syl > 0 else 0.0
        input_comp = args.ratio  # This is the "input compression rate"

        # Store
        perf_records.append({
            "cluster_id": cid,
            "rouge1_f": r1,
            "rouge2_f": r2,
            "rougeL_f": rl
        })

        summary_records.append({
            "cluster_id": cid,
            "input_text": raw_text,
            "summary": gen_sum,
            "input_compression_rate": input_comp,
            "actual_compression_rate": round(actual_comp, 4),
            "input_syllable": input_syl,
            "output_syllable": output_syl,
            "rouge1_f": r1,
            "rouge2_f": r2,
            "rougeL_f": rl
        })

        print(f"[{i}/{len(clusters)}] {cid} → R1:{r1:.4f} | Comp(I/O): {input_comp:.2f}/{actual_comp:.3f}")

    perf_df = pd.DataFrame(perf_records)
    avg = perf_df[["rouge1_f", "rouge2_f", "rougeL_f"]].mean()

    with open(perf_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_key} ({hf_model_name})\n")
        f.write(f"Input Compression Rate (ratio): {args.ratio}\n")
        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== Per-cluster ROUGE (max vs human) ===\n")
        f.write(perf_df.to_string(index=False))
        f.write("\n\n=== AVERAGE SCORES ===\n")
        f.write(avg.round(4).to_string())
    print(f"\nPerformance log → {perf_file}")

    new_df = pd.DataFrame(summary_records)
    
    if model_csv.exists():
        # Append mode
        existing_df = pd.read_csv(model_csv)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Appended {len(new_df)} rows to existing {model_csv}")
    else:
        combined_df = new_df
        print(f"Created new {model_csv}")

    combined_df.to_csv(model_csv, index=False, encoding="utf-8")
    print(f"Model results saved → {model_csv}")

    print("\n" + "="*60)
    print(f"FINAL AVERAGE ROUGE (Model: {model_key}, Ratio: {args.ratio})")
    print(avg.round(4).to_string())
    print("="*60)


if __name__ == "__main__":
    main()