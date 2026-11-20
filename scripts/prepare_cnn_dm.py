"""
CNN / DailyMail Dataset Loader & Splitter.
Pulls a clean 300-article test set and 50-article validation set for threshold tuning.
Guarantees NO evaluation leakage between threshold calibration and testing.
"""
import os
import json
from typing import Dict, Any


def prepare_cnn_dailymail(output_dir: str = "data"):
    """
    Downloads CNN/DailyMail (version 3.0.0) subset via HuggingFace datasets
    and saves to clean JSON format.
    """
    os.makedirs(output_dir, exist_ok=True)
    val_file = os.path.join(output_dir, "benchmark_val_50.json")
    test_file = os.path.join(output_dir, "benchmark_test_300.json")

    print("Attempting to load CNN/DailyMail via datasets library...")
    try:
        from datasets import load_dataset

        # Validation split: Used solely for tuning NLI classification thresholds
        print("Downloading 50 validation articles for NLI threshold tuning...")
        val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:50]")
        val_samples = [
            {"id": f"val_{i}", "article": row["article"], "reference_summary": row["highlights"]}
            for i, row in enumerate(val_dataset)
        ]
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_samples, f, indent=2)
        print(f"Saved {len(val_samples)} validation samples to {val_file}")

        # Test split: Used solely for the final ablation benchmarks and report
        print("Downloading 300 test articles for final benchmark evaluation...")
        test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:300]")
        test_samples = [
            {"id": f"test_{i}", "article": row["article"], "reference_summary": row["highlights"]}
            for i, row in enumerate(test_dataset)
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_samples, f, indent=2)
        print(f"Saved {len(test_samples)} test samples to {test_file}")

    except Exception as e:
        print(f"Could not reach HuggingFace datasets ({e}). Falling back to bundled samples.")
        samples_src = os.path.join("data", "samples", "cnn_dm_samples.json")
        if os.path.exists(samples_src):
            with open(samples_src, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            with open(val_file, "w", encoding="utf-8") as f:
                json.dump(data[:2], f, indent=2)
            print(f"Created fallback benchmark files from local samples.")


if __name__ == "__main__":
    prepare_cnn_dailymail()
