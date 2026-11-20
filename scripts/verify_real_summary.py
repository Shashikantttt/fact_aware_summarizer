"""
Verification script for real neural summarization with BART (facebook/bart-large-cnn).
Validates model loading, sidebar status indicator logic, device selection, and generation runtime.
"""
import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ExplainableSummarizationPipeline


def main():
    print("=== Step 4: Real Summary Verification ===")
    pipeline = ExplainableSummarizationPipeline(mock_mode=False)
    
    sample_article = (
        "The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday, "
        "continuing its campaign to tame inflation. The decision brings the central bank target federal funds rate "
        "to a range of 5 percent to 5.25 percent, the highest level in sixteen years. Fed Chair Jerome Powell signaled "
        "that the committee might pause future rate increases, stating that policy is approaching the level required "
        "to bring inflation back down to the target rate of 2 percent. However, Powell emphasized that officials have "
        "not made a definitive decision to halt increases and will monitor incoming economic data closely before the next meeting in June."
    )
    
    results = pipeline.process(sample_article, extractive_sentences=2)
    trans = pipeline.transformer_summarizer
    
    # Sidebar status check
    if not pipeline.mock_mode and trans.is_neural and trans.pipeline is not None:
        sidebar_status = "🟢 Real Neural Transformer Active"
    else:
        sidebar_status = f"🟡 Fallback Mode Active ({trans.load_error or 'Unknown'})"

    abs_data = results["abstractive"]
    
    print("\n" + "=" * 50)
    print("STATUS VERIFICATION:")
    print(f"Sidebar Status:    {sidebar_status}")
    print(f"Model Name:        {trans.model_name}")
    print(f"Device:            {trans.device_name}")
    print(f"Generation Time:   {abs_data['generation_time_sec']}s")
    print(f"Summary Mode:      {abs_data['mode']}")
    print(f"Is Abstractive:    {abs_data['is_abstractive']}")
    print("=" * 50)
    print(f"Generated BART Summary:\n\"{abs_data['summary']}\"")
    print("=" * 50)


if __name__ == "__main__":
    main()
