"""
Test script to verify the 3 Indian Standard Demo Samples:
1. Research Paper on Indic NLP (IIT Madras context)
2. ISRO Chandrayaan-3 Space News (The Hindu / PTI standard)
3. India UPI Digital Infrastructure News (Economic Times standard)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ExplainableSummarizationPipeline

def run_test():
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "demo_samples"))
    pipeline = ExplainableSummarizationPipeline(mock_mode=True)
    
    # 1. Test Indian Research Paper
    print("\n[1/3] Testing Research Paper (Indic NLP)...")
    paper_path = os.path.join(samples_dir, "research_paper_indic_nlp.txt")
    with open(paper_path, "r", encoding="utf-8") as f:
        paper_text = f.read()
    
    paper_res = pipeline.process(paper_text, extractive_sentences=3, is_academic_paper=True)
    assert "error" not in paper_res, f"Failed on research paper: {paper_res.get('error')}"
    assert paper_res["extractive"]["summary"], "Extractive summary must not be empty"
    assert paper_res["abstractive"]["summary"], "Abstractive summary must not be empty"
    assert paper_res["abstract_benchmark"] is not None, "Abstract benchmark should be calculated"
    print("  -> Research paper successfully cleaned and summarized.")
    print(f"  -> Extracted Abstract: {paper_res['source']['reference_abstract'][:80]}...")
    print(f"  -> TF-IDF Summary: {paper_res['extractive']['summary'][:100]}...")

    # 2. Test ISRO Space News
    print("\n[2/3] Testing ISRO Chandrayaan News Article...")
    isro_path = os.path.join(samples_dir, "news_isro_chandrayaan.txt")
    with open(isro_path, "r", encoding="utf-8") as f:
        isro_text = f.read()
        
    isro_res = pipeline.process(isro_text, extractive_sentences=3)
    assert "error" not in isro_res, f"Failed on ISRO article: {isro_res.get('error')}"
    assert len(isro_res["verification"]["claims"]) > 0, "Should extract and verify claims"
    print("  -> ISRO News successfully summarized.")
    print(f"  -> TF-IDF Summary: {isro_res['extractive']['summary'][:100]}...")
    print(f"  -> Factuality Score: {isro_res['verification']['factuality_score']}")

    # 3. Test UPI News
    print("\n[3/3] Testing UPI Digital Infrastructure News Article...")
    upi_path = os.path.join(samples_dir, "news_upi_digital_infrastructure.txt")
    with open(upi_path, "r", encoding="utf-8") as f:
        upi_text = f.read()
        
    upi_res = pipeline.process(upi_text, extractive_sentences=3)
    assert "error" not in upi_res, f"Failed on UPI article: {upi_res.get('error')}"
    assert len(upi_res["verification"]["claims"]) > 0, "Should extract and verify claims"
    print("  -> UPI News successfully summarized.")
    print(f"  -> TF-IDF Summary: {upi_res['extractive']['summary'][:100]}...")
    print(f"  -> Factuality Score: {upi_res['verification']['factuality_score']}")

    print("\n[SUCCESS] All 3 Indian Standard Demo Samples Processed Perfectly!")

if __name__ == "__main__":
    run_test()

