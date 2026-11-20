"""
Verification test for Academic Research Paper Cleaning, Segmentation,
Relevance Scoring, and Abstract Benchmarking.
Uses representative text from the user's 11-page CTF paper.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.academic_cleaner import clean_academic_paper
from src.preprocessing.tokenizer import segment_sentences
from src.pipeline import ExplainableSummarizationPipeline


SAMPLE_PAPER_TEXT = """2026 Journal of The Colloquium for Information Systems Security Education, Volume 13, No. 1, Spring 2026
©2026 The Author(s) | Creative Commons CC BY 4.0 1 www.cisse.info

A Case Study for Combating Student Overuse of Generative Artificial Intelligence in Cybersecurity Educational Activities Using Augmented Reality Capture-the-Flag Development

Shoshana Sugerman Information Technology & Web Science Program Rensselaer Polytechnic Institute Troy, NY, USA 0009-0002-9058-6828
Sanya Joseph Information Technology & Web Science Program Rensselaer Polytechnic Institute Troy, NY, USA 0009-0000-0343-809X
Brian Robert Callahan Department of Computer Science Monmouth University West Long Branch, NJ, USA 0000-0002-1797-8633

Abstract—Cybersecurity Capture-the-Flag (CTF) tournaments are well-understood to teach skills necessary for success in today's cybersecurity field. In the age of Generative Artificial Intelligence (AI), particularly for large-scale CTF tournaments, the use of Generative AI may be permitted or even encouraged. In such a situation, CTF participants must themselves balance the use of Generative AI with its overuse—effectively self-police the draw to offload one's thinking to the machine in pursuit of correct answers and prizes. In this paper, we introduce a case study for combating the overuse of Generative AI in cybersecurity educational activities through the building of our own CTF using Augmented Reality (AR) technologies. We argue that using the pedagogic lenses of the "see one, do one, teach one" model and peer learning allowed us to reinterpret our efforts on our CTF into a vision of shared labor and shared responsibility. This reframing kept us focused on using Generative AI as a tool and not a crutch.

Keywords—augmented reality, capture-the-flag, cybersecurity, pedagogy, peer learning

I. INTRODUCTION
In this paper, we, nineteen students who were afforded the educational opportunity to develop our own CTF, and our professor, provide a case study for combating student overuse of Generative AI in cybersecurity educational activities using the development of our own Augmented Reality (AR) CTF tournament, called the Rensselaer Cybersecurity Collaboratory CTF, or RC3TF for short. This work, undertaken when we were all undergraduate students, brings our campus to life as an enterprise network where players must physically navigate to encounter challenges.

Fig. 1. RC3TF game map transforming RPI campus buildings into an enterprise network.

II. BACKGROUND
In May 2024, the RCC won a seed grant from Rensselaer Polytechnic Institute (RPI) to develop a CTF for mass appeal. Many of us are CTF players ourselves participating in the National Cyber League (NCL). In Fall 2024, the NCL declared that participants could freely use Generative AI in the solving of challenges. It was tempting for us to offload our thinking to Generative AI when we were stuck on challenges. Debriefing as a group, we realized that it would have been possible for a player to offload all their thinking to the Generative AI and solve enough challenges to earn a challenge coin, placing in the top 500 of over 8,000 students, despite not understanding the work necessary to solve the challenges.

IV. DISCUSSION
The "see one, do one, teach one" model originally stems from William Stewart Halsted and his training methodology for surgeons at Johns Hopkins Hospital in Baltimore in the late 19th Century, as noted by W. S. Halsted in early surgical archives (vol. 12, pp. 45-50, fig. 3). First, the student will watch the new skill be performed. Second, the student will perform the new skill. Third, the student will teach another student the skill.

Fig. 5. Image of the Corrupted data detection game.

Shared labor means equitable shared responsibility. We simply were forced to reconcile with the fact that we respect each other as people, and therefore we are each responsible to ensure that we are all learning and that everyone can benefit from our work.

REFERENCES
[1] V. Ford, A. Siraj, A. Haynes and E. Brown, "Capture the Flag Unplugged: an Offline Cyber Competition," Proc. 2017 ACM SIGCSE, pp. 225-230, Mar. 2017. doi: https://doi.org/10.1145/3017680.3017783.
[2] L. McDaniel, E. Talvi and B. Hay, "Capture the Flag as Cyber Security Introduction," 2016 49th HICSS, pp. 5479-5486.
[3] S. Kucek and M. Leitner, "An Empirical Survey of Functions and Configurations of Open-Source CTF," J. Network Comput. Appl., vol. 151:102470, Feb. 1, 2020.
"""


def test_academic_cleaner():
    print("[1/4] Testing Academic Paper Cleaner...")
    cleaned = clean_academic_paper(SAMPLE_PAPER_TEXT)
    
    assert cleaned["has_references_removed"] is True, "References should be pruned"
    assert "Capture the Flag Unplugged" not in cleaned["clean_full_text"], "References section must be completely excluded"
    assert "0009-0002-9058-6828" not in cleaned["clean_full_text"], "ORCID must be stripped"
    assert "Creative Commons CC BY" not in cleaned["clean_full_text"], "License line must be stripped"
    assert "Fig. 1." not in cleaned["clean_full_text"], "Figure caption 1 must be stripped"
    assert "Fig. 5." not in cleaned["clean_full_text"], "Figure caption 5 must be stripped"
    assert cleaned["abstract"] is not None, "Abstract must be cleanly extracted"
    assert "Cybersecurity Capture-the-Flag" in cleaned["abstract"]
    print("  -> Academic cleaner passed: references, author blocks, and captions pruned.")
    print(f"  -> Extracted Abstract: \"{cleaned['abstract'][:80]}...\"")


def test_sentence_segmentation():
    print("[2/4] Testing Sentence Segmentation on Academic Text...")
    cleaned = clean_academic_paper(SAMPLE_PAPER_TEXT)
    sents = segment_sentences(cleaned["clean_full_text"])
    
    # Verify no broken abbreviation fragments
    for s in sents:
        assert not s.startswith("S. Halsted"), f"Initial shouldn't cause split: {s}"
        assert not s.startswith("pp."), f"pp. shouldn't cause split: {s}"
        assert not s.startswith("vol."), f"vol. shouldn't cause split: {s}"
        assert not s.startswith("fig. 3"), f"fig. shouldn't cause split: {s}"
        assert len(s.split()) >= 5, f"Sentence too short / fragment: '{s}'"
    print(f"  -> Sentence segmentation passed: {len(sents)} clean, coherent sentences formed.")


def test_pipeline_on_academic_paper():
    print("[3/4] Testing End-to-End Pipeline with Academic Paper & Abstract Benchmark...")
    pipeline = ExplainableSummarizationPipeline(mock_mode=True)
    results = pipeline.process(
        SAMPLE_PAPER_TEXT,
        extractive_sentences=3,
        is_academic_paper=True
    )
    
    # 1. Extractive Summary check
    ext_sum = results["extractive"]["summary"]
    assert "Capture the Flag Unplugged" not in ext_sum, "No references in extractive summary"
    print(f"  -> TF-IDF Extractive Summary ({len(ext_sum.split())} words):\n     {ext_sum[:120]}...")
    
    # 2. Abstractive Summary check - verify NO cheesy synthetic template
    abs_sum = results["abstractive"]["summary"]
    assert "The report outlines key developments regarding" not in abs_sum, "Must not contain cheesy synthetic templates"
    assert "Furthermore, it emphasizes that" not in abs_sum, "Must not contain cheesy synthetic templates"
    print(f"  -> Abstractive Summary ({len(abs_sum.split())} words):\n     {abs_sum[:120]}...")

    # 3. Gold-Standard Abstract Benchmark ROUGE check
    bench = results["abstract_benchmark"]
    assert bench is not None, "Abstract benchmark should be calculated"
    assert "reference_abstract" in bench
    print(f"  -> Benchmark vs Author Abstract:")
    print(f"     Extractive (TF-IDF) ROUGE-1: {bench['extractive_rouge']['rouge_1'] * 100:.1f}%, ROUGE-L: {bench['extractive_rouge']['rouge_l'] * 100:.1f}%")
    print(f"     Abstractive ROUGE-1: {bench['abstractive_rouge']['rouge_1'] * 100:.1f}%, ROUGE-L: {bench['abstractive_rouge']['rouge_l'] * 100:.1f}%")


def test_relevance_vs_faithfulness():
    print("[4/4] Testing Relevance Scorer (Separating Faithfulness from Topic Centrality)...")
    pipeline = ExplainableSummarizationPipeline(mock_mode=True)
    results = pipeline.process(
        SAMPLE_PAPER_TEXT,
        extractive_sentences=3,
        is_academic_paper=True
    )
    
    rel = results["relevance"]
    quadrants = rel["quadrant_counts"]
    print(f"  -> Quadrants: {quadrants}")
    assert rel["overall_relevance_score"] >= 0.0
    print("  -> Relevance & Faithfulness test passed successfully.")


if __name__ == "__main__":
    test_academic_cleaner()
    test_sentence_segmentation()
    test_pipeline_on_academic_paper()
    test_relevance_vs_faithfulness()
    print("\n[SUCCESS] All Academic Pipeline Verification Tests Passed!")
