"""
Streamlit Web Interface for Explainable, Fact-Aware Text Summarization.
Visualizes:
  1. Side-by-side comparison of Extractive vs. Abstractive Summaries.
  2. Claim-level factual verification badges (Supported, Contradicted, Uncertain).
  3. Linked source evidence retrieval for each summary statement.
  4. PDF Upload for Research Papers.
  5. Live Ablation Table generator.
"""
import os
import sys
import json
from typing import Optional, List, Dict, Any
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ExplainableSummarizationPipeline
from src.utils.pdf_loader import extract_text_from_pdf
from src.evaluation.ablation import run_ablation_study

# Streamlit Page Config
st.set_page_config(
    page_title="Fact-Aware Text Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Badges and Cards
st.markdown("""
<style>
    .badge-green, .badge-supported {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #c3e6cb;
    }
    .badge-red, .badge-contradicted {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #f5c6cb;
    }
    .badge-yellow, .badge-uncertain {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #ffeeba;
    }
    .badge-purple {
        background-color: #e2d9f3;
        color: #432874;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #cbbde5;
    }
    .badge-gray {
        background-color: #e9ecef;
        color: #495057;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #ced4da;
    }
    .evidence-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px 14px;
        margin-top: 6px;
        border-radius: 0 4px 4px 0;
        font-size: 0.92rem;
    }
    .claim-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline(
    model_name: str,
    mock_mode: bool,
    model_path: Optional[str] = None,
    backend: str = "local_cpu",
    colab_url: Optional[str] = None
):
    """Caches pipeline instance across Streamlit reruns so neural weights are loaded once."""
    import importlib
    import src.pipeline
    importlib.reload(src.pipeline)
    return src.pipeline.ExplainableSummarizationPipeline(
        abstractive_model=model_name,
        mock_mode=mock_mode,
        model_path=model_path,
        backend=backend,
        colab_url=colab_url
    )


# Sidebar Configuration
st.sidebar.title("⚙️ Pipeline Settings")

# 1. Compute Backend Selector (Colab T4 GPU vs Local CPU)
st.sidebar.subheader("Compute Backend")
backend_choice = st.sidebar.radio(
    "Execution Target",
    ["Local CPU", "Google Colab (T4 GPU)"],
    index=0,
    help="Select Google Colab (T4 GPU) for ~15x faster generation on long research papers."
)
is_colab = "Colab" in backend_choice
backend_str = "colab_gpu" if is_colab else "local_cpu"

colab_url = ""
if is_colab:
    colab_url = st.sidebar.text_input(
        "Colab Backend URL (ngrok)",
        value=st.session_state.get("colab_url", ""),
        placeholder="https://xxxx.ngrok-free.app",
        help="Run colab/colab_server.ipynb on Google Colab and paste the public ngrok URL here."
    ).strip()
    st.session_state["colab_url"] = colab_url

    if st.sidebar.button("🔌 Test Connection", key="btn_test_colab"):
        from src.remote.colab_client import ColabClient
        client = ColabClient(base_url=colab_url)
        with st.spinner("Pinging Colab server..."):
            health = client.check_health()
        if health.get("ok"):
            st.sidebar.success(f"🟢 Connected to Colab!\nGPU: {health.get('gpu_name', 'T4 GPU')} ({health.get('device', 'cuda').upper()})")
        else:
            st.sidebar.error(f"🔴 {health.get('error')}")

# 2. Local Model Selection
st.sidebar.subheader("Model Architecture")
model_choice = st.sidebar.selectbox(
    "Abstractive Model Architecture",
    [
        "facebook/bart-large-cnn (Default, ~1.6 GB)",
        "sshleifer/distilbart-cnn-12-6 (Fast / CPU-Friendly, ~1.2 GB)",
        "Custom / Local Offline Path"
    ],
    index=0
)

custom_model_path: Optional[str] = None
if "Custom" in model_choice:
    custom_path_input = st.sidebar.text_input(
        "Local Model Weights Directory (Offline Mode)",
        value="",
        placeholder="C:/path/to/downloaded/weights",
        help="Local directory containing model.safetensors/pytorch_model.bin and config.json"
    )
    if custom_path_input.strip():
        custom_model_path = custom_path_input.strip()
        selected_model_name = custom_model_path
    else:
        selected_model_name = "facebook/bart-large-cnn"
elif "distilbart" in model_choice:
    selected_model_name = "sshleifer/distilbart-cnn-12-6"
else:
    selected_model_name = "facebook/bart-large-cnn"

execution_mode = st.sidebar.radio(
    "Execution Mode",
    ["Standard / Auto-Detect", "Fast Mock / Dry-Run (No GPU Required)"],
    index=0
)
is_mock = "Fast Mock" in execution_mode

# Startup check with loading progress
if not is_mock and not is_colab:
    with st.spinner(f"Loading {selected_model_name} (Downloading BART, ~1.6 GB, first run only)..."):
        pipeline = get_pipeline(
            model_name=selected_model_name,
            mock_mode=is_mock,
            model_path=custom_model_path,
            backend="local_cpu"
        )
else:
    pipeline = get_pipeline(
        model_name=selected_model_name,
        mock_mode=is_mock,
        model_path=custom_model_path,
        backend=backend_str,
        colab_url=colab_url if is_colab else None
    )

# Model Status Indicator & Honest Fallback Error Reporter
if is_colab:
    if colab_url:
        st.sidebar.success(f"🚀 Colab T4 GPU Backend Configured")
        st.sidebar.caption(f"🔗 `{colab_url}`")
    else:
        st.sidebar.warning("⚠️ Colab URL not entered — Enter ngrok URL above or select Local CPU.")
else:
    trans = pipeline.transformer_summarizer
    if not is_mock and trans.is_neural and trans.pipeline is not None:
        st.sidebar.success(f"🟢 Real Neural Transformer Active")
        st.sidebar.caption(f"🧠 Model: `{trans.model_name}` | ⚡ Device: **{trans.device_name.upper()}**")
    else:
        failure_reason = trans.load_error or ("Fast Mock / Dry-Run selected by user" if is_mock else "PyTorch/BART weights offline")
        st.sidebar.warning(f"🟡 Fallback Mode Active\n\n**Reason**: `{failure_reason}`")
        if trans.load_traceback:
            with st.sidebar.expander("🔍 View Initialization Error Traceback"):
                st.code(trans.load_traceback, language="python")

extractive_sentences = st.sidebar.slider("Summary Sentences / Length", min_value=1, max_value=10, value=4)
top_k_evidence = st.sidebar.slider("Evidence Sentences per Claim", min_value=1, max_value=4, value=2)

st.sidebar.markdown("---")
st.sidebar.subheader("About This Project")
st.sidebar.info(
    "**Explainable, Fact-Aware Summarization**\n\n"
    "Abstractive summarizers can hallucinate unfaithful claims. "
    "This system pairs neural generation with **Sentence Embeddings** to retrieve grounding evidence, "
    "and **Natural Language Inference (NLI)** to classify each claim as Supported, Contradicted, or Uncertain.\n\n"
    "**New**: Includes an Academic Paper Cleaner to strip references, DOIs, and headers, "
    "plus a **Relevance Scorer** to separate faithful claims from irrelevant noise."
)

# Header
st.title("🔬 Explainable, Fact-Aware Text Summarizer")
st.caption("Research-oriented NLP Pipeline: Extractive Baselines vs. Abstractive Generation with NLI Grounding & Relevance Verification")

# Input Section Tabs
input_tab1, input_tab2, ablation_tab = st.tabs(["📝 Text Input & Challenge Cases", "📄 Upload PDF (Research Papers)", "📊 Ablation Study Benchmark"])

with input_tab1:
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "demo_samples"))
    col_preset, _ = st.columns([2, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Choose a Sample or Enter Custom Text:",
            [
                "Custom Text",
                "🇮🇳 [Research Paper] Low-Resource Indic NLP & Multilingual AI (IIT Madras)",
                "🇮🇳 [Indian News] ISRO Chandrayaan-3 Lunar South Pole Landing (The Hindu / PTI)",
                "🇮🇳 [Indian News] India's UPI & Digital Public Infrastructure Growth (Economic Times)",
                "Preset 1: Federal Reserve Interest Rate (CNN News)",
                "Preset 2: James Webb Telescope (NASA Science)",
                "Preset 3: [Challenge Test] Electric Vehicle Sales (Number Contradiction)",
                "Preset 4: [Challenge Test] Supreme Court Merger (Negation Flip)"
            ]
        )

    # Preset texts
    default_text = ""
    if "Indic NLP" in preset_choice:
        paper_file = os.path.join(samples_dir, "research_paper_indic_nlp.txt")
        if os.path.exists(paper_file):
            with open(paper_file, "r", encoding="utf-8") as f:
                default_text = f.read()
    elif "ISRO" in preset_choice:
        news_file1 = os.path.join(samples_dir, "news_isro_chandrayaan.txt")
        if os.path.exists(news_file1):
            with open(news_file1, "r", encoding="utf-8") as f:
                default_text = f.read()
    elif "UPI" in preset_choice:
        news_file2 = os.path.join(samples_dir, "news_upi_digital_infrastructure.txt")
        if os.path.exists(news_file2):
            with open(news_file2, "r", encoding="utf-8") as f:
                default_text = f.read()
    elif "Federal Reserve" in preset_choice:
        default_text = (
            "The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday, "
            "continuing its campaign to tame inflation. The decision brings the central bank's target federal funds rate "
            "to a range of 5 percent to 5.25 percent, the highest level in sixteen years. Fed Chair Jerome Powell signaled "
            "that the committee might pause future rate increases, stating that policy is approaching the level required "
            "to bring inflation back down to the target rate of 2 percent. However, Powell emphasized that officials have "
            "not made a definitive decision to halt increases and will monitor incoming economic data closely before the next meeting in June."
        )
    elif "James Webb" in preset_choice:
        default_text = (
            "NASA announced on Tuesday that the James Webb Space Telescope has detected water vapor in the inner disk "
            "of the planetary system PDS 70, located 370 light-years from Earth. PDS 70 is known to host two infant gas giant planets. "
            "The discovery marks the first time that water has been identified in the terrestrial region of a protoplanetary disk "
            "where rocky, Earth-like planets typically form. Astronomers used the telescope's Mid-Infrared Instrument (MIRI) "
            "to detect the chemical signatures. While water vapor has previously been found in planet-forming disks, scientists "
            "were uncertain whether water could survive harsh ultraviolet radiation from young stars."
        )
    elif "Electric Vehicle" in preset_choice:
        default_text = (
            "Electric vehicle sales reached 14 million units globally in 2023, representing an increase of 35 percent "
            "compared to 2022. Sales remained robust despite high borrowing costs and supply chain bottlenecks in key markets."
        )
    elif "Supreme Court" in preset_choice:
        default_text = (
            "The supreme court declined to hear the appeal, allowing the lower court's ruling against the merger to remain in effect. "
            "The decision effectively ends the two-year legal battle between the regulatory agency and the telecommunications giant."
        )

    input_text = st.text_area("Original Document Text", value=default_text, height=240, placeholder="Paste article or paper text here...")
    run_btn = st.button("🚀 Generate & Verify Summaries", type="primary", key="btn_text")

with input_tab2:
    st.subheader("Upload Research Paper / Academic Article PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    pdf_text = ""
    if uploaded_pdf is not None:
        with st.spinner("Extracting & cleaning text from PDF using PyMuPDF..."):
            extracted = extract_text_from_pdf(uploaded_pdf)
            if extracted["status"] == "success":
                st.session_state["pdf_abstract"] = extracted.get("abstract")
                st.success(
                    f"Extracted {extracted['page_count']} pages successfully! "
                    f"({extracted.get('original_word_count', 0)} raw words ➔ {extracted.get('cleaned_word_count', 0)} clean narrative words)"
                )
                if extracted.get("has_references_removed"):
                    st.info("✂️ **Academic Noise Removed**: Stripped reference bibliography, journal headers, author affiliation blocks, and figure captions.")
                
                if extracted.get("abstract"):
                    with st.expander("🎯 Extracted Paper Abstract (Gold-Standard Reference Summary)", expanded=True):
                        st.markdown(extracted["abstract"])

                pdf_text = extracted["full_text"]
                st.text_area("Cleaned Paper Content (Preview)", value=pdf_text[:1200] + "...", height=150)
            else:
                st.error(extracted["status"])

    run_pdf_btn = st.button("🚀 Summarize & Verify Uploaded PDF", type="primary", key="btn_pdf", disabled=not bool(pdf_text))

# Process execution
active_text = None
active_abstract = None
is_academic = False

if run_btn and input_text.strip():
    active_text = input_text.strip()
elif run_pdf_btn and pdf_text.strip():
    active_text = pdf_text.strip()
    is_academic = True
    active_abstract = st.session_state.get("pdf_abstract")

if active_text:
    with st.spinner("Executing pipeline: Cleaning ➔ Extractive Baselines ➔ Abstractive Generation ➔ Grounding ➔ NLI & Relevance..."):
        results = pipeline.process(
            active_text,
            extractive_sentences=extractive_sentences,
            top_k_evidence=top_k_evidence,
            reference_abstract=active_abstract,
            is_academic_paper=is_academic
        )

    st.markdown("---")
    st.subheader("📋 Summarization Comparison")

    src_words = len(active_text.split())
    st.caption(f"Original Document: **{src_words} words** across **{results['source']['total_sentences']} sentences**")

    col_ext, col_abs = st.columns(2)

    with col_ext:
        st.markdown("#### 1. Classical Extractive Baseline (TF-IDF)")
        st.caption("Verbatim sentences selected by Term Frequency - Inverse Document Frequency (Zero hallucinations)")
        
        tf_sum = results["extractive"]["tfidf"]["summary"]
        tf_words = len(tf_sum.split())
        tf_comp = int((1.0 - (tf_words / max(1, src_words))) * 100)
        st.info(tf_sum)
        st.caption(f"📊 Length: **{tf_words} words** ({tf_comp}% reduction) | ⏱️ Runtime: {results['extractive']['tfidf']['runtime_sec']}s")

    with col_abs:
        abs_mode = results["abstractive"].get("mode", "neural_bart")
        backend_used = results["abstractive"].get("backend", "local_cpu")
        is_fallback = (abs_mode not in ["neural_bart", "colab_bart"]) or not results["abstractive"].get("is_abstractive", True) or ("fallback" in backend_used)

        if backend_used == "colab_gpu":
            st.markdown("#### 2. Abstractive Transformer (Colab T4 GPU)")
            st.caption("⚡ Synthesized via neural beam search on **Google Colab NVIDIA T4 GPU** (Float16)")
        elif not is_fallback:
            model_disp = results["abstractive"].get("model_name", "BART")
            device_disp = results["abstractive"].get("device", "CPU").upper()
            st.markdown(f"#### 2. Abstractive Transformer ({model_disp})")
            st.caption(f"Synthesizes fluent new sentences via neural beam search on **{device_disp}**")
        elif "distilbart" in backend_used:
            st.markdown("#### 2. Local DistilBART Fallback (Colab Offline)")
            st.warning(f"⚠️ **Colab Unreachable**: Automatically switched to local DistilBART model.\n\n*Reason*: `{results['abstractive'].get('failure_reason')}`")
        else:
            st.markdown("#### 2. Extractive fallback (not abstractive)")
            exact_err = results["abstractive"].get("failure_reason") or pipeline.transformer_summarizer.load_error or "Weights offline / mock mode active"
            st.caption(f"Verbatim sentence selection — Failure Reason: `{exact_err}`")

        abs_sum = results["abstractive"]["summary"]
        abs_words = len(abs_sum.split())
        abs_comp = int((1.0 - (abs_words / max(1, src_words))) * 100)
        
        if not is_fallback:
            st.success(abs_sum)
        else:
            st.warning(abs_sum)

        gen_time = results["abstractive"].get("generation_time_sec", results["abstractive"].get("runtime_sec", 0))
        device_str = results["abstractive"].get("device", "CPU").upper()
        
        if backend_used == "colab_gpu":
            backend_badge = "🚀 Google Colab (T4 GPU)"
        elif "distilbart" in backend_used:
            backend_badge = "⚠️ Local DistilBART Fallback"
        elif is_fallback:
            backend_badge = "⚠️ Extractive Fallback"
        else:
            backend_badge = "💻 Local CPU"

        st.caption(
            f"📊 Length: **{abs_words} words** ({abs_comp}% reduction) | "
            f"⏱️ Generation Time: **{gen_time}s** | "
            f"🖥️ Backend: **{backend_badge}** ({device_str}) | "
            f"Chunks: {results['abstractive'].get('num_chunks', 1)}"
        )
        
        if is_fallback and "distilbart" not in backend_used:
            st.error(
                "⚠️ **Extractive fallback (not abstractive)**: "
                "This output was selected verbatim from source sentences because neural transformer weights could not be loaded. "
                "It is strictly an extractive fallback and was NOT generated by BART."
            )

    # Gold-Standard Abstract Benchmark (ROUGE Score Table)
    if results.get("abstract_benchmark"):
        bench = results["abstract_benchmark"]
        st.markdown("---")
        st.subheader("🏆 Author Abstract Benchmark (ROUGE Score)")
        st.caption(
            "🔒 **Zero Data Leakage**: The paper's Abstract was **strictly excluded** from the input document fed to the summarizers. "
            "The summaries were generated solely from the body text and are quantitatively evaluated against the author's reference abstract."
        )
        bench_data = {
            "Metric": ["ROUGE-1 (Unigram Overlap)", "ROUGE-2 (Bigram Overlap)", "ROUGE-L (Longest Common Subsequence)"],
            "Extractive Baseline (TF-IDF)": [
                f"{bench['extractive_rouge']['rouge_1'] * 100:.1f}%",
                f"{bench['extractive_rouge']['rouge_2'] * 100:.1f}%",
                f"{bench['extractive_rouge']['rouge_l'] * 100:.1f}%"
            ],
            "Abstractive / Synthesizer": [
                f"{bench['abstractive_rouge']['rouge_1'] * 100:.1f}%",
                f"{bench['abstractive_rouge']['rouge_2'] * 100:.1f}%",
                f"{bench['abstractive_rouge']['rouge_l'] * 100:.1f}%"
            ]
        }
        st.table(bench_data)

    # Factuality & Relevance Verification Summary Dashboard
    st.markdown("---")
    st.subheader("🔍 Explainability, Factuality & Relevance Layer")
    
    ver = results["verification"]
    counts = ver["counts"]
    rel = results.get("relevance", {})
    quadrants = rel.get("quadrant_counts", {})

    st.markdown("##### ⚖️ Faithfulness vs. Relevance (Separating Truth from Noise)")
    st.caption("A claim can be 100% faithful to the text (supported by NLI) but completely irrelevant (e.g. figure captions or boilerplate).")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Supported & Relevant (🟢)", quadrants.get("supported_and_relevant", 0), help="Gold standard: Factually true AND core topic")
    q2.metric("Supported but Peripheral (🟡)", quadrants.get("supported_but_peripheral", 0), help="Factually true, but secondary detail or noise")
    q3.metric("Contradicted (🔴)", quadrants.get("contradicted", 0), help="Factual conflict or hallucination")
    q4.metric("Uncertain / Speculative (⚪)", quadrants.get("uncertain", 0))

    m1, m2 = st.columns(2)
    m1.metric("Factuality Consistency Score", f"{int(ver['factuality_score'] * 100)}%")
    m2.metric("Topic Relevance Score", f"{rel.get('relevance_percentage', 0)}%")

    if ver["has_hallucinations"]:
        st.warning("⚠️ Warning: Factual discrepancies detected! At least one claim contradicts the source evidence.")
    elif quadrants.get("supported_but_peripheral", 0) > 0:
        st.info("ℹ️ Note: All claims are supported by the source text, though some represent peripheral details rather than core paper contributions.")
    else:
        st.success("✅ All claims appear substantiated and aligned with the core topic.")

    # Detailed Claim-by-Claim Evidence Cards
    st.markdown("### Claim-by-Claim Evidence Grounding & Relevance")
    for idx, claim_data in enumerate(ver["claims"]):
        status = str(claim_data.get("status", "uncertain"))
        badge = str(claim_data.get("badge", "yellow")).lower()
        badge_map = {
            "supported": "green",
            "contradicted": "red",
            "uncertain": "yellow",
            "green": "green",
            "red": "red",
            "yellow": "yellow"
        }
        badge_class = f"badge-{badge_map.get(badge, 'yellow')}"
        badge_text = status.upper()

        rel_class = "badge-purple" if claim_data.get("is_relevant") else "badge-gray"
        rel_label = "CORE TOPIC" if claim_data.get("is_relevant") else "PERIPHERAL / NOISE"

        # Safe extraction of confidence
        conf_val = claim_data.get("confidence")
        if conf_val is None:
            probs = claim_data.get("probabilities", {})
            conf_val = max(probs.values()) if probs else 0.85
        conf_pct = int(round(float(conf_val) * 100))

        rel_pct = int(round(float(claim_data.get("relevance_score", 0)) * 100))

        with st.expander(f"Claim {idx+1}: {claim_data.get('claim', '')}", expanded=True):
            st.markdown(
                f"<span class='{badge_class}'>{badge_text}</span> &nbsp; "
                f"<span class='{rel_class}'>{rel_label}</span> &nbsp; "
                f"Confidence: <b>{conf_pct}%</b> &nbsp; "
                f"Relevance: <b>{rel_pct}%</b>",
                unsafe_allow_html=True
            )
            
            if claim_data.get("overlap_keywords"):
                st.caption(f"Topic Keyword Matches: `{'`, `'.join(claim_data['overlap_keywords'])}`")

            st.markdown("**Retrieved Source Evidence Passages:**")
            sources = claim_data.get("evidence_sources", [])
            if sources:
                for s in sources:
                    st.markdown(
                        f"<div class='evidence-box'>"
                        f"<b>[Source Sentence #{s.get('source_index', 0) + 1} | Similarity: {s.get('similarity_score', 0)}]</b><br>"
                        f"\"{s.get('sentence', '')}\""
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.write("No matching source evidence retrieved.")

            # Probabilities breakdown
            probs = claim_data.get("probabilities", {})
            p_cols = st.columns(3)
            p_entail = max(0.0, min(1.0, float(probs.get("entailment", 0.0))))
            p_neutral = max(0.0, min(1.0, float(probs.get("neutral", 0.0))))
            p_contra = max(0.0, min(1.0, float(probs.get("contradiction", 0.0))))
            p_cols[0].progress(p_entail, text=f"Entailment: {p_entail:.2f}")
            p_cols[1].progress(p_neutral, text=f"Neutral: {p_neutral:.2f}")
            p_cols[2].progress(p_contra, text=f"Contradiction: {p_contra:.2f}")

# Ablation Benchmark Tab
with ablation_tab:
    st.subheader("📊 Ablation Study Benchmark (Page 5 Matrix)")
    st.write(
        "Runs all 4 configurations across benchmark articles to measure the trade-off "
        "between **ROUGE quality**, **Factual Consistency**, and **Processing Latency**."
    )
    
    sample_limit = st.slider("Benchmark Sample Count", min_value=1, max_value=10, value=3)
    if st.button("Run Full Ablation Benchmark", type="secondary"):
        samples_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "cnn_dm_samples.json")
        with open(samples_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        with st.spinner("Computing ablation metrics across configurations..."):
            ablation_results = run_ablation_study(
                bench_data,
                pipeline.textrank_summarizer,
                pipeline.transformer_summarizer,
                pipeline.evidence_retriever,
                pipeline.nli_verifier,
                max_samples=sample_limit
            )

        st.table(ablation_results["table"])
        st.success("Ablation study completed!")
