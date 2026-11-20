# Explainable, Fact-Aware Text Summarization

A research-oriented Natural Language Processing (NLP) system that:
1. Generates concise summaries using both **classical extractive** and **deep abstractive** models.
2. Retrieves and displays the exact source sentences supporting every summary statement (**Evidence Grounding**).
3. Verifies claims using **Natural Language Inference (NLI)** to flag hallucinations, unsupported assertions, or direct contradictions.
4. Provides quantitative evaluation across quality (ROUGE), factuality consistency, and execution latency.

---

## 1. System Architecture

```
                                  [ User Input ]
                         (Raw Text / Uploaded Document)
                                       │
                                       ▼
                         [ Layer 1: Preprocessing ]
               • Text Cleaning & Regex Normalization
               • Sentence Segmentation & Tokenization
               • Long Document Chunking (Token Budgeting)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
   [ Layer 2A: Classical Extractive ]         [ Layer 2B: Modern Abstractive ]
   • TF-IDF Sentence Scoring                  • Pretrained Transformer (BART / T5)
   • Length Normalization & Lead Bias              • Chunk-and-Combine Strategy
   • Pure Python / NumPy (Zero Download)      • Beam Search & Length Control
                 │                                           │
                 │                                           ▼
                 │                          [ Layer 3: Evidence Retrieval ]
                 │                          • Split abstractive summary into claims
                 │                          • Sentence Embeddings (MiniLM / Cosine Sim)
                 │                          • Top-k source sentence retrieval per claim
                 │                                           │
                 │                                           ▼
                 │                          [ Layer 4: Factual Verification ]
                 │                          • NLI Engine (Premise = Evidence, Hypothesis = Claim)
                 │                          • Classify: Entailed / Contradicted / Neutral
                 │                          • Confidence thresholding & hallucination flags
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       ▼
                           [ Layer 5: Evaluation ]
                 • Lexical Overlap: ROUGE-1, ROUGE-2, ROUGE-L
                 • Semantic Quality: BERTScore
                 • Faithfulness Metric: % Supported vs. % Contradicted claims
                 • Resource Profiling: Inference Latency & Compression Ratio
                 • Ablation Matrix generation across all 4 configurations
                                       │
                                       ▼
                   [ Layer 6: Explainable UI & Reports ]
                 • Side-by-side comparison (Extractive vs. Abstractive)
                 • Color-coded claims:
                   - 🟢 Green: Entailed (Supported with linked source text)
                   - 🔴 Red: Contradicted (Potential Hallucination)
                   - 🟡 Yellow: Neutral (Unsubstantiated claim)
```

---

## 2. Theoretical Foundations (Key Concepts)

### A. TF-IDF Extractive Scoring
Each sentence $s$ in the document is scored by the sum of TF-IDF values of its constituent terms $t$, normalized by sentence length:
$$\text{TF}(t, s) = \frac{\text{count}(t, s)}{|s|}, \quad \text{IDF}(t) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$
$$\text{Score}(s) = \frac{\sum_{t \in s} \text{TF}(t, s) \cdot \text{IDF}(t)}{|s|^{0.8}}$$

### B. Zero-Hallucination Extractive Guarantee
Extractive summarization directly isolates and ranks the most informative sentences using TF-IDF term salience.
* **Why Extractive Summaries Cannot Hallucinate**: Because sentences are extracted verbatim from the input document, they have 100% factual faithfulness and serve as a reliable baseline against abstractive hallucination.

### C. Abstractive Generation (BART / T5)
Encoder-decoder transformer architecture. The bidirectional encoder processes the full source context, and the autoregressive decoder generates summary tokens using cross-attention with beam search.
* **Why Abstractive Summaries Hallucinate**: During generation, the decoder can generate plausible-sounding tokens that alter dates, numbers, or polarity (e.g. turning "declined appeal" into "accepted appeal").

### D. Evidence Retrieval & NLI Verification
For each generated claim $C_i$:
1. **Sentence Embeddings**: Vector representation $\mathbf{e}(C_i)$ using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Top-K Retrieval**: Find source sentences $S_j$ maximizing $\cos(\mathbf{e}(C_i), \mathbf{e}(S_j))$.
3. **NLI Verification**: Treat retrieved evidence as **Premise** and claim $C_i$ as **Hypothesis**. The NLI model outputs probabilities for:
   * **Entailment (🟢 Supported)**: The premise guarantees the truth of the claim.
   * **Contradiction (🔴 Hallucination)**: The premise directly refutes the claim.
   * **Neutral (🟡 Uncertain)**: The premise neither confirms nor denies the claim.

---

## 3. Project Directory Structure

```text
NLP_PROJECT/
├── app/
│   └── app.py                            # Streamlit interactive UI (PDF upload + evidence grounding)
├── colab/
│   ├── colab_server.ipynb                # Fast T4 GPU backend server (FastAPI + ngrok)
│   └── explainable_summarizer_experiments.ipynb # 1-Click GPU benchmark runner for Colab
├── data/
│   └── samples/
│       ├── cnn_dm_samples.json           # Packaged benchmark news articles (zero download needed)
│       └── challenge_cases.json          # Controlled hallucination test suite (numbers, negations)
├── scripts/
│   └── prepare_cnn_dm.py                 # Dataset loader (300 test / 50 validation split)
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py                    # Whitespace, citation, and unicode normalization
│   │   └── tokenizer.py                  # Sentence splitting, stopwords, and chunking
│   ├── extractive/
│   │   ├── tfidf_summarizer.py           # First-principles TF-IDF sentence ranker
│   │   └── textrank_summarizer.py        # First-principles TextRank & PageRank
│   ├── abstractive/
│   │   └── transformer_summarizer.py     # BART/T5 loader with long-document chunking
│   ├── retrieval/
│   │   └── evidence_retriever.py         # Top-k semantic search via embeddings
│   ├── verification/
│   │   └── nli_verifier.py               # NLI entailment / contradiction decision engine
│   ├── evaluation/
│   │   ├── metrics.py                    # Pure Python ROUGE-1, ROUGE-2, ROUGE-L, factuality
│   │   └── ablation.py                   # 4-Configuration benchmark table runner
│   ├── utils/
│   │   └── pdf_loader.py                 # PyMuPDF text extractor for research papers
│   └── pipeline.py                       # Unified end-to-end controller
├── tests/
│   └── test_phase1.py                    # Verification suite for Phase 1 (Runs 100% offline)
├── requirements.txt                      # Minimal local requirements
├── requirements-gpu.txt                  # Full GPU/Colab requirements
└── README.md
```

---

## 4. How to Run Locally (Phase 1 & Sandbox)

### Step 1: Run the Phase 1 Test Suite (Zero External Downloads)
The classical NLP baselines, text cleaners, segmenters, and TF-IDF run with standard Python libraries:
```bash
python tests/test_phase1.py
```
This runs 5 verification tests and outputs the extractive summaries for sample articles.

### Step 2: Launch the Streamlit Interactive Interface
Install minimal packages:
```bash
pip install -r requirements.txt
```
Launch the application:
```bash
streamlit run app/app.py
```
* You can test custom text, preset news articles, or upload multi-page research paper PDFs.
* In environments without GPU/HuggingFace access, select **"Fast Mock / Dry-Run"** in the sidebar to test the full pipeline and UI interactions immediately.

---

## 5. How to Run Heavy Experiments on Google Colab

To run the full transformer models (`BART-large-cnn`, `all-MiniLM-L6-v2`, `deberta-v3-base NLI`) on 300 CNN/DailyMail benchmark articles:

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type** and select **T4 GPU** (free tier).
3. Upload and open `colab/explainable_summarizer_experiments.ipynb`.
4. Click **Runtime > Run all**.

The notebook will:
* Download CNN/DailyMail test articles and validation samples.
* Run all 4 ablation configurations.
* Produce the ablation table comparing ROUGE-1, Factual Consistency %, and Latency.
* Demonstrate NLI verification on controlled hallucination edge cases.

---

## 6. How to Connect Google Colab (T4 GPU) to Your Streamlit App

Run the heavy BART & NLI models on Google Colab's free NVIDIA T4 GPU for **~15x faster generation** on long research papers:

### Step 1: Open and Run the Colab Server
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `colab/colab_server.ipynb` from this project.
3. In Colab, go to **Runtime > Change runtime type** and select **T4 GPU** (free tier).
4. Click **Runtime > Run all**.
5. When prompted, enter your free [ngrok token](https://dashboard.ngrok.com/get-started/your-authtoken).

### Step 2: Copy the Public Tunnel URL
Once the models load, the final cell will print:
```text

```
Copy that URL.

### Step 3: Connect to Streamlit
1. Start your local Streamlit app:
   ```bash
   streamlit run app/app.py
   ```
2. In the left sidebar under **"Compute Backend"**, choose **Google Colab (T4 GPU)**.
3. Paste the URL into **"Colab Backend URL (ngrok)"**.
4. Click **"🔌 Test Connection"** to verify the GPU connection.
5. Upload research paper PDFs or paste text and generate summaries at GPU speeds!
   - If the Colab tunnel ever goes offline or closes, the app **automatically and gracefully falls back** to your local model with a clear failure explanation.

"# fact_aware_summarizer" 
