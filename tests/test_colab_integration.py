"""
Unit tests for Colab GPU client, endpoint communication, and automatic local fallback.
"""
import os
import sys
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.remote.colab_client import ColabClient, ColabClientError
from src.pipeline import ExplainableSummarizationPipeline


class MockColabServer(BaseHTTPRequestHandler):
    """Simulates the Colab GPU FastAPI endpoints for deterministic unit testing."""
    def log_message(self, format, *args):
        pass  # Suppress request logging in test output

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "backend": "colab_gpu",
                "device": "cuda",
                "gpu_name": "Tesla T4",
                "models_loaded": ["facebook/bart-large-cnn", "all-MiniLM-L6-v2", "nli-deberta-v3-base"]
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        req_json = json.loads(post_data) if post_data else {}

        if self.path == "/summarize":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "summary": "This is a lightning-fast abstractive summary synthesized on Colab T4 GPU.",
                "model_name": "facebook/bart-large-cnn",
                "device": "cuda",
                "gpu_name": "Tesla T4",
                "generation_time_sec": 0.42
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))

        elif self.path == "/verify":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            summary_sentences = req_json.get("summary_sentences", [])
            claims = []
            for s in summary_sentences:
                claims.append({
                    "claim": s,
                    "status": "Supported",
                    "badge": "supported",
                    "evidence_sources": [{"source_index": 0, "sentence": "Mock source sentence", "similarity_score": 0.95}],
                    "probabilities": {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}
                })
            response = {
                "claims": claims,
                "factuality_score": 1.0,
                "counts": {"supported": len(claims), "contradicted": 0, "uncertain": 0},
                "has_hallucinations": False,
                "device": "cuda",
                "verification_time_sec": 0.15
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def test_colab_client_url_normalization():
    print("[1/5] Testing ColabClient URL normalization...")
    client = ColabClient("https://abc-123.ngrok-free.app/")
    assert client.base_url == "https://abc-123.ngrok-free.app"

    client2 = ColabClient("abc-123.ngrok-free.app")
    assert client2.base_url == "https://abc-123.ngrok-free.app"
    print("  -> Passed URL normalization.")


def test_colab_client_invalid_url_health_and_retry():
    print("[2/5] Testing ColabClient error handling with unreachable URL...")
    client = ColabClient("https://invalid-nonexistent-colab-url-999.xyz", timeout_sec=1, max_retries=1)
    health = client.check_health()
    assert health["ok"] is False
    assert "error" in health
    print(f"  -> Health check reported expected failure: {health['error']}")

    try:
        client.summarize("Test document text.")
        assert False, "Expected ColabClientError was not raised!"
    except ColabClientError as e:
        print(f"  -> Summarize call raised expected ColabClientError: {e}")
    print("  -> Passed error handling on invalid URL.")


def test_pipeline_fallback_when_colab_url_is_wrong():
    print("[3/5] Testing automatic pipeline fallback when Colab URL is unreachable...")
    pipeline = ExplainableSummarizationPipeline(
        backend="colab_gpu",
        colab_url="https://invalid-nonexistent-colab-url-999.xyz",
        mock_mode=False
    )

    sample_text = (
        "The Federal Reserve held interest rates steady on Wednesday following a policy meeting. "
        "Chairman Jerome Powell stated that consumer spending has cooled and inflation has decreased."
    )

    res = pipeline.process(sample_text, extractive_sentences=2)
    abs_res = res["abstractive"]

    # Must NOT claim it ran on Colab GPU
    assert abs_res["backend"] != "colab_gpu", "Pipeline falsely claimed Colab GPU succeeded on unreachable URL!"
    assert "Colab GPU unreachable" in abs_res.get("failure_reason", "")
    assert abs_res["is_abstractive"] is True or abs_res["backend"] == "extractive_fallback"
    print(f"  -> Successfully triggered fallback to: {abs_res['backend']}")
    print(f"  -> Failure Reason: {abs_res['failure_reason']}")
    print(f"  -> Output Label: {abs_res['label']}")
    print("  -> Passed unreachable Colab URL fallback test.")


def test_colab_client_with_mock_server():
    print("[4/5] Testing ColabClient live communication with mock GPU server...")
    server = HTTPServer(("127.0.0.1", 8765), MockColabServer)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    try:
        client = ColabClient(base_url="http://127.0.0.1:8765")
        health = client.check_health()
        assert health["ok"] is True
        assert health["device"] == "cuda"
        assert health["gpu_name"] == "Tesla T4"
        print(f"  -> Health check: OK (GPU: {health['gpu_name']}, device: {health['device']})")

        sum_res = client.summarize("Sample news document", max_length=100)
        assert sum_res["backend"] == "colab_gpu"
        assert sum_res["device"] == "cuda"
        assert "lightning-fast" in sum_res["summary"]
        print(f"  -> Summarize returned: \"{sum_res['summary']}\" (Time: {sum_res['generation_time_sec']}s)")

        verify_res = client.verify(["Sample claim"], ["Sample source"])
        assert verify_res["backend"] == "colab_gpu"
        assert len(verify_res["claims"]) == 1
        print("  -> Verify returned valid claims.")
    finally:
        server.shutdown()
        server.server_close()
    print("  -> Passed mock Colab server test.")


def test_pipeline_with_colab_gpu_backend():
    print("[5/5] Testing full ExplainableSummarizationPipeline with Colab GPU backend...")
    server = HTTPServer(("127.0.0.1", 8766), MockColabServer)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    try:
        pipeline = ExplainableSummarizationPipeline(
            backend="colab_gpu",
            colab_url="http://127.0.0.1:8766",
            mock_mode=False
        )

        sample_article = (
            "The Federal Reserve held interest rates steady on Wednesday. "
            "Jerome Powell noted inflation has cooled significantly."
        )

        res = pipeline.process(sample_article)
        abs_res = res["abstractive"]

        assert abs_res["backend"] == "colab_gpu", f"Expected backend colab_gpu, got {abs_res['backend']}"
        assert abs_res["device"] == "cuda", f"Expected device cuda, got {abs_res['device']}"
        assert "Colab T4 GPU" in abs_res["label"]
        print(f"  -> Pipeline Result: Backend={abs_res['backend']}, Device={abs_res['device']}, Time={abs_res['generation_time_sec']}s")
        print(f"  -> Summary: \"{abs_res['summary']}\"")
    finally:
        server.shutdown()
        server.server_close()
    print("  -> Passed full pipeline Colab GPU integration test.")


if __name__ == "__main__":
    print("=== Running Colab GPU Integration & Fallback Tests ===\n")
    test_colab_client_url_normalization()
    test_colab_client_invalid_url_health_and_retry()
    test_pipeline_fallback_when_colab_url_is_wrong()
    test_colab_client_with_mock_server()
    test_pipeline_with_colab_gpu_backend()
    print("\n=== ALL COLAB INTEGRATION TESTS PASSED! ===")
