"""
Client module to connect the local application to the remote Google Colab GPU server.
Provides health checks, summarization, and claim verification over HTTP with retries and timeouts.
"""
import time
import requests
from typing import Dict, Any, List, Optional


class ColabClientError(Exception):
    """Raised when communication with Colab backend fails."""
    pass


class ColabClient:
    """
    Communicates with the Google Colab GPU FastAPI server via ngrok tunnel.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_sec: int = 35,
        max_retries: int = 2
    ):
        self.base_url = self._normalize_url(base_url) if base_url else ""
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        # Header to bypass ngrok's free tier browser warning interstitial page
        self.headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "DocAssist-Streamlit-Client/1.0"
        }

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Strips whitespace and trailing slashes."""
        cleaned = url.strip()
        if cleaned.endswith("/"):
            cleaned = cleaned[:-1]
        if cleaned and not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = "https://" + cleaned
        return cleaned

    def set_url(self, url: str):
        """Updates the active Colab base URL."""
        self.base_url = self._normalize_url(url)

    def is_configured(self) -> bool:
        """Returns True if a base URL has been set."""
        return bool(self.base_url)

    def check_health(self) -> Dict[str, Any]:
        """
        Pings /health endpoint on Colab server to test connectivity and GPU status.
        Returns dict with status, device, and GPU name, or error details.
        """
        if not self.base_url:
            return {
                "ok": False,
                "error": "Colab URL is not configured. Please enter your ngrok URL in the sidebar."
            }

        endpoint = f"{self.base_url}/health"
        try:
            resp = requests.get(endpoint, headers=self.headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "ok": True,
                    "status": "connected",
                    "device": data.get("device", "cuda"),
                    "gpu_name": data.get("gpu_name", "T4 GPU"),
                    "models_loaded": data.get("models_loaded", []),
                    "raw": data
                }
            else:
                return {
                    "ok": False,
                    "error": f"Colab returned HTTP {resp.status_code}: {resp.text[:200]}"
                }
        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "error": "Connection timed out (Colab server did not respond within 8s)."
            }
        except requests.exceptions.ConnectionError:
            return {
                "ok": False,
                "error": "Could not connect to Colab URL (Tunnel may be closed or offline)."
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Health check failed: {type(e).__name__} ({str(e)})"
            }

    def summarize(
        self,
        text: str,
        max_length: int = 140,
        min_length: int = 40,
        num_beams: int = 4
    ) -> Dict[str, Any]:
        """
        Sends text to Colab /summarize endpoint for fast GPU inference.
        Includes automatic retry logic.
        """
        if not self.base_url:
            raise ColabClientError("Colab base URL not provided.")

        endpoint = f"{self.base_url}/summarize"
        payload = {
            "text": text,
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start_req = time.time()
                resp = requests.post(
                    endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout_sec
                )
                if resp.status_code == 200:
                    data = resp.json()
                    data["backend"] = "colab_gpu"
                    data["roundtrip_sec"] = round(time.time() - start_req, 3)
                    return data
                else:
                    raise ColabClientError(
                        f"Colab GPU server returned status {resp.status_code}: {resp.text[:200]}"
                    )
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue

        raise ColabClientError(f"Colab GPU summarization request failed after {self.max_retries} attempts: {last_error}")

    def verify(
        self,
        summary_sentences: List[str],
        source_sentences: List[str],
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Sends claims and source sentences to Colab /verify endpoint for GPU NLI.
        """
        if not self.base_url:
            raise ColabClientError("Colab base URL not provided.")

        endpoint = f"{self.base_url}/verify"
        payload = {
            "summary_sentences": summary_sentences,
            "source_sentences": source_sentences,
            "top_k": top_k
        }

        try:
            resp = requests.post(
                endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout_sec
            )
            if resp.status_code == 200:
                data = resp.json()
                data["backend"] = "colab_gpu"
                
                # Guarantee consistent claim schema for downstream rendering
                for c in data.get("claims", []):
                    # 1. Guarantee confidence
                    if "confidence" not in c or c["confidence"] is None:
                        probs = c.get("probabilities", {})
                        if probs:
                            c["confidence"] = max(probs.values())
                        else:
                            c["confidence"] = 0.85
                    c["confidence"] = round(float(c["confidence"]), 4)

                    # 2. Guarantee badge color class (green/red/yellow)
                    badge = str(c.get("badge", "")).lower()
                    if badge in ["supported", "green"]:
                        c["badge"] = "green"
                    elif badge in ["contradicted", "red"]:
                        c["badge"] = "red"
                    else:
                        c["badge"] = "yellow"

                    # 3. Guarantee normalized status
                    if "status" in c:
                        c["status"] = str(c["status"]).lower()
                    else:
                        c["status"] = "supported" if c["badge"] == "green" else ("contradicted" if c["badge"] == "red" else "uncertain")

                return data
            else:
                raise ColabClientError(f"Colab verification returned status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            raise ColabClientError(f"Colab verification request failed: {e}")
