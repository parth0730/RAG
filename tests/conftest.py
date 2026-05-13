import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Register lightweight vertexai stubs so tests that patch GCP SDK paths
# (e.g. "vertexai.language_models.TextEmbeddingModel.from_pretrained")
# work without the real google-cloud-aiplatform package installed.
# ---------------------------------------------------------------------------
if "vertexai" not in sys.modules:
    from tests import _vertexai_stub
    from tests._vertexai_stub import language_models, generative_models

    sys.modules["vertexai"] = _vertexai_stub
    sys.modules["vertexai.language_models"] = language_models
    sys.modules["vertexai.generative_models"] = generative_models

