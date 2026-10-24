import pytest
from observability.tracing import setup_tracing

@pytest.fixture(autouse=True, scope="session")
def redirect_traces_for_testing(tmp_path_factory):
    """Guarantees that all tests write traces only to a pytest temp directory and never to eval/results/traces/traces.jsonl."""
    temp_dir = tmp_path_factory.mktemp("test_traces")
    test_trace_file = str(temp_dir / "pytest_traces.jsonl")
    setup_tracing(file_path=test_trace_file, force_reinit=True)
