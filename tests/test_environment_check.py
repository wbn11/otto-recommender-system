from tools.check_environment import collect_environment


def test_environment_check_returns_structured_report_without_requiring_gpu():
    report, failures = collect_environment(require_gpu=False)

    assert "python" in report
    assert "modules" in report
    assert report["failures"] == failures
    assert "torch" in report["modules"]
    assert "faiss" in report["modules"]
