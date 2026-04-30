from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_manager_log_files(tmp_path):
    """Prevent CLI tests from writing log files into the real logs/ directory
    or leaking the manager's logger-level configuration into other tests.

    The run_dir is redirected to a pytest tmp_path so report/manifest
    writes still work but nothing accumulates in the repo. Console-logging
    setup is suppressed so it doesn't lock the explainshell logger level
    and break tests that rely on caplog at DEBUG.
    """
    with (
        patch("explainshell.manager._attach_run_log", return_value=str(tmp_path)),
        patch("explainshell.manager._setup_console_logging"),
    ):
        yield
