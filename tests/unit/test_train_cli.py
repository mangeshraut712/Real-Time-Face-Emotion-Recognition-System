"""train.py setup that runs before any training work."""

import logging

import pytest

import train


class _StopAfterSetup(Exception):
    pass


def test_main_creates_a_new_output_dir_before_opening_the_log(tmp_path, monkeypatch):
    output_dir = tmp_path / "runs" / "new-experiment"
    monkeypatch.setattr("sys.argv", ["train.py", "--output-dir", str(output_dir)])

    def stop(_use_gpu):
        raise _StopAfterSetup

    monkeypatch.setattr(train, "setup_gpu", stop)
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        with pytest.raises(_StopAfterSetup):
            train.main()
    finally:
        for handler in root.handlers[:]:
            if handler not in before:
                root.removeHandler(handler)
                handler.close()

    assert output_dir.is_dir()
