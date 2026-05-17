"""Output module tests — behavior: saves report to markdown file with correct naming."""

import os
import tempfile
import pytest


class TestSaveReport:
    """save_report() writes content to a markdown file under output directory."""

    def test_creates_file_in_output_dir(self):
        from src.output import save_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            path = save_report("中国咖啡市场简报内容", topic="中国咖啡市场", output_dir=output_dir)

            assert os.path.exists(path)
            assert path.startswith(output_dir)
            with open(path) as f:
                assert f.read() == "中国咖啡市场简报内容"

    def test_filename_contains_topic_and_timestamp(self):
        from src.output import save_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            path = save_report("内容", topic="中国咖啡市场", output_dir=output_dir)

            basename = os.path.basename(path)
            assert basename.startswith("中国咖啡市场")
            assert basename.endswith(".md")
            # Should have timestamp pattern _YYYYMMDD_HHMMSS
            assert "_" in basename

    def test_sanitizes_special_characters_in_topic(self):
        from src.output import save_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            path = save_report("内容", topic='调研/报告:2024"测试"', output_dir=output_dir)

            basename = os.path.basename(path)
            assert "/" not in basename
            assert ":" not in basename
            assert '"' not in basename

    def test_truncates_long_topic(self):
        from src.output import save_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            long_topic = "A" * 200
            path = save_report("内容", topic=long_topic, output_dir=output_dir)

            basename = os.path.basename(path)
            # Topic part should be truncated (before timestamp)
            topic_part = basename.split("_")[0]
            assert len(topic_part) <= 50

    def test_creates_output_dir_if_missing(self):
        from src.output import save_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "output")
            path = save_report("内容", topic="测试", output_dir=output_dir)

            assert os.path.isdir(output_dir)
            assert os.path.exists(path)
