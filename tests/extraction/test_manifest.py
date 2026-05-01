"""Tests for explainshell.extraction.manifest."""

import os
import tempfile
import unittest

from explainshell.extraction.manifest import (
    BatchManifest,
    FileBatchManifestWriter,
)


def _read_manifest(path: str) -> BatchManifest:
    with open(path) as f:
        return BatchManifest.model_validate_json(f.read())


class TestFileBatchManifestWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.tmpdir, "batch-manifest.json")

    def tearDown(self) -> None:
        if os.path.exists(self.manifest_path):
            os.unlink(self.manifest_path)
        tmp = self.manifest_path + ".tmp"
        if os.path.exists(tmp):
            os.unlink(tmp)
        os.rmdir(self.tmpdir)

    def test_writes_valid_json(self) -> None:
        m = FileBatchManifestWriter(
            self.manifest_path, model="openai/gpt-5-mini", batch_size=50
        )
        m.set_total_batches(2)
        m.record_batch(
            batch_idx=1,
            batch_id="batch_abc",
            status="completed",
            files=["/path/a.gz", "/path/b.gz"],
        )

        data = _read_manifest(self.manifest_path)
        self.assertEqual(data.version, 1)
        self.assertEqual(data.model, "openai/gpt-5-mini")
        self.assertEqual(data.batch_size, 50)
        self.assertEqual(data.total_batches, 2)
        self.assertEqual(len(data.batches), 1)
        self.assertEqual(data.batches[0].batch_idx, 1)
        self.assertEqual(data.batches[0].batch_id, "batch_abc")
        self.assertEqual(data.batches[0].status, "completed")
        self.assertIsNone(data.batches[0].error)
        self.assertEqual(data.batches[0].files, ["/path/a.gz", "/path/b.gz"])

    def test_records_incrementally(self) -> None:
        m = FileBatchManifestWriter(
            self.manifest_path, model="openai/gpt-5-mini", batch_size=50
        )
        m.set_total_batches(3)

        m.record_batch(batch_idx=1, batch_id="b1", status="completed", files=["/a.gz"])
        data1 = _read_manifest(self.manifest_path)
        self.assertEqual(len(data1.batches), 1)

        m.record_batch(
            batch_idx=2,
            batch_id="b2",
            status="failed",
            files=["/b.gz"],
            error="expired",
        )
        data2 = _read_manifest(self.manifest_path)
        self.assertEqual(len(data2.batches), 2)

        m.record_batch(batch_idx=3, batch_id="b3", status="completed", files=["/c.gz"])
        data3 = _read_manifest(self.manifest_path)
        self.assertEqual(len(data3.batches), 3)

    def test_failed_batch_recorded(self) -> None:
        m = FileBatchManifestWriter(
            self.manifest_path, model="openai/gpt-5-mini", batch_size=50
        )
        m.set_total_batches(1)
        m.record_batch(
            batch_idx=1,
            batch_id="batch_fail",
            status="failed",
            files=["/x.gz"],
            error="Batch job expired",
        )

        data = _read_manifest(self.manifest_path)
        entry = data.batches[0]
        self.assertEqual(entry.status, "failed")
        self.assertEqual(entry.error, "Batch job expired")

    def test_null_batch_id_for_submit_failure(self) -> None:
        m = FileBatchManifestWriter(
            self.manifest_path, model="openai/gpt-5-mini", batch_size=50
        )
        m.set_total_batches(1)
        m.record_batch(
            batch_idx=1,
            batch_id=None,
            status="failed",
            files=["/x.gz"],
            error="submit failed",
        )

        data = _read_manifest(self.manifest_path)
        self.assertIsNone(data.batches[0].batch_id)

    def test_atomic_write_no_tmp_left(self) -> None:
        """After a successful write, no .tmp file should remain."""
        m = FileBatchManifestWriter(
            self.manifest_path, model="openai/gpt-5-mini", batch_size=50
        )
        m.set_total_batches(1)
        m.record_batch(batch_idx=1, batch_id="b1", status="completed", files=["/a.gz"])

        self.assertTrue(os.path.exists(self.manifest_path))
        self.assertFalse(os.path.exists(self.manifest_path + ".tmp"))

    def test_creates_parent_directory(self) -> None:
        """FileBatchManifestWriter creates the parent directory if it doesn't exist."""
        import shutil

        nested_dir = tempfile.mkdtemp()
        os.rmdir(nested_dir)  # remove so FileBatchManifestWriter has to create it
        nested = os.path.join(nested_dir, "sub", "batch-manifest.json")
        m = FileBatchManifestWriter(nested, model="openai/gpt-5-mini", batch_size=50)
        m.set_total_batches(1)
        m.record_batch(batch_idx=1, batch_id="b1", status="completed", files=["/a.gz"])

        self.assertTrue(os.path.isfile(nested))
        shutil.rmtree(nested_dir)

    def test_record_replaces_existing_entry(self) -> None:
        """A second record_batch with the same batch_idx replaces the first."""
        m = FileBatchManifestWriter(
            self.manifest_path, model="openai/gpt-5-mini", batch_size=50
        )
        m.set_total_batches(1)
        m.record_batch(batch_idx=1, batch_id="b1", status="submitted", files=["/a.gz"])

        data1 = _read_manifest(self.manifest_path)
        self.assertEqual(len(data1.batches), 1)
        self.assertEqual(data1.batches[0].status, "submitted")

        m.record_batch(
            batch_idx=1,
            batch_id="b1",
            status="completed",
            files=["/a.gz"],
        )

        data2 = _read_manifest(self.manifest_path)
        self.assertEqual(len(data2.batches), 1)
        self.assertEqual(data2.batches[0].status, "completed")


if __name__ == "__main__":
    unittest.main()
