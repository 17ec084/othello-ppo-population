from pathlib import Path
import tempfile
import unittest
from checkpoint_archive import archive_checkpoint,verify_archive


class ArchiveTest(unittest.TestCase):
    def test_replace_and_restore_are_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base.pt';target=root/'initial.pt'
            original=bytearray(b'base-data'*20000);original[50:55]=b'other';original[-20:-15]=b'final'
            base.write_bytes(b'base-data'*20000);target.write_bytes(original)
            result=archive_checkpoint(base,target,replace=True)
            self.assertFalse(target.exists())
            self.assertLess(result['archive_bytes'],len(original)//10)
            verify_archive(result['archive'],target)
            self.assertEqual(target.read_bytes(),original)
            self.assertEqual(base.read_bytes(),b'base-data'*20000)

    def test_changed_reference_is_rejected_without_creating_wrong_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base.pt';target=root/'initial.pt'
            base.write_bytes(b'a'*1000);target.write_bytes(b'a'*999+b'b')
            result=archive_checkpoint(base,target,replace=True)
            base.write_bytes(b'c'*1000)
            with self.assertRaisesRegex(ValueError,'hash differs'):verify_archive(result['archive'],target)
            self.assertFalse(target.exists())
            self.assertFalse((root/'initial.pt.restore.tmp').exists())
            self.assertTrue(Path(result['archive']).exists())


if __name__=='__main__':unittest.main(verbosity=2)
