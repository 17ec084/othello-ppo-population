import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from monitor import active_comparison,comparison_payload
from individual_train import save_checkpoint
import torch


class ComparisonTest(unittest.TestCase):
    def test_deleted_comparison_manifest_is_ignored_for_a_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'latest_comparison.txt').write_text(str(root/'deleted'),encoding='utf8')
            (root/'latest_run.txt').write_text(str(root/'new'),encoding='utf8')
            with patch('monitor.ROOT',root):self.assertIsNone(active_comparison())
    def test_algorithm_labels_are_available_before_both_methods_initialize(self):
        with tempfile.TemporaryDirectory() as d:
            manifest={'mode':'algorithm_comparison','models':20,'eval_games':128,
                      'runs':{m:{'path':str(Path(d)/m),'pid':0,'algorithm':m,'label':m}
                              for m in ('reinforce','ppo')}}
            result=comparison_payload(manifest)
            self.assertEqual(result['mode'],'algorithm_comparison')
            self.assertEqual(result['runs']['ppo']['label'],'ppo')
            self.assertIsNone(result['matched_step'])
    def test_failed_checkpoint_keeps_previous_snapshot_and_removes_incomplete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            save_checkpoint(path,{'step':0,'weight':torch.ones(3)})
            def disk_full(data,temporary):
                Path(temporary).write_bytes(b'partial checkpoint')
                raise OSError(28,'No space left on device')
            with patch('individual_train.torch.save',side_effect=disk_full),self.assertRaises(OSError):
                save_checkpoint(path,{'step':10})
            self.assertEqual(torch.load(path,weights_only=False)['step'],0)
            self.assertFalse(path.with_name('latest.pt.tmp').exists())

    def test_comparison_uses_latest_common_evaluation_step_and_keeps_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs={}
            for mode,steps in (('margin',[0,25,50]),('signed_win',[0,25])):
                out=root/mode;out.mkdir();runs[mode]={'path':str(out),'pid':0}
                (out/'progress.json').write_text(json.dumps({'status':'completed','models':{'transformer':{}},
                   'step':steps[-1],'controller':{'learner':11},'unfreezing':{'percent':.1}}),encoding='utf-8')
                rows=[{'step':step,'aggregate':{'random':{'games':2560,'win_rate':.5+step/1000},
                      'position':{'games':2560,'win_rate':.2}}} for step in steps]
                (out/'evaluations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            manifest={'runs':runs,'eval_games':128,'initial_weights_verified':True}
            result=comparison_payload(manifest)
            self.assertEqual(result['matched_step'],25)
            for run in result['runs'].values():
                self.assertEqual(run['matched_evaluation']['step'],25)
                self.assertEqual(run['matched_evaluation']['aggregate']['random']['games'],2560)
                self.assertEqual(run['baseline_evaluation']['step'],0)
            (root/'signed_win'/'evaluations.jsonl').write_text('',encoding='utf-8')
            self.assertIsNone(comparison_payload(manifest)['matched_step'])

    def test_old_comparison_pointer_does_not_replace_a_new_unrelated_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'pair';folder.mkdir()
            manifest={'runs':{'margin':{'path':str(folder/'margin')},'signed_win':{'path':str(folder/'signed_win')}}}
            (folder/'comparison.json').write_text(json.dumps(manifest),encoding='utf-8')
            (root/'latest_comparison.txt').write_text(str(folder),encoding='utf-8')
            (root/'latest_run.txt').write_text(str(root/'unrelated'),encoding='utf-8')
            with patch('monitor.ROOT',root):
                self.assertIsNone(active_comparison())
                (root/'latest_run.txt').write_text(str(folder/'margin'),encoding='utf-8')
                self.assertEqual(active_comparison(),manifest)


if __name__=='__main__':unittest.main(verbosity=2)
