"""Behavioral checks for judgment parsing, billing, caching and test gating."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import judge_openai as j


class JudgeTests(unittest.TestCase):
    def test_rejects_ambiguous_or_invalid_rubric_responses(self):
        for text in ('COUNT: -1\nNOTES: unknown', 'COUNT: 2', '2', 'COUNT: 2\nNOTES: ok\nCOUNT: 3'):
            self.assertIsNone(j.valid_label(text,'backtracking'))
        self.assertEqual(j.valid_label('COUNT: 2\nNOTES: Recomputed twice.','backtracking'),2)
        for text in ('4','-1','Grade: 2','2 and 3'):
            self.assertIsNone(j.valid_label(text,'coherence'))
        self.assertEqual(j.valid_label(' 2\n','coherence'),2)

    def test_usage_counts_reasoning_output_and_cached_discount(self):
        self.assertAlmostEqual(j.usage_cost({'input_tokens':1000000,
            'input_tokens_details':{'cached_tokens':500000},'output_tokens':1000000}), .555)
        self.assertAlmostEqual(j.usage_cost({'input_tokens':1000000, 'input_tokens_details':{'cache_write_tokens':1000000}, 'output_tokens':0}), .125)
        self.assertGreater(j.upper_cost(j.body_for('hello')),1024*.5/1e6)

    def test_budget_stops_before_dispatch(self):
        c=j.Campaign.__new__(j.Campaign)
        c.cached=lambda req:False
        c.rate_lock=asyncio.Lock()
        c.next_request=0
        c.args=SimpleNamespace(request_interval=0,budget=.001)
        c.spent=.0009
        c.reserved=0
        c.stop=False
        with self.assertRaisesRegex(RuntimeError,'Budget'):
            asyncio.run(c.judge({'body':j.body_for('hello')}))
        self.assertTrue(c.stop)

    def test_no_test_export_until_every_validation_selection_is_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            c=j.Campaign.__new__(j.Campaign)
            c.out=Path(d)
            (c.out/'pilot_receipt.json').write_text('{}')
            c.args=SimpleNamespace(stage='run')
            c.arms=[Path(d)/'a',Path(d)/'b']
            c.hist=Path(d)
            calls=[]
            c.exports=lambda partition:(calls.append('export:'+partition) or ([],{}))
            async def score(unique,stage): calls.append('score:'+stage)
            c.score=score
            c.import_partition=lambda partition,entries:calls.append('import:'+partition)
            def fail_second_selection(args):
                calls.append('select:'+args.workspace.name)
                if args.workspace.name=='b': raise ValueError('incomplete validation')
            with patch.object(j.s,'select_command',side_effect=fail_second_selection):
                with self.assertRaisesRegex(ValueError,'incomplete validation'):
                    asyncio.run(c.run())
            self.assertNotIn('export:test',calls)
            self.assertEqual(calls[-1],'select:b')

if __name__=='__main__': unittest.main()
