"""Budgeted, resumable OpenAI judging of the frozen held-out steering campaign.

Run pilot before run. Keeps the historical prompts verbatim and uses Responses
with explicit reasoning settings. Adapts Responses output to the existing
Batch-shaped importer without changing generation, scoring, or selection code.
Credentials never enter the results tree. No test grid is judged before every
validation arm has been scored and its signed magnitude frozen.
"""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import steering as s

MODEL = 'gpt-6-luna'
DELIVERY = {'model': MODEL, 'endpoint': '/v1/responses', 'reasoning_effort': 'low',
            'max_output_tokens': 1024, 'store': False, 'service_tier': 'default',
            'input_usd_per_million': .10, 'cached_input_usd_per_million': .01,
            'output_usd_per_million': .50,
            'pricing_source': 'https://developers.openai.com/api/docs/models/gpt-6-luna',
            'pricing_checked': '2026-09-30', 'rubric': 'historical prompts verbatim',
            'deduplication': 'exact request body including model and inference settings',
            'adapter': 'original Responses body archived; output_text mapped to choices[0].message.content'}


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def body_for(prompt):
    return {'model': MODEL, 'input': [{'role': 'user', 'content': prompt}],
            'reasoning': {'effort': DELIVERY['reasoning_effort']},
            'max_output_tokens': DELIVERY['max_output_tokens'],
            'store': False, 'service_tier': 'default'}


def valid_label(raw, rubric):
    if rubric == 'coherence':
        return int(raw.strip()) if re.fullmatch('[0-3]', raw.strip()) else None
    m = re.fullmatch(r'\s*COUNT:\s*(\d+)\s*\nNOTES:\s*\S[^\n]*\s*', raw)
    return int(m.group(1)) if m else None


def upper_cost(body):
    # UTF-8 byte count bounds token count conservatively; 512 tokens for framing.
    nbytes = len(json.dumps(body, ensure_ascii=False).encode())
    return ((nbytes + 512)*.125 + body['max_output_tokens']*.50)/1e6


def usage_cost(usage):
    details = usage.get('input_tokens_details') or {}
    cached = details.get('cached_tokens', 0)
    writes = details.get('cache_write_tokens', 0)
    return ((usage['input_tokens']-cached-writes)*.10 + cached*.01 + writes*.125 + usage['output_tokens']*.50)/1e6


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.out = self.root/'results/steering/judging'
        self.out.mkdir(exist_ok=True)
        self.hist = self.root/'historical/purified'
        self.arms = sorted((self.root/'results/steering/arms').iterdir())
        self.arms = [p for p in self.arms if (p/'validation/generations.jsonl').exists()]
        if len(self.arms) != 27:
            raise ValueError('Expected exactly 27 complete arms')
        s.frozen_json(self.out/'delivery_identity.json', DELIVERY)
        self.cache = self.out/'cache'
        self.cache.mkdir(exist_ok=True)
        self.ledger = self.out/'api_ledger.jsonl'
        self.spent = sum(usage_cost(r['usage']) if r.get('usage') else r['accounted_usd'] for r in s.rows(self.ledger))
        s.frozen_json(self.out/'accounting_policy.json', {'version': 2, 'cache_write_usd_per_million': .125, 'note': 'Recompute all ledger usage including pilot with explicit cache-write pricing; original pilot ledger retained.'})
        self.reserved = 0.
        self.api_calls = len(s.rows(self.ledger))
        self.completed = 0
        self.failed = 0
        self.next_request = 0.
        self.rate_lock = asyncio.Lock()
        self.stop = False
        from openai import AsyncOpenAI
        key = args.token_file.read_text().strip()
        if not key or args.token_file.stat().st_mode & 0o077:
            raise ValueError('Credential must be nonempty and mode 0600')
        self.client = AsyncOpenAI(api_key=key, max_retries=0, timeout=120)

    def status(self, stage, **extra):
        value = {'status': stage, 'model': MODEL, 'api_calls': self.api_calls,
                 'accounted_usd': self.spent, 'reserved_usd': self.reserved,
                 'budget_usd': self.args.budget, 'new_labels_this_invocation': self.completed,
                 'failed_requests': self.failed, 'updated_unix': time.time(), **extra}
        atomic(self.out/'progress.json', value)

    def exports(self, partition):
        entries, unique = [], {}
        for arm in self.arms:
            a = SimpleNamespace(workspace=arm, historical_root=self.hist,
                                partition=partition, judge_model=MODEL)
            s.export_judge_command(a)
            mapping = s.read(arm/partition/'judge_batch_index.json')
            for row in s.rows(arm/partition/'judge_batch_requests.jsonl'):
                prompt = row['body']['messages'][0]['content']
                body = body_for(prompt)
                key = s.canonical_hash(body)
                rubric = mapping[row['custom_id']]['rubric']
                entry = {'arm': arm.name, 'custom_id': row['custom_id'], 'cache_key': key, 'rubric': rubric}
                entries.append(entry)
                if key in unique and unique[key]['rubric'] != rubric:
                    raise ValueError('Cross-rubric cache collision')
                unique[key] = {'key': key, 'body': body, 'rubric': rubric}
        s.frozen_json(self.out/f'{partition}_request_manifest.json', {
            'partition': partition, 'logical_requests': len(entries),
            'unique_requests': len(unique), 'mapping': entries,
            'delivery_identity_sha256': s.digest(self.out/'delivery_identity.json')})
        return entries, unique

    def cached(self, request):
        path = self.cache/f"{request['key']}.json"
        if not path.exists():
            return False
        data = s.read(path)
        if data['request_body_sha256'] != request['key'] or valid_label(data['raw'], request['rubric']) is None:
            raise ValueError('Cached label identity or parse failure')
        return True

    async def judge(self, request):
        if self.cached(request):
            return
        body = request['body']
        bound = upper_cost(body)
        for attempt in range(10):
            async with self.rate_lock:
                await asyncio.sleep(max(0., self.next_request-time.monotonic()))
                self.next_request = time.monotonic()+self.args.request_interval
            if self.stop or self.spent+self.reserved+bound > self.args.budget:
                self.stop = True
                raise RuntimeError('Budget or fatal-error stop; resume only after review')
            self.reserved += bound
            started = time.time()
            try:
                response = await self.client.responses.create(**body)
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                accounted = 0. if status is not None and 400 <= status < 500 else bound
                self.spent += accounted
                self.reserved -= bound
                self.api_calls += 1
                # No exception message/header may expose a credential or request data.
                code = getattr(exc, 'code', None)
                headers = dict(getattr(getattr(exc, 'response', None), 'headers', {}) or {})
                rate_headers = {k:v for k,v in headers.items() if k.startswith('x-ratelimit-') or k == 'retry-after'}
                if status == 429 and code == 'rate_limit_exceeded':
                    try: cooldown = max(65., float(headers.get('retry-after', '65')))
                    except ValueError: cooldown = 65.
                    self.next_request = max(self.next_request, time.monotonic()+cooldown)
                s.append(self.ledger, {'request_body_sha256': request['key'],
                    'started_unix': started, 'error_type': type(exc).__name__,
                    'http_status': status, 'error_code': code, 'rate_headers': rate_headers,
                    'accounted_usd': accounted, 'accounting': 'zero for rejected 4xx; full bound for ambiguous transport/5xx'})
                if status in (400, 401, 403, 404) or code == 'insufficient_quota':
                    self.stop = True
                    raise RuntimeError(f'API rejected judging: {status}, {code}') from None
                if attempt == 9:
                    raise RuntimeError('Retry limit reached; output preserved') from None
                await asyncio.sleep(min(30, 3*2**attempt))
                continue
            data = response.model_dump(mode='json')
            raw = response.output_text
            usage = data.get('usage')
            cost = usage_cost(usage) if usage else bound
            self.spent += cost
            self.reserved -= bound
            self.api_calls += 1
            label = valid_label(raw, request['rubric']) if data['status']=='completed' else None
            event = {'request_body_sha256': request['key'], 'response_id': data['id'],
                     'started_unix': started, 'completed_unix': time.time(),
                     'usage': usage, 'accounted_usd': cost, 'status': data['status'],
                     'valid_label': label is not None, 'accounting': 'API usage at documented standard rates'}
            # Persist every response, including failures and incomplete reasoning.
            response_dir = self.out/'responses'
            response_dir.mkdir(exist_ok=True)
            atomic(response_dir/f"{data['id']}.json", data)
            s.append(self.ledger, event)
            if label is not None:
                s.frozen_json(self.cache/f"{request['key']}.json", {
                    'request_body_sha256': request['key'], 'raw': raw, 'label': label,
                    'rubric': request['rubric'], 'response_id': data['id'],
                    'response_sha256': s.digest(response_dir/f"{data['id']}.json"),
                    'model_returned': data['model'], 'usage': usage})
                self.completed += 1
                return
            self.failed += 1
            if attempt == 3:
                raise RuntimeError('Judge formatting or truncation failed four times; review required')

    async def score(self, unique, stage):
        pending = [v for k,v in sorted(unique.items()) if not self.cached(v)]
        queue = asyncio.Queue()
        for request in pending:
            queue.put_nowait(request)
        errors = []
        self.status(stage, pending_unique=len(pending), total_unique=len(unique))
        async def worker():
            while not queue.empty() and not self.stop:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await self.judge(item)
                except Exception as exc:
                    self.stop = True
                    errors.append(str(exc))
                finally:
                    queue.task_done()
                    self.status(stage, pending_unique=queue.qsize(), total_unique=len(unique))
                    if self.completed and self.completed%100 == 0:
                        print(json.dumps({'stage':stage,'new_labels':self.completed,'remaining':queue.qsize(),'usd':round(self.spent,4)}),flush=True)
        await asyncio.gather(*(worker() for _ in range(self.args.concurrency)))
        if errors:
            raise RuntimeError(errors[0])

    def import_partition(self, partition, entries):
        for arm in self.arms:
            output = arm/partition/'openai_outputs.jsonl'
            results = []
            for entry in entries:
                if entry['arm'] != arm.name:
                    continue
                c = s.read(self.cache/f"{entry['cache_key']}.json")
                results.append({'custom_id':entry['custom_id'], 'response': {'status_code':200,
                    'body': {'id':c['response_id'], 'model':c['model_returned'],
                        'choices':[{'message':{'content':c['raw']}}], 'usage':c['usage']}},
                    'adapter_provenance': {'request_body_sha256':entry['cache_key'],
                        'original_response_sha256':c['response_sha256'],
                        'delivery_identity_sha256':s.digest(self.out/'delivery_identity.json')}})
            text = ''.join(json.dumps(r,sort_keys=True)+'\n' for r in results)
            if output.exists() and output.read_text() != text:
                raise ValueError('Existing adapted outputs differ')
            output.write_text(text)
            s.import_judge_command(SimpleNamespace(workspace=arm, partition=partition,
                historical_root=self.hist, batch_output=output))

    async def run(self):
        entries, unique = self.exports('validation')
        if self.args.stage == 'pilot':
            # Fixed spread of validation prompts; no test content inspected.
            candidates = sorted(unique.values(), key=lambda r:(r['rubric'],len(json.dumps(r['body'])),r['key']))
            chosen = [candidates[round(i*(len(candidates)-1)/23)] for i in range(24)]
            await self.score({r['key']:r for r in chosen},'pilot')
            s.frozen_json(self.out/'pilot_receipt.json', {'request_keys':[r['key'] for r in chosen],
                'labels':[s.read(self.cache/f"{r['key']}.json") for r in chosen],
                'delivery_identity_sha256':s.digest(self.out/'delivery_identity.json')})
            self.status('pilot_complete', pilot_requests=24)
            return
        if not (self.out/'pilot_receipt.json').exists():
            raise ValueError('Run and inspect pilot before full judging')
        await self.score(unique,'validation')
        self.import_partition('validation', entries)
        # Finish and freeze all validation selections before releasing any test rows.
        for arm in self.arms:
            s.select_command(SimpleNamespace(workspace=arm))
        s.frozen_json(self.out/'validation_selection_gate.json', {
            'selections':{arm.name:s.digest(arm/'selection.json') for arm in self.arms},
            'policy':'All 27 validation selections frozen before test export or judging'})
        for arm in self.arms:
            s.unlock_command(SimpleNamespace(workspace=arm,historical_root=self.hist))
        entries, unique = self.exports('test')
        await self.score(unique,'test')
        self.import_partition('test',entries)
        for arm in self.arms:
            s.summarize_command(SimpleNamespace(workspace=arm))
        self.status('complete', arms=27, test_questions_per_arm=100,
                    validation_questions_per_arm=20, test_summaries=27)
        self.args.token_file.unlink(missing_ok=True)
        print(json.dumps({'status':'complete','api_calls':self.api_calls,'accounted_usd':self.spent}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/backtracking'))
    p.add_argument('--token-file',type=Path,default=Path('/workspace/.tokens/backtracking_openai_token'))
    p.add_argument('--stage',choices=['pilot','run'],required=True)
    p.add_argument('--budget',type=float,default=9.)
    p.add_argument('--concurrency',type=int,default=8)
    p.add_argument('--request-interval',type=float,default=.35)
    args=p.parse_args()
    if not 0 < args.budget <= 9.:
        raise ValueError('Budget must not exceed authorized $9 allocation')
    out=args.root/'results/steering/judging'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=Campaign(args)
        try:
            asyncio.run(campaign.run())
        except Exception as exc:
            campaign.status('blocked',error=str(exc))
            raise

if __name__=='__main__':
    main()
