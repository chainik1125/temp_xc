"""Exercise the real inherited rubrics and draft-to-real export without an API."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from steering import JUDGE_MODEL_PLACEHOLDER, export_judge_command

def main():
    with tempfile.TemporaryDirectory() as directory:
        work=Path(directory)
        part=work/'validation'
        part.mkdir()
        (part/'generation_identity.json').write_text(json.dumps(dict(qids=['synthetic-q'],magnitudes=[0,4])))
        records=[dict(question_id='synthetic-q',magnitude=m,problem_prompt='A synthetic smoke fixture.',
                      continuation='This is a synthetic fixture, not an experiment.',generation_sha256=f'synthetic-{m}') for m in (0,4)]
        (part/'generations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        args=SimpleNamespace(historical_root=Path('/workspace/backtracking/historical/purified'),
                             workspace=work,partition='validation',judge_model=JUDGE_MODEL_PLACEHOLDER)
        export_judge_command(args)
        assert not (part/'judge_identity.json').exists()
        assert json.loads((part/'judge_template/judge_batch_manifest.json').read_text())['request_count']==4
        args.judge_model='synthetic-never-submitted-model'
        export_judge_command(args)
        export_judge_command(args)
        assert json.loads((part/'judge_batch_manifest.json').read_text())['request_count']==4
        assert len((part/'judge_batch_requests.jsonl').read_text().splitlines())==4
    print('Real-rubric draft export, later model binding, and idempotence passed; zero API calls.')

if __name__=='__main__':
    main()
