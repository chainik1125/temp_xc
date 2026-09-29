"""Small read-only status snapshot, without reading generated answers."""
import json
from pathlib import Path

root=Path('/workspace/backtracking/results')
snapshot={'training':{},'steering':{}}
for path in sorted((root/'cells').glob('*/progress.json')):
    data=json.loads(path.read_text())
    snapshot['training'][path.parent.name]={k:data.get(k) for k in ('status','step','n_steps','elapsed_seconds')}
for path in sorted((root/'steering/arms').glob('*/*/progress.json')):
    data=json.loads(path.read_text())
    snapshot['steering'][str(path.parent.relative_to(root/'steering/arms'))]={k:data.get(k) for k in ('status','completed_records','expected_records')}
for name in ('queue_state.json','steering/queue_state.json','completion.json'):
    path=root/name
    if path.exists():
        snapshot[name]=json.loads(path.read_text())
print(json.dumps(snapshot,indent=2,sort_keys=True))
