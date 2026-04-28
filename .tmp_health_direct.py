import json
import app.main as main

resp = main.health()
body = json.loads(resp.body.decode('utf-8'))
print('status_code=', resp.status_code)
print('top_level_keys=', sorted(body.keys()))
for name in ('postgres', 'minio', 'qdrant'):
    dep = body.get(name, {})
    print(f'{name}_ok=', dep.get('ok'))
print('would_return_200=', resp.status_code == 200)
print('all_dependencies_ok=', all(body.get(name, {}).get('ok') is True for name in ('postgres', 'minio', 'qdrant')))
