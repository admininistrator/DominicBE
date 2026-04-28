import json
import app.main as main

resp = main.health()
payload = json.loads(resp.body.decode('utf-8'))
print(resp.body.decode('utf-8'))
print('status_code=', resp.status_code)
for name in ('postgres', 'minio', 'qdrant'):
    print(f"dependencies.{name}.ok=", payload['dependencies'][name]['ok'])
