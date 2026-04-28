$ErrorActionPreference = "Stop"
Write-Host "PWD=$(Get-Location)"
Get-ChildItem app -Recurse -File | Select-String -Pattern "def ingest_document|def delete_document_storage|class .*Qdrant|Minio|qdrant|knowledge_documents|document_id|owner" | Select-Object -First 200 Path,LineNumber,Line | Format-Table -AutoSize | Out-String -Width 260
