import sys
from ingestion.pipelines.ingestion_pipeline import run_ingestion

file_path = sys.argv[1]
document_id = sys.argv[2]

chunks = run_ingestion(file_path, document_id)
print(f"Done. {len(chunks)} chunks stored.")