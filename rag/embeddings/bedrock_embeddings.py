import json
import logging
from typing import List
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def embed_query_bedrock(text: str) -> List[float]:
    import boto3
    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]
