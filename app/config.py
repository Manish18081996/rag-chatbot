# Configuration variables (API keys, model config, etc.)

import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini-2024-07-18"