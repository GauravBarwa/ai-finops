"""Keep test collection offline; pricing comes from the repository registry."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
