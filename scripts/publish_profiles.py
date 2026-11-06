"""Publish a completed live verification report into active profiles without new API calls.

python -m scripts.publish_profiles --report data/results/provider_verification.json
"""

import argparse
from pathlib import Path

from app.llm.registry import ModelRegistry
from app.services.evaluation_service import EvaluationService
from app.services.profile_publication import publish_verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = publish_verification(
        args.report,
        EvaluationService(registry=ModelRegistry.from_config()),
    )
    print(
        f"Published {len(result.profiles)} model/task profiles from "
        f"{len(result.source_files)} benchmark files. No provider calls made."
    )


if __name__ == "__main__":
    main()
