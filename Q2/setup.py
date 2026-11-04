
import os
import sys
import json
import secrets
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup(output_dir: str):
    """Generate and save the watermark secret key."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate 32-byte random key
    key = secrets.token_bytes(32)

    key_path = os.path.join(output_dir, "watermark_key.bin")
    with open(key_path, "wb") as f:
        f.write(key)

    logger.info(f"Secret key generated and saved to: {key_path}")
    logger.info(f"Key (hex): {key.hex()}")

    # Also save watermark parameters for documentation
    params = {
        "gamma": 0.5,
        "delta": 2.0,
        "context_window": 4,
        "key_size_bytes": 32,
        "algorithm": "KGW-hard-greenlist",
        "reference": "Kirchenbauer et al. (2023) arXiv:2301.10226",
    }
    with open(os.path.join(output_dir, "watermark_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    logger.info("Setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Setup")
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    setup(args.output_dir)
