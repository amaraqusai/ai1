"""
Command Line Interface entry point.
"""
import argparse
import sys
import os
import logging

# Ensure src is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.config import setup_logging
from src.sdk.sdk import SDK

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Signal Frequency Extraction")
    parser.add_argument("--mode", choices=["generate", "train", "evaluate", "all", "ui"], required=True)
    parser.add_argument("--model", choices=["mlp", "rnn", "lstm", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8050)
    return parser

def main():
    setup_logging()
    logger = logging.getLogger("freq_extractor.main")
    
    parser = build_parser()
    args = parser.parse_args()
    
    # Simple hack to inject seed override into global setup if needed,
    # but the PRD says config file provides them and --seed is an override.
    from src.shared.config import get_setup
    config = get_setup()
    config["data"]["base_seed"] = args.seed
    
    sdk = SDK()
    
    try:
        if args.mode == "generate":
            sdk.generate_data()
        elif args.mode == "train":
            sdk.train_model(args.model)
        elif args.mode == "evaluate":
            sdk.evaluate_model(args.model)
        elif args.mode == "all":
            sdk.run_all()
        elif args.mode == "ui":
            # For AI Studio environments, default port 3000 mapping happens via Nginx,
            # but dash must bind internally. Wait, the docs say the platform only exposes port 3000.
            # But PRD FR-13 says: `uv run python src/main.py --mode ui` opens on http://localhost:8050
            # If the architecture requires port 3000 for visibility in this sandboxed env:
            # I will use args.port, but warn user if not 3000.
            sdk.launch_ui(port=args.port)
            
    except KeyboardInterrupt:
        logger.info("Graceful interrupt received. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
