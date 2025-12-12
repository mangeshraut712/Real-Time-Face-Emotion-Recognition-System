#!/usr/bin/env python3
"""
Web Interface for Emotion Recognition.

Launch the Flask web server for browser-based emotion detection.

Usage:
    python run_web.py
    python run_web.py --port 8080
    python run_web.py --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Web interface for real-time emotion recognition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_web.py                    # Start on port 5000
    python run_web.py --port 8080        # Start on port 8080
    python run_web.py --debug            # Enable debug mode
    python run_web.py --auto-start       # Auto-start camera on launch
        """,
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )
    
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically start the camera on launch (default: off)",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_false",
        dest="auto_start",
        help="Do not auto-start the camera (default)",
    )
    parser.set_defaults(auto_start=False)
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    
    # Check for Flask
    try:
        from flask import Flask
    except ImportError:
        logger.error("Flask is not installed. Install it with: pip install flask")
        sys.exit(1)
    
    from src.web.app import run_server
    
    logger.info(f"Starting web server at http://{args.host}:{args.port}")
    logger.info("Open this URL in your browser to use the web interface")
    
    try:
        run_server(
            host=args.host,
            port=args.port,
            debug=args.debug,
            auto_start=args.auto_start,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
