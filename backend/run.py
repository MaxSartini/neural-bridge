"""
Neural Bridge Backend Entry Point
"""

import os
import sys

# Solve Windows console Chinese character encoding issue: set UTF-8 encoding before all imports
if sys.platform == 'win32':
    # Set environment variable to ensure Python uses UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigure standard output stream to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def _check_services():
    """Verify LM Studio (LLM + embeddings) and Neo4j are reachable."""
    import httpx

    llm_url = Config.LLM_BASE_URL.rstrip('/') + '/models'
    embed_url = Config.EMBEDDING_BASE_URL.rstrip('/') + '/models'
    # Neo4j HTTP browser on 7474 (Bolt on 7687 is TCP-only)
    neo4j_http = Config.NEO4J_URI.replace('bolt://', 'http://').rsplit(':', 1)[0] + ':7474'

    services = [
        ("LM Studio LLM", llm_url),
        ("LM Studio Embeddings", embed_url),
        ("Neo4j", neo4j_http),
    ]
    failed = False
    for name, url in services:
        try:
            httpx.get(url, timeout=5).raise_for_status()
            print(f"  OK:   {name} ({url})")
        except Exception as e:
            print(f"  FAIL: {name} not reachable at {url} — {e}")
            failed = True
    if failed:
        sys.exit(1)


def main():
    """Main function"""
    # Validate configuration
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease check configuration in .env file")
        sys.exit(1)

    print("Checking services...")
    _check_services()

    # Create application
    app = create_app()

    # Get runtime configuration
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG

    # Start service
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

