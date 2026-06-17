"""
Neural Bridge Backend - Flask Application Factory
"""

import os
import warnings

# Suppress multiprocessing resource_tracker warnings (from third-party libraries like transformers)
# Must be set before all other imports
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

def create_app(config_class=None):
    """Flask application factory function"""
    from flask import Flask, request
    from flask_cors import CORS

    from .config import Config
    from .utils.logger import setup_logger, get_logger

    config_class = config_class or Config
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure JSON encoding: ensure Chinese displays directly (not as \uXXXX)
    # Flask >= 2.3 uses app.json.ensure_ascii, older versions use JSON_AS_ASCII config
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False

    # Setup logging
    logger = setup_logger('neural_bridge')

    # Only print startup info in reloader subprocess (avoid printing twice in debug mode)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("Neural Bridge Backend starting...")
        logger.info("=" * 50)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Request logging middleware
    @app.before_request
    def log_request():
        logger = get_logger('neural_bridge.request')
        logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Request body: {request.get_json(silent=True)}")

    @app.after_request
    def log_response(response):
        logger = get_logger('neural_bridge.request')
        logger.debug(f"Response: {response.status_code}")
        return response

    # Register blueprints
    from .api import neuro_viewer_bp
    app.register_blueprint(neuro_viewer_bp, url_prefix='/api/neuro-viewer')

    # Health check
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'Neural Bridge Backend'}

    if should_log_startup:
        logger.info("Neural Bridge Backend startup complete")

    return app
