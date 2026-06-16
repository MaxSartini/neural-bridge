"""
Scrape API — ingest social media screenshots into a knowledge graph.
"""

import os
import shutil
import tempfile

from flask import Blueprint, current_app, jsonify, request

from ..config import Config
from ..services.screenshot_processor import ScreenshotProcessor
from ..utils.logger import get_logger

scrape_bp = Blueprint('scrape', __name__)
logger = get_logger('neural_bridge.scrape')


@scrape_bp.route('/screenshots', methods=['POST'])
def process_screenshots():
    graph_id = request.form.get('graph_id')
    topic_context = request.form.get('topic_context', '')

    if not graph_id:
        return jsonify({"success": False, "error": "graph_id required"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"success": False, "error": "No files provided"}), 400

    allowed = Config.ALLOWED_IMAGE_EXTENSIONS
    valid_files = [
        f for f in files
        if f.filename and f.filename.rsplit('.', 1)[-1].lower() in allowed
    ]
    if not valid_files:
        return jsonify({"success": False, "error": "No valid image files"}), 400

    storage = current_app.extensions.get('neo4j_storage')
    if not storage:
        return jsonify({"success": False, "error": "Storage not initialised"}), 503

    tmp_dir = tempfile.mkdtemp()
    image_paths = []
    try:
        for f in valid_files:
            safe_name = os.path.basename(f.filename)
            path = os.path.join(tmp_dir, safe_name)
            f.save(path)
            image_paths.append(path)

        processor = ScreenshotProcessor()
        result = processor.process_batch(image_paths, graph_id, topic_context, storage)
        return jsonify({"success": True, **result})

    except Exception as e:
        logger.error(f"Screenshot processing failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
