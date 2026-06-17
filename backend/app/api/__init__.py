"""
API Routes Module
"""

from flask import Blueprint

neuro_viewer_bp = Blueprint('neuro_viewer', __name__)

from . import neuro_viewer  # noqa: E402, F401
