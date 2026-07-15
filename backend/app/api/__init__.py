"""
API Routes Module
"""

from flask import Blueprint

neuro_viewer_bp = Blueprint('neuro_viewer', __name__)
neural_bridge_results_bp = Blueprint('neural_bridge_results', __name__)

from . import neuro_viewer  # noqa: E402, F401
from . import neural_bridge_results  # noqa: E402, F401
