"""
Configuration Management
Loads configuration from .env file in project root directory
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
# Path: project root .env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If no .env in root, try to load environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""

    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'neural_bridge-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON configuration - disable ASCII escaping to display Chinese directly (not as \uXXXX)
    JSON_AS_ASCII = False

    # TRIBE is optional and loaded lazily.
    NEURO_PRIOR_MODE = os.environ.get('NEURO_PRIOR_MODE', 'apple_silicon_tribe')
    NEURO_PRIOR_STRICT = os.environ.get('NEURO_PRIOR_STRICT', 'false').lower() == 'true'
    NEURO_PRIOR_SAVE_RAW_OUTPUT = os.environ.get('NEURO_PRIOR_SAVE_RAW_OUTPUT', 'true').lower() == 'true'
    TRIBE_MODEL_ID = os.environ.get('TRIBE_MODEL_ID', 'facebook/tribev2')
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    NEURAL_BRIDGE_EXTERNAL_ROOT = os.environ.get('NEURAL_BRIDGE_EXTERNAL_ROOT', '')
    _EXTERNAL_ROOT = NEURAL_BRIDGE_EXTERNAL_ROOT or os.path.join(_PROJECT_ROOT, 'external_assets')
    TRIBE_CACHE_DIR = os.environ.get('TRIBE_CACHE_DIR', os.path.join(_EXTERNAL_ROOT, 'cache/tribev2'))
    TRIBE_DEVICE = os.environ.get('TRIBE_DEVICE', 'auto')
    TRIBE_APPLE_SILICON_SOURCE_DIR = os.environ.get(
        'TRIBE_APPLE_SILICON_SOURCE_DIR',
        './external_models/tribev2-apple-silicon'
    )
    TRIBE_MLX_MODEL_ID = os.environ.get('TRIBE_MLX_MODEL_ID', 'zimengxiong/tribev2-mlx')
    TRIBE_MLX_DIR = os.environ.get(
        'TRIBE_MLX_DIR',
        os.path.join(_EXTERNAL_ROOT, 'models/tribe-mlx/zimengxiong-tribev2-mlx')
    )
    TRIBE_MLX_ENABLED = os.environ.get('TRIBE_MLX_ENABLED', 'false').lower() == 'true'
    TRIBE_TEXT_ENCODER_ID = os.environ.get('TRIBE_TEXT_ENCODER_ID', 'meta-llama/Llama-3.2-3B')
    TRIBE_TEXT_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_TEXT_ENCODER_LOCAL_DIR',
        os.path.join(_EXTERNAL_ROOT, 'models/upstream-encoders/meta-llama-Llama-3.2-3B')
    )
    TRIBE_TEXT_ENCODER_MLX_DIR = os.environ.get(
        'TRIBE_TEXT_ENCODER_MLX_DIR',
        os.path.join(_EXTERNAL_ROOT, 'models/upstream-encoders-mlx/meta-llama-Llama-3.2-3B')
    )
    TRIBE_AUDIO_ENCODER_ID = os.environ.get('TRIBE_AUDIO_ENCODER_ID', 'facebook/w2v-bert-2.0')
    TRIBE_AUDIO_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_AUDIO_ENCODER_LOCAL_DIR',
        os.path.join(_EXTERNAL_ROOT, 'models/upstream-encoders/facebook-w2v-bert-2.0')
    )
    TRIBE_VIDEO_ENCODER_ID = os.environ.get('TRIBE_VIDEO_ENCODER_ID', 'facebook/vjepa2-vitg-fpc64-256')
    TRIBE_VIDEO_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_VIDEO_ENCODER_LOCAL_DIR',
        os.path.join(_EXTERNAL_ROOT, 'models/cortical-upstream/facebook-vjepa2-vitg-fpc64-256')
    )
    TRIBE_VIDEO_ENCODER_BACKEND = os.environ.get('TRIBE_VIDEO_ENCODER_BACKEND', 'mlx')
    TRIBE_VIDEO_ENCODER_MLX_DIR = os.environ.get(
        'TRIBE_VIDEO_ENCODER_MLX_DIR',
        os.path.join(_EXTERNAL_ROOT, 'models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256')
    )
    TRIBE_VIDEO_DEVICE = os.environ.get('TRIBE_VIDEO_DEVICE', 'auto')
    TRIBE_ALLOW_UNSAFE_VITG_MPS = os.environ.get(
        'TRIBE_ALLOW_UNSAFE_VITG_MPS', 'false'
    ).lower() == 'true'
    TRIBE_MPS_MEMORY_FRACTION = float(os.environ.get('TRIBE_MPS_MEMORY_FRACTION', '0.45'))
    TRIBE_TEXT_BATCH_SIZE = int(os.environ.get('TRIBE_TEXT_BATCH_SIZE', '4'))
    TRIBE_DATA_NUM_WORKERS = int(os.environ.get('TRIBE_DATA_NUM_WORKERS', '0'))
    TRIBE_VIDEO_DTYPE = os.environ.get('TRIBE_VIDEO_DTYPE', 'bfloat16')
    TRIBE_VIDEO_NUM_FRAMES = int(os.environ.get('TRIBE_VIDEO_NUM_FRAMES', '64'))
    TRIBE_VIDEO_FRAME_SAMPLER = os.environ.get('TRIBE_VIDEO_FRAME_SAMPLER', 'ffmpeg')
    TRIBE_VIDEO_WINDOW_BATCH_SIZE = int(os.environ.get('TRIBE_VIDEO_WINDOW_BATCH_SIZE', '1'))
    TRIBE_VJEPA21_IMAGE_SIZE = int(os.environ.get('TRIBE_VJEPA21_IMAGE_SIZE', '256'))
    TRIBE_VJEPA21_COMPILE_ENCODER = os.environ.get(
        'TRIBE_VJEPA21_COMPILE_ENCODER', 'true'
    ).lower() == 'true'
    TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW = os.environ.get(
        'TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW', 'false'
    ).lower() == 'true'
    TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO = os.environ.get(
        'TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO', 'true'
    ).lower() == 'true'
    TRIBE_COALESCE_DIRECT_VIDEO_CHUNKS = os.environ.get(
        'TRIBE_COALESCE_DIRECT_VIDEO_CHUNKS', 'true'
    ).lower() == 'true'
    TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS = int(os.environ.get(
        'TRIBE_ENCODER_RESTART_EVERY_N_VIDEOS', '25'
    ))
    _TRIBE_FEATURE_FREQUENCY_HZ = os.environ.get('TRIBE_FEATURE_FREQUENCY_HZ', '').strip()
    TRIBE_FEATURE_FREQUENCY_HZ = (
        float(_TRIBE_FEATURE_FREQUENCY_HZ) if _TRIBE_FEATURE_FREQUENCY_HZ else None
    )

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        return []
