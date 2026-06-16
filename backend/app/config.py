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

    # LLM configuration (unified OpenAI format)
    LLM_API_KEY = os.environ.get('LLM_API_KEY', 'lm-studio')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:1234/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'mlx-community/gemma-4-26b-a4b-it-4bit')

    # Neuro-prior / TRIBE configuration. TRIBE is optional and loaded lazily.
    NEURO_PRIOR_MODE = os.environ.get('NEURO_PRIOR_MODE', 'apple_silicon_tribe')
    NEURO_PRIOR_STRICT = os.environ.get('NEURO_PRIOR_STRICT', 'false').lower() == 'true'
    NEURO_PRIOR_FALLBACK_TO_PROXY = os.environ.get('NEURO_PRIOR_FALLBACK_TO_PROXY', 'true').lower() == 'true'
    NEURO_PRIOR_SAVE_RAW_OUTPUT = os.environ.get('NEURO_PRIOR_SAVE_RAW_OUTPUT', 'true').lower() == 'true'
    # Preserve persona identity and avoid applying the same population prior
    # repeatedly through both prose and numerical modifiers.
    NEURO_PRIOR_IN_PERSONA_PROMPTS = os.environ.get(
        'NEURO_PRIOR_IN_PERSONA_PROMPTS', 'false'
    ).lower() == 'true'
    NEURO_PRIOR_IN_CONFIG_PROMPTS = os.environ.get(
        'NEURO_PRIOR_IN_CONFIG_PROMPTS', 'false'
    ).lower() == 'true'
    NEURO_PRIOR_CAN_OVERRIDE_STANCE = os.environ.get(
        'NEURO_PRIOR_CAN_OVERRIDE_STANCE', 'false'
    ).lower() == 'true'
    NEURO_PRIOR_SHARED_SENTIMENT_SHIFT = os.environ.get(
        'NEURO_PRIOR_SHARED_SENTIMENT_SHIFT', 'false'
    ).lower() == 'true'
    # Hand-authored neuro modifiers are unvalidated research fallbacks. Keep
    # them observable for ablations, but never condition simulations by default.
    NEURO_HEURISTIC_MODIFIERS_ACTIVE = os.environ.get(
        'NEURO_HEURISTIC_MODIFIERS_ACTIVE', 'false'
    ).lower() == 'true'
    NEURO_PRIOR_BACKEND_PRIORITY = os.environ.get(
        'NEURO_PRIOR_BACKEND_PRIORITY',
        'apple_silicon_tribe,official_tribe,tribe_mlx,proxy,disabled'
    )
    TRIBE_MODEL_ID = os.environ.get('TRIBE_MODEL_ID', 'facebook/tribev2')
    TRIBE_CACHE_DIR = os.environ.get('TRIBE_CACHE_DIR', './models/cache/tribev2')
    TRIBE_DEVICE = os.environ.get('TRIBE_DEVICE', 'auto')
    TRIBE_ENABLE_SUBCORTICAL = os.environ.get('TRIBE_ENABLE_SUBCORTICAL', 'true').lower() == 'true'
    TRIBE_SUBCORTICAL_MODEL_ID = os.environ.get('TRIBE_SUBCORTICAL_MODEL_ID', 'loganf26/tribev2-subcortical')
    TRIBE_SUBCORTICAL_LOCAL_DIR = os.environ.get(
        'TRIBE_SUBCORTICAL_LOCAL_DIR',
        '/Volumes/onn. Drive/Neural Bridge/models/tribe/loganf26-tribev2-subcortical'
    )
    TRIBE_SUBCORTICAL_TEXT_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_SUBCORTICAL_TEXT_ENCODER_LOCAL_DIR',
        '/Volumes/onn. Drive/Neural Bridge/models/subcortical-upstream/Qwen-Qwen3-0.6B'
    )
    TRIBE_SUBCORTICAL_AUDIO_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_SUBCORTICAL_AUDIO_ENCODER_LOCAL_DIR',
        '/Volumes/onn. Drive/Neural Bridge/models/subcortical-upstream/facebook-w2v-bert-2.0'
    )
    TRIBE_SUBCORTICAL_VIDEO_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_SUBCORTICAL_VIDEO_ENCODER_LOCAL_DIR',
        '/Volumes/onn. Drive/Neural Bridge/models/subcortical-upstream/facebook-vjepa2-vitl-fpc64-256'
    )
    TRIBE_SUBCORTICAL_TEXT_BATCH_SIZE = int(os.environ.get('TRIBE_SUBCORTICAL_TEXT_BATCH_SIZE', '1'))
    TRIBE_SUBCORTICAL_TEXT_DEVICE = os.environ.get('TRIBE_SUBCORTICAL_TEXT_DEVICE', 'cpu')
    TRIBE_APPLE_SILICON_SOURCE_DIR = os.environ.get(
        'TRIBE_APPLE_SILICON_SOURCE_DIR',
        './external_models/tribev2-apple-silicon'
    )
    TRIBE_OFFICIAL_SOURCE_DIR = os.environ.get('TRIBE_OFFICIAL_SOURCE_DIR', './external_models/tribev2-official')
    TRIBE_MLX_MODEL_ID = os.environ.get('TRIBE_MLX_MODEL_ID', 'zimengxiong/tribev2-mlx')
    TRIBE_MLX_DIR = os.environ.get(
        'TRIBE_MLX_DIR',
        '/Volumes/onn. Drive/Neural Bridge/models/tribe-mlx/zimengxiong-tribev2-mlx'
    )
    TRIBE_MLX_ENABLED = os.environ.get('TRIBE_MLX_ENABLED', 'false').lower() == 'true'
    TRIBE_TEXT_ENCODER_ID = os.environ.get('TRIBE_TEXT_ENCODER_ID', 'meta-llama/Llama-3.2-3B')
    TRIBE_TEXT_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_TEXT_ENCODER_LOCAL_DIR',
        './models/upstream-encoders/meta-llama-Llama-3.2-3B'
    )
    TRIBE_TEXT_ENCODER_MLX_DIR = os.environ.get(
        'TRIBE_TEXT_ENCODER_MLX_DIR',
        os.path.expanduser('~/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit')
    )
    TRIBE_AUDIO_ENCODER_ID = os.environ.get('TRIBE_AUDIO_ENCODER_ID', 'facebook/w2v-bert-2.0')
    TRIBE_AUDIO_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_AUDIO_ENCODER_LOCAL_DIR',
        './models/upstream-encoders/facebook-w2v-bert-2.0'
    )
    TRIBE_VIDEO_ENCODER_ID = os.environ.get('TRIBE_VIDEO_ENCODER_ID', 'facebook/vjepa2-vitg-fpc64-256')
    TRIBE_VIDEO_ENCODER_LOCAL_DIR = os.environ.get(
        'TRIBE_VIDEO_ENCODER_LOCAL_DIR',
        './models/upstream-encoders/facebook-vjepa2-vitg-fpc64-256'
    )
    TRIBE_VIDEO_ENCODER_BACKEND = os.environ.get('TRIBE_VIDEO_ENCODER_BACKEND', 'mlx')
    TRIBE_VIDEO_ENCODER_MLX_DIR = os.environ.get(
        'TRIBE_VIDEO_ENCODER_MLX_DIR',
        '/Volumes/onn. Drive/Neural Bridge/models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256'
    )
    TRIBE_VIDEO_DEVICE = os.environ.get('TRIBE_VIDEO_DEVICE', 'auto')
    TRIBE_ALLOW_UNSAFE_VITG_MPS = os.environ.get(
        'TRIBE_ALLOW_UNSAFE_VITG_MPS', 'false'
    ).lower() == 'true'
    TRIBE_MPS_MEMORY_FRACTION = float(os.environ.get('TRIBE_MPS_MEMORY_FRACTION', '0.45'))
    TRIBE_TEXT_BATCH_SIZE = int(os.environ.get('TRIBE_TEXT_BATCH_SIZE', '4'))
    TRIBE_SUBCORTICAL_VIDEO_WINDOW_BATCH_SIZE = int(os.environ.get('TRIBE_SUBCORTICAL_VIDEO_WINDOW_BATCH_SIZE', '4'))
    TRIBE_VIDEO_DTYPE = os.environ.get('TRIBE_VIDEO_DTYPE', 'float16')
    TRIBE_VIDEO_NUM_FRAMES = int(os.environ.get('TRIBE_VIDEO_NUM_FRAMES', '64'))

    # Neo4j configuration
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'neural_bridge')
    NEO4J_LEGACY_PASSWORD = os.environ.get('NEO4J_LEGACY_PASSWORD', 'mirofish')

    # Embedding configuration
    EMBEDDING_MODEL = os.environ.get(
        'EMBEDDING_MODEL',
        'nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf'
    )
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:1235/v1')
    EMBEDDING_DIMENSIONS = int(os.environ.get('EMBEDDING_DIMENSIONS', '768'))

    # File upload configuration
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB — allow bulk screenshot / spreadsheet uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown', 'csv', 'xlsx', 'xls'}
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500  # Default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # Default overlap size

    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'FOLLOW'
    ]

    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    # ==============================================================
    # Performance tuning — bounded-state execution for local Gemma
    # All overridable via env. Defaults target M2 Max + LM Studio.
    # ==============================================================

    # OASIS env semaphore — concurrent LLM calls into LM Studio.
    # Single-instance local Gemma serialises; high values just queue.
    # Sequential execution — one agent at a time to save VRAM on local Gemma.
    OASIS_ENV_SEMAPHORE = int(os.environ.get('OASIS_ENV_SEMAPHORE', '1'))

    # Cap how many agents act in any single simulated round.
    OASIS_MAX_ACTIVE_AGENTS_PER_ROUND = int(os.environ.get('OASIS_MAX_ACTIVE_AGENTS_PER_ROUND', '8'))

    # Empty-round guard — after N consecutive zero-action rounds, stop invoking.
    OASIS_EMPTY_ROUND_SKIP_THRESHOLD = int(os.environ.get('OASIS_EMPTY_ROUND_SKIP_THRESHOLD', '2'))

    # Skip dead/off-peak hours entirely.
    OASIS_SKIP_DEAD_HOURS = os.environ.get('OASIS_SKIP_DEAD_HOURS', 'true').lower() == 'true'

    # Local 27B models should not run Twitter and Reddit environments against
    # LM Studio simultaneously. Keep platforms serial unless explicitly
    # overridden for smaller/cloud models.
    OASIS_SERIAL_PLATFORMS = os.environ.get('OASIS_SERIAL_PLATFORMS', 'true').lower() == 'true'
    OASIS_NEURO_PRIOR_IN_ROUND_PROMPTS = os.environ.get(
        'OASIS_NEURO_PRIOR_IN_ROUND_PROMPTS', 'false'
    ).lower() == 'true'
    OASIS_RANDOM_SEED = int(os.environ.get('OASIS_RANDOM_SEED', '33'))
    OASIS_LLM_TEMPERATURE = float(os.environ.get('OASIS_LLM_TEMPERATURE', '0.6'))
    OASIS_LLM_TOP_P = float(os.environ.get('OASIS_LLM_TOP_P', '0.9'))

    # Persona generator — bounded prompt / output.
    PERSONA_MAX_TOKENS = int(os.environ.get('PERSONA_MAX_TOKENS', '700'))
    PERSONA_CONTEXT_CHAR_LIMIT = int(os.environ.get('PERSONA_CONTEXT_CHAR_LIMIT', '1500'))
    PERSONA_TARGET_WORDS = int(os.environ.get('PERSONA_TARGET_WORDS', '250'))
    PERSONA_SKIP_HYBRID_SEARCH_IF_LOCAL = os.environ.get(
        'PERSONA_SKIP_HYBRID_SEARCH_IF_LOCAL', 'true'
    ).lower() == 'true'

    # Graph memory updater — delta-oriented batch writes.
    GRAPH_MEMORY_BATCH_SIZE = int(os.environ.get('GRAPH_MEMORY_BATCH_SIZE', '25'))
    GRAPH_MEMORY_SEND_INTERVAL = float(os.environ.get('GRAPH_MEMORY_SEND_INTERVAL', '0.2'))
    GRAPH_MEMORY_SKIP_LOW_INFO_ACTIONS = os.environ.get(
        'GRAPH_MEMORY_SKIP_LOW_INFO_ACTIONS', 'true'
    ).lower() == 'true'

    # Simulation horizon defaults — shorter than the 72h legacy default.
    SIM_DEFAULT_HOURS = int(os.environ.get('SIM_DEFAULT_HOURS', '36'))
    SIM_DEFAULT_MINUTES_PER_ROUND = int(os.environ.get('SIM_DEFAULT_MINUTES_PER_ROUND', '60'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY not configured (set to any non-empty value, e.g. 'ollama')")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI not configured")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD not configured")
        return errors
