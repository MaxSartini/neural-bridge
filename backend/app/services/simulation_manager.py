"""
Neural Bridge simulation manager.

Local demo mode uses one internal OASIS social environment. Historical
Twitter/Reddit adapters remain as compatibility shims, but they are not the
product concept.
"""

import os
import json
import csv
import shutil
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityReader, FilteredEntities, EntityNode
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_config_generator import SimulationConfigGenerator, SimulationParameters
from .market_data_consolidator import MarketDataConsolidator
from .neuro_prior_mapper import NeuroPriorMapper
from .neuro_prior_service import NeuroPriorService

logger = get_logger('neural_bridge.simulation')

OHLCV_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}


def _contains_ohlcv_csv(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    for name in os.listdir(directory):
        if not name.lower().endswith(".csv"):
            continue
        try:
            with open(os.path.join(directory, name), "r", encoding="utf-8", errors="replace") as handle:
                columns = set(next(csv.reader(handle), []))
            if OHLCV_COLUMNS.issubset(columns):
                return True
        except OSError:
            continue
    return False


class SimulationStatus(str, Enum):
    """Simulation status"""
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"      # Simulation manually stopped
    COMPLETED = "completed"  # Simulation completed naturally
    FAILED = "failed"


class PlatformType(str, Enum):
    """Platform type"""
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class SimulationState:
    """Simulation status"""
    simulation_id: str
    project_id: str
    graph_id: str
    
    # Internal OASIS adapter state. Default product path is single-channel
    # Neural Bridge execution using the Reddit-shaped JSON profile adapter.
    enable_twitter: bool = False
    enable_reddit: bool = True

    # Simulation topic passed through to profile generation + agent personas.
    simulation_requirement: str = ""

    # Optional neuro-prior calibration.
    enable_neuro_priors: bool = False
    stimulus_text: str = ""
    stimulus_type: str = "text"
    stimulus_media_path: str = ""
    neuro_prior_generated: bool = False
    neuro_prior_path: str = ""
    neuro_modifiers_path: str = ""
    neuro_prior_mode: str = ""
    
    # Status
    status: SimulationStatus = SimulationStatus.CREATED
    
    # Preparation phase data
    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = field(default_factory=list)
    
    # Config generation information
    config_generated: bool = False
    config_reasoning: str = ""
    
    # Runtime data
    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Error message
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Complete status dict (internal use)"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
            "simulation_requirement": self.simulation_requirement,
            "enable_neuro_priors": self.enable_neuro_priors,
            "stimulus_text": self.stimulus_text,
            "stimulus_type": self.stimulus_type,
            "stimulus_media_path": self.stimulus_media_path,
            "neuro_prior_generated": self.neuro_prior_generated,
            "neuro_prior_path": self.neuro_prior_path,
            "neuro_modifiers_path": self.neuro_modifiers_path,
            "neuro_prior_mode": self.neuro_prior_mode,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round": self.current_round,
            "twitter_status": self.twitter_status,
            "reddit_status": self.reddit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }
    
    def to_simple_dict(self) -> Dict[str, Any]:
        """Simplified status dict (API return use)"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "enable_neuro_priors": self.enable_neuro_priors,
            "neuro_prior_generated": self.neuro_prior_generated,
            "neuro_prior_mode": self.neuro_prior_mode,
            "error": self.error,
        }


class SimulationManager:
    """
    Simulation Manager
    
    Core Functions:
    1. Read entities from graph and filter
    2. Generate OASIS Agent Profile
    3. Use LLM intelligent generation of simulation config parameters
    4. Prepare all files required by preset scripts
    """
    
    # Simulation data storage directory
    SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), 
        '../../uploads/simulations'
    )
    
    def __init__(self):
        # Ensure directory exists
        os.makedirs(self.SIMULATION_DATA_DIR, exist_ok=True)
        
        # In-memory simulation state cache
        self._simulations: Dict[str, SimulationState] = {}
    
    def _get_simulation_dir(self, simulation_id: str) -> str:
        """Get simulation data directory"""
        sim_dir = os.path.join(self.SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir
    
    def _save_simulation_state(self, state: SimulationState):
        """Save simulation state to file"""
        sim_dir = self._get_simulation_dir(state.simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        state.updated_at = datetime.now().isoformat()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        
        self._simulations[state.simulation_id] = state
    
    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        """Load simulation state from file"""
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]
        
        sim_dir = self._get_simulation_dir(simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        if not os.path.exists(state_file):
            return None
        
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            enable_twitter=data.get("enable_twitter", False),
            enable_reddit=data.get("enable_reddit", True),
            simulation_requirement=data.get("simulation_requirement", ""),
            enable_neuro_priors=data.get("enable_neuro_priors", False),
            stimulus_text=data.get("stimulus_text", ""),
            stimulus_type=data.get("stimulus_type", "text"),
            stimulus_media_path=data.get("stimulus_media_path", ""),
            neuro_prior_generated=data.get("neuro_prior_generated", False),
            neuro_prior_path=data.get("neuro_prior_path", ""),
            neuro_modifiers_path=data.get("neuro_modifiers_path", ""),
            neuro_prior_mode=data.get("neuro_prior_mode", ""),
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            twitter_status=data.get("twitter_status", "not_started"),
            reddit_status=data.get("reddit_status", "not_started"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
        )
        
        self._simulations[simulation_id] = state
        return state
    
    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = False,
        enable_reddit: bool = True,
        enable_neuro_priors: bool = False,
        stimulus_text: str = "",
        stimulus_type: str = "text",
        stimulus_media_path: str = "",
    ) -> SimulationState:
        """
        Create new simulation
        
        Args:
            project_id: Project ID
            graph_id: Graph ID
            enable_twitter: Legacy internal adapter, disabled by default
            enable_reddit: Internal single-channel Neural Bridge adapter
            
        Returns:
            SimulationState
        """
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            enable_neuro_priors=enable_neuro_priors,
            stimulus_text=stimulus_text or "",
            stimulus_type=stimulus_type or "text",
            stimulus_media_path=stimulus_media_path or "",
            status=SimulationStatus.CREATED,
        )
        
        self._save_simulation_state(state)
        logger.info(f"Create simulation: {simulation_id}, project={project_id}, graph={graph_id}")
        
        return state
    
    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 3,
        storage: 'GraphStorage' = None,
    ) -> SimulationState:
        """
        Prepare simulation environment (fully automated)
        
        Steps:
        1. Read and filter entities from graph
        2. Generate OASIS Agent Profile for each entity (optional LLM enhancement, parallel support)
        3. Use LLM intelligent generation of simulation config parameters (time, activity, speaking frequency, etc.)
        4. Save config files and Profile files
        5. Copy preset scripts to simulation directory
        
        Args:
            simulation_id: Simulation ID
            simulation_requirement: Simulation requirement description (for LLM config generation)
            document_text: Original document content (for LLM background understanding)
            defined_entity_types: Predefined entity types (optional)
            use_llm_for_profiles: Whether to use LLM to generate detailed profiles
            progress_callback: Progress callback function (stage, progress, message)
            parallel_profile_count: Number of parallel profile generations, default 3
            
        Returns:
            SimulationState
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation does not exist: {simulation_id}")
        
        try:
            state.status = SimulationStatus.PREPARING
            state.simulation_requirement = simulation_requirement or ""
            self._save_simulation_state(state)
            
            sim_dir = self._get_simulation_dir(simulation_id)

            neuro_prior = None
            neuro_modifiers = None
            active_neuro_modifiers = None
            if state.enable_neuro_priors:
                stimulus_type = (state.stimulus_type or "text").lower()
                media_path = os.path.abspath(os.path.expanduser(state.stimulus_media_path or ""))
                has_media = stimulus_type in {"audio", "video"} and os.path.isfile(media_path)
                stimulus = (state.stimulus_text or "").strip()
                if stimulus_type == "text":
                    stimulus = (
                        stimulus
                        or (simulation_requirement or "").strip()
                        or (document_text or "")[:8000].strip()
                    )
                if stimulus or has_media:
                    if progress_callback:
                        progress_callback("reading", 10, "Generating neuro-prior profile...")
                    neuro_prior = NeuroPriorService().generate(
                        stimulus_text=stimulus,
                        stimulus_type=stimulus_type,
                        simulation_requirement=simulation_requirement,
                        document_text=document_text,
                        simulation_id=simulation_id,
                        output_dir=sim_dir,
                        media_path=media_path if has_media else None,
                    )
                    neuro_modifiers = NeuroPriorMapper().map_to_modifiers(neuro_prior)
                    if Config.NEURO_HEURISTIC_MODIFIERS_ACTIVE:
                        active_neuro_modifiers = neuro_modifiers
                        logger.warning(
                            "Experimental unvalidated heuristic neuro modifiers are ACTIVE "
                            "because NEURO_HEURISTIC_MODIFIERS_ACTIVE=true."
                        )
                    neuro_prior_path = os.path.join(sim_dir, "neuro_prior.json")
                    neuro_modifiers_path = os.path.join(sim_dir, "neuro_prior_modifiers.json")
                    with open(neuro_prior_path, "w", encoding="utf-8") as f:
                        json.dump(neuro_prior.to_dict(), f, ensure_ascii=False, indent=2)
                    with open(neuro_modifiers_path, "w", encoding="utf-8") as f:
                        json.dump(neuro_modifiers, f, ensure_ascii=False, indent=2)
                    state.neuro_prior_generated = True
                    state.neuro_prior_path = neuro_prior_path
                    state.neuro_modifiers_path = neuro_modifiers_path
                    state.neuro_prior_mode = neuro_prior.mode
                    self._save_simulation_state(state)
                else:
                    logger.warning(
                        "Neuro-priors enabled but no valid %s stimulus was available.",
                        stimulus_type,
                    )
            
            # ========== Phase 1: Read and filter entities ==========
            if progress_callback:
                progress_callback("reading", 0, "Connecting to graph...")

            if not storage:
                raise ValueError("storage (GraphStorage) is required for prepare_simulation")
            reader = EntityReader(storage)
            
            if progress_callback:
                progress_callback("reading", 30, "Reading node data...")
            
            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True
            )
            
            state.entities_count = filtered.filtered_count
            state.entity_types = list(filtered.entity_types)
            
            if progress_callback:
                progress_callback(
                    "reading", 100, 
                    f"Completed, total {filtered.filtered_count} entities",
                    current=filtered.filtered_count,
                    total=filtered.filtered_count
                )
            
            if filtered.filtered_count == 0:
                # Do not fail outright — we may synthesize personas from the user's prompt.
                logger.warning("No entities found in graph; will attempt persona synthesis from prompt.")
            
            # ========== Phase 2: Generate Agent Profile ==========
            total_entities = len(filtered.entities)
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 0, 
                    "Starting generation...",
                    current=0,
                    total=total_entities
                )
            
            # Pass graph_id AND the simulation topic so personas are anchored to what's being simulated.
            generator = OasisProfileGenerator(
                storage=storage,
                graph_id=state.graph_id,
                simulation_requirement=getattr(state, "simulation_requirement", "") or "",
                neuro_prior=neuro_prior.to_dict() if neuro_prior else None,
                neuro_modifiers=active_neuro_modifiers,
            )
            
            def profile_progress(current, total, msg):
                if progress_callback:
                    progress_callback(
                        "generating_profiles", 
                        int(current / total * 100), 
                        msg,
                        current=current,
                        total=total,
                        item_name=msg
                    )
            
            # Set real-time save file path (prefer Reddit JSON format)
            realtime_output_path = None
            realtime_platform = "reddit"
            if state.enable_reddit:
                realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
                realtime_platform = "reddit"
            elif state.enable_twitter:
                realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
                realtime_platform = "twitter"
            
            # Separate screenshot-sourced commentors from pure data nodes.
            # Commentors from screenshots (type=Commenter/Person) become agents directly.
            # Data nodes (Asset, Metric, Index, etc.) never become agents.
            AGENT_ELIGIBLE_TYPES = {"commenter", "person", "user", "analyst", "trader",
                                    "investor", "reporter", "expert", "participant"}
            DATA_ONLY_TYPES = {"asset", "metric", "index", "exchange", "commodity",
                               "sector", "marketcorrelation", "dataset",
                               "publication", "document", "article", "paper",
                               "report", "table", "image"}

            screenshot_entities = []
            for e in filtered.entities:
                etype = (e.get_entity_type() or "").lower()
                src = (e.attributes or {}).get("source", "").lower()
                # Commenter nodes injected by screenshot_processor carry source="screenshot"
                if src == "screenshot" or etype in AGENT_ELIGIBLE_TYPES:
                    screenshot_entities.append(e)

            # Decide agent source: screenshots take precedence over prompt synthesis
            if screenshot_entities:
                logger.info(f"Using {len(screenshot_entities)} screenshot commentors as agent personas")
                human_like_entities = screenshot_entities
            else:
                # No screenshots — synthesize personas from the simulation prompt
                synthetic = self._synthesize_persona_seeds(
                    state.simulation_requirement, active_neuro_modifiers
                )
                if synthetic:
                    human_like_entities = synthetic
                    logger.info(f"Synthesized {len(synthetic)} persona seeds from simulation prompt")
                else:
                    human_like_entities = []

            profiles = generator.generate_profiles_from_entities(
                entities=human_like_entities,
                use_llm=use_llm_for_profiles,
                progress_callback=profile_progress,
                graph_id=state.graph_id,
                parallel_count=parallel_profile_count,
                realtime_output_path=realtime_output_path,
                output_platform=realtime_platform
            )

            state.profiles_count = len(profiles)
            state.entities_count = len(human_like_entities)

            if progress_callback:
                progress_callback(
                    "generating_profiles", 95,
                    "Saving Profile files...",
                    current=total_entities,
                    total=total_entities
                )

            if state.enable_reddit:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                    platform="reddit"
                )

            if state.enable_twitter:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                    platform="twitter"
                )

            if progress_callback:
                progress_callback(
                    "generating_profiles", 100,
                    f"Completed, total {len(profiles)} Profiles",
                    current=len(profiles),
                    total=len(profiles)
                )

            # ========== Phase 3: LLM intelligent generation of simulation config ==========
            if progress_callback:
                progress_callback(
                    "generating_config", 0,
                    "Analyzing simulation requirements...",
                    current=0,
                    total=3
                )

            config_generator = SimulationConfigGenerator()

            if progress_callback:
                progress_callback(
                    "generating_config", 30,
                    "Calling LLM to generate config...",
                    current=1,
                    total=3
                )

            config_entities = human_like_entities
            project_uploads_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../../uploads/projects', state.project_id)
            )
            csv_data_dir = os.path.join(project_uploads_dir, 'files')
            has_chronological_csv = _contains_ohlcv_csv(csv_data_dir)
            # Full historical CSV extraction can contain future outcomes. Use
            # only the requirement and graph-derived personas when generating
            # config; the runner exposes CSV rows chronologically per round.
            config_document_text = "" if has_chronological_csv else document_text

            sim_params = config_generator.generate_config(
                simulation_id=simulation_id,
                project_id=state.project_id,
                graph_id=state.graph_id,
                simulation_requirement=simulation_requirement,
                document_text=config_document_text,
                entities=config_entities,
                enable_twitter=state.enable_twitter,
                enable_reddit=state.enable_reddit,
                neuro_prior=neuro_prior.to_dict() if neuro_prior else None,
                neuro_modifiers=active_neuro_modifiers,
            )

            sim_params.market_context = "" if has_chronological_csv else (document_text or "")

            config_path = os.path.join(sim_dir, "simulation_config.json")
            config_dict = sim_params.to_dict()
            config_dict['csv_data_dir'] = csv_data_dir if has_chronological_csv else ""
            config_dict['chronology_guard'] = {
                "enabled": has_chronological_csv,
                "full_document_excluded_from_config_generation": has_chronological_csv,
                "full_document_excluded_from_round_context": has_chronological_csv,
            }
            config_dict['neuro_integration_contract'] = {
                "heuristic_mapper_status": "experimental_unvalidated_fallback_only",
                "heuristic_modifiers_active": bool(active_neuro_modifiers),
                "active_runner_modifiers": (
                    [
                        "activity_multiplier",
                        "posting_multiplier",
                        "commenting_multiplier",
                        "response_speed_multiplier",
                    ]
                    if active_neuro_modifiers else []
                ),
                "recorded_only_modifiers": [
                    "sentiment_shift",
                    "sharing_multiplier",
                    "risk_aversion_shift",
                    "trust_shift",
                    "polarisation_multiplier",
                    "echo_chamber_shift",
                    "viral_threshold_shift",
                    "memory_persistence_multiplier",
                ],
                "note": (
                    "Heuristic fields are recorded for ablation only unless the explicit "
                    "experimental override is enabled. Production conditioning requires a "
                    "held-out calibrated numerical agent-state/action model."
                ),
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                import json as _json
                f.write(_json.dumps(config_dict, ensure_ascii=False, indent=2))

            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning

            if progress_callback:
                progress_callback(
                    "generating_config", 100,
                    "Config generation completed",
                    current=3,
                    total=3
                )

            state.status = SimulationStatus.READY
            self._save_simulation_state(state)

            logger.info(f"Simulation preparation completed: {simulation_id}, "
                       f"entities={state.entities_count}, profiles={state.profiles_count}")

            return state

        except Exception as e:
            logger.error(f"Simulation preparation failed: {simulation_id}, error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            raise

    
    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        """Get simulation state"""
        return self._load_simulation_state(simulation_id)

    def _synthesize_persona_seeds(
        self,
        simulation_requirement: str,
        neuro_modifiers: Optional[Dict[str, Any]] = None,
    ) -> List['EntityNode']:
        """Create synthetic `EntityNode` seeds dynamically generated from the user's prompt."""
        import uuid as _uuid
        from openai import OpenAI
        from ..config import Config
        from ..utils.logger import get_logger
        from .entity_reader import EntityNode
        
        # Safe binding if not imported universally
        logger = get_logger('neural_bridge.simulation')
        text = (simulation_requirement or "").strip()
        roles = []
        
        if text:
            try:
                client = OpenAI(
                    base_url=getattr(Config, "LLM_BASE_URL", "http://localhost:11434/v1"),
                    api_key=getattr(Config, "LLM_API_KEY", "ollama")
                )
                model_name = getattr(Config, "LLM_MODEL_NAME", "qwen2.5:32b")
                
                prompt_content = (
                    f"Based on the following simulation requirement, generate a JSON array of 15 highly specific "
                    f"and highly diverse agent personas/roles that would be relevant to simulate public reaction. "
                    f"Adapt to the domain: politics, public health, finance, crisis comms, consumer products, "
                    f"advertising, regulation, investor relations, or any other topic. "
                    f"Include a balanced mix of supporters, skeptics, neutral observers, institutional voices, "
                    f"amplifiers, affected public groups, journalists/media, experts, regulators/compliance, "
                    f"and opportunistic or opposition actors where relevant. "
                    f"Return ONLY a raw JSON list of strings. No markdown, no explanations.\n\n"
                    + (
                        f"Behavioural-prior context: {(neuro_modifiers or {}).get('persona_instruction', '')}\n\n"
                        if Config.NEURO_PRIOR_IN_PERSONA_PROMPTS
                        else ""
                    )
                    +
                    f"Simulation prompt: {text}"
                )
                
                logger.info("Calling LLM to dynamically generate persona seeds based on prompt...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a simulation data structuralizer. Output valid JSON array of strings ONLY."},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.8
                )
                
                # Standardize stripping
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                
                import json
                roles = json.loads(content)
                if not isinstance(roles, list):
                    roles = []
            except Exception as e:
                logger.error(f"Failed to generate dynamic roles from LLM, falling back to emergency default. Error: {e}")
                
        if not roles:
            # Domain-neutral emergency fallback if LLM parsing fails.
            roles = [
                "Concerned Member Of The Public",
                "Supportive Early Amplifier",
                "Skeptical Community Commentator",
                "Investigative Journalist",
                "Policy Or Compliance Analyst",
                "Affected Consumer Or Citizen",
                "Institutional Decision Maker",
                "Opposition Or Critic Voice",
                "Subject Matter Expert",
                "Social Media Influencer",
                "Risk Averse Professional",
                "Pragmatic Neutral Observer"
            ]
            
        seeds = []
        for r in roles:
            if not isinstance(r, str): continue
            name = (r[:40] + '...') if len(r) > 40 else r
            seeds.append(
                EntityNode(
                    uuid=str(_uuid.uuid4()),
                    name=name.title(),
                    labels=["Entity", "Person"],
                    summary=f"Synthetic organically generated persona role: {name}",
                    attributes={"role": name}
                )
            )
        
        logger.info(f"Initialized {len(seeds)} dynamic Agent Persona Seeds.")
        return seeds

    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        """List all simulations"""
        simulations = []
        
        if os.path.exists(self.SIMULATION_DATA_DIR):
            for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
                # Skip hidden files (such as .DS_Store) and non-directory files
                sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
                if sim_id.startswith('.') or not os.path.isdir(sim_path):
                    continue
                
                state = self._load_simulation_state(sim_id)
                if state:
                    if project_id is None or state.project_id == project_id:
                        simulations.append(state)
        
        return simulations
    
    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        """Get Agent Profiles for simulation"""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation does not exist: {simulation_id}")
        
        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")
        
        if not os.path.exists(profile_path):
            return []
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation config"""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        """Get run instructions"""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        backend_python = os.path.join(project_root, "backend", ".venv", "bin", "python")
        python_cmd = backend_python if os.path.exists(backend_python) else "python"
        
        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "neural_bridge": f"{python_cmd} {scripts_dir}/run_parallel_simulation.py --config {config_path} --reddit-only",
            },
            "instructions": (
                f"1. Open the project: cd {project_root}\n"
                f"2. Run simulation (scripts located in {scripts_dir}):\n"
                f"   - Run local Neural Bridge single-channel demo: {python_cmd} {scripts_dir}/run_parallel_simulation.py --config {config_path} --reddit-only"
            )
        }
