"""
Simulation Configuration Intelligent Generator
Use LLM to automatically generate detailed simulation parameters based on simulation requirements, document content, and knowledge graph information
Implement full process automation without manual parameter setting

Adopt step-by-step generation strategy to avoid failures from generating too long content at once:
1. Generate time configuration
2. Generate event configuration
3. Generate agent configurations in batches
4. Generate platform configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

import os
from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityNode

logger = get_logger('neural_bridge.simulation_config')

# Domain-neutral public reaction session configuration.
# The simulator may model finance, politics, public health, marketing, crisis
# comms, or any other public-facing stimulus. Keep defaults broadly plausible
# instead of hard-coding one market or country.
PUBLIC_SESSION_CONFIG = {
    # Off-hours: late night/early morning
    "dead_hours": [1, 2, 3, 4],
    # Morning commute / early news cycle
    "morning_hours": [5, 6, 7, 8],
    # Workday / institutional response window
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16],
    # Evening public discussion peak
    "peak_hours": [17, 18, 19, 20, 21],
    # Late-night recap / niche discussion
    "night_hours": [22, 23, 0],
    "activity_multipliers": {
        "dead": 0.15,      # Low but not silent
        "morning": 0.7,    # Asian open — active
        "work": 0.9,       # European session active
        "peak": 1.5,       # US session peak
        "night": 0.5       # After-hours recap
    }
}


@dataclass
class AgentActivityConfig:
    """Activity configuration for a single Agent"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Activity configuration (0.0-1.0)
    activity_level: float = 0.5  # Overall activity level

    # Speech frequency (expected posts per hour)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Active time periods (24-hour format, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Response speed (reaction delay to trending events, unit: simulation minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment tendency (-1.0 to 1.0, negative to positive)
    sentiment_bias: float = 0.0

    # Stance (attitude toward specific topics)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Influence weight (determines probability of their speech being seen by other agents)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Time simulation configuration for local public-reaction demos."""
    # Total simulation time (simulation hours) — shortened default for local LLM.
    total_simulation_hours: int = getattr(Config, "SIM_DEFAULT_HOURS", 36)

    # Time represented per round (simulation minutes).
    minutes_per_round: int = getattr(Config, "SIM_DEFAULT_MINUTES_PER_ROUND", 60)

    # Range of agents activated per hour — lowered to bound per-round LLM cost.
    agents_per_hour_min: int = 3
    agents_per_hour_max: int = 10

    # Peak hours (evening public-discussion window)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Off-peak hours — very early morning only
    off_peak_hours: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    off_peak_activity_multiplier: float = 0.15  # Low but never silent

    # Morning hours
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Work hours / institutional response window
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16])
    work_activity_multiplier: float = 0.9  # European session highly active


@dataclass
class EventConfig:
    """Event configuration"""
    # Initial posts (triggering events at the start of simulation)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Scheduled events (events triggered at specific times)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Hot topic keywords
    hot_topics: List[str] = field(default_factory=list)

    # Opinion narrative direction
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    platform: str  # twitter or reddit

    # Recommendation algorithm weights
    recency_weight: float = 0.4  # Time freshness
    popularity_weight: float = 0.3  # Popularity
    relevance_weight: float = 0.3  # Relevance

    # Viral threshold (number of interactions before triggering spread)
    viral_threshold: int = 10

    # Echo chamber effect strength (degree of similar opinion clustering)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete simulation parameter configuration"""
    # Basic information
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Time configuration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Agent configuration list
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Event configuration
    event_config: EventConfig = field(default_factory=EventConfig)

    # Platform configuration
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # LLM configuration
    llm_model: str = ""
    llm_base_url: str = ""

    # Generation metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM reasoning explanation
    
    # Mathematical and Quantitative Context
    market_context: str = ""

    neuro_prior: Dict[str, Any] = field(default_factory=dict)
    neuro_modifiers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
            "market_context": self.market_context,
            "neuro_prior": self.neuro_prior,
            "neuro_modifiers": self.neuro_modifiers,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Simulation Configuration Intelligent Generator

    Use LLM to analyze simulation requirements, document content, knowledge graph entity information,
    and automatically generate optimal simulation parameter configuration

    Adopt step-by-step generation strategy:
    1. Generate time configuration and event configuration (lightweight)
    2. Generate agent configurations in batches (10-20 per batch)
    3. Generate platform configuration
    """

    # Maximum context length in characters — bounded for local investor demos.
    # Unlimited context makes arbitrary uploads brittle and can stall local LLMs.
    MAX_CONTEXT_LENGTH = int(os.environ.get("LLM_CONTEXT_MAX_CHARS", "60000"))
    # Number of agents per batch
    AGENTS_PER_BATCH = 15

    # Context truncation length for each step (characters)
    # Use the same configurable cap for step-specific contexts
    TIME_CONFIG_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH   # Time configuration
    EVENT_CONFIG_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH   # Event configuration
    ENTITY_SUMMARY_LENGTH = 300          # Entity summary
    AGENT_SUMMARY_LENGTH = 300           # Entity summary in agent configuration
    ENTITIES_PER_TYPE_DISPLAY = 20       # Number of entities to display per type

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        neuro_prior: Optional[Dict[str, Any]] = None,
        neuro_modifiers: Optional[Dict[str, Any]] = None,
    ) -> SimulationParameters:
        """
        Intelligently generate complete simulation configuration (step-by-step generation)

        Args:
            simulation_id: Simulation ID
            project_id: Project ID
            graph_id: Knowledge graph ID
            simulation_requirement: Simulation requirement description
            document_text: Original document content
            entities: Filtered entity list
            enable_twitter: Whether to enable Twitter
            enable_reddit: Whether to enable Reddit
            progress_callback: Progress callback function(current_step, total_steps, message)

        Returns:
            SimulationParameters: Complete simulation parameters
        """
        logger.info(f"Starting intelligent simulation configuration generation: simulation_id={simulation_id}, entities={len(entities)}")
        
        # Calculate total steps
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # time config + event config + N batch agents + platform config
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. Build basic context information
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
            neuro_prior=neuro_prior,
            neuro_modifiers=neuro_modifiers,
        )
        
        reasoning_parts = []
        
        # ========== Step 1: Generate time configuration ==========
        report_progress(1, "Generating time configuration...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"Time config: {time_config_result.get('reasoning', 'Success')}")

        # ========== Step 2: Generate event configuration ==========
        report_progress(2, "Generating event configuration and hot topics...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"Event config: {event_config_result.get('reasoning', 'Success')}")

        # ========== Step 3-N: Generate agent configurations in batches ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Generating agent configuration ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"Agent config: Successfully generated {len(all_agent_configs)}")

        # Keep demo agents observable without erasing role differences.
        # Very low activity causes silent simulations; forcing 1.0 makes all
        # personas behave identically. Use a floor instead.
        for cfg in all_agent_configs:
            if isinstance(cfg, AgentActivityConfig):
                cfg.activity_level = self._clamp(cfg.activity_level, 0.65, 1.0)
            elif isinstance(cfg, dict):
                cfg["activity_level"] = self._clamp(cfg.get("activity_level", 0.65), 0.65, 1.0)

        # ========== Assign initial post agents ==========
        logger.info("Assigning appropriate publisher agents to initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Initial posts assigned: {assigned_count} posts assigned publishers")

        # ========== Final step: Generate platform configuration ==========
        report_progress(total_steps, "Generating platform configuration...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )

        if neuro_modifiers:
            all_agent_configs = [
                self._apply_neuro_modifiers_to_agent(cfg, neuro_modifiers, getattr(cfg, "entity_type", ""))
                for cfg in all_agent_configs
            ]
            if twitter_config:
                twitter_config = self._apply_neuro_modifiers_to_platform(twitter_config, neuro_modifiers)
            if reddit_config:
                reddit_config = self._apply_neuro_modifiers_to_platform(reddit_config, neuro_modifiers)
        
        # Build final parameters
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        params.neuro_prior = neuro_prior or {}
        params.neuro_modifiers = neuro_modifiers or {}
        
        logger.info(f"Simulation configuration generation complete: {len(params.agent_configs)} agent configurations")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        neuro_prior: Optional[Dict[str, Any]] = None,
        neuro_modifiers: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build bounded LLM context from uploaded documents and graph entities."""
        # Entity summary
        entity_summary = self._summarize_entities(entities)

        # Build base context
        context_parts = [
            f"## Simulation Requirements\n{simulation_requirement}",
            f"\n## Entity Information ({len(entities)})\n{entity_summary}",
        ]

        if Config.NEURO_PRIOR_IN_CONFIG_PROMPTS and (neuro_prior or neuro_modifiers):
            context_parts.append(
                "\n## Neuro-Prior Behavioural Calibration\n"
                "This is a population-level prior for social reaction dynamics. It should bias activity, "
                "trust, threat sensitivity, uncertainty, sharing, and polarisation without overriding persona identity.\n"
                f"Prior JSON: {json.dumps(neuro_prior or {}, ensure_ascii=False)}\n"
                f"Modifier JSON: {json.dumps(neuro_modifiers or {}, ensure_ascii=False)}"
            )

        # Helper: split aggregated extracted_text into per-document segments
        def split_docs(agg_text: str):
            import re
            # Expect sections like: "=== filename ===\n<content>"
            pattern = re.compile(r"(^|\n)===\s+(.+?)\s+===\n", re.MULTILINE)
            parts = []
            last_end = 0
            last_title = None
            for m in pattern.finditer(agg_text):
                start, end = m.span()
                title = m.group(2).strip()
                # Flush previous section
                if last_title is not None:
                    parts.append((last_title, agg_text[last_end:start]))
                last_title = title
                last_end = end
            # Tail
            if last_title is not None:
                parts.append((last_title, agg_text[last_end:]))
            # Fallback when no markers found
            if not parts and agg_text:
                parts = [("Document", agg_text)]
            return [(t, c.strip()) for t, c in parts if c and c.strip()]

        # Include documents with an explicit local-demo bound. This preserves
        # arbitrary-input handling without silently blowing local context/latency.
        if document_text:
            docs = split_docs(document_text)
            if docs:
                remaining = self.MAX_CONTEXT_LENGTH
                doc_sections = []
                for title, content in docs:
                    if remaining <= 0:
                        break
                    chunk = content[:remaining]
                    remaining -= len(chunk)
                    suffix = "\n[TRUNCATED FOR LOCAL CONTEXT BUDGET]" if len(content) > len(chunk) else ""
                    doc_sections.append(f"\n### Document: {title}\n{chunk}{suffix}")
                context_parts.append("\n## Original Document Content (full text from each upload)\n" + "\n".join(doc_sections))
            else:
                suffix = "\n[TRUNCATED FOR LOCAL CONTEXT BUDGET]" if len(document_text) > self.MAX_CONTEXT_LENGTH else ""
                context_parts.append(f"\n## Original Document Content\n{document_text[:self.MAX_CONTEXT_LENGTH]}{suffix}")

        return "\n".join(context_parts)

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _apply_neuro_modifiers_to_agent(
        self,
        agent: AgentActivityConfig,
        modifiers: Dict[str, Any],
        entity_type: str,
    ) -> AgentActivityConfig:
        agent.activity_level = self._clamp(agent.activity_level * float(modifiers.get("activity_multiplier", 1.0)), 0.05, 1.0)
        agent.posts_per_hour = self._clamp(agent.posts_per_hour * float(modifiers.get("posting_multiplier", 1.0)), 0.05, 10.0)
        agent.comments_per_hour = self._clamp(agent.comments_per_hour * float(modifiers.get("commenting_multiplier", 1.0)), 0.05, 20.0)
        if Config.NEURO_PRIOR_SHARED_SENTIMENT_SHIFT:
            agent.sentiment_bias = self._clamp(
                agent.sentiment_bias + float(modifiers.get("sentiment_shift", 0.0)),
                -1.0,
                1.0,
            )

        speed = float(modifiers.get("response_speed_multiplier", 1.0))
        if speed > 0:
            agent.response_delay_min = int(self._clamp(agent.response_delay_min / speed, 1, 240))
            agent.response_delay_max = int(self._clamp(agent.response_delay_max / speed, agent.response_delay_min + 1, 480))

        entity_type_l = (entity_type or "").lower()
        virality = float(modifiers.get("sharing_multiplier", 1.0))
        if entity_type_l in {"journalist", "influencer", "activist", "reporter", "mediaoutlet"}:
            agent.influence_weight = self._clamp(agent.influence_weight * (1.0 + (virality - 1.0) * 0.25), 0.1, 5.0)

        if Config.NEURO_PRIOR_CAN_OVERRIDE_STANCE:
            risk = float(modifiers.get("risk_aversion_shift", 0.0))
            sentiment = float(modifiers.get("sentiment_shift", 0.0))
            if risk > 0.18 and sentiment < -0.05:
                agent.stance = "opposing"
            elif sentiment > 0.18 and risk < 0.15:
                agent.stance = "supportive"
            elif risk > 0.25:
                agent.stance = "observer"
        return agent

    def _apply_neuro_modifiers_to_platform(
        self,
        platform_config: PlatformConfig,
        modifiers: Dict[str, Any],
    ) -> PlatformConfig:
        platform_config.echo_chamber_strength = self._clamp(
            platform_config.echo_chamber_strength + float(modifiers.get("echo_chamber_shift", 0.0)),
            0.0,
            1.0,
        )
        platform_config.viral_threshold = max(
            2,
            int(platform_config.viral_threshold + int(modifiers.get("viral_threshold_shift", 0))),
        )
        return platform_config

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generate entity summary"""
        lines = []

        # Group by type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            # Use configured display quantity and summary length
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... and {len(type_entities) - display_count} more")

        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM call with retry, including JSON repair logic"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                # LM Studio rejects response_format={"type":"json_object"}; parse JSON manually below.
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7 - (attempt * 0.1)  # Lower temperature with each retry
                    # Don't set max_tokens, let LLM generate freely
                )

                content = response.choices[0].message.content or ""
                # Strip optional ```json fences and any preamble before the first '{'.
                content = re.sub(r'^```(?:json)?\s*\n?', '', content.strip(), flags=re.IGNORECASE)
                content = re.sub(r'\n?```\s*$', '', content).strip()
                if content and not content.startswith('{'):
                    m = re.search(r'\{[\s\S]*\}', content)
                    if m:
                        content = m.group(0)
                finish_reason = response.choices[0].finish_reason

                # Check if output was truncated
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)

                # Try to parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed (attempt {attempt+1}): {str(e)[:80]}")

                    # Try to fix JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON"""
        content = content.strip()

        # Count unclosed parentheses
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Check for unclosed strings
        if content and content[-1] not in '",}]':
            content += '"'

        # Close parentheses
        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to fix configuration JSON"""
        import re

        # Fix truncated case
        content = self._fix_truncated_json(content)

        # Extract JSON portion
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # Remove newlines in strings
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # Try removing all control characters
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate time configuration"""
        # Use configured context truncation length
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calculate maximum allowed value (90% of agents)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Based on the following simulation requirements, generate a time configuration for a public reaction simulation.

{context_truncated}

## Task
Please generate time configuration JSON.

### Basic principles:
- Adapt to the topic and audience. Do not assume a financial-market-only or China-only schedule.
- 1-4am is usually low activity unless the scenario is a breaking crisis.
- 5-8am may reflect early news, commute, or market-open attention.
- 9-16 often captures institutional, journalist, regulator, executive, and workplace response.
- 17-22 is usually high public discussion and social sharing.
- Late-night discussion may increase for scandals, crises, politics, entertainment, or viral content.

### Return JSON format (no markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Explanation of time configuration for this event"
}}

Field description:
- total_simulation_hours (int): Total simulation time, 12-96 hours for local demos; short for breaking news, longer for ongoing topics
- minutes_per_round (int): Time per round, 30-120 minutes, recommend 60 minutes
- agents_per_hour_min (int): Minimum agents activated per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Maximum agents activated per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): Peak hours, adjust based on event participants
- off_peak_hours (int array): Off-peak hours, usually late night/early morning
- morning_hours (int array): Morning hours
- work_hours (int array): Work hours
- reasoning (string): Brief explanation for this configuration"""

        system_prompt = "You are a domain-adaptive social simulation architect. Return pure JSON only. Configure time for the specific public, market, political, consumer, or institutional reaction described by the prompt."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM generation failed: {e}, using default configuration")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Get default time configuration."""
        return {
            "total_simulation_hours": getattr(Config, "SIM_DEFAULT_HOURS", 36),
            "minutes_per_round": 60,  # 1 hour per round, speed up time
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [17, 18, 19, 20, 21],
            "off_peak_hours": [1, 2, 3, 4],
            "morning_hours": [5, 6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16],
            "reasoning": "Default domain-neutral public reaction schedule."
        }

    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Parse time configuration result and verify agents_per_hour doesn't exceed total agents"""
        # Get original values
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))

        # Verify and correct: ensure not exceeding total agents
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total agents ({num_entities}), corrected")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # Ensure min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}")

        return TimeSimulationConfig(
            total_simulation_hours=int(self._clamp(result.get("total_simulation_hours", getattr(Config, "SIM_DEFAULT_HOURS", 36)), 12, 96)),
            minutes_per_round=result.get("minutes_per_round", 60),  # Default 1 hour per round
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [1, 2, 3, 4]),
            off_peak_activity_multiplier=0.15,
            morning_hours=result.get("morning_hours", [5, 6, 7, 8]),
            morning_activity_multiplier=0.7,
            work_hours=result.get("work_hours", list(range(9, 17))),
            work_activity_multiplier=0.9,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generate event configuration"""

        # Get available entity types list for LLM reference
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))

        # List representative entity names for each type
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}"
            for t, examples in type_examples.items()
        ])

        # Use configured context truncation length
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Based on the following simulation requirements, generate event configuration.

Simulation Requirements: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

## Task
Please generate event configuration JSON:
- Extract hot topic keywords
- Describe opinion development direction
- Design initial post content, **each post must specify poster_type (publisher type)**

**Important**: poster_type must be selected from the "Available Entity Types" above so initial posts can be assigned to appropriate agents for publishing.
Example: Official statements should be published by Official/University type, news by MediaOutlet, student opinions by Student type.

Return JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<description of opinion development direction>",
    "initial_posts": [
        {{"content": "post content", "poster_type": "entity type (must select from available types)"}},
        ...
    ],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = "You are a domain-adaptive social simulation architect. Return pure JSON only. Generate initial posts that fit the specific stimulus, audience, and platform dynamics. Do not force financial-market framing unless the prompt is financial."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}, using default configuration")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default configuration"
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse event configuration result"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign appropriate publisher agents to initial posts

        Match agent_id based on each post's poster_type
        """
        if not event_config.initial_posts:
            return event_config

        # Build agent index by entity type
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Type mapping table — covers common demo personas and prompt-synthesized roles.
        type_aliases = {
            "analyst": ["analyst", "researchanalyst", "macroanalyst", "asset"],
            "trader": ["trader", "commoditytrader", "growthstrategist"],
            "investor": ["investor", "defensiveinvestor", "institutionalallocator", "bankinginvestor"],
            "reporter": ["reporter", "marketreporter", "contrarian"],
            "journalist": ["journalist", "reporter", "mediaoutlet", "media"],
            "activist": ["activist", "campaigner", "advocate"],
            "influencer": ["influencer", "creator", "commentator"],
            "consumer": ["consumer", "customer", "retailpublic"],
            "regulator": ["regulator", "governmentagency", "compliance"],
            "executive": ["executive", "leader", "manager"],
            "voter": ["voter", "citizen", "resident", "public"],
            "asset": ["asset", "index", "metric", "sector"],
        }

        # Track used agent indices for each type to avoid reusing same agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # Try to find matching agent
            matched_agent_id = None

            # 1. Direct match
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Match using aliases
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            # 3. If still not found, use agent with highest influence
            if matched_agent_id is None:
                logger.warning(f"No matching agent found for type '{poster_type}', using agent with highest influence")
                if agent_configs:
                    # Sort by influence, select highest
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })

            logger.info(f"Initial post assigned: poster_type='{poster_type}' -> agent_id={matched_agent_id}")

        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Generate agent configurations in batch"""

        # Build entity information (using configured summary length)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })

        prompt = f"""Based on the following information, generate social media activity configuration for each agent/persona.

Simulation Requirements: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity configuration for each entity, noting:
- Adapt activity to role and scenario. Do not assume finance unless the prompt is finance.
- Institutions/regulators/executives: slower, more cautious, active mainly during work hours, high influence.
- Journalists/media/influencers: faster, high posting/sharing, active across news-cycle and peak hours.
- Activists/opposition/supporters: high response intensity when threat, identity, or moral stakes are high.
- Consumers/voters/general public: mixed activity; stronger evening and peak-hour participation.
- Analysts/investors/traders: market-hours and event-driven activity when the topic is financial.

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <comment frequency>,
            "active_hours": [<active hours list, matched to the role, geography, platform, and scenario urgency>],
            "response_delay_min": <minimum response delay minutes>,
            "response_delay_max": <maximum response delay minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a domain-adaptive social simulation architect. Return pure JSON. Configure each persona for the specific stimulus and audience. Keep activity observable for demos but preserve role differences."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent config batch LLM generation failed: {e}, using rule-based generation")
            llm_configs = {}

        # Build AgentActivityConfig objects
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # If LLM didn't generate, use rule-based generation
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)

        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Generate single agent configuration based on broad role rules."""
        role = ((entity.get_entity_type() or "") + " " + entity.name).lower()
        if any(x in role for x in ["journalist", "media", "reporter", "influencer", "creator"]):
            return {
                "activity_level": 0.9,
                "posts_per_hour": 1.6,
                "comments_per_hour": 2.2,
                "active_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.0,
            }
        if any(x in role for x in ["regulator", "executive", "lawyer", "compliance", "official"]):
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.0,
                "active_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                "response_delay_min": 30,
                "response_delay_max": 180,
                "sentiment_bias": -0.05,
                "stance": "observer",
                "influence_weight": 2.2,
            }
        return {
            "activity_level": 0.8,
            "posts_per_hour": 1.0,
            "comments_per_hour": 1.8,
            "active_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "response_delay_min": 1,
            "response_delay_max": 45,
            "sentiment_bias": 0.0,
            "stance": "neutral",
            "influence_weight": 1.0
        }
    
