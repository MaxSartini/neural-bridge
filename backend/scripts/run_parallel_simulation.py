"""
OASIS dual-platform parallel simulation preset script
Run Twitter and Reddit simulations simultaneously with the same configuration file

Features:
- Dual-platform (Twitter + Reddit) parallel simulation
- Keep environment running after simulation completes (enter wait mode)
- Support Interview commands via IPC
- Support single Agent interview and batch interview
- Support remote environment shutdown command

Usage:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait  # Close immediately after completion
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

Log structure:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter platform action log
    ├── reddit/
    │   └── actions.jsonl    # Reddit platform action log
    ├── simulation.log       # Main simulation process log
    └── run_state.json       # Run state (for API queries)
"""

# ============================================================
# Fix Windows encoding issue: Set UTF-8 encoding before all imports
# This is to fix the issue that OASIS third-party library doesn't specify encoding when reading files
# ============================================================
import sys
import os

if sys.platform == 'win32':
    # Set Python default I/O encoding to UTF-8
    # This affects all open() calls without specified encoding
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    # Reconfigure standard output stream to UTF-8 (fix console encoding issues)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Force set default encoding (affects default encoding of open() function)
    # Note: This must be set when Python starts, runtime configuration may not work
    # So we also need to monkey-patch the built-in open function
    import builtins
    _original_open = builtins.open

    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None,
                   newline=None, closefd=True, opener=None):
        """
        Wrap open() function to use UTF-8 encoding by default for text mode
        This can fix the issue that third-party libraries (like OASIS) don't specify encoding when reading files
        """
        # Only set default encoding for text mode (non-binary) without specified encoding
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors,
                              newline, closefd, opener)

    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import math
import multiprocessing
import random
import signal
import sqlite3
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# Global variables: for signal handling
_shutdown_event = None
_cleanup_done = False

# Add backend directory to path
# Script is fixed in backend/scripts/ directory
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Load .env file from project root (contains LLM_API_KEY and other configurations)
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"Loaded environment configuration: {_env_file}")
else:
    # Try to load backend/.env
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Loaded environment configuration: {_backend_env}")


class MaxTokensWarningFilter(logging.Filter):
    """Filter out camel-ai max_tokens warnings (we intentionally don't set max_tokens to let the model decide)"""

    def filter(self, record):
        # Filter out logs containing max_tokens warnings
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# Add filter immediately when module loads, ensure it takes effect before camel code executes
logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging():
    """
    Disable verbose logging output from OASIS library
    OASIS logging is too verbose (logs every agent's observation and action), we use our own action_logger
    """
    # Disable all OASIS loggers
    oasis_loggers = [
        "social.agent",
        "social.twitter",
        "social.rec",
        "oasis.env",
        "table",
    ]

    for logger_name in oasis_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # Only log critical errors
        logger.handlers.clear()
        logger.propagate = False


def init_logging_for_simulation(simulation_dir: str):
    """
    Initialize simulation log configuration

    Args:
        simulation_dir: Simulation directory path
    """
    # Disable OASIS verbose logging
    disable_oasis_logging()

    # Clean up old log directory (if exists)
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)


from action_logger import SimulationLogManager, PlatformActionLogger

import re as _re


def _clean_interview_response(text):
    """Strip tool-call fragments Gemma sometimes emits during interviews."""
    if not isinstance(text, str) or not text.strip():
        return text
    s = text.strip()
    s = _re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', s, flags=_re.IGNORECASE).strip()
    s = _re.sub(r'^```(?:json)?\s*\n?', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'\n?```\s*$', '', s).strip()
    if s.startswith('{') and s.endswith('}'):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                name = obj.get('name') or obj.get('tool') or obj.get('function')
                args = obj.get('arguments') or obj.get('parameters') or obj.get('args') or {}
                if name:
                    if isinstance(args, dict):
                        content = args.get('content') or args.get('text') or args.get('message')
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                    return f"(Agent chose action '{name}' instead of replying.)"
                for key in ('content', 'message', 'response', 'text'):
                    v = obj.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except (json.JSONDecodeError, ValueError):
            pass
    return s or text

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"Error: Missing dependency {e}")
    print("Please install first: pip install oasis-ai camel-ai")
    sys.exit(1)


# Twitter available actions (INTERVIEW not included, INTERVIEW can only be triggered manually via ManualAction)
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.QUOTE_POST,
]

# Reddit available actions (INTERVIEW not included, INTERVIEW can only be triggered manually via ManualAction)
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.FOLLOW,
]


# ---------------------------------------------------------------------------
# Rolling round-summary memory management
# ---------------------------------------------------------------------------
# Instead of letting CAMEL's ChatAgent accumulate the full conversation
# history (which causes the LLM to see stale observations and regurgitate
# old content), we:
#   1. clear_memory() before each round  (keeps the system prompt / persona)
#   2. inject a compact summary of what happened in the *previous* round
#   3. after the round, build next round's summary from DB actions
# This keeps context lean (~1-2k tokens) while preserving continuity.
# ---------------------------------------------------------------------------

from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole

# Maximum summary length in characters.
# With local instruction models at 28k context:
#   - System prompt/persona:        ~500 tokens
#   - Scenario evidence (current):  ~700 tokens
#   - Rolling summary (prior):     ~1250 tokens  (~5000 chars)
#   - Agent response budget:        ~600 tokens
#   - Total:                       ~3050 tokens  (well inside 28k)
# Rolling chain: round N only ever sees round N-1's compressed digest.
_SUMMARY_MAX_CHARS = 3500
_SUMMARY_MAX_ITEMS = 12


def _prepare_agents_for_round(
    active_agents: list,
    round_summary: str,
    round_num: int,
    simulated_hour: int,
    simulated_day: int,
    scenario_context: str = ""
):
    """Clear each agent's memory and inject the previous round's summary.

    This prevents unbounded context growth while preserving short-term
    continuity. Only the immediately previous round is carried forward.
    """
    for _, agent in active_agents:
        # clear_memory() wipes conversation history but re-injects the
        # system message (agent persona), so identity is preserved.
        agent.clear_memory()

        evidence = scenario_context.strip() or "(No additional scenario evidence supplied.)"
        evidence_label = "SCENARIO EVIDENCE AND CONTEXT"

        # Inject time context + previous round summary
        if round_summary:
            context_msg = BaseMessage.make_user_message(
                role_name="System",
                content=(
                    f"--- TEMPORAL ANCHOR (FORWARD PROJECTION) ---\n"
                    f"Current Simulation Clock: Day {simulated_day}, {simulated_hour:02d}:00.\n"
                    f"This is Round {round_num + 1} of the Forward-Prediction Window.\n"
                    f"Anchor your reaction to this exact simulated time.\n\n"
                    f"--- {evidence_label} ---\n"
                    f"{evidence}\n\n"
                    f"--- MOST RECENT PLATFORM ACTIVITY ONLY ---\n"
                    f"{round_summary}\n\n"
                    f"CRITICAL DIRECTIVES FOR YOUR RESPONSE:\n"
                    f"1. React as your persona to the scenario and platform activity. Use concrete facts only when they appear in the evidence above.\n"
                    f"2. Do not invent statistics, dates, quotes, prices, events, or scientific claims. If evidence is thin, state uncertainty naturally.\n"
                    f"3. You are actively forbidden from copy-pasting or repeating exact sentences from the activity feed above.\n"
                    f"4. Maintain your persona's independent reasoning. Do not agree or disagree merely to imitate other agents.\n"
                    f"5. Prefer an underexplored evidence-grounded angle. If the evidence supports no meaningful update, say so rather than manufacturing novelty.\n"
                    f"6. If the scenario is financial and numerical market data is present, cite the exact numbers. Otherwise do qualitative public-reaction reasoning."
                ),
            )
        else:
            context_msg = BaseMessage.make_user_message(
                role_name="System",
                content=(
                    f"--- TEMPORAL ANCHOR (FORWARD PROJECTION) ---\n"
                    f"Current Simulation Clock: Day {simulated_day}, {simulated_hour:02d}:00.\n"
                    f"This is Round {round_num + 1} of the Forward-Prediction Window.\n"
                    f"This is the beginning of the simulation. Ground your initial reaction in the scenario evidence.\n\n"
                    f"--- {evidence_label} ---\n"
                    f"{evidence}\n\n"
                    f"CRITICAL DIRECTIVES FOR YOUR RESPONSE:\n"
                    f"1. React as your persona. Use evidence when available and uncertainty when facts are missing.\n"
                    f"2. Do not invent statistics, dates, quotes, prices, events, or scientific claims.\n"
                    f"3. Do not copy default templates. If the evidence supports no meaningful reaction, express that uncertainty."
                ),
            )
        agent.update_memory(context_msg, OpenAIBackendRole.USER)


def _build_round_summary(
    previous_summary: str,
    actual_actions: list,
    round_num: int,
    agent_names: dict,
) -> str:
    """Build a rolling summary for the next round.

    Only carries the immediately previous round's notable activity. This
    prevents topic lock-in and avoids quadratic context growth where each
    round repeatedly re-summarises old summaries.
    """
    latest_by_agent = {}
    seen_content = set()
    for action_data in actual_actions:
        name = action_data.get('agent_name', f"Agent_{action_data.get('agent_id', '?')}")
        atype = action_data.get('action_type', 'unknown')
        args = action_data.get('action_args', {})

        # Extract the interesting content from the action
        if atype in ('CREATE_POST', 'CREATE_COMMENT'):
            content = args.get('content', '')[:160]
            normalized = " ".join(content.lower().split())
            if content and normalized not in seen_content:
                seen_content.add(normalized)
                latest_by_agent[name] = f"- {name} posted: \"{content}\""
        elif atype == 'QUOTE_POST':
            content = args.get('quote_content', '')[:160]
            post_id = args.get('post_id', '?')
            normalized = " ".join(content.lower().split())
            if content and normalized not in seen_content:
                seen_content.add(normalized)
                latest_by_agent[name] = f"- {name} quoted post #{post_id}: \"{content}\""
        # Skip low-info actions (likes, follows, refresh, trend, search, etc.)

    new_parts = list(latest_by_agent.values())[:_SUMMARY_MAX_ITEMS]
    new_activity = "\n".join(new_parts) if new_parts else "(No notable new activity)"
    summary = (
        f"Round {round_num + 1} activity:\n{new_activity}\n\n"
        "Independence instruction for next round: do not restate these lines. "
        "Use them only as immediate context and make no update when evidence is insufficient."
    )

    # Truncate to keep tokens bounded — older content gets trimmed first
    if len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[-_SUMMARY_MAX_CHARS:]
        # Clean up — don't start mid-line
        first_newline = summary.find('\n')
        if first_newline > 0:
            summary = summary[first_newline + 1:]

    return summary


# IPC-related constants
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """Command type constants"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class ParallelIPCHandler:
    """
    Dual-platform IPC command handler
    
    Manage environments of both platforms, handle Interview commands
    """
    
    def __init__(
        self,
        simulation_dir: str,
        twitter_env=None,
        twitter_agent_graph=None,
        reddit_env=None,
        reddit_agent_graph=None
    ):
        self.simulation_dir = simulation_dir
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph
        
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        
        # Ensure directory exists
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """Update environment status"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "twitter_available": self.twitter_env is not None,
                "reddit_available": self.reddit_env is not None,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Poll for pending commands"""
        if not os.path.exists(self.commands_dir):
            return None
        
        # Get command files (sorted by time)
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """Send response"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # Delete command file
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def _get_env_and_graph(self, platform: str):
        """
        Get environment and agent_graph for specified platform
        
        Args:
            platform: Platform name ("twitter" or "reddit")
            
        Returns:
            (env, agent_graph, platform_name) or (None, None, None)
        """
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        else:
            return None, None, None
    
    async def _interview_single_platform(self, agent_id: int, prompt: str, platform: str) -> Dict[str, Any]:
        """
        Execute Interview on a single platform
        
        Returns:
            Dictionary containing result, or dictionary containing error
        """
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)
        
        if not env or not agent_graph:
            return {"platform": platform, "error": f"{platform}platform unavailable"}
        
        try:
            agent = agent_graph.get_agent(agent_id)
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            actions = {agent: interview_action}
            await env.step(actions)
            
            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result
            
        except Exception as e:
            return {"platform": platform, "error": str(e)}
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str, platform: str = None) -> bool:
        """
        Handle single Agent interview command
        
        Args:
            command_id: Command ID
            agent_id: Agent ID
            prompt: Interview question
            platform: Specify platform (optional)
                - "twitter": Interview only Twitter platform
                - "reddit": Interview only Reddit platform
                - None/unspecified: Interview both platforms simultaneously, return integrated result
            
        Returns:
            True means success, False means failure
        """
        # If platform is specified, only interview that platform
        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)
            
            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                print(f"  Interview failed: agent_id={agent_id}, platform={platform}, error={result['error']}")
                return False
            else:
                self.send_response(command_id, "completed", result=result)
                print(f"  Interview completed: agent_id={agent_id}, platform={platform}")
                return True
        
        # Platform not specified: interview both platforms simultaneously
        if not self.twitter_env and not self.reddit_env:
            self.send_response(command_id, "failed", error="No available simulation environment")
            return False
        
        results = {
            "agent_id": agent_id,
            "prompt": prompt,
            "platforms": {}
        }
        success_count = 0
        
        # Interview both platforms in parallel
        tasks = []
        platforms_to_interview = []
        
        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platforms_to_interview.append("twitter")
        
        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platforms_to_interview.append("reddit")
        
        # Execute in parallel
        platform_results = await asyncio.gather(*tasks)
        
        for platform_name, platform_result in zip(platforms_to_interview, platform_results):
            results["platforms"][platform_name] = platform_result
            if "error" not in platform_result:
                success_count += 1
        
        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            print(f"  Interview completed: agent_id={agent_id}, success_platforms={success_count}/{len(platforms_to_interview)}")
            return True
        else:
            errors = [f"{p}: {r.get('error', 'Unknown error')}" for p, r in results["platforms"].items()]
            self.send_response(command_id, "failed", error="; ".join(errors))
            print(f"  Interview failed: agent_id={agent_id}, All platforms failed")
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict], platform: str = None) -> bool:
        """
        Handle batch interview command
        
        Args:
            command_id: Command ID
            interviews: [{"agent_id": int, "prompt": str, "platform": str(optional)}, ...]
            platform: default platform (can be overridden by each interview item)
                - "twitter": Interview only Twitter platform
                - "reddit": Interview only Reddit platform
                - None/unspecified: Interview both platforms simultaneously for each Agent
        """
        # Group by platform
        twitter_interviews = []
        reddit_interviews = []
        both_platforms_interviews = []  # Need to interview both platforms simultaneously
        
        for interview in interviews:
            item_platform = interview.get("platform", platform)
            if item_platform == "twitter":
                twitter_interviews.append(interview)
            elif item_platform == "reddit":
                reddit_interviews.append(interview)
            else:
                # Platform not specified: interview both platforms
                both_platforms_interviews.append(interview)
        
        # Split both_platforms_interviews to two platforms
        if both_platforms_interviews:
            if self.twitter_env:
                twitter_interviews.extend(both_platforms_interviews)
            if self.reddit_env:
                reddit_interviews.extend(both_platforms_interviews)
        
        results = {}
        
        # Handle Twitter platform interview
        if twitter_interviews and self.twitter_env:
            try:
                twitter_actions = {}
                for interview in twitter_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.twitter_agent_graph.get_agent(agent_id)
                        twitter_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  Warning: Unable to get Twitter Agent {agent_id}: {e}")
                
                if twitter_actions:
                    await self.twitter_env.step(twitter_actions)
                    
                    for interview in twitter_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "twitter")
                        result["platform"] = "twitter"
                        results[f"twitter_{agent_id}"] = result
            except Exception as e:
                print(f"  Twitter batch Interview failed: {e}")
        
        # Handle Reddit platform interview
        if reddit_interviews and self.reddit_env:
            try:
                reddit_actions = {}
                for interview in reddit_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.reddit_agent_graph.get_agent(agent_id)
                        reddit_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  Warning: Unable to get Reddit Agent {agent_id}: {e}")
                
                if reddit_actions:
                    await self.reddit_env.step(reddit_actions)
                    
                    for interview in reddit_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "reddit")
                        result["platform"] = "reddit"
                        results[f"reddit_{agent_id}"] = result
            except Exception as e:
                print(f"  Reddit batch Interview failed: {e}")
        
        if results:
            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Batch Interview completed: {len(results)} Agents")
            return True
        else:
            self.send_response(command_id, "failed", error="No successful interviews")
            return False
    
    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        """Get the latest Interview result from database"""
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")
        
        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }
        
        if not os.path.exists(db_path):
            return result
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query the latest Interview record
            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))
            
            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    raw_response = info.get("response", info)
                    result["response"] = _clean_interview_response(raw_response) if isinstance(raw_response, str) else raw_response
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = _clean_interview_response(info_json)
            
            conn.close()
            
        except Exception as e:
            print(f"  Failed to read Interview result: {e}")
        
        return result
    
    async def process_commands(self) -> bool:
        """
        Process all pending commands
        
        Returns:
            True means continue running, False means should exit
        """
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\nReceived IPC command: {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", ""),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", []),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("Received close environment command")
            self.send_response(command_id, "completed", result={"message": "Environment will close"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"Unknown command type: {command_type}")
            return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Optional Chronological CSV Timeline
# ---------------------------------------------------------------------------
# If a scenario includes OHLCV-style CSVs, each simulation round maps to a
# proportional window of historical data. Non-financial scenarios simply use
# the static scenario/evidence context produced by the config generator.
# ---------------------------------------------------------------------------

class MarketDataTimeline:
    """Slice OHLCV CSV data chronologically — one compact table per round.

    Design constraints:
    - Must stay within ~700 tokens per context block for local 28k models.
    - One round = one proportional window through the full date range.
    - Table format is dense/machine-readable (for agent inference, not UI).
    """

    def __init__(self, csv_data_dir: str):
        self.frames: Dict[str, Any] = {}  # ticker -> dataframe-like list of rows
        self.all_dates: list = []
        self._available = False
        if csv_data_dir and os.path.isdir(csv_data_dir):
            self._load(csv_data_dir)

    def _load(self, directory: str):
        """Load all CSV files in directory into memory as sorted row lists."""
        import csv

        all_dates_set = set()
        filename_map = {}
        project_path = os.path.join(os.path.dirname(directory), "project.json")
        if os.path.isfile(project_path):
            try:
                project = load_config(project_path)
                filename_map = {
                    item["saved_filename"]: item["filename"]
                    for item in project.get("files", [])
                    if item.get("saved_filename") and item.get("filename")
                }
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"[Timeline] Warning: could not load original filenames: {exc}")

        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(directory, fname)
            display_name = filename_map.get(fname, fname)
            ticker = display_name.rsplit('.csv', 1)[0]
            rows = []
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Normalise the Date field — strip timezone info
                        raw_date = row.get('Date', '').strip()
                        # "2000-01-04 00:00:00+02:00" → "2000-01-04"
                        date_str = raw_date[:10]
                        try:
                            d = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except ValueError:
                            continue
                        try:
                            rows.append({
                                'date': d,
                                'open':   float(row.get('Open', 0) or 0),
                                'high':   float(row.get('High', 0) or 0),
                                'low':    float(row.get('Low',  0) or 0),
                                'close':  float(row.get('Close', 0) or 0),
                                'volume': int(float(row.get('Volume', 0) or 0)),
                            })
                            all_dates_set.add(d)
                        except (ValueError, TypeError):
                            continue
                rows.sort(key=lambda r: r['date'])
                if rows:
                    self.frames[ticker] = rows
                    print(f"[Timeline] Loaded {ticker}: {len(rows)} rows "
                          f"({rows[0]['date']} → {rows[-1]['date']})")
            except Exception as e:
                print(f"[Timeline] Warning: could not load {fname}: {e}")

        self.all_dates = sorted(all_dates_set)
        self._available = bool(self.all_dates and self.frames)
        if self._available:
            print(f"[Timeline] Ready: {len(self.frames)} tickers, "
                  f"{len(self.all_dates)} unique trading dates "
                  f"({self.all_dates[0]} → {self.all_dates[-1]})")

    @property
    def available(self) -> bool:
        return self._available

    def get_terminal(self, round_num: int, total_rounds: int) -> str:
        """Return a compact market-data string for this round's date window.

        The full date range is divided evenly across total_rounds.
        Each round sees one slice. The output is verbatim OHLCV-style data
        only; no leaders, laggards, momentum, returns, or interpretation.
        Token budget target: ≤700 tokens (~2800 chars).
        """
        if not self._available:
            return "(No chronological CSV timeline available)"

        n_dates = len(self.all_dates)
        # Map round_num → proportional date index range
        start_idx = int(round_num / total_rounds * n_dates)
        end_idx   = int((round_num + 1) / total_rounds * n_dates)
        start_idx = min(start_idx, n_dates - 1)
        end_idx   = min(max(end_idx, start_idx + 1), n_dates)

        window_dates = self.all_dates[start_idx:end_idx]
        if not window_dates:
            return "(Empty date window for this round)"

        start_date = window_dates[0]
        end_date   = window_dates[-1]
        n_days     = len(window_dates)
        window_set = set(window_dates)

        lines = [
            f"=== CHRONOLOGICAL CSV DATA ONLY | {start_date} → {end_date} | Round {round_num+1}/{total_rounds} | {n_days}d ===",
            "Instruction: verbatim price/volume rows only. No interpretation is included in this block.",
            f"{'DATE':<12} {'TICKER':<14} {'OPEN':>10} {'HIGH':>10} {'LOW':>10} {'CLOSE':>10} {'VOLUME':>12}",
        ]

        for ticker, rows in sorted(self.frames.items()):
            # Rows in this window
            win_rows = [r for r in rows if r['date'] in window_set]
            if not win_rows:
                continue
            if len(win_rows) > 8:
                positions = [
                    round(index * (len(win_rows) - 1) / 7)
                    for index in range(8)
                ]
                display_rows = [win_rows[index] for index in positions]
            else:
                display_rows = win_rows
            for row in display_rows:
                lines.append(
                    f"{row['date'].isoformat():<12} {ticker:<14} "
                    f"{row['open']:>10.2f} {row['high']:>10.2f} "
                    f"{row['low']:>10.2f} {row['close']:>10.2f} "
                    f"{row['volume']:>12,}"
                )
            if len(win_rows) > 8:
                lines.append(
                    f"... {ticker}: displayed 8 evenly spaced rows from {len(win_rows)} total"
                )

        lines.append("=" * 88)
        return "\n".join(lines)


def _build_market_timeline(config: Dict[str, Any]) -> 'MarketDataTimeline':
    """Build a MarketDataTimeline from the simulation config.

    Looks for CSV files in `csv_data_dir` (injected by simulation_manager).
    Falls back to a no-op timeline if the directory is absent or empty.
    """
    csv_dir = config.get('csv_data_dir', '')
    return MarketDataTimeline(csv_dir)


def _build_round_scenario_context(config: Dict[str, Any], timeline: MarketDataTimeline, round_num: int, total_rounds: int) -> str:
    """Build compact per-round evidence context for arbitrary scenarios."""
    parts = []

    base_context = (
        config.get("scenario_context")
        or config.get("market_context")
        or config.get("simulation_requirement")
        or ""
    )
    if base_context:
        parts.append(str(base_context)[:3500])

    if timeline.available:
        parts.append(timeline.get_terminal(round_num, total_rounds))

    neuro_prior = config.get("neuro_prior") or {}
    neuro_modifiers = config.get("neuro_modifiers") or {}
    if _NEURO_PRIOR_IN_ROUND_PROMPTS and (neuro_prior or neuro_modifiers):
        prior_lines = ["### Behavioural Prior Calibration"]
        prior_lines.append(
            "Population-level behavioural bias only; do not mention BOLD, brain scans, or brain reading."
        )
        for key, label in [
            ("salience_score", "salience"),
            ("threat_score", "threat sensitivity"),
            ("reward_score", "reward/hope sensitivity"),
            ("arousal_score", "arousal"),
            ("uncertainty_score", "uncertainty"),
            ("approach_bias", "approach bias"),
            ("avoidance_bias", "avoidance bias"),
            ("polarisation_risk", "polarisation risk"),
            ("virality_pressure", "virality pressure"),
        ]:
            if key in neuro_prior:
                prior_lines.append(f"- {label}: {neuro_prior.get(key)}")
        instruction = neuro_modifiers.get("persona_instruction")
        if instruction:
            prior_lines.append(f"- behavioural instruction: {instruction}")
        parts.append("\n".join(prior_lines))

    return "\n\n".join(part for part in parts if part).strip()


# Non-core action types to be filtered (these actions have low analytical value)
FILTERED_ACTIONS = {'refresh', 'sign_up'}

# Action type mapping table (Database name -> standard name)
ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}


def get_agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Get mapping of agent_id -> entity_name from simulation_config
    
    This allows displaying real entity names in actions.jsonl instead of codes like "Agent_0"
    
    Args:
        config: Content of simulation_config.json
        
    Returns:
        Mapping dictionary of agent_id -> entity_name
    """
    agent_names = {}
    agent_configs = config.get("agent_configs", [])
    
    for agent_config in agent_configs:
        agent_id = agent_config.get("agent_id")
        entity_name = agent_config.get("entity_name", f"Agent_{agent_id}")
        if agent_id is not None:
            agent_names[agent_id] = entity_name
    
    return agent_names


def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Get new action records from Database and supplement complete context information
    
    Args:
        db_path: Database file path
        last_rowid: Maximum rowid value from last read (use rowid instead of created_at because different platforms have different created_at formats)
        agent_names: agent_id -> agent_name mapping
        
    Returns:
        (actions_list, new_last_rowid)
        - actions_list: List of actions, each element contains agent_id, agent_name, action_type, action_args (including context information)
        - new_last_rowid: New maximum rowid value
    """
    actions = []
    new_last_rowid = last_rowid
    
    if not os.path.exists(db_path):
        return actions, new_last_rowid
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Use rowid to track processed records (rowid is SQLite's built-in auto-increment field)
        # This avoids created_at format differences (Twitter uses integers, Reddit uses datetime strings)
        cursor.execute("""
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """, (last_rowid,))
        
        for rowid, user_id, action, info_json in cursor.fetchall():
            # Update maximum rowid
            new_last_rowid = rowid
            
            # Filter non-core actions
            if action in FILTERED_ACTIONS:
                continue
            
            # Parse action arguments
            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}
            
            # Simplify action_args, keep only key fields (keep full content, no truncation)
            simplified_args = {}
            if 'content' in action_args:
                simplified_args['content'] = action_args['content']
            if 'post_id' in action_args:
                simplified_args['post_id'] = action_args['post_id']
            if 'comment_id' in action_args:
                simplified_args['comment_id'] = action_args['comment_id']
            if 'quoted_id' in action_args:
                simplified_args['quoted_id'] = action_args['quoted_id']
            if 'new_post_id' in action_args:
                simplified_args['new_post_id'] = action_args['new_post_id']
            if 'follow_id' in action_args:
                simplified_args['follow_id'] = action_args['follow_id']
            if 'query' in action_args:
                simplified_args['query'] = action_args['query']
            if 'like_id' in action_args:
                simplified_args['like_id'] = action_args['like_id']
            if 'dislike_id' in action_args:
                simplified_args['dislike_id'] = action_args['dislike_id']
            
            # Convert action type names
            action_type = ACTION_TYPE_MAP.get(action, action.upper())
            
            # Supplement context information (post content, usernames, etc.)
            _enrich_action_context(cursor, action_type, simplified_args, agent_names)
            
            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified_args,
            })
        
        conn.close()
    except Exception as e:
        print(f"Failed to read Database actions: {e}")
    
    return actions, new_last_rowid


def _enrich_action_context(
    cursor,
    action_type: str,
    action_args: Dict[str, Any],
    agent_names: Dict[int, str]
) -> None:
    """
    for actionSupplement context information (post content, usernames, etc.)
    
    Args:
        cursor: Database cursor
        action_type: Action type
        action_args: Action arguments (will be modified)
        agent_names: agent_id -> agent_name mapping
    """
    try:
        # Like/dislike post: supplement post content and author
        if action_type in ('LIKE_POST', 'DISLIKE_POST'):
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
        
        # Repost: supplement original post content and author
        elif action_type == 'REPOST':
            new_post_id = action_args.get('new_post_id')
            if new_post_id:
                # Repost's original_post_id points to original post
                cursor.execute("""
                    SELECT original_post_id FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    original_post_id = row[0]
                    original_info = _get_post_info(cursor, original_post_id, agent_names)
                    if original_info:
                        action_args['original_content'] = original_info.get('content', '')
                        action_args['original_author_name'] = original_info.get('author_name', '')
        
        # Quote post: supplement original post content, author, and quote comment
        elif action_type == 'QUOTE_POST':
            quoted_id = action_args.get('quoted_id')
            new_post_id = action_args.get('new_post_id')
            
            if quoted_id:
                original_info = _get_post_info(cursor, quoted_id, agent_names)
                if original_info:
                    action_args['original_content'] = original_info.get('content', '')
                    action_args['original_author_name'] = original_info.get('author_name', '')
            
            # Get quote post comment content (quote_content)
            if new_post_id:
                cursor.execute("""
                    SELECT quote_content FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    action_args['quote_content'] = row[0]
        
        # Follow user: supplement followed user name
        elif action_type == 'FOLLOW':
            follow_id = action_args.get('follow_id')
            if follow_id:
                # Get followee_id from follow table
                cursor.execute("""
                    SELECT followee_id FROM follow WHERE follow_id = ?
                """, (follow_id,))
                row = cursor.fetchone()
                if row:
                    followee_id = row[0]
                    target_name = _get_user_name(cursor, followee_id, agent_names)
                    if target_name:
                        action_args['target_user_name'] = target_name
        
        # Mute user: supplement muted user name
        elif action_type == 'MUTE':
            # Get user_id or target_id from action_args
            target_id = action_args.get('user_id') or action_args.get('target_id')
            if target_id:
                target_name = _get_user_name(cursor, target_id, agent_names)
                if target_name:
                    action_args['target_user_name'] = target_name
        
        # Like/dislike comment: supplement comment content and author
        elif action_type in ('LIKE_COMMENT', 'DISLIKE_COMMENT'):
            comment_id = action_args.get('comment_id')
            if comment_id:
                comment_info = _get_comment_info(cursor, comment_id, agent_names)
                if comment_info:
                    action_args['comment_content'] = comment_info.get('content', '')
                    action_args['comment_author_name'] = comment_info.get('author_name', '')
        
        # Post comment: supplement commented post information
        elif action_type == 'CREATE_COMMENT':
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
    
    except Exception as e:
        # Context supplement failure does not affect main process
        print(f"Failed to supplement action context: {e}")


def _get_post_info(
    cursor,
    post_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Get post information
    
    Args:
        cursor: Database cursor
        post_id: Post ID
        agent_names: agent_id -> agent_name mapping
        
    Returns:
        Dictionary containing content and author_name, or None
    """
    try:
        cursor.execute("""
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
        """, (post_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            
            # Preferentially use name from agent_names
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # Get name from user table
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''
            
            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def _get_user_name(
    cursor,
    user_id: int,
    agent_names: Dict[int, str]
) -> Optional[str]:
    """
    Get user name
    
    Args:
        cursor: Database cursor
        user_id: User ID
        agent_names: agent_id -> agent_name mapping
        
    Returns:
        User name, or None
    """
    try:
        cursor.execute("""
            SELECT agent_id, name, user_name FROM user WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            agent_id = row[0]
            name = row[1]
            user_name = row[2]
            
            # Preferentially use name from agent_names
            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ''
    except Exception:
        pass
    return None


def _get_comment_info(
    cursor,
    comment_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Get comment information
    
    Args:
        cursor: Database cursor
        comment_id: Comment ID
        agent_names: agent_id -> agent_name mapping
        
    Returns:
        Dictionary containing content and author_name, or None
    """
    try:
        cursor.execute("""
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        """, (comment_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            
            # Preferentially use name from agent_names
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # Get name from user table
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''
            
            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    Create LLM model
    
    Support dual LLM configuration for acceleration during parallel simulation：
    - Common configuration：LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
    - Acceleration configuration (optional)：LLM_BOOST_API_KEY, LLM_BOOST_BASE_URL, LLM_BOOST_MODEL_NAME
    
    If acceleration LLM is configured, different platforms can use different API providers during parallel simulation to improve concurrency.
    
    Args:
        config: Simulation configuration dictionary
        use_boost: Whether to use acceleration LLM configuration (if available)
    """
    # Check if acceleration configuration exists
    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)
    
    # Choose which LLM to use based on parameters and configuration
    if use_boost and has_boost_config:
        # Use acceleration configuration
        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[Acceleration LLM]"
    else:
        # useCommon configuration
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[Common LLM]"
    
    # If model name is not in .env, use config as fallback
    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    
    # Fall back to config-file base_url if env is empty (sim subprocesses may miss .env).
    if not llm_base_url:
        llm_base_url = config.get("llm_base_url", "")
    if not llm_api_key:
        llm_api_key = config.get("llm_api_key", "") or "lm-studio"

    # Set env vars for any downstream OpenAI SDK consumer.
    os.environ["OPENAI_API_KEY"] = llm_api_key
    if llm_base_url:
        # Both names — legacy and current — so CAMEL + openai-python both resolve to local LM Studio.
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        os.environ["OPENAI_BASE_URL"] = llm_base_url
        os.environ["OPENAI_API_BASE"] = llm_base_url

    print(f"{config_label} model={llm_model}, base_url={llm_base_url or 'default'}")

    # Pass url/api_key explicitly — ModelFactory does not read env on all camel-ai builds.
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
        url=llm_base_url or None,
        api_key=llm_api_key,
        model_config_dict={
            "temperature": float(
                config.get("llm_temperature", os.environ.get("OASIS_LLM_TEMPERATURE", "0.6"))
            ),
            "top_p": float(config.get("llm_top_p", os.environ.get("OASIS_LLM_TOP_P", "0.9")))
            # WARNING: frequency_penalty and presence_penalty removed because they physically crash
            # local LLM server schemas (LM Studio 'Channel Error') when used alongside Tool Calls.
        }
    )


# Performance envelope — read once from env so the simulation subprocess
# honours user overrides without importing the Flask Config class.
_ENV_SEMAPHORE = int(os.environ.get("OASIS_ENV_SEMAPHORE", "1"))
_MAX_ACTIVE_AGENTS = int(os.environ.get("OASIS_MAX_ACTIVE_AGENTS_PER_ROUND", "8"))
_EMPTY_ROUND_THRESHOLD = int(os.environ.get("OASIS_EMPTY_ROUND_SKIP_THRESHOLD", "2"))
_SKIP_DEAD_HOURS = os.environ.get("OASIS_SKIP_DEAD_HOURS", "true").lower() == "true"
_SERIAL_PLATFORMS = os.environ.get("OASIS_SERIAL_PLATFORMS", "true").lower() == "true"
_NEURO_PRIOR_IN_ROUND_PROMPTS = os.environ.get(
    "OASIS_NEURO_PRIOR_IN_ROUND_PROMPTS", "false"
).lower() == "true"
_RANDOM_SEED = int(os.environ.get("OASIS_RANDOM_SEED", "33"))


def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int
) -> List:
    """Select a bounded, role-aware set of agents for the current round.

    The simulator needs visible demo activity without making every persona
    speak every round. Selection uses generated activity config, active hours,
    and a local max cap so LM Studio usage stays predictable.
    """
    agent_configs = config.get("agent_configs", [])

    if not agent_configs:
        return []

    available_agents = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        try:
            agent = env.agent_graph.get_agent(agent_id)
            available_agents.append((cfg, agent_id, agent))
        except Exception:
            pass

    if not available_agents:
        return []

    # Keep rounds active for demos, but do not exceed the local inference cap.
    max_active = max(1, min(_MAX_ACTIVE_AGENTS, len(available_agents)))
    min_active = min(max_active, max(1, min(3, len(available_agents))))

    selected = []
    scored_fallback = []

    minutes_per_round = float(config.get("time_config", {}).get("minutes_per_round", 60))
    for cfg, agent_id, agent in available_agents:
        probability, hour_match = _agent_activation_probability(
            cfg, current_hour, minutes_per_round
        )
        # Slight deterministic jitter avoids the exact same cohort each round.
        jitter = random.random() * 0.08
        probability = max(0.03, min(0.95, probability + jitter))
        score = probability + (0.15 if hour_match else 0.0)
        scored_fallback.append((score, cfg, agent_id, agent))

        if random.random() <= probability:
            selected.append((agent_id, agent))

    if len(selected) < min_active:
        selected_ids = {agent_id for agent_id, _ in selected}
        scored_fallback.sort(key=lambda item: item[0], reverse=True)
        for _, _, agent_id, agent in scored_fallback:
            if agent_id in selected_ids:
                continue
            selected.append((agent_id, agent))
            selected_ids.add(agent_id)
            if len(selected) >= min_active:
                break

    if len(selected) > max_active:
        random.shuffle(selected)
        selected = selected[:max_active]

    return selected


def _agent_activation_probability(
    cfg: Dict[str, Any],
    current_hour: int,
    minutes_per_round: float,
) -> Tuple[float, bool]:
    """Convert configured activity/rates/delay into a bounded round probability."""
    try:
        activity_level = float(cfg.get("activity_level", 0.65))
    except (TypeError, ValueError):
        activity_level = 0.65
    activity_level = max(0.05, min(1.0, activity_level))

    active_hours = cfg.get("active_hours") or list(range(24))
    try:
        hour_match = current_hour in {int(hour) % 24 for hour in active_hours}
    except (TypeError, ValueError):
        hour_match = True
    hour_factor = 1.0 if hour_match else (0.25 if not _SKIP_DEAD_HOURS else 0.12)

    try:
        actions_per_hour = max(
            0.0,
            float(cfg.get("posts_per_hour", 1.0)) + float(cfg.get("comments_per_hour", 2.0)),
        )
    except (TypeError, ValueError):
        actions_per_hour = 3.0
    try:
        mean_delay = (
            float(cfg.get("response_delay_min", 5))
            + float(cfg.get("response_delay_max", 60))
        ) / 2.0
    except (TypeError, ValueError):
        mean_delay = 32.5
    speed_factor = max(0.25, min(2.0, 60.0 / max(1.0, mean_delay)))
    expected_actions = actions_per_hour * max(1.0, minutes_per_round) / 60.0 * speed_factor
    opportunity_probability = 1.0 - math.exp(-expected_actions)
    return (
        max(0.03, min(0.95, activity_level * hour_factor * opportunity_probability)),
        hour_match,
    )


class PlatformSimulation:
    """Platform simulation result container"""
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0


async def run_twitter_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """Run Twitter simulation
    
    Args:
        config: Simulation configuration
        simulation_dir: Simulation directory
        action_logger: Action logger
        main_logger: Main logger manager
        max_rounds: Maximum simulation rounds (optional, used to truncate long simulations)
        
    Returns:
        PlatformSimulation: Result object containing env and agent_graph
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Twitter] {msg}")
        print(f"[Twitter] {msg}")
    
    log_info("Initializing...")
    
    # Twitter use common LLM configuration
    model = create_model(config, use_boost=False)
    
    # OASIS Twitter uses CSV format
    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        log_info(f"Error: Profile file does not exist: {profile_path}")
        return result
    
    result.agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )
    
    # Get Agent real name mapping from config (use entity_name instead of default Agent_X)
    agent_names = get_agent_names_from_config(config)
    # If an agent is not in config, use OASIS default name
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        semaphore=_ENV_SEMAPHORE,  # Tuned for single-instance local Gemma; see OASIS_ENV_SEMAPHORE
    )
    
    await result.env.reset()
    log_info("Environment started")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # Track last processed row in Database (use rowid to avoid created_at format differences)
    
    # Execute initial events
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    # Log round 0 start (initial event phase)
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
            except Exception as exc:
                log_info(f"Skipping invalid initial post for agent_id={agent_id}: {exc}")
        
        if initial_actions:
            await result.env.step(initial_actions)
            initial_action_count = len(initial_actions)
            total_actions += initial_action_count
            log_info(f"Published {len(initial_actions)} initial posts")
    
    # Log round 0 end
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    # Main simulation loop
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # If maximum rounds specified, truncate
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"Rounds truncated: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()

    consecutive_empty = 0
    rolling_summary = ""  # Rolling round summary for memory management
    # Build optional chronological CSV timeline. Non-financial demos fall back
    # to static scenario evidence plus behavioural prior calibration.
    timeline = _build_market_timeline(config)

    for round_num in range(total_rounds):
        # Check if received exit signal
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Received exit signal，at round {round_num + 1} stop simulation")
            break

        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1

        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )

        # Log round start regardless of active agents
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)

        if not active_agents:
            consecutive_empty += 1
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            # Empty-round guard: skip forward without invoking agents.
            if _EMPTY_ROUND_THRESHOLD > 0 and consecutive_empty >= _EMPTY_ROUND_THRESHOLD:
                # Fast-forward — cheap no-op; simulated clock still advances via loop.
                pass
            continue

        consecutive_empty = 0

        scenario_context = _build_round_scenario_context(config, timeline, round_num, total_rounds)

        # --- Rolling summary memory management ---
        # Clear stale history and inject condensed summary from prior rounds
        _prepare_agents_for_round(
            active_agents, rolling_summary, round_num,
            simulated_hour, simulated_day, scenario_context
        )

        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)

        # Get actual executed actions from Database and log
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )

        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
            total_actions += 1
            round_action_count += 1

        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)

        # Update rolling summary with this round's actions
        rolling_summary = _build_round_summary(
            rolling_summary, actual_actions, round_num, agent_names
        )

        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")

    # Note: Do not close environment, keep for Interview use

    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)

    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation loop completed! Time taken: {elapsed:.1f}seconds, Total actions: {total_actions}")

    return result


async def run_reddit_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """Run Reddit simulation
    
    Args:
        config: Simulation configuration
        simulation_dir: Simulation directory
        action_logger: Action logger
        main_logger: Main logger manager
        max_rounds: Maximum simulation rounds (optional, used to truncate long simulations)
        
    Returns:
        PlatformSimulation: Result object containing env and agent_graph
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")
    
    log_info("Initializing...")
    
    # Reddit use acceleration LLM configuration(if available，otherwise fallback toCommon configuration）
    model = create_model(config, use_boost=True)
    
    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"Error: Profile file does not exist: {profile_path}")
        return result
    
    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )
    
    # Get Agent real name mapping from config (use entity_name instead of default Agent_X)
    agent_names = get_agent_names_from_config(config)
    # If an agent is not in config, use OASIS default name
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=_ENV_SEMAPHORE,  # Tuned for single-instance local Gemma; see OASIS_ENV_SEMAPHORE
    )
    
    await result.env.reset()
    log_info("Environment started")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # Track last processed row in Database (use rowid to avoid created_at format differences)
    
    # Execute initial events
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    # Log round 0 start (initial event phase)
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                if agent in initial_actions:
                    if not isinstance(initial_actions[agent], list):
                        initial_actions[agent] = [initial_actions[agent]]
                    initial_actions[agent].append(ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    ))
                else:
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
            except Exception as exc:
                log_info(f"Skipping invalid initial post for agent_id={agent_id}: {exc}")
        
        if initial_actions:
            await result.env.step(initial_actions)
            initial_action_count = len(initial_actions)
            total_actions += initial_action_count
            log_info(f"Published {len(initial_actions)} initial posts")
    
    # Log round 0 end
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    # Main simulation loop
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # If maximum rounds specified, truncate
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"Rounds truncated: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()

    consecutive_empty = 0
    rolling_summary = ""  # Rolling round summary for memory management
    timeline = _build_market_timeline(config)

    for round_num in range(total_rounds):
        # Check if received exit signal
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Received exit signal，at round {round_num + 1} stop simulation")
            break

        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1

        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )

        # Log round start regardless of active agents
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)

        if not active_agents:
            consecutive_empty += 1
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue

        consecutive_empty = 0

        scenario_context = _build_round_scenario_context(config, timeline, round_num, total_rounds)

        # --- Rolling summary memory management ---
        # Clear stale history and inject condensed summary from prior rounds
        _prepare_agents_for_round(
            active_agents, rolling_summary, round_num,
            simulated_hour, simulated_day, scenario_context
        )

        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)
        
        # Get actual executed actions from Database and log
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
            total_actions += 1
            round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        # Update rolling summary with this round's actions
        rolling_summary = _build_round_summary(
            rolling_summary, actual_actions, round_num, agent_names
        )

        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    # Note: Do not close environment, keep for Interview use
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Simulation loop completed! Time taken: {elapsed:.1f}seconds, Total actions: {total_actions}")
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='OASIS Dual-Platform Parallel Simulation')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Configuration file path (simulation_config.json)'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='Only run Twitter simulation'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='Only run Reddit simulation'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='Maximum simulation rounds (optional, used to truncate long simulations)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='Close environment immediately after simulation completes, do not enter wait mode'
    )
    
    args = parser.parse_args()
    
    # Create shutdown event at the start of main function to ensure the whole program can respond to exit signal
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file does not exist: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    random.seed(int(config.get("random_seed", _RANDOM_SEED)))
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait
    
    # Initialize logging configuration (disable OASIS logs, clean up old files)
    init_logging_for_simulation(simulation_dir)
    
    # Create log manager
    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()
    
    log_manager.info("=" * 60)
    log_manager.info("OASIS dual-platform simulation")
    log_manager.info(f"Configuration file: {args.config}")
    log_manager.info(f"Simulation ID: {config.get('simulation_id', 'unknown')}")
    log_manager.info(f"Wait mode: {'Enabled' if wait_for_commands else 'Disabled'}")
    log_manager.info("=" * 60)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    
    log_manager.info(f"Simulation parameters:")
    log_manager.info(f"  - Total simulation duration: {total_hours}hours")
    log_manager.info(f"  - Time per round: {minutes_per_round}minutes")
    log_manager.info(f"  - Configured total rounds: {config_total_rounds}")
    log_manager.info(f"  - Agent LLM semaphore: {_ENV_SEMAPHORE} (1 = one local model call at a time)")
    log_manager.info(f"  - Max active agents per round: {_MAX_ACTIVE_AGENTS}")
    log_manager.info(f"  - Platform execution: {'serial' if _SERIAL_PLATFORMS else 'parallel'}")
    if args.max_rounds:
        log_manager.info(f"  - Maximum rounds limit: {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(f"  - Actual execution rounds: {args.max_rounds} (Truncated)")
    log_manager.info(f"  - Number of Agents: {len(config.get('agent_configs', []))}")
    
    log_manager.info("Log structure:")
    log_manager.info(f"  - Main log: simulation.log")
    log_manager.info(f"  - Twitter actions: twitter/actions.jsonl")
    log_manager.info(f"  - Reddit actions: reddit/actions.jsonl")
    log_manager.info("=" * 60)
    
    start_time = datetime.now()
    
    # Store simulation results of both platforms
    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    
    if args.twitter_only:
        twitter_result = await run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds)
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds)
    else:
        if _SERIAL_PLATFORMS:
            log_manager.info("Running platforms serially for local single-model inference")
            twitter_result = await run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds)
            reddit_result = await run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds)
        else:
            # Run in parallel only when explicitly enabled for smaller/cloud models.
            results = await asyncio.gather(
                run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds),
                run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds),
            )
            twitter_result, reddit_result = results
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"Simulation loop completed! Total time: {total_elapsed:.1f}seconds")
    
    # Whether to enter wait mode
    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info("Enter wait mode - environment keeps running")
        log_manager.info("Supported commands: interview, batch_interview, close_env")
        log_manager.info("=" * 60)
        
        # Create IPC handler
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None
        )
        ipc_handler.update_status("alive")
        
        # Command wait loop (using global _shutdown_event)
        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break
                # Use wait_for instead of sleep to respond to shutdown_event
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break  # Received exit signal
                except asyncio.TimeoutError:
                    pass  # Timeout continue loop
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        except asyncio.CancelledError:
            print("\nTask was cancelled")
        except Exception as e:
            print(f"\nError processing command: {e}")
        
        log_manager.info("\nClose environment...")
        ipc_handler.update_status("stopped")
    
    # Close environment
    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] Environment closed")
    
    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] Environment closed")
    
    log_manager.info("=" * 60)
    log_manager.info(f"All completed!")
    log_manager.info(f"Log files:")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)


def setup_signal_handlers(loop=None):
    """
    Set signal handlers to ensure proper exit when receiving SIGTERM/SIGINT
    
    Persistent simulation scenario：Simulation completeafter does not exit，Wait for interview command
    When receiving termination signal, need to：
    1. Notify asyncio loop to exit wait
    2. Give program a chance to clean up resources properly (close database, environment, etc.)
    3. Then exit
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nReceived {sig_name} signal, exiting...")
        
        if not _cleanup_done:
            _cleanup_done = True
            # Set event to notify asyncio loop to exit (give loop a chance to clean up)
            if _shutdown_event:
                _shutdown_event.set()
        
        # Don't directly sys.exit(), let asyncio loop exit normally and clean up
        # Force exit only if signal is received repeatedly
        else:
            print("Force exit...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except SystemExit:
        pass
    finally:
        # Clean up multiprocessing resource tracker (prevent warning on exit)
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Simulation process exited")
