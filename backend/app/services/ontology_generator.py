"""
Ontology generation service
Interface 1: Analyze text content and generate entity and relationship type definitions suitable for social simulation
"""

import json
import os
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """You are a professional knowledge graph ontology design expert. Analyze the given documents and simulation topic, then design an appropriate graph topology that best represents the subject matter.

**Important: You must output valid JSON format data, do not output anything else.**

## Core Task

Design an ontology with:
- The right **entity types** for this domain (what the important subjects/objects are)
- The right **relationship types** (how they connect)

## Rules

### Entity Types (6-10 types)
- Match types to the DOMAIN of the documents/topic:
  - Financial data: Asset, Metric, Index, Sector, Commodity, Company
  - Public opinion / social: Person, Organization, Topic, Event, Claim, MediaOutlet
  - Political: Politician, Party, Policy, Institution, Constituency
  - General: infer from document content
- Use PascalCase names
- 1-2 key attributes per type
- **Do NOT include simulation-persona archetypes** such as GrowthStrategist, DefensiveInvestor, MarketReporter. These are agent roles, not graph subjects.
- Attribute names cannot use: name, uuid, group_id, use full_name, org_name, etc.

### Relationship Types (5-10 types)
- Reflect real connections in the domain
- Financial examples: CORRELATES_WITH, LEADS_MARKET, BELONGS_TO_SECTOR, HAS_METRIC
- Social/opinion examples: COMMENTED_ON, SUPPORTS, OPPOSES, REPLIED_TO, MENTIONS
- Political examples: VOTED_FOR, AFFILIATED_WITH, PROPOSED, ENACTED
- Use UPPER_SNAKE_CASE

## Output Format

```json
{
    "entity_types": [
        {
            "name": "EntityTypeName",
            "description": "Brief description (max 100 chars)",
            "attributes": [
                {
                    "name": "attribute_name",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example 1", "Example 2"]
        }
    ],
    "edge_types": [
        {
            "name": "RELATIONSHIP_TYPE",
            "description": "Brief description (max 100 chars)",
            "source_targets": [
                {"source": "SourceType", "target": "TargetType"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief explanation of the domain and chosen topology"
}
```

## Design Guidelines

- Quantity: 6-10 entity types accurately representing the domain
- Types must reflect what ACTUALLY APPEARS in the documents
- Prefer specific types over generic ones
- Relationships: 5-10 types reflecting real data connections
- FLOWS_INTO: Capital flows from one sector to another.
"""


class OntologyGenerator:
    """
    Ontology generator
    Analyze text content and generate entity and relationship type definitions
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definition

        Args:
            document_texts: List of document texts
            simulation_requirement: Description of simulation requirements
            additional_context: Additional context

        Returns:
            Ontology definition (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Call LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Validate and post-process
        result = self._validate_and_process(result)

        # Parse any ticker → preferred-name aliases present in the simulation requirement
        # Lines like: "PRX.JO.csv = Prosus N.V." or "PRX.JO = Prosus N.V."
        result["aliases"] = self._parse_aliases(simulation_requirement)

        return result

    # Maximum text length for LLM — configurable to match local model window
    MAX_TEXT_LENGTH_FOR_LLM = int(os.environ.get("LLM_CONTEXT_MAX_CHARS", "20000"))

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build user message"""

        # Combine texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # If text exceeds 50,000 characters, truncate (only affects LLM input, not graph construction)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Original text has {original_length} characters, first {self.MAX_TEXT_LENGTH_FOR_LLM} characters extracted for ontology analysis)..."

        message = f"""## Simulation Requirements

{simulation_requirement}

## Document Content

{combined_text}
"""

        if additional_context:
            message += f"""
## Additional Explanation

{additional_context}
"""

        message += """
Based on the above content, design entity types and relationship types suitable for social opinion simulation.

**Rules to follow**:
1. Must output exactly 12 entity types.
2. Last 4 must be fallback types: Person, Organization, Asset, and Metric.
3. First 8 are specific types designed based on text content.
4. All entity types must be real-world subjects that can voice opinions, not abstract concepts
5. Attribute names cannot use reserved words like name, uuid, group_id, use full_name, org_name, etc. instead
"""

        return message

    @staticmethod
    def _parse_aliases(simulation_requirement: str) -> Dict[str, str]:
        """Extract ticker → preferred name mappings from freeform instructions.

        Supports lines like:
          TICKER.csv = Preferred Name
          TICKER = Preferred Name
        Returns mapping for both raw ticker and ticker without .csv suffix.
        """
        import re
        aliases: Dict[str, str] = {}
        if not simulation_requirement:
            return aliases
        for line in simulation_requirement.splitlines():
            m = re.match(r"\s*([A-Za-z0-9_.-]+)(?:\.csv)?\s*=\s*(.+?)\s*$", line)
            if m:
                ticker = m.group(1)
                name = m.group(2)
                if ticker and name:
                    aliases[ticker] = name
        return aliases
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process result"""

        # Ensure necessary fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Validate entity types
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description doesn't exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Validate relationship types
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Display/complexity limits
        MAX_ENTITY_TYPES = 14
        MAX_EDGE_TYPES = 18

        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        asset_fallback = {
            "name": "Asset",
            "description": "A tradable financial instrument, ticker, index, commodity, or named dataset.",
            "attributes": [{"name": "asset_class", "type": "text", "description": "Asset Class"}, {"name": "ticker", "type": "text", "description": "Ticker"}],
            "examples": ["BTC-USD", "SP500"]
        }
        
        metric_fallback = {
            "name": "Metric",
            "description": "A named numeric indicator of an asset.",
            "attributes": [{"name": "unit", "type": "text", "description": "Unit"}, {"name": "direction", "type": "text", "description": "Direction"}],
            "examples": ["BTC-USD.Close", "SP500.Volume"]
        }

        # Check if fallback types already exist
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        has_asset = "Asset" in entity_names
        has_metric = "Metric" in entity_names

        # Fallback types to add
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        if not has_asset:
            fallbacks_to_add.append(asset_fallback)
        if not has_metric:
            fallbacks_to_add.append(metric_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # If adding would exceed 14, need to remove some existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from end (keep more important specific types in front)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Add fallback types
            result["entity_types"].extend(fallbacks_to_add)

        # Ensure a broader canonical set appears when the model returned too few
        # This addresses cases where local models over-focus on Asset/Metric only.
        canonical_specific_types = [
            {"name": "MediaOutlet", "description": "News or media organization covering markets.", "attributes": [{"name": "outlet_name", "type": "text", "description": "Name"}]},
            {"name": "ResearchAnalyst", "description": "Sell-side or independent analyst producing research.", "attributes": [{"name": "firm", "type": "text", "description": "Affiliation"}]},
            {"name": "InstitutionalAllocator", "description": "Fund or allocator moving capital.", "attributes": [{"name": "strategy", "type": "text", "description": "Strategy"}]},
            {"name": "DataTrader", "description": "Trader acting on quantitative signals.", "attributes": [{"name": "style", "type": "text", "description": "Style"}]},
            {"name": "Company", "description": "Operating company or issuer.", "attributes": [{"name": "ticker", "type": "text", "description": "Ticker"}]},
            {"name": "GovernmentAgency", "description": "Regulatory or policy authority.", "attributes": [{"name": "jurisdiction", "type": "text", "description": "Jurisdiction"}]},
            {"name": "Exchange", "description": "Marketplace where assets trade.", "attributes": [{"name": "code", "type": "text", "description": "Exchange code"}]},
            {"name": "Index", "description": "Composite market index.", "attributes": [{"name": "ticker", "type": "text", "description": "Symbol"}]},
        ]

        # If model already supplied some of these, don’t duplicate
        existing = {e["name"] for e in result["entity_types"]}
        for et in canonical_specific_types:
            if len(result["entity_types"]) >= MAX_ENTITY_TYPES:
                break
            if et["name"] not in existing:
                result["entity_types"].insert(0, et)  # prioritize above fallbacks
                existing.add(et["name"])

        # Ensure a healthy relation set; if model returned too few, augment with defaults
        default_edges = [
            {"name": "REPORTS_ON", "description": "Media or analysts report on a subject.", "source_targets": [{"source": "MediaOutlet", "target": "Asset"}, {"source": "MediaOutlet", "target": "Organization"}]},
            {"name": "COMMENTS_ON", "description": "An entity comments on another entity or asset.", "source_targets": [{"source": "Person", "target": "Asset"}, {"source": "Person", "target": "Organization"}]},
            {"name": "RESPONDS_TO", "description": "Reactive reply or response.", "source_targets": [{"source": "Organization", "target": "MediaOutlet"}, {"source": "Person", "target": "Person"}]},
            {"name": "SUPPORTS", "description": "Expresses support.", "source_targets": [{"source": "Person", "target": "Organization"}]},
            {"name": "OPPOSES", "description": "Expresses opposition.", "source_targets": [{"source": "Person", "target": "Organization"}]},
            {"name": "TRADES", "description": "Trades or allocates toward an asset.", "source_targets": [{"source": "InstitutionalAllocator", "target": "Asset"}, {"source": "DataTrader", "target": "Asset"}]},
            {"name": "ANALYZES", "description": "Analyst interprets or evaluates a subject.", "source_targets": [{"source": "ResearchAnalyst", "target": "Asset"}, {"source": "ResearchAnalyst", "target": "Company"}]},
            {"name": "REGULATES", "description": "Regulatory oversight.", "source_targets": [{"source": "GovernmentAgency", "target": "Company"}, {"source": "GovernmentAgency", "target": "Exchange"}]},
            {"name": "AFFILIATED_WITH", "description": "Organizational affiliation.", "source_targets": [{"source": "Company", "target": "Exchange"}, {"source": "MediaOutlet", "target": "Organization"}]},
            {"name": "CORRELATES_WITH", "description": "Moves in statistical correlation.", "source_targets": [{"source": "Asset", "target": "Asset"}, {"source": "Metric", "target": "Metric"}]},
            {"name": "COMPETES_WITH", "description": "Competitor structure.", "source_targets": [{"source": "Company", "target": "Company"}, {"source": "Asset", "target": "Asset"}]},
            {"name": "LEADS_MARKET", "description": "Historically moves before another asset.", "source_targets": [{"source": "Asset", "target": "Asset"}]},
            {"name": "LAGS_MARKET", "description": "Historically moves after another asset.", "source_targets": [{"source": "Asset", "target": "Asset"}]},
            {"name": "ROTATES_INTO", "description": "Capital or volume shifts into another asset.", "source_targets": [{"source": "Asset", "target": "Asset"}]},
        ]

        existing_edge_names = {e.get("name") for e in result["edge_types"]}
        
        # ALWAYS inject missing critical edges to guarantee a rich semantic graph
        for e in default_edges:
            if e["name"] not in existing_edge_names:
                result["edge_types"].append(e)
                
        # De-dup just in case
        deduped = []
        seen = set()
        for e in result["edge_types"]:
            name = e.get("name")
            if name and name not in seen:
                deduped.append(e)
                seen.add(name)
        result["edge_types"] = deduped

        # Final check to ensure limits not exceeded (defensive programming)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result
