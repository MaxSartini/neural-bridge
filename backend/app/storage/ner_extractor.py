"""
NER/RE Extractor — entity and relation extraction via local LLM

Replaces Zep Cloud's built-in NER/RE pipeline.
Uses LLMClient.chat_json() with a structured prompt to extract
entities and relations from text chunks, guided by the graph's ontology.
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient

logger = logging.getLogger('neural_bridge.ner_extractor')

# Simulation-persona role types that must NEVER appear as graph nodes.
# These are LLM-generated archetypes used as agent identifiers, not real entities.
# Real entities (Person, Organization, Asset, etc.) ARE valid graph nodes —
# especially when created by the screenshot pipeline (source='screenshot').
_FORBIDDEN_ENTITY_TYPES = {
    "growthstrategist", "defensiveinvestor", "researchanalyst", "macroanalyst",
    "marketreporter", "contrarian", "commoditytrader", "institutionalallocator",
    "bankinginvestor", "agent", "commentator",
}

# System prompt template for NER/RE extraction
_SYSTEM_PROMPT = """You are a knowledge graph extraction system.
Given a document and an ontology, extract all relevant entities and their relationships.

ONTOLOGY:
{ontology_description}

RELATION TYPE USAGE GUIDE:
{relation_type_guidance}

RULES:
1. Extract entities matching the defined ontology types. Do NOT invent types not in the ontology.
2. Normalize entity names — use canonical names, strip extra whitespace.
3. Each entity must have: name, type (from ontology), and optional attributes.
4. Each relation must have: source, target, type (from ontology), and a fact sentence.
5. Choose the MOST SPECIFIC relation type. Use diverse relation types rather than defaulting to generic ones.
6. ENTITY RESOLUTION: If text references both a full name and a short identifier (ticker, handle, etc.), merge into ONE entity using the most recognizable identifier as the name.
7. Extract the maximum number of meaningful relationships.
8. If no entities or relations are found, return empty lists.
9. Do NOT extract: navigation labels, ads, button text, or UI elements.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "fact": "..."}}
  ]
}}"""

_USER_PROMPT = """Extract entities and relations from the following text:

{text}"""


class NERExtractor:
    """Extract entities and relations from text using local LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        self.llm = llm_client or LLMClient()
        self.max_retries = max_retries

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relations from text, guided by ontology.

        Args:
            text: Input text chunk
            ontology: Dict with 'entity_types' and 'relation_types' from graph

        Returns:
            Dict with 'entities' and 'relations' lists:
            {
                "entities": [{"name": str, "type": str, "attributes": dict}],
                "relations": [{"source": str, "target": str, "type": str, "fact": str}]
            }
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        relation_guidance = self._build_relation_type_guidance(ontology)
        system_msg = _SYSTEM_PROMPT.format(
            ontology_description=ontology_desc,
            relation_type_guidance=relation_guidance
        )
        user_msg = _USER_PROMPT.format(text=text.strip())

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,  # Low temp for extraction precision
                    max_tokens=4096,
                )
                cleaned = self._validate_and_clean(result, ontology)
                return self._inject_heuristic_edges(text, cleaned)

            except ValueError as e:
                last_error = e
                logger.warning(
                    f"NER extraction failed (attempt {attempt + 1}): invalid JSON — {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"NER extraction error: {e}")
                if attempt >= self.max_retries:
                    break

        logger.error(
            f"NER extraction failed after {self.max_retries + 1} attempts: {last_error}"
        )
        extraction = {"entities": [], "relations": []}
        return self._inject_heuristic_edges(text, extraction)
        
    def _inject_heuristic_edges(self, text: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Bypass the LLM organically and forcefully wire up assets that co-occur in text."""
        import re
        import difflib
        
        # 1. DYNAMIC TICKER/ASSET IDENTIFICATION
        # Safely capture financial tickers (e.g. STX40, JSE.JO, PRX.JO)
        # MUST either contain a dot suffix OR contain a mix of letters and numbers.
        # This prevents capturing standard ALL CAPS english words like "AND" or "RULES"
        dynamic_tickers = re.findall(r'\b([A-Z0-9]{2,8}\.[A-Z]{1,3}|[A-Z]*[0-9]+[A-Z]+|[A-Z]+[0-9]+[A-Z]*)\b', text)
        stop_words = {"AND", "THE", "RULES", "MARKET", "COMPARE", "BROAD", "INDEX", "ETF", "CSV", "FOR", "THAT", "THIS"}
        tickers = list(dict.fromkeys(
            t for t in dynamic_tickers if not t.isdigit() and t not in stop_words
        ))
        
        if len(tickers) > 1:
            for t in tickers:
                if not any(e.get("name") == t for e in extraction.get("entities", [])):
                    extraction.setdefault("entities", []).append({"name": t, "type": "Asset", "attributes": {}})
            
            lower_text = text.lower()
            rel_type = "CORRELATES_WITH"
            fact_prefix = "Algorithmic Correlation"
            
            if "compete" in lower_text:
                rel_type = "COMPETES_WITH"
            elif "lead" in lower_text:
                rel_type = "LEADS_MARKET"
            elif "lag" in lower_text:
                rel_type = "LAGS_MARKET"
            elif "rotat" in lower_text or "flow" in lower_text or "fleeing" in lower_text:
                rel_type = "ROTATES_INTO"
            
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    
                    exists = False
                    for r in extraction.get("relations", []):
                        if (r.get("source") == t1 and r.get("target") == t2) or (r.get("source") == t2 and r.get("target") == t1):
                            exists = True
                            break
                    
                    if not exists:
                        extraction.setdefault("relations", []).append({
                            "source": t1,
                            "target": t2,
                            "type": rel_type,
                            "fact": f"{fact_prefix}: Discovered systemic interaction between {t1} and {t2} via heuristic bypass."
                        })
                        
        # 2. DYNAMIC GRAPH ENTITY RESOLUTION (Deduplication)
        # Prevent LLM from splitting synonymous market concepts. Instead of hardcoded maps, we algorithmically 
        # cluster entities that share high string similarity or are subsets of each other.
        entities = extraction.get("entities", [])
        
        canonical_map = {}
        
        # Sort entities to prefer shorter, standard ticker-like names as the canonical root
        def get_score(e):
            name = str(e.get("name", ""))
            # If it strictly matches a ticker format, it's the highest preference
            if re.match(r'^[A-Z0-9]{3,8}(?:\.[A-Z]{1,3})?$', name): return 100
            if name.isupper(): return 50
            return max(0, 40 - len(name))
            
        entities.sort(key=get_score, reverse=True)
        
        for e in entities:
            orig_name = str(e.get("name", ""))
            if not orig_name: continue
            
            norm_name = re.sub(r'[^a-z0-9]', '', orig_name.lower())
            
            found_canonical = None
            for can_orig, can_norm in canonical_map.items():
                if not can_norm or not norm_name: continue
                
                # Exact alphanumeric match
                if norm_name == can_norm:
                    found_canonical = can_orig
                    break
                    
                # High fuzzy similarity match
                if len(norm_name) > 3 and len(can_norm) > 3:
                    similarity = difflib.SequenceMatcher(None, norm_name, can_norm).ratio()
                    if similarity >= 0.85:
                        found_canonical = can_orig
                        break
                    
                    # Direct subset (e.g. 'jse top 40' vs 'top 40')
                    if norm_name in can_norm or can_norm in norm_name:
                        # Ensure the subset is substantial
                        if min(len(norm_name), len(can_norm)) > 4:
                            found_canonical = can_orig
                            break

            if found_canonical:
                e["name"] = found_canonical
            else:
                canonical_map[orig_name] = norm_name

        # Map relations to their dynamically resolved names
        for r in extraction.get("relations", []):
            for node_type in ["source", "target"]:
                if node_type in r:
                    orig_name = str(r.get(node_type, ""))
                    norm_name = re.sub(r'[^a-z0-9]', '', orig_name.lower())
                    for can_orig, can_norm in canonical_map.items():
                        if norm_name == can_norm:
                            r[node_type] = can_orig
                            break
                        elif len(norm_name) > 3 and len(can_norm) > 3:
                            if difflib.SequenceMatcher(None, norm_name, can_norm).ratio() >= 0.85:
                                r[node_type] = can_orig
                                break
                            if min(len(norm_name), len(can_norm)) > 4 and (norm_name in can_norm or can_norm in norm_name):
                                r[node_type] = can_orig
                                break

        # Final pass: Filter out absolute identical duplicate dictionaries from pushing to the DB
        seen_entities = set()
        deduped = []
        
        # Stop words to purge garbage nodes hallucinated by the LLM (like "AND", "RULES" extracted as Assets)
        garbage_words = {"AND", "THE", "RULES", "MARKET", "COMPARE", "BROAD", "INDEX", "ETF", "CSV", "FOR", "THAT", "THIS", "OR"}
        
        for e in entities:
            ename = str(e.get("name", "")).strip()
            up_name = ename.upper()
            
            if ename and ename not in seen_entities and up_name not in garbage_words and len(ename) > 1:
                seen_entities.add(ename)
                deduped.append(e)
                
        extraction["entities"] = deduped
        
        return extraction

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Format ontology dict into readable text for the LLM prompt."""
        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            parts.append("Entity Types:")
            for et in entity_types:
                if isinstance(et, dict):
                    name = et.get("name", str(et))
                    desc = et.get("description", "")
                    attrs = et.get("attributes", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if attrs:
                        attr_names = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in attrs]
                        line += f" (attributes: {', '.join(attr_names)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {et}")

        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        if relation_types:
            parts.append("\nRelation Types:")
            for rt in relation_types:
                if isinstance(rt, dict):
                    name = rt.get("name", str(rt))
                    desc = rt.get("description", "")
                    source_targets = rt.get("source_targets", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    # Intentionally omitting strict source_targets from prompt to free the LLM
                    # from refusing to draw edges if it mildly misclassifies a node type.
                    parts.append(line)
                else:
                    parts.append(f"  - {rt}")

        if not parts:
            parts.append("No specific ontology defined. Extract all entities and relations you find.")

        return "\n".join(parts)

    def _build_relation_type_guidance(self, ontology: Dict[str, Any]) -> str:
        """
        Build context-aware guidance for relation type usage based on ontology.
        
        Generates guidance dynamically from the defined relation types and their descriptions,
        so the guidance is specific to the domain/context of the current ontology.
        """
        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        
        if not relation_types:
            return "No specific relation types defined. Extract any relations you find."
        
        lines = ["When extracting relations, choose the most semantically accurate type:"]
        for rt in relation_types:
            if isinstance(rt, dict):
                name = rt.get("name", "").strip()
                desc = rt.get("description", "").strip()
                if name:
                    if desc:
                        lines.append(f"- {name}: {desc}")
                    else:
                        lines.append(f"- {name}: Use for this type of relationship.")
            else:
                name = str(rt).strip()
                if name:
                    lines.append(f"- {name}: Use for this type of relationship.")
        
        lines.append("CRITICAL INSTRUCTION: DO NOT default to 'REPORTS_ON' or 'INTERPRETS_DATA' for everything. You MUST actively hunt for text that implies 'CORRELATES_WITH', 'LEADS_MARKET', 'COMPETES_WITH', and 'ROTATES_INTO' and explicitly use those complex edges between Assets. Build a complex, multi-dimensional web of relationships.")
        return "\n".join(lines)

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize LLM output."""
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Get valid type names from ontology
        valid_entity_types = set()
        for et in ontology.get("entity_types", []):
            if isinstance(et, dict):
                valid_entity_types.add(et.get("name", "").strip())
            else:
                valid_entity_types.add(str(et).strip())

        valid_relation_types = set()
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(rt, dict):
                valid_relation_types.add(rt.get("name", "").strip())
            else:
                valid_relation_types.add(str(rt).strip())

        # Clean entities — hard-filter any human/agent types regardless of what the LLM returns
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", "Entity")).strip()
            if not name:
                continue

            # HARD GATE: drop any entity whose type is a known human/agent category
            if etype.lower() in _FORBIDDEN_ENTITY_TYPES:
                logger.debug(f"Dropped forbidden entity '{name}' (type='{etype}')")
                continue

            # Deduplicate by normalized name
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": entity.get("attributes", {}),
            })

        # Clean relations
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            rtype = str(relation.get("type", "RELATED_TO")).strip()
            fact = str(relation.get("fact", "")).strip()

            if not source or not target:
                continue

            # Ensure source and target entities exist
            # (they might not if LLM hallucinated a relation without the entity)
            # Only add as Asset type (never as generic 'Entity') to prevent pollution
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Asset",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Asset",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
