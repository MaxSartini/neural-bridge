"""
ScreenshotProcessor — extract social media comments from screenshots.

Two-stage pipeline:
  1. Tesseract performs OCR on the raw screenshot (robust on small UI text where
     the Gemma vision projector struggles).
  2. Gemma-4 (text-only) structures the OCR output into platform/topic/comments
     JSON matching our Neo4j schema.

Falls back to Gemma vision if Tesseract is unavailable at runtime.
Embedding-based semantic dedupe merges OCR jitter across chunks.
"""

import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from ..config import Config
from ..storage.embedding_service import EmbeddingService
from ..utils.logger import get_logger

logger = get_logger('neural_bridge.screenshot')

VISION_MAX_EDGE = 1152          # used only by the vision fallback
OCR_UPSCALE_MIN_WIDTH = 1600    # tesseract accuracy rises sharply above ~1600px
SEMANTIC_DUP_THRESHOLD = 0.93
MAX_STRUCTURE_CHARS = 12000     # cap OCR text sent to LLM to stay well below context
TESSERACT_BIN = shutil.which('tesseract') or '/opt/homebrew/bin/tesseract'


class ScreenshotProcessor:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self._client = OpenAI(
            base_url=Config.LLM_BASE_URL,
            api_key=Config.LLM_API_KEY or 'lm-studio',
            timeout=600.0,
        )
        self._model = Config.LLM_MODEL_NAME
        self._max_tokens = int(os.environ.get('VISION_TOKEN_BUDGET', 4096))
        self._embedding = embedding_service or EmbeddingService()
        self._tesseract_available = bool(TESSERACT_BIN and os.path.exists(TESSERACT_BIN))
        if not self._tesseract_available:
            logger.warning("tesseract binary not found — will fall back to Gemma vision.")

    # ---------- image prep ----------

    def _load_rgb(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _upscale_for_ocr(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w >= OCR_UPSCALE_MIN_WIDTH:
            return img
        scale = OCR_UPSCALE_MIN_WIDTH / w
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ---------- OCR stage ----------

    def _run_tesseract(self, img: Image.Image) -> str:
        img = self._upscale_for_ocr(img)
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, 'in.png')
            img.save(img_path, format='PNG')
            out_base = os.path.join(td, 'out')
            # --psm 6: assume a single uniform block of text. Good for threaded comments.
            cmd = [
                TESSERACT_BIN, img_path, out_base,
                '-l', 'eng', '--psm', '6', '--oem', '1',
                '-c', 'preserve_interword_spaces=1',
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"tesseract failed: {result.stderr.strip()}")
            with open(out_base + '.txt', 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

    # ---------- structuring stage (text-only LLM) ----------

    def _structure_ocr(self, ocr_text: str, topic_context: str, label: str) -> Dict:
        trimmed = ocr_text.strip()
        if not trimmed:
            return {"platform": "unknown", "topic": "", "comments": []}
        if len(trimmed) > MAX_STRUCTURE_CHARS:
            trimmed = trimmed[:MAX_STRUCTURE_CHARS]

        prompt = f"""You are given raw OCR text extracted from a screenshot of a social media
comment section. Your job is to identify individual comments and return structured JSON.

Topic context: {topic_context or 'not specified'}

Respond with ONLY valid JSON, no prose, no markdown fences:
{{
  "platform": "twitter|reddit|facebook|instagram|youtube|news|other",
  "topic": "short inferred topic of the discussion",
  "comments": [
    {{"username": "...", "text": "...", "sentiment": "positive|negative|neutral|mixed", "reaction_count": null, "is_reply": false}}
  ]
}}

Rules:
- A comment is one author expressing one message. The username usually appears immediately
  before or above the comment text and often looks like "u/name", "@name", or a standalone word.
- OCR noise is common: fix obvious character errors (e.g. "rn" → "m") only when confident.
- Skip navigation labels, ads, button text, and promotional content.
- Skip fragments where you cannot identify both a username and at least one sentence of comment text.
- reaction_count: integer if a number is clearly associated with the comment; else null.
- is_reply: true if the comment is visibly indented or nested under another.

OCR TEXT:
\"\"\"
{trimmed}
\"\"\"
"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"[{label}] LLM structuring call failed: {e}")
            return {"platform": "unknown", "topic": "", "comments": [], "_error": str(e)}

        raw = (response.choices[0].message.content or "").strip()
        finish = response.choices[0].finish_reason
        logger.info(f"[{label}] structure finish={finish} raw_len={len(raw)}")

        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned).strip()
        if cleaned and not cleaned.startswith('{'):
            m = re.search(r'\{[\s\S]*\}', cleaned)
            if m:
                cleaned = m.group(0)

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON is not an object")
            parsed.setdefault("comments", [])
            if not isinstance(parsed["comments"], list):
                parsed["comments"] = []
            return parsed
        except Exception as e:
            logger.warning(f"[{label}] structure JSON parse failed: {e}; raw={raw[:400]!r}")
            return {"platform": "unknown", "topic": "", "comments": [], "_error": f"parse: {e}"}

    # ---------- vision fallback (text-only failed / no tesseract) ----------

    def _encode(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    def _vision_extract(self, img: Image.Image, topic_context: str, label: str) -> Dict:
        w, h = img.size
        scale = min(1.0, VISION_MAX_EDGE / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        b64 = self._encode(img)
        prompt = (
            "Extract every social media comment visible in this screenshot. "
            f"Topic context: {topic_context or 'not specified'}. "
            "Return ONLY valid JSON: "
            '{"platform":"...","topic":"...","comments":[{"username":"...","text":"...",'
            '"sentiment":"positive|negative|neutral|mixed","reaction_count":null,"is_reply":false}]}'
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=self._max_tokens,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"[{label}] vision fallback call failed: {e}")
            return {"platform": "unknown", "topic": "", "comments": [], "_error": str(e)}

        raw = (response.choices[0].message.content or "").strip()
        logger.info(f"[{label}] vision finish={response.choices[0].finish_reason} raw_len={len(raw)}")
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', raw, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned).strip()
        if cleaned and not cleaned.startswith('{'):
            m = re.search(r'\{[\s\S]*\}', cleaned)
            if m:
                cleaned = m.group(0)
        try:
            parsed = json.loads(cleaned)
            parsed.setdefault("comments", [])
            return parsed
        except Exception as e:
            logger.warning(f"[{label}] vision JSON parse failed: {e}; raw={raw[:300]!r}")
            return {"platform": "unknown", "topic": "", "comments": [], "_error": f"parse: {e}"}

    # ---------- dedupe ----------

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _dedupe(self, comments: List[Dict]) -> List[Dict]:
        if not comments:
            return []
        keys = [f"{(c.get('username') or '').strip()}: {(c.get('text') or '').strip()}" for c in comments]
        try:
            vectors = self._embedding.embed_batch(keys)
        except Exception as e:
            logger.warning(f"Dedupe embedding failed, falling back to exact-string dedupe: {e}")
            seen, unique = set(), []
            for c, k in zip(comments, keys):
                if k in seen:
                    continue
                seen.add(k)
                unique.append(c)
            return unique

        unique: List[Dict] = []
        unique_vecs: List[List[float]] = []
        for c, v in zip(comments, vectors):
            if any(self._cosine(v, u) >= SEMANTIC_DUP_THRESHOLD for u in unique_vecs):
                continue
            unique.append(c)
            unique_vecs.append(v)
        return unique

    # ---------- public API ----------

    def process_image(self, image_path: str, topic_context: str = "") -> Dict:
        name = os.path.basename(image_path)
        try:
            img = self._load_rgb(image_path)
        except Exception as e:
            logger.warning(f"Could not open {image_path}: {e}")
            return {"platform": "unknown", "topic": "", "comments": [], "error": str(e)}

        errors: List[str] = []
        used = "tesseract"
        ocr_text = ""
        if self._tesseract_available:
            try:
                ocr_text = self._run_tesseract(img)
                logger.info(f"{name}: tesseract OCR produced {len(ocr_text)} chars")
            except Exception as e:
                logger.warning(f"{name}: tesseract failed: {e}")
                errors.append(f"ocr: {e}")
                ocr_text = ""

        if ocr_text.strip():
            result = self._structure_ocr(ocr_text, topic_context, label=f"{name}#structure")
        else:
            used = "vision"
            logger.info(f"{name}: OCR empty, falling back to Gemma vision")
            result = self._vision_extract(img, topic_context, label=f"{name}#vision")

        if result.get("_error"):
            errors.append(result["_error"])

        platform = result.get("platform") or "unknown"
        topic = result.get("topic") or topic_context or ""

        raw_comments = []
        for c in result.get("comments", []):
            if not isinstance(c, dict):
                continue
            uname = (c.get("username") or "").strip()
            text = (c.get("text") or "").strip()
            if not uname or not text:
                continue
            raw_comments.append(c)

        unique = self._dedupe(raw_comments)
        logger.info(
            f"{name}: pipeline={used} raw={len(raw_comments)} unique={len(unique)} errors={len(errors)}"
        )

        return {
            "platform": platform,
            "topic": topic,
            "comments": unique,
            "pipeline": used,
            "ocr_chars": len(ocr_text),
            "errors": errors,
        }

    def inject_into_graph(self, result: Dict, graph_id: str, storage) -> List[str]:
        node_uuids: List[str] = []
        platform = result.get("platform", "unknown")
        topic = result.get("topic", "")

        for comment in result.get("comments", []):
            username = (comment.get("username") or "").strip()
            text = (comment.get("text") or "").strip()
            if not username or not text:
                continue

            attrs = {
                "platform": platform,
                "sentiment": comment.get("sentiment", "neutral"),
                "reaction_count": comment.get("reaction_count"),
                "is_reply": bool(comment.get("is_reply", False)),
                "topic": topic,
                "source": "screenshot",
            }
            try:
                uid = storage.add_commenter_node(
                    graph_id=graph_id,
                    username=username,
                    text=text,
                    attributes=attrs,
                )
                node_uuids.append(uid)
            except Exception as e:
                logger.warning(f"Failed to inject commenter {username}: {e}")

        return node_uuids

    def process_batch(
        self,
        image_paths: List[str],
        graph_id: str,
        topic_context: str,
        storage,
    ) -> Dict:
        total_comments = 0
        all_uuids: List[str] = []
        per_image: List[Dict] = []
        errors: List[Dict] = []

        for path in image_paths:
            name = os.path.basename(path)
            try:
                result = self.process_image(path, topic_context)
                uuids = self.inject_into_graph(result, graph_id, storage)
                n_comments = len(result.get("comments", []))
                total_comments += n_comments
                all_uuids.extend(uuids)
                per_image.append({
                    "file": name,
                    "pipeline": result.get("pipeline"),
                    "ocr_chars": result.get("ocr_chars", 0),
                    "comments": n_comments,
                    "errors": result.get("errors", []),
                })
                if n_comments == 0:
                    errors.append({
                        "file": name,
                        "error": "no comments extracted",
                        "pipeline": result.get("pipeline"),
                        "stage_errors": result.get("errors", []),
                    })
            except Exception as e:
                logger.warning(f"process_image crashed for {name}: {e}", exc_info=True)
                errors.append({"file": name, "error": str(e)})

        return {
            "total_images": len(image_paths),
            "total_comments": total_comments,
            "node_uuids": all_uuids,
            "per_image": per_image,
            "errors": errors,
        }
