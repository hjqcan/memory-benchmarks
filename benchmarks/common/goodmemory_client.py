"""
GoodMemory Client
=================

Async client for GoodMemory, exposing the same interface the benchmark
harness already uses for Mem0 (``add`` / ``search`` / ``delete_user``), so the
LoCoMo, LongMemEval, and BEAM runners can drive GoodMemory unchanged.

GoodMemory is served through its self-hosted HTTP bridge — a single Bun
process backed by SQLite, with bearer-token auth required by default:

    GOODMEMORY_HTTP_BRIDGE_TOKEN=<token> docker compose up -d
    # or, from the GoodMemory repo:
    GOODMEMORY_HTTP_BRIDGE_TOKEN=<token> bun scripts/goodmemory-http-bridge.ts

The bridge's default retrieval path needs no embedding service (deterministic
BM25 + lexical union), so a 1 GB / 1 vCPU instance runs the full suite.

Interface parity with ``Mem0Client`` (OSS mode):
    client.add(messages, user_id, ...)      -> {"results": [...]}
    client.search(query, user_id, top_k=200) -> [{"memory", "score", "id"}, ...]
    client.delete_user(user_id)             -> bool

Endpoint mapping (bridge contract phase-39.http-memory.v1):
    add         -> POST /memory/remember        {"messages": [...]}
    search      -> POST /memory/recall-context  {"query": ..., "maxTokens": ...}
    delete_user -> POST /memory/export + N x POST /memory/forget  (best-effort)

Scope is carried per request via x-goodmemory-* headers; the bearer token is
the security boundary. Search scores are assigned by GoodMemory's recall rank
(first item = highest) so the harness's ``sort(score desc)`` + ``[:cutoff]``
slice preserves GoodMemory's own ranking rather than reordering by a
memory-quality confidence that is not a retrieval-relevance signal.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import aiohttp
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# recall-context is budgeted by token count, not a top-k count. Request a
# generous budget so enough ranked items come back to satisfy the harness's
# top_k, then cap client-side. ~40 tokens/item is a safe over-estimate.
_TOKENS_PER_ITEM = 40
_MIN_RECALL_TOKENS = 2048
_MAX_RECALL_TOKENS = 16000


class GoodMemoryClient:
    """Async GoodMemory client speaking the HTTP bridge contract.

    Args:
        host: Bridge URL. Defaults to GOODMEMORY_BRIDGE_HOST env or
              http://localhost:8739.
        token: Bearer token. Falls back to GOODMEMORY_HTTP_BRIDGE_TOKEN /
               GOODMEMORY_BRIDGE_TOKEN env vars. Required unless the bridge
               was started with --allow-insecure.
        workspace_id: Optional workspace scope applied to every request.
        max_retries: Maximum retry attempts for API calls.
        retry_delay: Base delay in seconds between retries (grows each attempt).
        rpm: Requests per minute rate limit (0/None = unlimited).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        workspace_id: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        rpm: int | None = 60,
        timeout: float = 300.0,
        extraction_strategy: str = "llm-assisted",
        # Accepted for signature parity with Mem0Client; unused by the bridge.
        mode: str = "oss",
        **_ignored: Any,
    ):
        self.host = (host or os.getenv("GOODMEMORY_BRIDGE_HOST", "http://localhost:8739")).rstrip("/")
        self.token = (
            token
            or os.getenv("GOODMEMORY_HTTP_BRIDGE_TOKEN")
            or os.getenv("GOODMEMORY_BRIDGE_TOKEN")
            or ""
        )
        self.workspace_id = workspace_id or os.getenv("GOODMEMORY_BRIDGE_WORKSPACE_ID") or None
        # The bridge defaults remember() to rules-only extraction; benchmark
        # reproduction needs GoodMemory's full LLM-assisted extraction stack
        # (an extractor provider must be configured on the bridge). Override to
        # "rules-only" only when deliberately measuring the deterministic floor.
        self.extraction_strategy = os.getenv("GOODMEMORY_BRIDGE_EXTRACTION_STRATEGY") or extraction_strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        # Match Mem0Client's effectively-unlimited limiter unless rpm is set.
        self.limiter = AsyncLimiter(rpm if rpm and rpm > 0 else 100000, 60)
        self._session: aiohttp.ClientSession | None = None

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _scope_headers(self, user_id: str, operations: str = "*") -> dict[str, str]:
        # Caller identity + authorized operations (the auth boundary).
        headers = {
            "x-goodmemory-user-id": user_id,
            "x-goodmemory-operations": operations,
        }
        if self.workspace_id:
            headers["x-goodmemory-workspace-id"] = self.workspace_id
        return headers

    def _scope_body(self, user_id: str) -> dict[str, str]:
        # Target memory scope carried in the body; the bridge requires it to
        # match the caller identity (caller.userId === scope.userId).
        scope: dict[str, str] = {"userId": user_id}
        if self.workspace_id:
            scope["workspaceId"] = self.workspace_id
        return scope

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self.timeout,
                connector=connector,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> GoodMemoryClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # =========================================================================
    # Add
    # =========================================================================

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Ingest a conversation turn as durable memory.

        Returns ``{"results": [...]}`` mirroring Mem0's add response, or None on
        failure. ``timestamp``/``observation_date`` are threaded into metadata
        for provenance; GoodMemory dates memories by ingestion, so temporal
        fidelity is best-effort at the bridge boundary.
        """
        session = await self._get_session()

        payload: dict[str, Any] = {
            "messages": messages,
            "scope": self._scope_body(user_id),
            "extractionStrategy": self.extraction_strategy,
        }
        request_metadata = dict(metadata or {})
        if timestamp is not None:
            request_metadata["observationEpoch"] = timestamp
        elif observation_date is not None:
            try:
                d = datetime.strptime(observation_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                request_metadata["observationEpoch"] = int(d.timestamp())
            except ValueError:
                pass
        if custom_instructions:
            request_metadata["customInstructions"] = custom_instructions
        if request_metadata:
            payload["metadata"] = request_metadata

        headers = self._scope_headers(user_id)

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(
                        f"{self.host}/memory/remember", json=payload, headers=headers
                    ) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        data = await resp.json()
                return {"results": _normalise_add_results(data)}

            except Exception as exc:
                logger.warning(
                    "ADD attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries, user_id, str(exc)[:200],
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("ADD failed after %d attempts for user=%s", self.max_retries, user_id)
                    return None

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Recall memories for a query. Returns a list of
        ``{"memory", "score", "id"}`` sorted by score descending, where score
        encodes GoodMemory's recall rank (first item = highest).
        """
        session = await self._get_session()
        max_tokens = max(_MIN_RECALL_TOKENS, min(_MAX_RECALL_TOKENS, top_k * _TOKENS_PER_ITEM))
        payload: dict[str, Any] = {
            "query": query,
            "maxTokens": max_tokens,
            "retrievalProfile": "coding_agent",
            # hybrid activates GoodMemory's semantic recall (embedding + BM25 +
            # semantic candidate union). The bridge forces any non-"hybrid"
            # strategy to rules-only, which cannot surface semantically-relevant
            # facts — so this is required for benchmark-representative recall.
            "strategy": "hybrid",
            "scope": self._scope_body(user_id),
        }
        headers = self._scope_headers(user_id)

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(
                        f"{self.host}/memory/recall-context", json=payload, headers=headers
                    ) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        data = await resp.json()

                items = data.get("items", []) if isinstance(data, dict) else []
                return map_recall_items_to_results(items, top_k)

            except Exception as exc:
                logger.warning(
                    "SEARCH attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries, user_id, str(exc)[:200],
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("SEARCH failed after %d attempts for user=%s", self.max_retries, user_id)
                    return []

    # =========================================================================
    # Delete (best-effort; not exercised by the benchmark harnesses)
    # =========================================================================

    async def delete_user(self, user_id: str) -> bool:
        """Delete all memories for a user by exporting then forgetting each.

        The bridge exposes per-memory forget only, so a full-user reset is
        export + N forgets. Returns True if the sweep completed without error.
        """
        session = await self._get_session()
        headers = self._scope_headers(user_id)
        try:
            scope_body = self._scope_body(user_id)
            async with self.limiter:
                async with session.post(
                    f"{self.host}/memory/export", json={"scope": scope_body}, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    exported = await resp.json()

            memory_ids = _collect_exported_memory_ids(exported)
            for memory_id in memory_ids:
                async with self.limiter:
                    async with session.post(
                        f"{self.host}/memory/forget",
                        json={"memoryId": memory_id, "scope": scope_body},
                        headers=headers,
                    ) as resp:
                        resp.raise_for_status()
            logger.info("Deleted %d memories for user %s", len(memory_ids), user_id)
            return True
        except Exception as exc:
            logger.warning("Failed to delete user %s: %s", user_id, exc)
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def map_recall_items_to_results(items: Any, top_k: int) -> list[dict]:
    """Map bridge recall-context ``items`` to Mem0-shaped search results.

    ``items`` arrive already ordered by GoodMemory's recall ranking (best
    first). Each result is assigned a rank-descending score so the harness's
    ``sorted(..., reverse=True)`` + ``[:cutoff]`` slice preserves that ranking
    rather than reordering by a memory-quality ``confidence`` (which is not a
    retrieval-relevance signal). Returns ``[{"memory", "score", "id", ...}]``.
    """
    if not isinstance(items, list):
        return []
    total = len(items)
    results: list[dict[str, Any]] = []
    for rank, item in enumerate(items[:top_k]):
        if not isinstance(item, dict):
            continue
        text = item.get("content", "")
        if not text:
            continue
        results.append({
            "memory": text,
            "score": (total - rank) / total if total else 0.0,
            "id": item.get("memoryId", ""),
            "metadata": {
                "type": item.get("type"),
                "goodmemory_confidence": item.get("confidence"),
            },
        })
    return results


def _normalise_add_results(data: Any) -> list[dict]:
    """Map a GoodMemory remember response to Mem0's add-results shape."""
    if not isinstance(data, dict):
        return []
    result = data.get("result")
    events = result.get("events") if isinstance(result, dict) else None
    if not isinstance(events, list):
        events = data.get("events") if isinstance(data.get("events"), list) else []
    results: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        outcome = str(event.get("outcome", "")).upper() or "ADD"
        results.append({
            "id": event.get("memoryId", ""),
            "memory": event.get("content", event.get("memory", "")),
            "event": "ADD" if outcome == "WRITTEN" else outcome,
        })
    return results


def _collect_exported_memory_ids(exported: Any) -> list[str]:
    """Pull every durable memory id out of a bridge export response."""
    ids: list[str] = []
    if not isinstance(exported, dict):
        return ids
    durable = exported.get("exported", exported)
    durable = durable.get("durable", durable) if isinstance(durable, dict) else {}
    if not isinstance(durable, dict):
        return ids
    for bucket in durable.values():
        if not isinstance(bucket, list):
            continue
        for record in bucket:
            if isinstance(record, dict) and record.get("id"):
                ids.append(record["id"])
    return ids


def format_search_results(search_results: list[dict]) -> tuple[list[dict], dict | None]:
    """Normalize search results for benchmark output.

    Mirrors ``mem0_client.format_search_results`` so runners can import either.
    Returns (formatted results list, query_debug dict or None).
    """
    if not search_results:
        return [], None
    if isinstance(search_results, dict):
        search_results = search_results.get("results", [])
    sorted_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)
    formatted = []
    for r in sorted_results:
        entry: dict[str, Any] = {
            "memory": r.get("memory", ""),
            "score": r.get("score", 0),
            "id": r.get("id", ""),
        }
        if r.get("metadata"):
            entry["metadata"] = r["metadata"]
        formatted.append(entry)
    return formatted, None
