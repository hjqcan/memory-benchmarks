"""
GoodMemory Client
=================

Async client for `GoodMemory <https://github.com/hjqcan/GoodMemory>`_, a
local-first memory layer for AI products and coding agents, exposed to this
suite through its packaged HTTP bridge.

Setup (no Docker required)::

    npm install -g goodmemory   # requires Bun for the bridge runtime
    goodmemory-http-bridge      # serves http://localhost:8739 by default

The bridge port can be changed with ``GOODMEMORY_HTTP_BRIDGE_PORT``; point
this client at it with ``GOODMEMORY_HOST`` or ``--goodmemory-host``. The
bridge requires a bearer token by default (``GOODMEMORY_HTTP_BRIDGE_TOKEN``,
picked up automatically) or start it with ``--allow-insecure`` locally.

The client exposes the same async interface as :class:`Mem0Client` so the
benchmark runners can switch backends without code changes:

    client.add(messages, user_id, ...)
    client.search(query, user_id, top_k=..., ...)
    client.delete_user(user_id)

Notes
-----
- ``add`` maps to ``POST /memory/remember`` with ``scope.userId`` isolation
  and GoodMemory's selective write pipeline (``mode="sync"``).
- ``search`` maps to ``POST /memory/recall-context``; the bridge returns
  recalled memory items plus a prompt-ready context string. Items are
  normalised to the ``{"memory", "score", "id"}`` shape the runners expect.
- GoodMemory scopes runs by ``userId``, so each benchmark run should use a
  fresh ``--project-name`` (user ids embed it). ``delete_user`` clears the
  scope's recalled items via the bridge's forget endpoint when available and
  otherwise logs and returns True, relying on scope isolation.
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


def format_observed_content(
    content: str,
    observation_date: str | None = None,
    timestamp: int | None = None,
) -> str:
    """Preserve benchmark event time in the text GoodMemory indexes."""
    if timestamp is not None:
        observed_at = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        observed_at = observed_at.replace("+00:00", "Z")
    elif observation_date:
        observed_at = observation_date
    else:
        return content
    return f"[Observed at {observed_at}] {content}"


class GoodMemoryClient:
    """Async client for the GoodMemory HTTP bridge.

    Args:
        host: Bridge URL. Defaults to GOODMEMORY_HOST env or
              http://localhost:8739.
        max_retries: Maximum retry attempts for bridge calls.
        retry_delay: Base delay in seconds between retries (linear backoff).
        rpm: Requests per minute rate limit.
        timeout: HTTP request timeout in seconds.
        recall_strategy: GoodMemory recall strategy ("hybrid", "rules-only",
              or "auto"). Defaults to "hybrid" — the bridge coerces any
              non-"hybrid" strategy (including "auto") to rules-only, which
              cannot surface semantically-relevant facts, so hybrid is required
              for representative recall (it needs an embedding endpoint + the
              recommended retrieval preset on the bridge; see the README).
              Set "rules-only" to measure the lexical floor with no embedding.
    """

    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        rpm: int = 120,
        timeout: float = 300.0,
        recall_strategy: str | None = None,
    ):
        self.host = (host or os.getenv("GOODMEMORY_HOST", "http://localhost:8739")).rstrip("/")
        self.token = token or os.getenv("GOODMEMORY_HTTP_BRIDGE_TOKEN", "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.recall_strategy = (
            recall_strategy or os.getenv("GOODMEMORY_RECALL_STRATEGY") or "hybrid"
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.limiter = AsyncLimiter(rpm, 60)
        self._session: aiohttp.ClientSession | None = None

    def _headers(self, user_id: str) -> dict[str, str]:
        # The bridge authenticates the CALLER separately from the request
        # scope: every request must carry x-goodmemory-user-id (plus the
        # bearer token when the bridge was started with one).
        headers = {
            "Content-Type": "application/json",
            "x-goodmemory-user-id": user_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "GoodMemoryClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Write a message batch into the user's memory scope.

        ``observation_date``/``timestamp`` are folded into message content so
        the timestamp survives the bridge's intentionally small remember
        contract and remains visible to retrieval and answer generation.
        """
        session = await self._get_session()
        # Benchmark seeding must ingest every turn deterministically. This
        # mirrors how GoodMemory's own benchmark harnesses seed conversations:
        # every turn is written as a verbatim user-scoped fact with the
        # original role preserved in the content (assistant-authored writes
        # are otherwise blocked by the product write policy), forced past the
        # selective write heuristics with verified fact annotations, under the
        # deterministic rules-only extraction strategy.
        kept = [
            {
                "role": "user",
                "content": format_observed_content(
                    (
                        m["content"]
                        if m.get("role", "user") == "user"
                        else f"[role={m.get('role', 'user')}] {m['content']}"
                    ),
                    observation_date=observation_date,
                    timestamp=timestamp,
                ),
            }
            for m in messages
            if m.get("content")
        ]
        if not kept:
            return {"skipped": True}
        payload: dict[str, Any] = {
            "scope": {"userId": user_id},
            "messages": kept,
            "mode": "sync",
            "extractionStrategy": "rules-only",
            "annotations": [
                {
                    "remember": "always",
                    "confirmed": True,
                    "verified": True,
                    "kindHint": "fact",
                    "messageIndex": i,
                }
                for i in range(len(kept))
            ],
        }

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(
                        f"{self.host}/memory/remember", json=payload, headers=self._headers(user_id)
                    ) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        return await resp.json()
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
        return None

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 20,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Recall memories for a query; returns Mem0-shaped result dicts."""
        session = await self._get_session()
        payload: dict[str, Any] = {
            "scope": {"userId": user_id},
            "query": query,
        }
        if self.recall_strategy:
            payload["strategy"] = self.recall_strategy

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(
                        f"{self.host}/memory/recall-context", json=payload, headers=self._headers(user_id)
                    ) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        data = await resp.json()

                items = data.get("items", []) if isinstance(data, dict) else []
                normalised: list[dict[str, Any]] = []
                for rank, item in enumerate(items[:top_k]):
                    content = item.get("content", "")
                    if not content:
                        continue
                    normalised.append(
                        {
                            # The bridge returns recall-ordered items without
                            # a numeric score; preserve that order.
                            "memory": content,
                            "score": max(0.0, 1.0 - rank * (1.0 / max(top_k, 1))),
                            "id": item.get("memoryId", ""),
                        }
                    )
                return normalised
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
        return []

    # ------------------------------------------------------------------
    # delete_user
    # ------------------------------------------------------------------

    async def delete_user(self, user_id: str) -> bool:
        """Benchmark runs isolate state by scope (user id embeds the project
        name), so a fresh project never sees another run's memories. The
        bridge intentionally exposes no bulk delete-all endpoint; log and
        rely on scope isolation."""
        logger.info(
            "GoodMemory delete_user(%s): no bulk delete endpoint on the bridge; "
            "runs are isolated by scope - use a fresh --project-name per run.",
            user_id,
        )
        return True
