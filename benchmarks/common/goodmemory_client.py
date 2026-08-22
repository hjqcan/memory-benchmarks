"""
Async benchmark adapter for the official GoodMemory Python bridge client.

Setup (no Docker required)::

    npm install -g goodmemory@0.7.5
    pip install goodmemory-client==0.1.0
    GOODMEMORY_HTTP_BRIDGE_TOKEN=replace-me \
      goodmemory-http-bridge --recommended

The official client owns the HTTP wire contract, caller/scope identity,
authorization headers, retries, and recall-routing metadata. This module only
adapts that synchronous client to the async ``Mem0Client`` interface used by
the benchmark runners.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from aiolimiter import AsyncLimiter
from goodmemory_client import (
    GoodMemoryClient as BridgeClient,
    GoodMemoryClientError,
    Scope,
)

logger = logging.getLogger(__name__)

PUBLISHED_RECALL_ITEM_LIMIT = 12


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
    """Expose the benchmark suite's async memory-client interface."""

    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        rpm: int = 120,
        timeout: float = 300.0,
        recall_strategy: str | None = None,
    ) -> None:
        self.host = (host or os.getenv("GOODMEMORY_HOST", "http://localhost:8739")).rstrip("/")
        self.token = token or os.getenv("GOODMEMORY_HTTP_BRIDGE_TOKEN") or None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.recall_strategy = (
            recall_strategy or os.getenv("GOODMEMORY_RECALL_STRATEGY") or "auto"
        )
        self.limiter = AsyncLimiter(rpm, 60)

    def _bridge_client(self, user_id: str) -> BridgeClient:
        return BridgeClient(
            self.host,
            scope=Scope(user_id=user_id),
            token=self.token,
            timeout_seconds=self.timeout,
            max_attempts=self.max_retries,
            retry_delay_seconds=self.retry_delay,
        )

    async def close(self) -> None:
        """The stdlib bridge client has no persistent session to close."""

    async def __aenter__(self) -> "GoodMemoryClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Import each turn as a verified fact, preserving its source role."""
        source_messages = [message for message in messages if message.get("content")]
        kept = [
            {
                "role": "user",
                "content": format_observed_content(
                    (
                        message["content"]
                        if (message.get("role") or "user") == "user"
                        else f"[role={message.get('role') or 'user'}] {message['content']}"
                    ),
                    observation_date=observation_date,
                    timestamp=timestamp,
                ),
            }
            for message in source_messages
        ]
        if not kept:
            return {"skipped": True}

        annotations = [
            {
                "remember": "always",
                "confirmed": True,
                "verified": True,
                "kindHint": "fact",
                "messageIndex": index,
                "metadataPatch": {
                    "attributes": {
                        "sourceRole": source_messages[index].get("role") or "user",
                    }
                },
            }
            for index in range(len(kept))
        ]

        try:
            async with self.limiter:
                return await asyncio.to_thread(
                    self._bridge_client(user_id).remember,
                    kept,
                    mode="sync",
                    extraction_strategy="rules-only",
                    annotations=annotations,
                )
        except (GoodMemoryClientError, OSError, ValueError) as error:
            logger.error("ADD failed for user=%s: %s", user_id, str(error)[:200])
            return None

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 20,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Recall memories and normalize them to the Mem0 result shape."""
        effective_top_k = min(top_k, PUBLISHED_RECALL_ITEM_LIMIT)
        if top_k > PUBLISHED_RECALL_ITEM_LIMIT:
            logger.warning(
                "GoodMemory 0.7.5 recall-context requested top_k=%d but returns "
                "at most %d selected items; use cutoff 10 for comparable runs.",
                top_k,
                PUBLISHED_RECALL_ITEM_LIMIT,
            )

        try:
            async with self.limiter:
                result = await asyncio.to_thread(
                    self._bridge_client(user_id).recall_context,
                    query,
                    strategy=self.recall_strategy,
                )
        except (GoodMemoryClientError, OSError, ValueError) as error:
            logger.error("SEARCH failed for user=%s: %s", user_id, str(error)[:200])
            return []

        routing = result.routing
        if (
            routing.fallback_reason
            or (
                routing.requested_strategy not in ("", "auto")
                and routing.resolved_strategy != routing.requested_strategy
            )
        ):
            logger.warning(
                "GoodMemory recall routing: requested=%s resolved=%s fallback=%s",
                routing.requested_strategy,
                routing.resolved_strategy,
                routing.fallback_reason,
            )

        normalised: list[dict[str, Any]] = []
        for item in result.items[:effective_top_k]:
            content = item.get("content", "")
            if not content:
                continue
            normalised.append(
                {
                    "memory": content,
                    "score": 0.0,
                    "id": item.get("memoryId", ""),
                }
            )
        return normalised

    async def delete_user(self, user_id: str) -> bool:
        """Runs are isolated by the project-derived GoodMemory user id."""
        logger.info(
            "GoodMemory delete_user(%s): no bulk delete endpoint; "
            "use a fresh --project-name per run.",
            user_id,
        )
        return True
