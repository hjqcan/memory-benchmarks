import unittest
from types import SimpleNamespace
from unittest.mock import patch

from goodmemory_client import Scope

from benchmarks.common.goodmemory_client import GoodMemoryClient, format_observed_content


class GoodMemoryClientTimestampTest(unittest.TestCase):
    def test_formats_unix_timestamp_as_utc_observation_prefix(self) -> None:
        self.assertEqual(
            format_observed_content("Deployed the API.", timestamp=1683504000),
            "[Observed at 2023-05-08T00:00:00Z] Deployed the API.",
        )

    def test_uses_observation_date_when_timestamp_is_absent(self) -> None:
        self.assertEqual(
            format_observed_content(
                "Reviewed the launch plan.",
                observation_date="2023-05-09",
            ),
            "[Observed at 2023-05-09] Reviewed the launch plan.",
        )

    def test_leaves_undated_content_unchanged(self) -> None:
        self.assertEqual(
            format_observed_content("No date was supplied."),
            "No date was supplied.",
        )


class GoodMemoryClientOfficialBridgeTest(unittest.IsolatedAsyncioTestCase):
    @patch("benchmarks.common.goodmemory_client.BridgeClient")
    async def test_add_uses_official_client_with_scoped_verified_writes(
        self,
        bridge_client_type,
    ) -> None:
        bridge = bridge_client_type.return_value
        bridge.remember.return_value = {"ok": True}
        client = GoodMemoryClient(
            host="http://bridge.test/",
            token="test-token",
            max_retries=2,
            retry_delay=0.1,
            timeout=7,
        )

        result = await client.add(
            [
                {"role": "user", "content": "Deployed the API."},
                {"role": "assistant", "content": "Deployment succeeded."},
            ],
            "run-1",
            timestamp=1683504000,
        )

        self.assertEqual(result, {"ok": True})
        bridge_client_type.assert_called_once_with(
            "http://bridge.test",
            scope=Scope(user_id="run-1"),
            token="test-token",
            timeout_seconds=7,
            max_attempts=2,
            retry_delay_seconds=0.1,
        )
        bridge.remember.assert_called_once_with(
            [
                {
                    "role": "user",
                    "content": "[Observed at 2023-05-08T00:00:00Z] Deployed the API.",
                },
                {
                    "role": "user",
                    "content": (
                        "[Observed at 2023-05-08T00:00:00Z] "
                        "[role=assistant] Deployment succeeded."
                    ),
                },
            ],
            mode="sync",
            extraction_strategy="rules-only",
            annotations=[
                {
                    "remember": "always",
                    "confirmed": True,
                    "verified": True,
                    "kindHint": "fact",
                    "messageIndex": 0,
                },
                {
                    "remember": "always",
                    "confirmed": True,
                    "verified": True,
                    "kindHint": "fact",
                    "messageIndex": 1,
                },
            ],
        )

    @patch("benchmarks.common.goodmemory_client.BridgeClient")
    async def test_search_normalises_official_recall_result(
        self,
        bridge_client_type,
    ) -> None:
        bridge = bridge_client_type.return_value
        bridge.recall_context.return_value = SimpleNamespace(
            items=[
                {"content": "First", "memoryId": "m-1"},
                {"content": "Second", "memoryId": "m-2"},
            ],
            routing=SimpleNamespace(
                requested_strategy="auto",
                resolved_strategy="hybrid",
                fallback_reason=None,
            ),
        )
        client = GoodMemoryClient(recall_strategy="auto")

        result = await client.search("deployment", "run-2", top_k=1)

        bridge.recall_context.assert_called_once_with(
            "deployment",
            strategy="auto",
        )
        self.assertEqual(
            result,
            [{"memory": "First", "score": 1.0, "id": "m-1"}],
        )


if __name__ == "__main__":
    unittest.main()
