import unittest

from benchmarks.common.goodmemory_client import format_observed_content


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


if __name__ == "__main__":
    unittest.main()
