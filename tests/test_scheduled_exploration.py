from __future__ import annotations

import unittest

from nim_router.schemas import ModelInfo
from nim_router.scoring import scheduled_exploration_candidate
from nim_router.stats import StatsStore


class ScheduledExplorationTests(unittest.TestCase):
    def test_claims_only_one_slot_per_interval(self) -> None:
        stats = StatsStore()

        self.assertTrue(stats.claim_exploration(60, now=100))
        self.assertFalse(stats.claim_exploration(60, now=120))
        self.assertTrue(stats.claim_exploration(60, now=160))

    def test_prefers_least_observed_candidate(self) -> None:
        stats = StatsStore()
        stats.record_success("a")
        stats.record_success("a")
        stats.record_success("b")

        selected = scheduled_exploration_candidate(
            [ModelInfo(id="a"), ModelInfo(id="b"), ModelInfo(id="c")], stats
        )

        self.assertEqual(selected.id, "c")
