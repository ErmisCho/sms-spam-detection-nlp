from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AzureInfrastructureContractTest(unittest.TestCase):
    def test_container_app_scales_to_zero_and_one_replica_maximum(self) -> None:
        template = (PROJECT_ROOT / "infra/azure/main.bicep").read_text(encoding="utf-8")

        self.assertIn("minReplicas: 0", template)
        self.assertIn("maxReplicas: 1", template)
        self.assertIn("cpu: json('0.25')", template)
        self.assertIn("memory: '0.5Gi'", template)
        self.assertIn("destination: 'none'", template)
        self.assertIn("@sha256:${imageDigest}", template)

    def test_baseline_avoids_persistent_paid_resource_types(self) -> None:
        template = (PROJECT_ROOT / "infra/azure/main.bicep").read_text(encoding="utf-8").lower()
        prohibited_resource_types = (
            "microsoft.containerregistry/",
            "microsoft.storage/",
            "microsoft.network/",
            "microsoft.operationalinsights/",
            "microsoft.insights/components",
        )

        for resource_type in prohibited_resource_types:
            self.assertNotIn(resource_type, template)

    def test_dedicated_resource_group_and_budget_are_declared_separately(self) -> None:
        resource_group = (PROJECT_ROOT / "infra/azure/resource-group.bicep").read_text(
            encoding="utf-8"
        )
        budget = (PROJECT_ROOT / "infra/azure/budget.bicep").read_text(encoding="utf-8")

        self.assertIn("Microsoft.Resources/resourceGroups", resource_group)
        self.assertIn("Microsoft.Consumption/budgets", budget)
        self.assertIn("ResourceGroupName", budget)
        self.assertIn("This is an alert, not a hard spending cap", budget)


if __name__ == "__main__":
    unittest.main()
