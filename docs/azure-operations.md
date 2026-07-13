# Azure deployment and operations runbook

This runbook deploys the portfolio demo to Azure Container Apps with a free-tier-oriented baseline. It minimizes idle resources, but it does **not** guarantee a zero invoice. Azure pricing, subscription eligibility, outbound traffic, abuse, and future platform changes can still create charges.

## Baseline architecture

- One dedicated resource group with expiry tags.
- One Azure Container Apps Consumption environment.
- One public Container App using the default Azure hostname.
- A public, immutable GHCR image containing a trusted synthetic demo model.
- `minReplicas: 0`, `maxReplicas: 1`, 0.25 vCPU, and 0.5 GiB memory.
- No ACR, Azure Storage/Azure Files, Log Analytics workspace, Application Insights, custom DNS, VNet, NAT gateway, private endpoint, or Dedicated workload profile.
- Container application logs are not persisted by the baseline environment. GitHub Actions retains deployment results and failed container builds retain local logs for that workflow run.

The image's synthetic model exists only for the hosted product demonstration. Benchmark results in the repository are produced from the public UCI dataset and are not claimed for the synthetic model.

## Prerequisites

Install Azure CLI, authenticate interactively, and select the intended subscription:

```bash
az login
az account set --subscription "<subscription-id-or-name>"
az account show --output table
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.Consumption --wait
```

Use an Azure region where Container Apps Consumption is available. The examples below use `westeurope`. Choose a unique prefix of 3–24 characters and a dedicated resource-group name.

## One-time bootstrap

Set an expiry 30–60 days after deployment and create the dedicated resource group:

```bash
az deployment sub create \
  --name sms-spam-resource-group \
  --location westeurope \
  --template-file infra/azure/resource-group.bicep \
  --parameters resourceGroupName=sms-spam-portfolio-rg location=westeurope \
    owner="<your-name>" expiryDate="<YYYY-MM-DD>"
```

The resource group is deliberately separate. Deleting it removes the complete baseline deployment without searching across the subscription.

Follow [Azure OIDC setup](azure-oidc-setup.md) to create the GitHub federated identity. Scope its deployment role to `sms-spam-portfolio-rg`; do not give the workflow subscription-wide Owner or Contributor access.

### Optional budget alert

If the subscription supports Cost Management budgets, deploy a small monthly alert filtered to the dedicated resource group:

```bash
az deployment sub create \
  --name sms-spam-budget \
  --location westeurope \
  --template-file infra/azure/budget.bicep \
  --parameters resourceGroupName=sms-spam-portfolio-rg monthlyAmount=5 \
    startDate="<first-day-of-month-YYYY-MM-DD>" endDate="<YYYY-MM-DD>" \
    contactEmails='["<your-email>"]'
```

Budget data and notifications can be delayed. A budget sends alerts; it is not a spending cap and does not stop or delete resources. Some sponsorship and restricted subscription types do not expose the same budget features.

## Publish and deploy

1. Run the `Publish container` GitHub Actions workflow with a new version such as `v1.0.0`, or push an intentional version tag.
2. Confirm the package is public in GitHub Packages. Azure must be able to pull it without registry credentials.
3. Copy the complete `sha256:...` digest from the workflow summary.
4. Run the protected `Deploy to Azure` workflow and supply that digest.
5. Approve the configured GitHub Environment deployment.

The publishing workflow tests Python and React, builds the non-root image, then verifies readiness, UI delivery, and prediction without mounting storage. Deployment uses the exact digest, performs the same live checks against Azure, and automatically redeploys the previous digest if a replacement fails its smoke test.

## Verify scale-to-zero and cold start

Record the configuration and public URL:

```bash
az containerapp show \
  --resource-group sms-spam-portfolio-rg \
  --name <name-prefix>-app \
  --query "{url:properties.configuration.ingress.fqdn,min:properties.template.scale.minReplicas,max:properties.template.scale.maxReplicas,image:properties.template.containers[0].image}" \
  --output yaml
```

Leave the application unused, then inspect active replicas:

```bash
az containerapp replica list \
  --resource-group sms-spam-portfolio-rg \
  --name <name-prefix>-app \
  --output table
```

With `minReplicas: 0`, an idle app can have no active replica. The next request starts a new replica and can be noticeably slower. Test the cold path with `/health/ready`, then open the root URL and submit one sample prediction.

Azure may remove an empty managed environment after an extended idle period. The IaC and immutable image digest make recreation deterministic; do not treat the hosted URL as permanent portfolio storage.

## Cost and resource audit

Run before deployment, after deployment, and monthly:

```bash
bash scripts/azure_audit.sh sms-spam-portfolio-rg
```

```powershell
.\scripts\azure_audit.ps1 -ResourceGroup sms-spam-portfolio-rg
```

The audit lists the resource inventory, flags resource types outside the baseline, requests month-to-date actual cost, and lists budget alerts. Cost Management data can lag behind resource use, so recheck on the following day after a traffic spike or teardown.

Expected project resources are the managed environment and Container App. The optional budget is a control-plane resource. Investigate and remove any registry, storage account, workspace, monitoring component, public IP, NAT gateway, private endpoint, or custom DNS resource associated with this demo.

## Public-demo safeguards

The baseline caps scale at one replica. The API additionally defaults to:

- 30 prediction requests per client per 60 seconds;
- a 16 KiB HTTP request-body limit before JSON parsing;
- a 10-second prediction timeout;
- a 10,000-character validated SMS field;
- structured logs that contain route, status, request ID, and latency, never SMS text.

These values can be changed with `SMS_SPAM_RATE_LIMIT_REQUESTS`, `SMS_SPAM_RATE_LIMIT_WINDOW_SECONDS`, `SMS_SPAM_MAX_REQUEST_BYTES`, and `SMS_SPAM_REQUEST_TIMEOUT_SECONDS`. Do not raise them merely to hide an abuse test failure.

The limiter is intentionally in-process because the baseline permits only one replica. It is a portfolio safeguard, not a substitute for an authenticated gateway or distributed production rate limiter.

## Temporarily disable and re-enable public access

Disable ingress without deleting or rebuilding the app:

```bash
az containerapp ingress disable --resource-group sms-spam-portfolio-rg --name <name-prefix>-app
```

Re-enable the same public endpoint:

```bash
az containerapp ingress enable \
  --resource-group sms-spam-portfolio-rg \
  --name <name-prefix>-app \
  --type external --target-port 8000 --transport auto --allow-insecure false
```

Run the readiness and prediction smoke checks again after re-enabling ingress.

## Upgrade and rollback

For an upgrade, publish a new version and deploy its immutable digest. Never deploy `latest`. The protected workflow serializes deployments and records the application URL and digest.

If a smoke test fails, the workflow redeploys the previously recorded digest. For a manual rollback, dispatch `Deploy to Azure` again with the last known-good digest from an earlier workflow summary or package version.

## Incident triage

1. Check GitHub Actions build and deployment summaries.
2. Confirm the app image is a public GHCR digest and not a mutable tag.
3. Inspect `provisioningState`, replica state, and revision health with `az containerapp show`, `az containerapp revision list`, and `az containerapp replica list`.
4. Distinguish cold-start delay from persistent readiness failure by retrying `/health/live` and `/health/ready` for up to five minutes.
5. If usage looks abusive, disable ingress immediately, run the cost audit, and leave it disabled until the cause is understood.
6. Redeploy the last known-good digest or tear down the dedicated resource group.

Because baseline logs are not persisted, temporarily enabling a reviewed Azure Monitor diagnostic destination is an incident-only change that may incur ingestion and retention cost. Revert it after diagnosis.

## Monthly maintenance checklist

- Review Dependabot/security alerts and update locked Python, Node, action, and base-image versions.
- Run the complete Python, frontend, container, and IaC contract tests.
- Open the hosted URL and verify readiness plus HAM and SPAM samples.
- Confirm `minReplicas: 0`, `maxReplicas: 1`, and the expected immutable image digest.
- Run the audit and inspect delayed Azure cost data and budget alerts.
- Confirm there is no custom domain or certificate to renew in the baseline.
- Review the `expiry` tag and delete the deployment when it is no longer supporting active applications or interviews.
- Put the expiry date in a personal calendar; Azure tags do not delete resources automatically.

## Verified teardown

The scripts require the resource-group name twice to reduce accidental deletion risk:

```bash
bash scripts/azure_teardown.sh sms-spam-portfolio-rg --confirm sms-spam-portfolio-rg
```

```powershell
.\scripts\azure_teardown.ps1 \
  -ResourceGroup sms-spam-portfolio-rg \
  -Confirm sms-spam-portfolio-rg
```

After deletion, the script verifies the group is absent. Check Cost Management again the following day because usage data is delayed. The subscription-scoped budget may remain after the filtered resource group is deleted; remove it explicitly if it is no longer useful:

```bash
az consumption budget delete --budget-name sms-spam-portfolio-budget
```

## Clean-subscription rehearsal checklist

Run this before calling the hosted demo complete:

1. Create only the dedicated resource group from `resource-group.bicep`.
2. Run the pre-deployment audit and save its output.
3. Publish a versioned image and record its digest.
4. Deploy through the protected OIDC workflow.
5. Verify UI, liveness, readiness, and prediction.
6. Leave the app idle and verify zero active replicas.
7. Run the post-deployment resource and cost audit.
8. Exercise a known-good digest rollback.
9. Disable and re-enable ingress.
10. Tear down the dedicated resource group and verify no project resources remain.
11. Recheck delayed costs and delete the optional subscription budget.

The rehearsal requires an authenticated Azure subscription. Local and CI validation cannot prove subscription eligibility, regional capacity, actual scale-to-zero timing, or billing behavior.
