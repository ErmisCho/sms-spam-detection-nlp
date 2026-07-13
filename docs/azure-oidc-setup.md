# Azure OIDC deployment setup

The `Deploy to Azure` workflow accepts only a complete GHCR `sha256:` digest and authenticates without a stored Azure password. Run it manually after the container publication workflow reports the digest.

## GitHub Environment

Create a protected GitHub Environment named `production` (or select another environment when dispatching). Configure required reviewers and prevent self-review for production deployments. Add these environment variables; none of these identifiers is a password:

- `AZURE_CLIENT_ID`: application/client ID of the deployment identity.
- `AZURE_TENANT_ID`: Microsoft Entra tenant ID.
- `AZURE_SUBSCRIPTION_ID`: target Azure subscription ID.
- `AZURE_RESOURCE_GROUP`: pre-created deployment resource group.
- `AZURE_LOCATION`: Azure region used by `infra/azure/main.bicep`.
- `AZURE_NAME_PREFIX`: 3–24 character resource-name prefix used by the Bicep template.
- `AZURE_EXPIRY_DATE`: cost-governance expiry date in `YYYY-MM-DD` form.

Do not create a client secret. After creating the dedicated resource group from `infra/azure/resource-group.bicep`, the following Azure CLI bootstrap creates an app registration, service principal, federated GitHub identity, and resource-group-scoped assignment:

```bash
client_id=$(az ad app create \
  --display-name sms-spam-github-deployer \
  --query appId --output tsv)
az ad sp create --id "$client_id" >/dev/null
principal_id=$(az ad sp show --id "$client_id" --query id --output tsv)
tenant_id=$(az account show --query tenantId --output tsv)
subscription_id=$(az account show --query id --output tsv)
resource_group_id=$(az group show \
  --name sms-spam-portfolio-rg \
  --query id --output tsv)

az ad app federated-credential create \
  --id "$client_id" \
  --parameters '{"name":"github-production","issuer":"https://token.actions.githubusercontent.com","subject":"repo:ErmisCho/sms-spam-detection-nlp:environment:production","description":"GitHub Actions protected production environment","audiences":["api://AzureADTokenExchange"]}'

az role assignment create \
  --assignee-object-id "$principal_id" \
  --assignee-principal-type ServicePrincipal \
  --role Contributor \
  --scope "$resource_group_id"

printf 'AZURE_CLIENT_ID=%s\nAZURE_TENANT_ID=%s\nAZURE_SUBSCRIPTION_ID=%s\n' \
  "$client_id" "$tenant_id" "$subscription_id"
```

Adjust the repository owner/name, environment, and resource-group name when necessary. The federated subject must exactly match `repo:<owner>/<repository>:environment:<environment>`, and the audience must be `api://AzureADTokenExchange`.

The example uses Contributor only at the dedicated resource-group scope so it can create and update the two `Microsoft.App` resources but cannot assign IAM roles. Avoid subscription-wide Owner or Contributor assignments. The supported baseline requires a public GHCR package; it does not pass a registry token through the workflow or Azure configuration.

## Infrastructure contract

The workflow uses `infra/azure/main.bicep`, which:

- accepts `location`, `namePrefix`, `ghcrImagePath`, `imageDigest`, and `expiryDate`;
- constructs the container reference from the fixed GHCR path and exact digest without using a mutable tag;
- is deployed under the stable deployment name `sms-spam-api`;
- outputs `containerAppName` and an HTTPS `applicationUrl`;
- provides the model artifact/configuration needed for `/health/ready` to succeed;
- retains revisions or otherwise permits a previous immutable image to be redeployed.

Before changing Azure, the workflow validates the Bicep deployment and records the currently configured image. It checks both `/health/live` and `/health/ready` after deployment. If either fails, it redeploys the previous image and leaves the workflow failed so the bad release cannot appear successful. The environment concurrency group serializes deployments, and `cancel-in-progress: false` prevents a running rollback path from being cancelled by a newer dispatch.
