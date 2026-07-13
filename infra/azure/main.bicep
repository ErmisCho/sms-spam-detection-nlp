targetScope = 'resourceGroup'

@description('Azure region for the Container Apps resources.')
param location string = resourceGroup().location

@description('Short, globally unique deployment name used to derive resource names.')
@minLength(3)
@maxLength(24)
param namePrefix string

@description('Public GHCR repository path without registry or tag, for example org/image.')
@minLength(3)
param ghcrImagePath string

@description('Immutable SHA-256 image digest as exactly 64 hexadecimal characters, without the sha256: prefix. Obtain it from the published GHCR package, not a mutable tag.')
@minLength(64)
@maxLength(64)
param imageDigest string

@description('UTC expiry date (YYYY-MM-DD) used by cost-governance automation and reviewers.')
param expiryDate string

@description('Owner recorded on every deployed resource.')
param owner string = 'portfolio'

@description('Additional tags merged onto every deployed resource.')
param additionalTags object = {}

var resourceTags = union(additionalTags, {
  application: 'sms-spam-detection-nlp'
  environment: 'portfolio'
  owner: owner
  expiry: expiryDate
  managedBy: 'bicep'
})
var environmentName = '${namePrefix}-env'
var appName = '${namePrefix}-app'
var containerImage = 'ghcr.io/${ghcrImagePath}@sha256:${imageDigest}'

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: resourceTags
  properties: {
    zoneRedundant: false
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  tags: resourceTags
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        allowInsecure: false
        targetPort: 8000
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'web'
          image: containerImage
          env: [
            {
              name: 'PORT'
              value: '8000'
            }
            {
              name: 'SMS_SPAM_TRUST_X_FORWARDED_FOR'
              value: 'true'
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/health/live'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 1
              periodSeconds: 3
              timeoutSeconds: 2
              failureThreshold: 20
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health/live'
                port: 8000
                scheme: 'HTTP'
              }
              periodSeconds: 10
              timeoutSeconds: 2
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health/ready'
                port: 8000
                scheme: 'HTTP'
              }
              periodSeconds: 5
              timeoutSeconds: 2
              failureThreshold: 6
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http-requests'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

@description('Public HTTPS URL assigned by Azure Container Apps.')
output applicationUrl string = 'https://${app.properties.configuration.ingress.fqdn}'

@description('Azure Container App resource name.')
output containerAppName string = app.name

@description('Azure Container Apps environment resource name.')
output containerAppsEnvironmentName string = environment.name
