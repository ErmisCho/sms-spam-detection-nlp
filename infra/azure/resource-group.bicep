targetScope = 'subscription'

@description('Dedicated resource group name for the portfolio deployment.')
param resourceGroupName string

@description('Azure region for the resource group and Container Apps resources.')
param location string

@description('Owner recorded on the resource group.')
param owner string = 'portfolio'

@description('UTC expiry date (YYYY-MM-DD) for the temporary portfolio deployment.')
param expiryDate string

resource deploymentResourceGroup 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: {
    application: 'sms-spam-detection-nlp'
    environment: 'portfolio'
    owner: owner
    expiry: expiryDate
    managedBy: 'bicep'
  }
}

output resourceGroupId string = deploymentResourceGroup.id
