targetScope = 'subscription'

@description('Unique budget name within the subscription.')
param budgetName string = 'sms-spam-portfolio-budget'

@description('Dedicated resource group containing only this portfolio deployment.')
param resourceGroupName string

@description('Monthly alert amount in the subscription billing currency. This is an alert, not a hard spending cap.')
@minValue(1)
param monthlyAmount int = 5

@description('First day of a month in YYYY-MM-DD format.')
param startDate string

@description('Budget end date in YYYY-MM-DD format.')
param endDate string

@description('Addresses that receive actual-cost budget notifications.')
@minLength(1)
param contactEmails array

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: budgetName
  properties: {
    amount: monthlyAmount
    category: 'Cost'
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: startDate
      endDate: endDate
    }
    filter: {
      dimensions: {
        name: 'ResourceGroupName'
        operator: 'In'
        values: [
          resourceGroupName
        ]
      }
    }
    notifications: {
      ActualCost50Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 50
        thresholdType: 'Actual'
        contactEmails: contactEmails
        contactGroups: []
        contactRoles: []
      }
      ActualCost100Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Actual'
        contactEmails: contactEmails
        contactGroups: []
        contactRoles: []
      }
    }
  }
}

output budgetResourceId string = budget.id
