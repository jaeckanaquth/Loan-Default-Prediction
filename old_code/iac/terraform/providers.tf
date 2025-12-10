terraform {
  required_version = ">= 1.0.0"
}
provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  endpoints {
    s3             = "http://localhost:4566"
    ecr            = "http://localhost:4566"
    ecs            = "http://localhost:4566"
    iam            = "http://localhost:4566"
    secretsmanager = "http://localhost:4566"
  }
}
