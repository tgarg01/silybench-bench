variable "project_id" { type = string }

variable "project_number" {
  type        = string
  description = "Numeric project ID (used by the budget filter)"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "bucket_name" {
  type        = string
  description = "Globally unique GCS bucket for code bundles and results"
}

variable "network" {
  type    = string
  default = "default"
}

variable "billing_account" {
  type        = string
  default     = ""
  description = "Billing account ID for the budget alert; empty = no budget"
}

variable "monthly_budget_usd" {
  type    = number
  default = 200
}
