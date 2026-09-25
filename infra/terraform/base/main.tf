# Long-lived resources shared by every benchmark VM: APIs, results bucket,
# VM service account, HF token secret, IAP SSH firewall rule, optional budget.
# Apply once; GPU VMs live in envs/* with their own short-lived state.

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 7.0" }
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

locals {
  # cloudresourcemanager + iam must be on before Terraform can manage APIs and service
  # accounts, so they're bootstrapped once with `gcloud services enable` and listed here too.
  apis = [
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "iap.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudbilling.googleapis.com", # price catalog lookups for $/1M tokens
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

resource "google_storage_bucket" "results" {
  name                        = var.bucket_name
  location                    = upper(var.region)
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  versioning { enabled = true }
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "runner" {
  account_id   = "gpubench-runner"
  display_name = "gpubench benchmark VM"
}

# Read code / write results in the bucket.
resource "google_storage_bucket_iam_member" "runner_bucket" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = google_service_account.runner.member
}

# Lets the VM delete itself when the run finishes, and write logs.
resource "google_project_iam_member" "runner_roles" {
  for_each = toset(["roles/compute.instanceAdmin.v1", "roles/logging.logWriter"])
  project  = var.project_id
  role     = each.value
  member   = google_service_account.runner.member
}

# The secret's value is added by hand (never in Terraform state):
#   printf '%s' "$HF_TOKEN" | gcloud secrets versions add hf-token --data-file=-
resource "google_secret_manager_secret" "hf_token" {
  secret_id = "hf-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "runner_hf_token" {
  secret_id = google_secret_manager_secret.hf_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.runner.member
}

# SSH only through Identity-Aware Proxy (no public SSH).
resource "google_compute_firewall" "iap_ssh" {
  name          = "gpubench-allow-iap-ssh"
  network       = var.network
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["gpubench"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_billing_budget" "monthly" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "gpubench monthly budget"

  budget_filter {
    projects = ["projects/${var.project_number}"]
  }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_usd)
    }
  }
  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }
  depends_on = [google_project_service.apis]
}
