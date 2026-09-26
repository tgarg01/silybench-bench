terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 7.0" }
  }
}

provider "google" {
  project = var.project_id
}

module "vm" {
  source          = "../../modules/gpu_vm"
  name            = "gpubench-h100-1g"
  campaign        = var.campaign
  machine_type    = "a3-highgpu-1g"
  zone            = var.zone
  spot            = true
  max_run_hours   = var.max_run_hours
  service_account = "gpubench-runner@${var.project_id}.iam.gserviceaccount.com"
  bucket          = var.bucket
  code_uri        = var.code_uri
  config_path     = var.config_path
  self_delete     = var.self_delete
  run_args        = var.run_args
  image           = "projects/deeplearning-platform-release/global/images/${var.image}"
}

output "vm_name" { value = module.vm.name }
output "vm_zone" { value = module.vm.zone }
