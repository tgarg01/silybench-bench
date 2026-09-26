variable "project_id" { type = string }

variable "zone" {
  type    = string
  default = "us-central1-a"
}

variable "bucket" { type = string }
variable "campaign" { type = string }
variable "code_uri" { type = string }
variable "config_path" { type = string }

variable "max_run_hours" {
  type    = number
  default = 12
}

variable "self_delete" {
  type    = bool
  default = true
}

variable "run_args" {
  type    = string
  default = ""
}

# Read by infra/scripts/up.sh (recorded in result.json for $ per token); unused by Terraform.
variable "price_per_hour" {
  type    = number
  default = null
}

# Exact boot image (NVIDIA driver + CUDA). Pinned, not a family, so the driver can't change
# between an experiment and its reproductions. Override with up.sh --image.
variable "image" {
  type    = string
  default = "common-cu129-ubuntu-2404-nvidia-580-v20260909"
}
