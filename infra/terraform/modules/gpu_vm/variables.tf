variable "name" { type = string }
variable "campaign" { type = string }
variable "machine_type" { type = string }
variable "zone" { type = string }

variable "image" {
  type        = string
  description = "Deep Learning VM image (NVIDIA driver preinstalled)"
  default     = "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2404-nvidia-580"
}

variable "boot_disk_gb" {
  type        = number
  default     = 200
  description = "OS + vLLM image (~20 GB) + model weights; counts against the region's SSD_TOTAL_GB quota"
}

variable "boot_disk_type" {
  type    = string
  default = "pd-balanced"
}

variable "accelerator_type" {
  type    = string
  default = ""
}

variable "accelerator_count" {
  type    = number
  default = 0
}

variable "network" {
  type    = string
  default = "default"
}

variable "spot" {
  type    = bool
  default = true
}

variable "max_run_hours" {
  type        = number
  default     = 12
  description = "GCP deletes the VM after this many hours (cost guardrail)"
}

variable "service_account" { type = string }
variable "bucket" { type = string }

variable "code_uri" {
  type        = string
  description = "gs:// URI of the repo tarball uploaded by up.sh"
}

variable "config_path" {
  type        = string
  description = "Config path relative to repo root, e.g. configs/smoke.yaml"
}

variable "self_delete" {
  type        = bool
  default     = true
  description = "Delete the VM when the benchmark finishes (false = keep it for debugging)"
}

variable "run_args" {
  type        = string
  default     = ""
  description = "Extra `gpubench run` flags, e.g. \"--precision fp8\""
}
