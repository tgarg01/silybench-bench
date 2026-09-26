# One benchmark VM. Parameterized by machine type so the same module covers
# a3-highgpu-1g (1xH100) today and a3-highgpu-8g / a3-ultragpu / a4 / a2 / g2 later.

terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 7.0" }
  }
}

resource "google_compute_instance" "vm" {
  name         = var.name
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["gpubench"]
  labels       = { app = "gpubench", campaign = replace(lower(var.campaign), "/[^a-z0-9_-]/", "-") }

  boot_disk {
    auto_delete = true
    initialize_params {
      image = var.image
      size  = var.boot_disk_gb
      type  = var.boot_disk_type
    }
  }

  # A2/A3/A4 machine types include their GPUs; only G2/N1 need guest_accelerator.
  dynamic "guest_accelerator" {
    for_each = var.accelerator_type == "" ? [] : [1]
    content {
      type  = var.accelerator_type
      count = var.accelerator_count
    }
  }

  network_interface {
    network  = var.network
    nic_type = "GVNIC"
    access_config {} # ephemeral public IP for egress (HF / Docker Hub downloads)
  }

  scheduling {
    provisioning_model          = var.spot ? "SPOT" : "STANDARD"
    preemptible                 = var.spot
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    instance_termination_action = "DELETE"
    # Hard cost cap enforced by GCP: the VM is deleted after this long no matter what.
    max_run_duration {
      seconds = var.max_run_hours * 3600
    }
  }

  service_account {
    email  = var.service_account
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script       = file("${path.module}/../../../scripts/startup-script.sh")
    gpubench-code-uri    = var.code_uri
    gpubench-config      = var.config_path
    gpubench-bucket      = var.bucket
    gpubench-self-delete = var.self_delete ? "true" : "false"
    gpubench-run-args    = var.run_args
    gpubench-image       = var.image
    enable-oslogin       = "TRUE"
  }
}
