variable "region" {
  default = "eu-central-1"
}

variable "instance_type" {
  default = "g4dn.xlarge" # T4 GPU — same Turing generation as the local 1660 Ti
}

# Hard price ceiling for the spot bid. On-demand is ~$0.658/hr in eu-central-1;
# spot typically lands at $0.15-0.25. If spot price exceeds this, we simply
# don't get the instance (which is the correct failure mode for a budget).
variable "spot_max_price" {
  default = "0.30"
}

# Who may reach the endpoints. 0.0.0.0/0 = the whole internet — fine for a
# few-hour demo with no secrets, but tighten to "<your-ip>/32" if it runs longer.
variable "allowed_cidr" {
  default = "0.0.0.0/0"
}

variable "model_s3_prefix" {
  default = "s3://ppe-mlops-dvc-428232898120/triton-models"
}
