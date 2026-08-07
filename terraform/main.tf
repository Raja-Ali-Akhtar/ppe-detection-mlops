# Stage 3 serving infrastructure. `terraform apply` stands it up,
# `terraform destroy` guarantees $0. No console clicking.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

# ---------------------------------------------------------------- container registry
resource "aws_ecr_repository" "gateway" {
  name         = "ppe-gateway"
  force_delete = true # demo repo: destroy must succeed even with images inside
}

# ----------------------------------------------------------------------- networking
# Default VPC keeps this minimal — a real deployment would define its own.
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "serving" {
  name_prefix = "ppe-serving-"
  vpc_id      = data.aws_vpc.default.id

  # gateway API + grafana; triton stays internal to the instance
  dynamic "ingress" {
    for_each = [9000, 3000]
    content {
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = [var.allowed_cidr]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ------------------------------------------------------------------- instance role
# The instance authenticates by ROLE, not by copying access keys onto it —
# this is the pattern that makes leaked-key incidents structurally impossible.
resource "aws_iam_role" "instance" {
  name_prefix = "ppe-serving-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "s3_read" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

# SSM = browser/CLI shell into the instance with no SSH keys and no port 22
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name_prefix = "ppe-serving-"
  role        = aws_iam_role.instance.name
}

# -------------------------------------------------------------------------- the GPU
# Deep Learning Base AMI: NVIDIA driver + docker + nvidia-container-toolkit
# preinstalled — user_data only has to fetch the model and start compose.
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
}

resource "aws_instance" "serving" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.serving.id]
  iam_instance_profile   = aws_iam_instance_profile.instance.name

  # NOTE: launched ON-DEMAND (~$0.658/hr) — AWS granted on-demand G/VT quota (4 vCPU)
  # but denied the spot request for this account age. Spot appeal still open; when
  # granted, re-enable the block below for ~$0.16/hr.
  # instance_market_options {
  #   market_type = "spot"
  #   spot_options {
  #     max_price          = var.spot_max_price
  #     spot_instance_type = "one-time"
  #   }
  # }

  root_block_device {
    volume_size = 80 # AMI + triton image (~17 GB unpacked) need headroom
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region          = var.region
    ecr_gateway     = aws_ecr_repository.gateway.repository_url
    model_s3_prefix = var.model_s3_prefix
  })

  tags = { Name = "ppe-serving", Project = "ppe-detection-mlops" }
}
