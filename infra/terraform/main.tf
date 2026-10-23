# Infrastructure sketch for running the identity-resolution batch on AWS.
# Checked with `tofu fmt` and `tofu validate` (OpenTofu 1.8.5, AWS provider 5.72.1).
# NOT applied: no AWS account was used, so nothing here has been deployed or cost-tested.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

locals {
  name = "${var.project}-${var.env}"
  tags = { project = var.project, env = var.env, owner = "ml-platform" }
}

# ---------------------------------------------------------------- storage
resource "aws_s3_bucket" "lake" {
  bucket = "${local.name}-lake"
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration { status = "Enabled" }
}

# raw/ holds PII and expires; curated/identities/ holds only hashed + masked fields
resource "aws_s3_bucket_lifecycle_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    id     = "expire-raw"
    status = "Enabled"
    filter { prefix = "raw/" }
    expiration { days = var.raw_retention_days }
  }
}

# ---------------------------------------------------------------- secrets
resource "aws_secretsmanager_secret" "hash_salt" {
  name = "${local.name}/hash-salt"
  tags = local.tags
}

# ---------------------------------------------------------------- compute: EMR Serverless (Spark)
resource "aws_emrserverless_application" "spark" {
  name          = "${local.name}-spark"
  release_label = var.emr_release
  type          = "spark"

  maximum_capacity {
    cpu    = "64 vCPU"
    memory = "256 GB"
  }
  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }
  tags = local.tags
}

data "aws_iam_policy_document" "emr_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["emr-serverless.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "job" {
  name               = "${local.name}-emr-job"
  assume_role_policy = data.aws_iam_policy_document.emr_assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "job" {
  statement {
    sid       = "ReadWriteLake"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.lake.arn, "${aws_s3_bucket.lake.arn}/*"]
  }
  statement {
    sid       = "ReadSalt"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.hash_salt.arn]
  }
  statement {
    sid = "GlueCatalog"
    actions = ["glue:GetDatabase", "glue:GetTable", "glue:CreateTable", "glue:UpdateTable",
    "glue:GetPartitions", "glue:BatchCreatePartition"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "job" {
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job.json
}

# ---------------------------------------------------------------- catalog for consumers (Athena / MCP server)
resource "aws_glue_catalog_database" "identity" {
  name = replace("${local.name}_identity", "-", "_")
}

resource "aws_glue_catalog_table" "identities" {
  name          = "identities"
  database_name = aws_glue_catalog_database.identity.name
  table_type    = "EXTERNAL_TABLE"
  parameters    = { classification = "parquet" }

  partition_keys {
    name = "run"
    type = "string"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.lake.bucket}/curated/identities/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    dynamic "columns" {
      for_each = {
        record_id  = "string", identity_id = "string", source = "string", name_display = "string",
        name_key   = "string", zip5 = "string", email_hash = "string", email_masked = "string",
        phone_hash = "string", phone_masked = "string"
      }
      content {
        name = columns.key
        type = columns.value
      }
    }
  }
}
