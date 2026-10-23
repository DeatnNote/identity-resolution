variable "project" {
  type    = string
  default = "identity-resolution"
}
variable "env" {
  type    = string
  default = "dev"
}
variable "region" {
  type    = string
  default = "us-east-1"
}
variable "emr_release" {
  type    = string
  default = "emr-7.5.0"
}
variable "raw_retention_days" {
  type        = number
  default     = 30
  description = "Raw extracts contain PII; keep them only as long as reprocessing needs."
}
