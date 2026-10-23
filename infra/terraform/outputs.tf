output "lake_bucket" { value = aws_s3_bucket.lake.bucket }
output "emr_application_id" { value = aws_emrserverless_application.spark.id }
output "emr_job_role_arn" { value = aws_iam_role.job.arn }
output "glue_database" { value = aws_glue_catalog_database.identity.name }
