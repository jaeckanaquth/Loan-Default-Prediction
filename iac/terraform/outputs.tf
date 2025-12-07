output "s3_bucket" {
  value = aws_s3_bucket.data_bucket.bucket
}
output "ecr_repo" {
  value = aws_ecr_repository.app_repo.repository_url
}
