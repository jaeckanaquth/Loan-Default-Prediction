resource "aws_s3_bucket" "data_bucket" {
  bucket = "loan-default-data"
  acl    = "private"
  tags = {
    Name = "loan-default-data"
  }
}
resource "aws_ecr_repository" "app_repo" {
  name = "loan-predict-repo"
}
