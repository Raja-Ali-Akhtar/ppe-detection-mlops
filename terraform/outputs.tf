output "ecr_gateway_url" {
  value = aws_ecr_repository.gateway.repository_url
}

output "instance_public_ip" {
  value = aws_instance.serving.public_ip
}

output "try_it" {
  value = <<EOT
detect:  curl -F "file=@image.jpg" http://${aws_instance.serving.public_ip}:9000/detect
grafana: http://${aws_instance.serving.public_ip}:3000
shell:   aws ssm start-session --target ${aws_instance.serving.id} --region ${var.region}
EOT
}
