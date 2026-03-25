from central_management import _post_cpu_update, update_network_latency

deployments = {
        "microservice1-deployment": {
            "replicas": 1,
            "cpu_limit_m": "500",
            "memory_limit": "512Mi"
        }
    }
for name, config in deployments.items():
    _post_cpu_update(name, config["cpu_limit_m"], config["memory_limit"])

update_network_latency(20)


