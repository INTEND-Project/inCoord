Microservice1_url = 'http://10.0.0.100:31656'

webhooks = 'http://10.0.0.100:31656,http://10.0.0.100:30171,http://10.0.0.100:31178/bw,http://10.0.0.100:31178/notify'


central_db_url = 'http://10.0.0.100:31075/track_time'
db_url_get_time = 'http://10.0.0.100:31075/get_time'

logs_url = 'http://10.0.0.100:31075/log'


master_ip_address = '10.0.0.100'
worker1_ip_address = '10.0.0.101'
worker2_ip_address = '10.0.0.102'
worker3_ip_address = '10.0.0.103'

all_worker_node_ips = [
    '10.0.0.101',
    '10.0.0.102',
    '10.0.0.103'
]

node_port_microservice1 = 30001

k3s_config_file = "/etc/rancher/k3s/k3s.yaml"
