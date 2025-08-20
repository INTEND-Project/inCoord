CREATE (:Drone:Source {name: 'Source Drone 1'});
CREATE (:Drone:Source {name: 'Source Drone 2'});
CREATE (:Drone:Source {name: 'Source Drone 3'});
CREATE (:Drone:Source {name: 'Source Drone 4'});

CREATE
  (k8s:KubernetesCluster {name: 'K3s Cluster', type: 'K3s'}),

  (ms1:Service {name: 'Resize image', podCount: 2}),
  (m3:Metric {name: 'Service Latency', value: 20, unit: 'ms', tunable: true}),
  (m4:Metric {name: 'Service CPU Limit Cores', value: 0.5, unit: 'cores', tunable: false}),
  (m5:Metric {name: 'Service Memory Limit MiB', value: 512, unit: 'MiB', tunable: false}),
  (ms1)-[:HAS_METRIC]->(m3),
  (ms1)-[:HAS_METRIC]->(m4),
  (ms1)-[:HAS_METRIC]->(m5),

  (ms2: Service {name: 'Black and white', podCount: 3}),
  (m6:Metric {name: 'Service Latency', value: 50, unit: 'ms', tunable: true}),
  (m7:Metric {name: 'Service CPU Limit Cores', value: 0.5, unit: 'cores', tunable: false}),
  (m8:Metric {name: 'Service Memory Limit MiB', value: 512, unit: 'MiB', tunable: false}),
  (ms2)-[:HAS_METRIC]->(m6),
  (ms2)-[:HAS_METRIC]->(m7),
  (ms2)-[:HAS_METRIC]->(m8),

  (ms3: Service {name: 'Object detection', podCount: 6}),
  (m9:Metric {name: 'Service Latency', value: 200, unit: 'ms', tunable: true}),
  (m10:Metric {name: 'Service CPU Limit Cores', value: 0.5, unit: 'cores', tunable: false}),
  (m11:Metric {name: 'Service Memory Limit MiB', value: 512, unit: 'MiB', tunable: false}),
  (ms3)-[:HAS_METRIC]->(m9),
  (ms3)-[:HAS_METRIC]->(m10),
  (ms3)-[:HAS_METRIC]->(m11),

  (ms4: Service {name: 'Alarm', podCount: 1}),
  (m12:Metric {name: 'Service Latency', value: 20, unit: 'ms', tunable: true}),
  (m13:Metric {name: 'Service CPU Limit Cores', value: 0.5, unit: 'cores', tunable: false}),
  (m14:Metric {name: 'Service Memory Limit MiB', value: 512, unit: 'MiB', tunable: false}),
  (ms4)-[:HAS_METRIC]->(m12),
  (ms4)-[:HAS_METRIC]->(m13),
  (ms4)-[:HAS_METRIC]->(m14),


  (ms5: Service {name: 'DB', podCount: 2}),
  (m15:Metric {name: 'Service Latency', value: 10, unit: 'ms', tunable: true}),
  (m16:Metric {name: 'Service CPU Limit Cores', value: 0.5, unit: 'cores', tunable: false}),
  (m17:Metric {name: 'Service Memory Limit MiB', value: 512, unit: 'MiB', tunable: false}),
  (ms5)-[:HAS_METRIC]->(m15),
  (ms5)-[:HAS_METRIC]->(m16),
  (ms5)-[:HAS_METRIC]->(m17),

  (k1:K3sHost {name: 'master'}),
  (m18:Metric {name: 'Host CPU Usage', value: 75.3, unit: '%', tunable: false}),
  (k1)-[:HAS_METRIC]->(m18),
  (k2:K3sHost {name: 'worker1'}),
  (m19:Metric {name: 'Host CPU Usage', value: 50, unit: '%', tunable: false}),
  (k2)-[:HAS_METRIC]->(m19),

  (k3:K3sHost {name: 'worker2'}),
  (m20:Metric {name: 'Host CPU Usage', value: 40, unit: '%', tunable: false}),
  (k3)-[:HAS_METRIC]->(m20),

  (k4:K3sHost {name: 'worker3'}),
  (m21:Metric {name: 'Host CPU Usage', value: 30, unit: '%', tunable: false}),
  (k4)-[:HAS_METRIC]->(m21),


  (n:NetworkTopology {name: 'Network Topology'}),
  (m23:Metric {name: 'Network Latency', value: 100, unit: 'ms', tunable: true}),
  (n)-[:HAS_METRIC]->(m23),
  (s1:Switch {name: 'switch1'})-[:PART_OF]->(n),
  (s2:Switch {name: 'switch2'})-[:PART_OF]->(n),
  (s3:Switch {name: 'switch3'})-[:PART_OF]->(n),
  (s4:Switch {name: 'switch4'})-[:PART_OF]->(n),
  (s5:Switch {name: 'switch5'})-[:PART_OF]->(n),
  (s6:Switch {name: 'switch6'})-[:PART_OF]->(n);

MATCH
  (s1:Switch {name: 'switch1'}),
  (s2:Switch {name: 'switch2'}),
  (s3:Switch {name: 'switch3'}),
  (s4:Switch {name: 'switch4'}),
  (s5:Switch {name: 'switch5'}),
  (s6:Switch {name: 'switch6'})
CREATE
  (s1)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s2),
  (s1)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s3),
  (s1)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s4),
  (s1)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s5),
  (s2)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s3),
  (s2)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s4),
  (s2)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s5),
  (s3)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s4),
  (s3)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s6),
  (s4)-[:CONNECTED_TO {latency: 50, bandwidth: 50}]->(s6);

MATCH
  (s1:Switch {name: 'switch1'}),
  (s2:Switch {name: 'switch2'}),
  (s3:Switch {name: 'switch3'}),
  (s4:Switch {name: 'switch4'}),
  (k1:K3sHost {name: 'master'}),
  (k2:K3sHost {name: 'worker1'}),
  (k3:K3sHost {name: 'worker2'}),
  (k4:K3sHost {name: 'worker3'})
CREATE
  (k1)-[:CONNECTED_TO {latency: 5, bandwidth: 50}]->(s1),
  (k2)-[:CONNECTED_TO {latency: 5, bandwidth: 50}]->(s2),
  (k3)-[:CONNECTED_TO {latency: 5, bandwidth: 50}]->(s3),
  (k4)-[:CONNECTED_TO {latency: 5, bandwidth: 50}]->(s4);

MATCH
  (k8s:KubernetesCluster {name: 'K3s Cluster'}),
  (k1:K3sHost {name: 'master'}),
  (k2:K3sHost {name: 'worker1'}),
  (k3:K3sHost {name: 'worker2'}),
  (k4:K3sHost {name: 'worker3'})
CREATE
  (k8s)-[:HOSTS]->(k1),
  (k8s)-[:HOSTS]->(k2),
  (k8s)-[:HOSTS]->(k3),
  (k8s)-[:HOSTS]->(k4);

MATCH
  (ms1:Service {name: 'Resize image'}),
  (ms2:Service {name: 'Black and white'}),
  (ms3:Service {name: 'Object detection'}),
  (ms4:Service {name: 'Alarm'}),
  (ms5:Service {name: 'DB'}),
  (k1:K3sHost {name: 'worker1'}),
  (k2:K3sHost {name: 'worker2'}),
  (k3:K3sHost {name: 'worker3'})
CREATE
  (ms1)-[:RUNS_ON]->(k3),
  (ms2)-[:RUNS_ON]->(k3),
  (ms3)-[:RUNS_ON]->(k1),
  (ms4)-[:RUNS_ON]->(k2),
  (ms5)-[:RUNS_ON]->(k2);

MATCH
  (d:Drone:Source),
  (k8s:KubernetesCluster {name: 'K3s Cluster'})
CREATE
  (d)-[:SENDS_TO]->(k8s);