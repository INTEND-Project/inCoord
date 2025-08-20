// Metric Categories
CREATE (availability:SLO {name: 'Availability', description: 'System uptime and successful request ratio'});
CREATE (latency:SLO {name: 'Latency', description: 'Time taken for responses or page loads'});
CREATE (throughput:SLO {name: 'Throughput', description: 'Volume of requests or data processed'});
CREATE (saturation:SLO {name: 'Saturation', description: 'System resource usage'});
CREATE (freshness:SLO {name: 'Freshness', description: 'Data age and replication delays'});

//Availability
CREATE (m1:Metric:Availability {name: 'Successful Request Rate', formula: 'Successful Requests / Total Requests'});
CREATE (m2:Metric:Availability {name: 'Uptime Ratio', formula: 'Uptime / Total Time'});
MATCH (a:SLO {name: 'Availability'}), (m:Metric:Availability)
CREATE (a)-[:HAS_METRIC]->(m);

// Latency
CREATE (m1:Metric:Latency {name: 'API Response Time', description: 'Time taken for API response'});
CREATE (m2:Metric:Latency {name: 'Page Load Time', description: 'Time taken for web page load'});
MATCH (l:SLO {name: 'Latency'}), (m:Metric:Latency)
CREATE (l)-[:HAS_METRIC]->(m);


// Throughput
CREATE (m1:Metric:Throughput {name: 'Requests per Second', description: 'Number of requests handled per second'});
CREATE (m2:Metric:Throughput {name: 'Data Volume per Time Unit', description: 'Volume of data processed within a specific time frame'});
MATCH (t:SLO {name: 'Throughput'}), (m:Metric:Throughput)
CREATE (t)-[:HAS_METRIC]->(m);

// Saturation
CREATE (m1:Metric:Saturation {name: 'CPU/RAM Utilization', description: 'Percentage of resource utilization, such as CPU or RAM'});
CREATE (m2:Metric:Saturation {name: 'Storage Utilization', description: 'Used storage relative to total available storage'});
MATCH (s:SLO {name: 'Saturation'}), (m:Metric:Saturation)
CREATE (s)-[:HAS_METRIC]->(m);

// Coverage
CREATE (m1:Metric:Coverage {name: 'Feature Update Reach', description: 'Percentage of users who receive a new feature update within a given time frame'});
CREATE (m2:Metric:Coverage {name: 'Cache Hit Ratio', description: 'Ratio of cached responses vs. total responses delivered'});
MATCH (c:SLO {name: 'Coverage'}), (m:Metric:Coverage)
CREATE (c)-[:HAS_METRIC]->(m);


// Freshness
CREATE (m1:Metric:Freshness {name: 'Data Age', description: 'Age of the data being read relative to when it was written'});
CREATE (m2:Metric:Freshness {name: 'Replication Lag', description: 'Time taken for data replication across multiple databases or systems'});
MATCH (f:SLO {name: 'Freshness'}), (m:Metric:Freshness)
CREATE (f)-[:HAS_METRIC]->(m);

// Capacity
CREATE (m1:Metric:Capacity {name: 'Max Concurrent Users', description: 'Maximum number of users or sessions the system can handle simultaneously'});
CREATE (m2:Metric:Capacity {name: 'Max Data Volume', description: 'Maximum data volume the system can handle without degradation'});
MATCH (c:SLO {name: 'Capacity'}), (m:Metric:Capacity)
CREATE (c)-[:HAS_METRIC]->(m);

// Latency Factors

CREATE (f1_l:SLOFactor:Latency {name: 'Network Latency', description: 'Time taken for data to travel over the network'});
CREATE (f2_l:SLOFactor:Latency {name: 'Compute Latency', description: 'Time taken by backend servers or functions to compute the response'});
CREATE (f3_l:SLOFactor:Latency {name: 'Disk I/O Latency', description: 'Time taken to read/write from storage'});
CREATE (f4_l:SLOFactor:Latency {name: 'Queueing Delay', description: 'Time spent waiting in processing queues'});
CREATE (f5_l:SLOFactor:Latency {name: 'Client-Side Rendering Time', description: 'Time the client takes to render the content after receiving it'});
MATCH (l:SLO {name: 'Latency'}), (f:SLOFactor:Latency)
CREATE (l)-[:DEPENDENT_ON]->(f);

// AVAILABILITY Dependencies
CREATE (f1_a:SLOFactor:Availability {name: 'Redundancy', description: 'Redundant systems to ensure failover'});
CREATE (f2_a:SLOFactor:Availability {name: 'Monitoring', description: 'Availability monitoring and alerting'});
CREATE (f3_a:SLOFactor:Availability {name: 'Failover Strategy', description: 'Automatic failover configuration'});
MATCH (a:SLO {name: 'Availability'}), (f:SLOFactor:Availability)
CREATE (a)-[:DEPENDENT_ON]->(f);


// THROUGHPUT Dependencies
CREATE (f1_t:SLOFactor:Throughput {name: 'Concurrency Model', description: 'Ability to handle simultaneous requests'});
CREATE (f2_t:SLOFactor:Throughput {name: 'DB Write Performance', description: 'Speed of database writes'});
CREATE (f3_t:SLOFactor:Throughput {name: 'Network Bandwidth', description: 'Available throughput at the network level'});
MATCH (t:SLO {name: 'Throughput'}), (f:SLOFactor:Throughput)
CREATE (t)-[:DEPENDENT_ON]->(f);

// SATURATION Dependencies
CREATE (f1_s:SLOFactor:Saturation {name: 'CPU Utilization', description: 'Processor usage under load'});
CREATE (f2_s:SLOFactor:Saturation {name: 'RAM Usage', description: 'Memory allocation and limits'});
CREATE (f3_s:SLOFactor:Saturation {name: 'Disk Throughput', description: 'I/O operations per second'});
MATCH (s:SLO {name: 'Saturation'}), (f:SLOFactor:Saturation)
CREATE (s)-[:DEPENDENT_ON]->(f);

// FRESHNESS Dependencies
CREATE (f1_f:SLOFactor:Freshness {name: 'Replication Delay', description: 'Lag in syncing data across DBs or services'});
CREATE (f2_f:SLOFactor:Freshness {name: 'Caching Policies', description: 'Staleness of cached content'});
CREATE (f3_f:SLOFactor:Freshness {name: 'Data Pipeline Latency', description: 'Delays in data ingestion and processing'});
MATCH (f:SLO {name: 'Freshness'}), (f1:SLOFactor:Freshness)
CREATE (f)-[:DEPENDENT_ON]->(f1);


//// Example Intent: Real-Time Video Streaming
//CREATE (video:Intent {name: 'Real-Time Video Streaming', description: 'Users watch live or real-time video content', priority: 'High'});
//
//// Link to Latency, Throughput
//MATCH (latency:SLO {name: 'Latency'}), (throughput:SLO {name: 'Throughput'}), (saturation:SLO {name: 'Saturation'}), (video:Intent)
//MERGE (video)-[:REQUIRES]->(latency)
//MERGE (video)-[:REQUIRES]->(throughput)
//MERGE (video)-[:REQUIRES]->(saturation);
//
//// Add a buffering concept
//CREATE (buffering:ExperienceIssue {name: 'Buffering', description: 'Playback is paused while waiting for data'})
//
//// Connect to SLIs that cause buffering
//MATCH (latency:SLO {name: 'Latency'})
//MATCH (throughput:SLO {name: 'Throughput'})
//MATCH (saturation:SLO {name: 'Saturation'})
//MERGE (buffering)-[:CAUSED_BY]->(latency)
//MERGE (buffering)-[:CAUSED_BY]->(throughput)
//MERGE (buffering)-[:CAUSED_BY]->(saturation)
//
//// Tie it to the business intent
//MATCH (intent:Intent {name: 'Real-Time Video Streaming'})
//MERGE (intent)-[:AIMS_TO_AVOID]->(buffering)
//
//// Example Intent: E-commerce Checkout
//CREATE (checkout:Intent {name: 'E-commerce Checkout', description: 'Completing a product purchase in an online store', priority: 'High'})
//MATCH (availability:SLO {name: 'Availability'})
//MATCH (latency2:SLO {name: 'Latency'})
//MATCH (capacity:SLO {name: 'Capacity'})
//MERGE (checkout)-[:REQUIRES]->(availability)
//MERGE (checkout)-[:REQUIRES]->(latency2)
//
//// Example Intent: News Feed Scrolling
//CREATE (feed:Intent {name: 'News Feed Scrolling', description: 'Smooth scrolling through dynamically loaded content'})
//MATCH (latency3:SLO {name: 'Latency'})
//MATCH (freshness:SLO {name: 'Freshness'})
//MATCH (saturation:SLO {name: 'Saturation'})
//MERGE (feed)-[:REQUIRES]->(latency3)
//MERGE (feed)-[:REQUIRES]->(freshness)
//MERGE (feed)-[:REQUIRES]->(saturation)
