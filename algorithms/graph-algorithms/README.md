
# Graph Algorithms for Network Analysis

This module provides production-ready graph algorithms and network analysis tools tailored for backend systems. It is designed for analyzing microservices architectures, optimizing network routing, mapping infrastructure dependencies, and monitoring service health.

---

## Features

- Directed and undirected weighted graph modeling
- Dijkstra's shortest path (latency/cost optimization)
- Tarjan's Strongly Connected Components (SCC) for dependency and partition analysis
- BFS for minimum-hop routing
- Comprehensive path analysis with health and bottleneck detection
- Server health metrics integration
- Criticality analysis for infrastructure planning
- Network partition detection and reporting
- Caching for efficient repeated queries

## Algorithms Included

- **Dijkstra's Algorithm**: Find shortest weighted paths (latency, cost, etc.)
- **Tarjan's SCC Algorithm**: Detect strongly connected components (service clusters, cycles)
- **BFS**: Find shortest path by hop count
- **Criticality Analysis**: Identify most critical servers in the network
- **Alternative Path Discovery**: Enumerate multiple viable routes

## Use Cases

- Optimize API call paths between microservices
- Detect circular dependencies and service clusters
- Identify single points of failure in infrastructure
- Analyze network partitions and isolated groups
- Monitor and report on service health and bottlenecks
- Plan for redundancy and disaster recovery

## Usage

See [`graph_algorithms.py`](./graph_algorithms.py) for full implementation and advanced usage examples.

### Example: Microservices Network Analysis

```python
from graph_algorithms import NetworkAnalyzer

# Initialize analyzer and build network
network = NetworkAnalyzer()
network.add_server_connection("api-gateway", "auth-service", 5)
network.add_server_connection("auth-service", "database", 15)
network.update_server_metrics("database", cpu_usage=95, memory_usage=90, load=95)

# Find optimal path
path_info = network.find_optimal_path("api-gateway", "database")
print(f"Optimal path: {path_info.path}")
print(f"Total latency: {path_info.total_latency}ms")
print(f"Health status: {path_info.status.value}")

# Analyze server criticality
criticality = network.analyze_server_criticality()
print(f"Most critical server: {max(criticality.items(), key=lambda x: x[1])}")
```

---

For more details, see the code and docstrings in [`graph_algorithms.py`](./graph_algorithms.py).
