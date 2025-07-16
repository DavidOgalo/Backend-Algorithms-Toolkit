
"""
Graph Algorithms for Backend Network Analysis
--------------------------------------------
Production-grade graph algorithms and network analysis tools for backend systems.

Features:
- Directed/undirected weighted graph modeling
- Dijkstra's shortest path, BFS, Tarjan's SCC
- Health-aware path analysis and criticality detection
- Designed for microservices, infrastructure, and backend topologies
"""

from collections import defaultdict, deque
from typing import Set, List, Dict, Tuple, Optional, Any, Union
import heapq
import logging
from dataclasses import dataclass
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class HealthStatus(Enum):
    """
    Health status classifications for network paths and services.
    Used to represent the health of servers and network routes.
    """
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNREACHABLE = "unreachable"



@dataclass
class ServerMetrics:
    """
    Encapsulates server health metrics for network analysis.

    Attributes:
        cpu_usage (float): CPU usage percentage (0-100)
        memory_usage (float): Memory usage percentage (0-100)
        load (float): System load percentage (0-100)
        timestamp (float): Time the metrics were recorded (epoch seconds)
    """
    cpu_usage: float
    memory_usage: float
    load: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def health_score(self) -> float:
        """
        Calculate a composite health score (0-100, lower is better).
        Returns the average of CPU, memory, and load.
        """
        return (self.cpu_usage + self.memory_usage + self.load) / 3

    @property
    def status(self) -> HealthStatus:
        """
        Determine health status based on the composite score.
        Returns a HealthStatus enum value.
        """
        score = self.health_score
        if score < 70:
            return HealthStatus.HEALTHY
        elif score < 90:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL



@dataclass
class PathAnalysis:
    """
    Comprehensive path analysis result for network routing.

    Attributes:
        path (Optional[List[str]]): The optimal path as a list of server names
        total_latency (float): Total latency along the path
        average_health_score (float): Average health score of servers on the path
        hop_count (int): Number of hops (edges) in the path
        status (HealthStatus): Health status of the path
        bottleneck_server (Optional[str]): Server with the worst health on the path
        alternative_paths (List[List[str]]): Alternative paths (if requested)
    """
    path: Optional[List[str]]
    total_latency: float
    average_health_score: float
    hop_count: int
    status: HealthStatus
    bottleneck_server: Optional[str] = None
    alternative_paths: List[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the path analysis result to a dictionary for JSON serialization.
        """
        return {
            'path': self.path,
            'total_latency': self.total_latency,
            'average_health_score': self.average_health_score,
            'hop_count': self.hop_count,
            'status': self.status.value,
            'bottleneck_server': self.bottleneck_server,
            'alternative_paths': self.alternative_paths or []
        }


class GraphValidationError(Exception):
    """Custom exception for graph validation errors"""
    pass


class Graph:
    """
    High-performance graph implementation for backend network modeling.

    Supports both directed and undirected graphs with weighted edges.
    Optimized for:
    - Network analysis
    - Microservices routing
    - Infrastructure mapping
    - Service dependency and partition detection
    """
    
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adjacency_list: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.vertices: Set[str] = set()
        self._edge_count = 0
    
    def add_edge(self, source: str, destination: str, weight: float = 1.0) -> None:
        """
        Add weighted edge between vertices.
        
        Args:
            source: Source vertex
            destination: Destination vertex  
            weight: Edge weight (latency, distance, cost, etc.)
            
        Raises:
            GraphValidationError: If weight is negative
        """
        if weight < 0:
            raise GraphValidationError(f"Negative weight not allowed: {weight}")
        
        self.vertices.add(source)
        self.vertices.add(destination)
        
        # Add edge to adjacency list
        self.adjacency_list[source].append((destination, weight))
        self._edge_count += 1
        
        # Add reverse edge for undirected graphs
        if not self.directed:
            self.adjacency_list[destination].append((source, weight))
            self._edge_count += 1
    
    def remove_edge(self, source: str, destination: str) -> bool:
        """
        Remove edge between vertices.
        Returns True if the edge existed and was removed, False otherwise.
        """
        if source not in self.vertices or destination not in self.vertices:
            return False
        
        # Remove from adjacency list
        original_length = len(self.adjacency_list[source])
        self.adjacency_list[source] = [
            (dest, weight) for dest, weight in self.adjacency_list[source]
            if dest != destination
        ]
        
        edge_removed = len(self.adjacency_list[source]) < original_length
        
        if edge_removed:
            self._edge_count -= 1
            
            # Remove reverse edge for undirected graphs
            if not self.directed:
                self.adjacency_list[destination] = [
                    (dest, weight) for dest, weight in self.adjacency_list[destination]
                    if dest != source
                ]
                self._edge_count -= 1
        
        return edge_removed
    
    def get_neighbors(self, vertex: str) -> List[Tuple[str, float]]:
        """
        Get all neighbors of a vertex with their edge weights.
        Returns a list of (neighbor, weight) tuples.
        """
        return self.adjacency_list.get(vertex, [])
    
    def bfs_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        Find shortest path (minimum hops) using BFS.
        
        Optimal for unweighted graphs or when hop count matters more than total weight.
        Time complexity: O(V + E)
        
        Args:
            start: Starting vertex
            end: Target vertex
            
        Returns:
            List of vertices in shortest path, or None if no path exists
        """
        if start not in self.vertices or end not in self.vertices:
            logger.warning(f"BFS failed: Vertex not found. Start: {start}, End: {end}")
            return None
        
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor, _ in self.adjacency_list[current]:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def dijkstra_shortest_path(self, start: str, end: str) -> Tuple[Optional[List[str]], float]:
        """
        Find shortest weighted path using Dijkstra's algorithm.
        
        Optimal for weighted graphs where total weight (latency, cost) matters.
        Time complexity: O((V + E) log V)
        
        Args:
            start: Starting vertex
            end: Target vertex
            
        Returns:
            Tuple of (path, total_weight) or (None, inf) if unreachable
        """
        if start not in self.vertices or end not in self.vertices:
            logger.warning(f"Dijkstra failed: Vertex not found. Start: {start}, End: {end}")
            return None, float('inf')
        
        if start == end:
            return [start], 0
        
        # Priority queue: (distance, vertex, path)
        pq = [(0, start, [start])]
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        visited = set()
        
        while pq:
            current_dist, current, path = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end:
                return path, current_dist
            
            for neighbor, weight in self.adjacency_list[current]:
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor, path + [neighbor]))
        
        return None, float('inf')
    
    def find_all_paths(self, start: str, end: str, max_depth: int = 10) -> List[Tuple[List[str], float]]:
        """
        Find all paths between two vertices (with depth limit to prevent infinite loops).
        
        Useful for finding alternative routes in network analysis.
        
        Args:
            start: Starting vertex
            end: Target vertex
            max_depth: Maximum path length to explore
            
        Returns:
            List of (path, total_weight) tuples
        """
        if start not in self.vertices or end not in self.vertices:
            return []
        
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str], total_weight: float, depth: int):
            if depth > max_depth:
                return
            
            if current == target:
                all_paths.append((path[:], total_weight))
                return
            
            for neighbor, weight in self.adjacency_list[current]:
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs(neighbor, target, path, total_weight + weight, depth + 1)
                    path.pop()
        
        dfs(start, end, [start], 0, 0)
        return sorted(all_paths, key=lambda x: x[1])  # Sort by total weight
    
    def find_strongly_connected_components(self) -> List[List[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.
        
        Essential for detecting:
        - Service clusters in microservices
        - Circular dependencies
        - Network partitions
        
        Time complexity: O(V + E)
        
        Returns:
            List of strongly connected components
            
        Raises:
            GraphValidationError: If called on undirected graph
        """
        if not self.directed:
            raise GraphValidationError("SCC analysis only works on directed graphs")
        
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        result = []
        
        def strongconnect(vertex: str):
            index[vertex] = index_counter[0]
            lowlinks[vertex] = index_counter[0]
            index_counter[0] += 1
            stack.append(vertex)
            on_stack[vertex] = True
            
            for neighbor, _ in self.adjacency_list[vertex]:
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[vertex] = min(lowlinks[vertex], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[vertex] = min(lowlinks[vertex], index[neighbor])
            
            if lowlinks[vertex] == index[vertex]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == vertex:
                        break
                result.append(component)
        
        for vertex in self.vertices:
            if vertex not in index:
                strongconnect(vertex)
        
        return result
    
    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get graph statistics including vertex/edge count and density.
        """
        return {
            'vertices': len(self.vertices),
            'edges': self._edge_count,
            'directed': self.directed,
            'density': self._edge_count / (len(self.vertices) * (len(self.vertices) - 1)) if len(self.vertices) > 1 else 0
        }


class NetworkAnalyzer:
    """
    Production-ready network analysis toolkit for backend systems.

    Provides comprehensive analysis of network topologies including:
    - Optimal path finding with health and latency considerations
    - Critical infrastructure identification
    - Service dependency mapping
    - Performance bottleneck detection
    - Network partitioning and reporting
    """
    
    def __init__(self, enable_caching: bool = True):
        self.network_graph = Graph(directed=True)
        self.server_metrics: Dict[str, ServerMetrics] = {}
        self.enable_caching = enable_caching
        self._path_cache: Dict[Tuple[str, str], PathAnalysis] = {}
    
    def add_server_connection(self, server1: str, server2: str, latency: float) -> None:
        """
        Add bidirectional connection between servers.
        
        Args:
            server1: Source server
            server2: Destination server
            latency: Connection latency in milliseconds
        """
        if latency < 0:
            raise GraphValidationError(f"Latency cannot be negative: {latency}")
        
        self.network_graph.add_edge(server1, server2, latency)
        logger.info(f"Added connection: {server1} -> {server2} ({latency}ms)")
    
    def update_server_metrics(self, server: str, cpu_usage: float, memory_usage: float, load: float) -> None:
        """
        Update health metrics for a server.
        
        Args:
            server: Server identifier
            cpu_usage: CPU usage percentage (0-100)
            memory_usage: Memory usage percentage (0-100)
            load: System load percentage (0-100)
        """
        self.server_metrics[server] = ServerMetrics(cpu_usage, memory_usage, load)
        
        # Clear cache for paths involving this server
        if self.enable_caching:
            self._invalidate_cache_for_server(server)
    
    def _invalidate_cache_for_server(self, server: str) -> None:
        """
        Invalidate cached paths involving a specific server.
        Used to ensure cache consistency when server health changes.
        """
        keys_to_remove = [
            key for key in self._path_cache.keys()
            if server in key
        ]
        for key in keys_to_remove:
            del self._path_cache[key]
    
    def find_optimal_path(self, source: str, destination: str, 
                         include_alternatives: bool = False) -> PathAnalysis:
        """
        Find optimal path considering both latency and server health.
        
        Args:
            source: Source server
            destination: Destination server
            include_alternatives: Whether to include alternative paths
            
        Returns:
            PathAnalysis object with comprehensive path information
        """
        cache_key = (source, destination)
        
        # Check cache first
        if self.enable_caching and cache_key in self._path_cache:
            cached_result = self._path_cache[cache_key]
            # Check if cache is still valid (within last 5 minutes)
            if time.time() - cached_result.average_health_score < 300:
                return cached_result
        
        # Find primary path
        path, total_latency = self.network_graph.dijkstra_shortest_path(source, destination)
        
        if path is None:
            result = PathAnalysis(
                path=None,
                total_latency=float('inf'),
                average_health_score=0,
                hop_count=0,
                status=HealthStatus.UNREACHABLE
            )
        else:
            # Calculate health metrics
            health_scores = []
            bottleneck_server = None
            worst_health = 0
            
            for server in path:
                if server in self.server_metrics:
                    metrics = self.server_metrics[server]
                    health_score = metrics.health_score
                    health_scores.append(health_score)
                    
                    if health_score > worst_health:
                        worst_health = health_score
                        bottleneck_server = server
            
            avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
            
            # Determine status
            if avg_health < 70:
                status = HealthStatus.HEALTHY
            elif avg_health < 90:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.CRITICAL
            
            # Find alternative paths if requested
            alternative_paths = []
            if include_alternatives:
                all_paths = self.network_graph.find_all_paths(source, destination, max_depth=8)
                alternative_paths = [path for path, _ in all_paths[1:6]]  # Top 5 alternatives
            
            result = PathAnalysis(
                path=path,
                total_latency=total_latency,
                average_health_score=avg_health,
                hop_count=len(path) - 1,
                status=status,
                bottleneck_server=bottleneck_server,
                alternative_paths=alternative_paths
            )
        
        # Cache result
        if self.enable_caching:
            self._path_cache[cache_key] = result
        
        return result
    
    def detect_network_partitions(self) -> List[List[str]]:
        """
        Detect network partitions using strongly connected components.
        
        Returns:
            List of server groups that are isolated from each other
        """
        try:
            return self.network_graph.find_strongly_connected_components()
        except GraphValidationError as e:
            logger.error(f"Partition detection failed: {e}")
            return []
    
    def analyze_server_criticality(self) -> Dict[str, float]:
        """
        Analyze how critical each server is to overall network connectivity.
        
        Criticality score represents the percentage of server pairs that would
        lose connectivity if this server fails.
        
        Returns:
            Dictionary mapping server names to criticality scores (0-100)
        """
        criticality_scores = {}
        
        for server in self.network_graph.vertices:
            affected_paths = 0
            total_paths = 0
            
            # Test all server pairs
            for source in self.network_graph.vertices:
                for dest in self.network_graph.vertices:
                    if source != dest:
                        total_paths += 1
                        path, _ = self.network_graph.dijkstra_shortest_path(source, dest)
                        
                        # Check if path exists and server is a critical node
                        if path and server in path and server != source and server != dest:
                            affected_paths += 1
            
            criticality_scores[server] = (affected_paths / total_paths * 100) if total_paths > 0 else 0
        
        return criticality_scores
    
    def generate_network_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive network analysis report.
        
        Returns:
            Dictionary containing complete network analysis
        """
        return {
            'network_stats': self.network_graph.stats,
            'server_health': {
                server: {
                    'metrics': {
                        'cpu_usage': metrics.cpu_usage,
                        'memory_usage': metrics.memory_usage,
                        'load': metrics.load,
                        'health_score': metrics.health_score
                    },
                    'status': metrics.status.value
                }
                for server, metrics in self.server_metrics.items()
            },
            'criticality_analysis': self.analyze_server_criticality(),
            'network_partitions': self.detect_network_partitions(),
            'cache_stats': {
                'enabled': self.enable_caching,
                'cached_paths': len(self._path_cache)
            }
        }



def main():
    """
    Demonstration of network analysis capabilities using a sample microservices network.
    Shows optimal path analysis, server criticality, network partitions, and reporting.
    """
    # Initialize network analyzer
    network = NetworkAnalyzer(enable_caching=True)

    # Build sample microservices network
    logger.info("Building sample microservices network...")
    connections = [
        ("api-gateway", "auth-service", 5),
        ("api-gateway", "user-service", 8),
        ("api-gateway", "product-service", 10),
        ("auth-service", "database", 15),
        ("user-service", "database", 12),
        ("product-service", "database", 20),
        ("product-service", "cache", 3),
        ("user-service", "cache", 4),
        ("database", "backup-db", 50),
        ("cache", "database", 8),  # Cache can read from DB
        ("api-gateway", "load-balancer", 2),
        ("load-balancer", "web-server-1", 3),
        ("load-balancer", "web-server-2", 4),
    ]
    for server1, server2, latency in connections:
        network.add_server_connection(server1, server2, latency)

    # Add server health metrics
    server_stats = {
        "api-gateway": (45, 60, 70),
        "auth-service": (30, 40, 35),
        "user-service": (55, 70, 65),
        "product-service": (80, 85, 90),
        "database": (95, 90, 95),
        "cache": (20, 25, 30),
        "backup-db": (60, 55, 50),
        "load-balancer": (35, 45, 40),
        "web-server-1": (25, 30, 35),
        "web-server-2": (28, 32, 38),
    }
    for server, (cpu, memory, load) in server_stats.items():
        network.update_server_metrics(server, cpu, memory, load)

    # Demonstrate network analysis
    print("\n=== Network Analysis Demonstration ===")

    # 1. Find optimal paths
    print("\n1. Optimal Path Analysis:")
    path_analysis = network.find_optimal_path("api-gateway", "database", include_alternatives=True)
    print(f"   Path: {' -> '.join(path_analysis.path) if path_analysis.path else 'UNREACHABLE'}")
    print(f"   Total latency: {path_analysis.total_latency}ms")
    print(f"   Health status: {path_analysis.status.value}")
    print(f"   Bottleneck server: {path_analysis.bottleneck_server}")

    # 2. Server criticality analysis
    print("\n2. Server Criticality Analysis:")
    criticality = network.analyze_server_criticality()
    for server, score in sorted(criticality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {server}: {score:.1f}% criticality")

    # 3. Network partitions
    print("\n3. Network Partitions:")
    partitions = network.detect_network_partitions()
    for i, partition in enumerate(partitions, 1):
        print(f"   Partition {i}: {partition}")

    # 4. Generate comprehensive report
    print("\n4. Network Statistics:")
    report = network.generate_network_report()
    stats = report['network_stats']
    print(f"   Vertices: {stats['vertices']}")
    print(f"   Edges: {stats['edges']}")
    print(f"   Network density: {stats['density']:.3f}")
    print(f"   Cached paths: {report['cache_stats']['cached_paths']}")


if __name__ == "__main__":
    main()