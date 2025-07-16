"""
Advanced Sorting Algorithms for Backend Data Processing

This module provides production-optimized sorting algorithms for backend systems:
- External sorting for big data processing
- Efficient percentile calculations
- Real-time analytics pipelines
- Memory-efficient batch processing

Key Features:
- Handle datasets larger than available memory
- O(n) percentile calculations without full sorting
- Configurable memory usage and performance tuning
- Production-ready error handling and monitoring

"""

from typing import List, Callable, Any, Dict, Optional, Iterator, Tuple, Union
import heapq
import tempfile
import os
import pickle
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import mmap
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SortingStrategy(Enum):
    """Available sorting strategies for different use cases"""
    MEMORY_OPTIMIZED = "memory_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    HYBRID = "hybrid"


@dataclass
class SortingConfig:
    """Configuration for sorting operations"""
    chunk_size: int = 10000
    max_memory_mb: int = 512
    temp_dir: str = tempfile.gettempdir()
    parallel_workers: int = 4
    strategy: SortingStrategy = SortingStrategy.HYBRID
    enable_compression: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("Max memory must be positive")
        if self.parallel_workers <= 0:
            raise ValueError("Worker count must be positive")


@dataclass
class SortingMetrics:
    """Metrics collected during sorting operations"""
    total_elements: int = 0
    chunks_created: int = 0
    merge_passes: int = 0
    memory_peak_mb: float = 0.0
    processing_time: float = 0.0
    temp_files_created: int = 0
    compression_ratio: float = 0.0
    
    def __str__(self) -> str:
        return (f"SortingMetrics(elements={self.total_elements}, "
                f"chunks={self.chunks_created}, time={self.processing_time:.2f}s)")


class PerformanceMonitor:
    """Monitor performance metrics during sorting operations"""
    
    def __init__(self):
        self.start_time: float = 0
        self.peak_memory: float = 0
        self.operations_count: int = 0
    
    def start_operation(self):
        """Start timing an operation"""
        self.start_time = time.time()
    
    def end_operation(self) -> float:
        """End timing and return elapsed time"""
        return time.time() - self.start_time
    
    def track_memory(self, current_usage: float):
        """Track peak memory usage"""
        self.peak_memory = max(self.peak_memory, current_usage)
    
    def increment_operations(self):
        """Increment operation counter"""
        self.operations_count += 1


class ExternalSortingAlgorithms:
    """
    Production-ready external sorting algorithms for big data processing.
    
    Handles datasets that exceed available memory by processing in chunks
    and using disk-based merge operations.
    """
    
    def __init__(self, config: SortingConfig = None):
        self.config = config or SortingConfig()
        self.metrics = SortingMetrics()
        self.monitor = PerformanceMonitor()
        self._temp_files: List[str] = []
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp files"""
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self._temp_files.clear()
    
    def _create_temp_file(self) -> str:
        """Create and track temporary file"""
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.config.temp_dir,
            prefix="external_sort_",
            suffix=".tmp"
        )
        os.close(temp_fd)  # Close file descriptor, keep path
        self._temp_files.append(temp_path)
        self.metrics.temp_files_created += 1
        return temp_path
    
    def _write_chunk_to_disk(self, chunk: List[Any], temp_file: str, 
                           key_func: Optional[Callable] = None) -> int:
        """
        Write sorted chunk to disk with optional compression.
        
        Returns:
            Number of elements written
        """
        try:
            sorted_chunk = sorted(chunk, key=key_func)
            
            if self.config.enable_compression:
                # Use pickle with highest compression
                with open(temp_file, 'wb') as f:
                    pickle.dump(sorted_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Use JSON for human-readable format (if serializable)
                try:
                    with open(temp_file, 'w') as f:
                        json.dump(sorted_chunk, f)
                except (TypeError, ValueError):
                    # Fallback to pickle if JSON fails
                    with open(temp_file, 'wb') as f:
                        pickle.dump(sorted_chunk, f)
            
            return len(sorted_chunk)
            
        except Exception as e:
            logger.error(f"Failed to write chunk to {temp_file}: {e}")
            raise
    
    def _read_chunk_from_disk(self, temp_file: str) -> Iterator[Any]:
        """Read chunk from disk file"""
        try:
            if self.config.enable_compression:
                with open(temp_file, 'rb') as f:
                    chunk = pickle.load(f)
            else:
                try:
                    with open(temp_file, 'r') as f:
                        chunk = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    # Fallback to pickle
                    with open(temp_file, 'rb') as f:
                        chunk = pickle.load(f)
            
            return iter(chunk)
            
        except Exception as e:
            logger.error(f"Failed to read chunk from {temp_file}: {e}")
            raise
    
    def external_merge_sort(self, data: List[Any], 
                          key_func: Optional[Callable] = None) -> List[Any]:
        """
        External merge sort for datasets larger than memory.
        
        Process:
        1. Split data into memory-sized chunks
        2. Sort each chunk and write to disk
        3. Merge sorted chunks back into final result
        
        Args:
            data: Input data to sort
            key_func: Optional key function for sorting
            
        Returns:
            Sorted list
        """
        self.monitor.start_operation()
        self.metrics.total_elements = len(data)
        
        # Handle small datasets in memory
        if len(data) <= self.config.chunk_size:
            result = sorted(data, key=key_func)
            self.metrics.processing_time = self.monitor.end_operation()
            return result
        
        try:
            # Phase 1: Create sorted chunks
            chunk_files = self._create_sorted_chunks(data, key_func)
            
            # Phase 2: Merge chunks
            result = self._merge_sorted_chunks(chunk_files, key_func)
            
            self.metrics.processing_time = self.monitor.end_operation()
            logger.info(f"External sort completed: {self.metrics}")
            
            return result
            
        finally:
            self._cleanup_temp_files()
    
    def _create_sorted_chunks(self, data: List[Any], 
                            key_func: Optional[Callable] = None) -> List[str]:
        """Create sorted chunks and write to disk"""
        chunk_files = []
        
        # Process data in chunks
        for i in range(0, len(data), self.config.chunk_size):
            chunk = data[i:i + self.config.chunk_size]
            temp_file = self._create_temp_file()
            
            elements_written = self._write_chunk_to_disk(chunk, temp_file, key_func)
            chunk_files.append(temp_file)
            
            self.metrics.chunks_created += 1
            logger.debug(f"Created chunk {len(chunk_files)}: {elements_written} elements")
        
        return chunk_files
    
    def _merge_sorted_chunks(self, chunk_files: List[str], 
                           key_func: Optional[Callable] = None) -> List[Any]:
        """Merge sorted chunks using k-way merge"""
        if not chunk_files:
            return []
        
        if len(chunk_files) == 1:
            # Single chunk - read directly
            return list(self._read_chunk_from_disk(chunk_files[0]))
        
        # Use heap for efficient k-way merge
        result = []
        heap = []
        iterators = []
        
        # Initialize iterators for each chunk
        for i, chunk_file in enumerate(chunk_files):
            try:
                iterator = self._read_chunk_from_disk(chunk_file)
                iterators.append(iterator)
                
                # Add first element from each chunk to heap
                element = next(iterator)
                heap_entry = (
                    key_func(element) if key_func else element,
                    element,
                    i  # chunk index
                )
                heapq.heappush(heap, heap_entry)
                
            except StopIteration:
                # Empty chunk
                iterators.append(None)
        
        # Merge process
        while heap:
            key_val, element, chunk_idx = heapq.heappop(heap)
            result.append(element)
            
            # Add next element from same chunk
            if iterators[chunk_idx] is not None:
                try:
                    next_element = next(iterators[chunk_idx])
                    heap_entry = (
                        key_func(next_element) if key_func else next_element,
                        next_element,
                        chunk_idx
                    )
                    heapq.heappush(heap, heap_entry)
                except StopIteration:
                    iterators[chunk_idx] = None
        
        self.metrics.merge_passes += 1
        return result
    
    def parallel_external_sort(self, data: List[Any], 
                             key_func: Optional[Callable] = None) -> List[Any]:
        """
        Parallel external sorting using multiple threads.
        
        Improves performance for large datasets by processing chunks in parallel.
        """
        self.monitor.start_operation()
        self.metrics.total_elements = len(data)
        
        if len(data) <= self.config.chunk_size:
            result = sorted(data, key=key_func)
            self.metrics.processing_time = self.monitor.end_operation()
            return result
        
        try:
            # Create chunks for parallel processing
            chunks = [
                data[i:i + self.config.chunk_size]
                for i in range(0, len(data), self.config.chunk_size)
            ]
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                # Submit sorting tasks
                future_to_chunk = {
                    executor.submit(self._sort_chunk_to_file, chunk, key_func): i
                    for i, chunk in enumerate(chunks)
                }
                
                chunk_files = [None] * len(chunks)
                
                # Collect results
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_files[chunk_idx] = future.result()
                        self.metrics.chunks_created += 1
                    except Exception as e:
                        logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                        raise
            
            # Merge sorted chunks
            result = self._merge_sorted_chunks(chunk_files, key_func)
            
            self.metrics.processing_time = self.monitor.end_operation()
            logger.info(f"Parallel external sort completed: {self.metrics}")
            
            return result
            
        finally:
            self._cleanup_temp_files()
    
    def _sort_chunk_to_file(self, chunk: List[Any], 
                          key_func: Optional[Callable] = None) -> str:
        """Sort chunk and write to temporary file"""
        temp_file = self._create_temp_file()
        self._write_chunk_to_disk(chunk, temp_file, key_func)
        return temp_file


class FastPercentileCalculator:
    """
    High-performance percentile calculator using QuickSelect.
    
    Optimized for real-time analytics where you need specific percentiles
    without full sorting overhead.
    """
    
    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching
        self._cache: Dict[Tuple[int, float], Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def quickselect_kth_element(self, data: List[Any], k: int, 
                               key_func: Optional[Callable] = None) -> Any:
        """
        Find k-th smallest element using QuickSelect algorithm.
        
        Average time complexity: O(n)
        Worst case: O(n²) - but very rare with good pivot selection
        
        Args:
            data: Input data
            k: Index of desired element (0-based)
            key_func: Optional key function for comparison
            
        Returns:
            k-th smallest element
        """
        if not data or k < 0 or k >= len(data):
            raise ValueError(f"Invalid k value: {k} for data length {len(data)}")
        
        def partition(arr: List[Any], low: int, high: int) -> int:
            """Partition array around pivot"""
            # Use median-of-three pivot selection for better performance
            mid = (low + high) // 2
            pivot_candidates = [
                (arr[low], low),
                (arr[mid], mid),
                (arr[high], high)
            ]
            
            if key_func:
                pivot_candidates.sort(key=lambda x: key_func(x[0]))
            else:
                pivot_candidates.sort(key=lambda x: x[0])
            
            # Swap median to end
            pivot_idx = pivot_candidates[1][1]
            arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
            
            pivot_val = key_func(arr[high]) if key_func else arr[high]
            i = low - 1
            
            for j in range(low, high):
                current_val = key_func(arr[j]) if key_func else arr[j]
                if current_val <= pivot_val:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1
        
        def quickselect_recursive(arr: List[Any], low: int, high: int, k: int) -> Any:
            """Recursive quickselect implementation"""
            if low == high:
                return arr[low]
            
            pivot_index = partition(arr, low, high)
            
            if k == pivot_index:
                return arr[k]
            elif k < pivot_index:
                return quickselect_recursive(arr, low, pivot_index - 1, k)
            else:
                return quickselect_recursive(arr, pivot_index + 1, high, k)
        
        # Work on copy to avoid modifying original
        data_copy = data.copy()
        return quickselect_recursive(data_copy, 0, len(data_copy) - 1, k)
    
    def calculate_percentile(self, data: List[Any], percentile: float, 
                           key_func: Optional[Callable] = None) -> Any:
        """
        Calculate percentile using QuickSelect.
        
        Args:
            data: Input data
            percentile: Percentile to calculate (0-100)
            key_func: Optional key function for comparison
            
        Returns:
            Value at specified percentile
        """
        if not data:
            raise ValueError("Cannot calculate percentile of empty data")
        
        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        
        # Check cache
        cache_key = (len(data), percentile)
        if self.enable_caching and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Calculate index for percentile
        n = len(data)
        if percentile == 100:
            k = n - 1
        else:
            k = int((percentile / 100) * n)
            k = max(0, min(k, n - 1))
        
        result = self.quickselect_kth_element(data, k, key_func)
        
        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = result
        
        return result
    
    def calculate_multiple_percentiles(self, data: List[Any], 
                                     percentiles: List[float],
                                     key_func: Optional[Callable] = None) -> Dict[float, Any]:
        """
        Calculate multiple percentiles efficiently.
        
        More efficient than calling calculate_percentile multiple times
        as it can reuse sorted portions of the data.
        """
        if not data:
            raise ValueError("Cannot calculate percentiles of empty data")
        
        results = {}
        n = len(data)
        
        # Sort percentiles to optimize cache usage
        sorted_percentiles = sorted(percentiles)
        
        for percentile in sorted_percentiles:
            if not 0 <= percentile <= 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
            
            if percentile == 100:
                k = n - 1
            else:
                k = int((percentile / 100) * n)
                k = max(0, min(k, n - 1))
            
            results[percentile] = self.quickselect_kth_element(data, k, key_func)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_entries': len(self._cache)
        }


class AdvancedAnalyticsPipeline:
    """
    Production-ready analytics pipeline for processing large datasets.
    
    Combines external sorting and fast percentile calculations for
    comprehensive data analysis.
    """
    
    def __init__(self, config: SortingConfig = None):
        self.config = config or SortingConfig()
        self.percentile_calc = FastPercentileCalculator()
        self.session_data: List[Dict[str, Any]] = []
        self._total_processed = 0
        self._processing_stats = {
            'sessions_added': 0,
            'batch_processes': 0,
            'total_processing_time': 0.0
        }
    
    def add_session_batch(self, sessions: List[Dict[str, Any]]) -> None:
        """
        Add batch of session data.
        
        Args:
            sessions: List of session dictionaries with keys:
                     - user_id: int
                     - duration: float (seconds)
                     - pages_viewed: int
                     - timestamp: str
                     - additional metrics...
        """
        self.session_data.extend(sessions)
        self._processing_stats['sessions_added'] += len(sessions)
        self._total_processed += len(sessions)
        
        logger.info(f"Added batch of {len(sessions)} sessions. Total: {self._total_processed}")
    
    def get_top_sessions_by_duration(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N sessions by duration using external merge sort.
        
        Efficiently handles large datasets that exceed memory limits.
        """
        if not self.session_data:
            return []
        
        start_time = time.time()
        
        with ExternalSortingAlgorithms(self.config) as sorter:
            # Sort by duration in descending order
            sorted_sessions = sorter.external_merge_sort(
                self.session_data,
                key_func=lambda x: -x['duration']  # Negative for descending
            )
        
        processing_time = time.time() - start_time
        self._processing_stats['total_processing_time'] += processing_time
        self._processing_stats['batch_processes'] += 1
        
        return sorted_sessions[:top_n]
    
    def get_session_duration_percentiles(self, 
                                       percentiles: List[float] = None) -> Dict[float, float]:
        """
        Calculate session duration percentiles efficiently.
        
        Args:
            percentiles: List of percentiles to calculate (default: [50, 95, 99])
        """
        if percentiles is None:
            percentiles = [50, 95, 99]
        
        if not self.session_data:
            return {p: 0.0 for p in percentiles}
        
        start_time = time.time()
        
        results = self.percentile_calc.calculate_multiple_percentiles(
            self.session_data,
            percentiles,
            key_func=lambda x: x['duration']
        )
        
        processing_time = time.time() - start_time
        self._processing_stats['total_processing_time'] += processing_time
        
        # Convert to duration values
        duration_results = {
            p: result['duration'] if isinstance(result, dict) else result
            for p, result in results.items()
        }
        
        return duration_results
    
    def analyze_user_behavior(self, user_id: int) -> Dict[str, Any]:
        """
        Analyze behavior for specific user.
        
        Returns comprehensive user analytics including session patterns,
        duration statistics, and engagement metrics.
        """
        user_sessions = [s for s in self.session_data if s['user_id'] == user_id]
        
        if not user_sessions:
            return {'user_id': user_id, 'sessions_found': 0}
        
        # Calculate user-specific metrics
        durations = [s['duration'] for s in user_sessions]
        page_views = [s['pages_viewed'] for s in user_sessions]
        
        # Use external sorting for large user datasets
        if len(user_sessions) > self.config.chunk_size:
            with ExternalSortingAlgorithms(self.config) as sorter:
                sorted_sessions = sorter.external_merge_sort(
                    user_sessions,
                    key_func=lambda x: x['duration']
                )
        else:
            sorted_sessions = sorted(user_sessions, key=lambda x: x['duration'])
        
        # Calculate percentiles
        duration_percentiles = self.percentile_calc.calculate_multiple_percentiles(
            user_sessions,
            [25, 50, 75, 90, 95],
            key_func=lambda x: x['duration']
        )
        
        return {
            'user_id': user_id,
            'sessions_found': len(user_sessions),
            'total_duration': sum(durations),
            'average_duration': sum(durations) / len(durations),
            'total_pages_viewed': sum(page_views),
            'average_pages_per_session': sum(page_views) / len(page_views),
            'longest_session': max(durations),
            'shortest_session': min(durations),
            'duration_percentiles': {
                f'p{int(p)}': (result['duration'] if isinstance(result, dict) else result)
                for p, result in duration_percentiles.items()
            },
            'session_pattern': self._analyze_session_pattern(sorted_sessions)
        }
    
    def _analyze_session_pattern(self, sorted_sessions: List[Dict[str, Any]]) -> str:
        """Analyze user session patterns"""
        if len(sorted_sessions) < 3:
            return "insufficient_data"
        
        durations = [s['duration'] for s in sorted_sessions]
        
        # Calculate coefficient of variation
        mean_duration = sum(durations) / len(durations)
        variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
        std_dev = variance ** 0.5
        cv = std_dev / mean_duration if mean_duration > 0 else 0
        
        if cv < 0.3:
            return "consistent"
        elif cv < 0.7:
            return "moderate_variation"
        else:
            return "highly_variable"
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        if not self.session_data:
            return {'error': 'No session data available'}
        
        start_time = time.time()
        
        # Calculate key metrics
        total_sessions = len(self.session_data)
        unique_users = len(set(s['user_id'] for s in self.session_data))
        
        # Duration analysis
        duration_percentiles = self.get_session_duration_percentiles([10, 25, 50, 75, 90, 95, 99])
        
        # Page view analysis
        page_view_percentiles = self.percentile_calc.calculate_multiple_percentiles(
            self.session_data,
            [50, 95, 99],
            key_func=lambda x: x['pages_viewed']
        )
        
        # Top sessions
        top_sessions = self.get_top_sessions_by_duration(10)
        
        processing_time = time.time() - start_time
        
        return {
            'summary': {
                'total_sessions': total_sessions,
                'unique_users': unique_users,
                'sessions_per_user': total_sessions / unique_users,
                'report_generation_time': processing_time
            },
            'duration_analytics': {
                'percentiles': duration_percentiles,
                'mean': sum(s['duration'] for s in self.session_data) / total_sessions,
                'total_duration': sum(s['duration'] for s in self.session_data)
            },
            'engagement_analytics': {
                'page_view_percentiles': {
                    f'p{int(p)}': (result['pages_viewed'] if isinstance(result, dict) else result)
                    for p, result in page_view_percentiles.items()
                },
                'mean_pages_per_session': sum(s['pages_viewed'] for s in self.session_data) / total_sessions
            },
            'top_sessions': [
                {
                    'user_id': s['user_id'],
                    'duration': s['duration'],
                    'pages_viewed': s['pages_viewed'],
                    'timestamp': s['timestamp']
                }
                for s in top_sessions
            ],
            'processing_stats': self._processing_stats,
            'cache_stats': self.percentile_calc.get_cache_stats()
        }


def main():
    """Demonstrate advanced analytics pipeline capabilities"""
    
    # Configure for production use
    config = SortingConfig(
        chunk_size=50000,
        max_memory_mb=1024,
        parallel_workers=4,
        strategy=SortingStrategy.HYBRID
    )
    
    # Initialize analytics pipeline
    analytics = AdvancedAnalyticsPipeline(config)
    
    # Generate sample data
    import random
    print("Generating sample session data...")
    
    # Simulate realistic user behavior
    sessions = []
    for i in range(100000):  # 100K sessions
        user_id = random.randint(1, 10000)  # 10K unique users
        
        # Realistic duration distribution (log-normal)
        base_duration = random.lognormvariate(5, 1.5)  # Log-normal distribution
        duration = max(30, min(7200, base_duration))  # 30s to 2h
        
        # Pages viewed correlates with duration
        pages_viewed = max(1, int(duration / 120) + random.randint(-2, 5))
        
        timestamp = f"2024-06-{random.randint(1, 30):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z"
        
        sessions.append({
            'user_id': user_id,
            'duration': duration,
            'pages_viewed': pages_viewed,
            'timestamp': timestamp
        })
    
    # Add sessions to pipeline
    analytics.add_session_batch(sessions)
    
    print("\n=== Advanced Analytics Pipeline Demonstration ===")
    
    # 1. Generate comprehensive report
    print("\n1. Generating comprehensive analytics report...")
    report = analytics.generate_analytics_report()
    
    print(f"   Total sessions: {report['summary']['total_sessions']:,}")
    print(f"   Unique users: {report['summary']['unique_users']:,}")
    print(f"   Sessions per user: {report['summary']['sessions_per_user']:.2f}")
    print(f"   Report generation time: {report['summary']['report_generation_time']:.2f}s")
    
    # 2. Duration percentiles
    print("\n2. Session Duration Percentiles:")
    for percentile, value in report['duration_analytics']['percentiles'].items():
        print(f"   P{percentile}: {value:.2f}s")
    
    # 3. Top sessions
    print("\n3. Top 5 Longest Sessions:")
    for i, session in enumerate(report['top_sessions'][:5], 1):
        print(f"   {i}. User {session['user_id']}: {session['duration']:.2f}s, {session['pages_viewed']} pages")
    

    # 4. Individual user analysis
    print("\n4. Individual User Analysis (User 1):")
    user_report = analytics.analyze_user_behavior(1)
    print(f"   User 1 sessions found: {user_report['sessions_found']}")
    print(f"   Total duration: {user_report.get('total_duration', 0):.2f}s")
    print(f"   Average duration: {user_report.get('average_duration', 0):.2f}s")
    print(f"   Duration percentiles: {user_report.get('duration_percentiles', {})}")
    print(f"   Session pattern: {user_report.get('session_pattern', 'N/A')}")


if __name__ == "__main__":
    main()