# Sorting Algorithms for Data Processing

Production-ready sorting and analytics algorithms for backend data processing, analytics, and big data scenarios. This module is designed to handle massive datasets, real-time analytics, and memory-efficient batch processing in Python.

---

## ✨ Features

- External merge sort for datasets larger than available memory
- Parallel chunked sorting for speed and scalability
- O(n) percentile/median calculation with QuickSelect
- Advanced analytics pipeline for user/session data
- Configurable memory usage, chunk size, and parallelism
- Production-grade error handling and monitoring

---

## 🧩 Algorithms Included

- **External Merge Sort:** Memory-efficient sorting for large datasets
- **Parallel External Sort:** Multi-threaded chunked sorting
- **QuickSelect:** Fast percentile/median calculation (O(n) average)
- **Advanced Analytics Pipeline:** Combines sorting and percentiles for backend analytics

---

## 🚀 Use Cases

- Handle datasets too large for memory (log files, analytics, ETL)
- Real-time analytics and percentile calculations (SLA, latency, user behavior)
- Efficiently sort and analyze massive log files or user sessions
- Track and report on backend performance metrics

---

## 📦 Usage

See [`sorting_algorithms.py`](./sorting_algorithms.py) for full implementation and advanced usage examples.

### Example: Sorting and Analytics

```python
from sorting_algorithms import ExternalSortingAlgorithms, FastPercentileCalculator, AdvancedAnalyticsPipeline

# External merge sort
with ExternalSortingAlgorithms() as sorter:
    sorted_data = sorter.external_merge_sort(data, key_func=lambda x: x['duration'])

# Fast percentile calculation
percentile_calc = FastPercentileCalculator()
p95 = percentile_calc.calculate_percentile([s['duration'] for s in data], 95)

# Analytics pipeline
analytics = AdvancedAnalyticsPipeline()
analytics.add_session_batch(data)
top_sessions = analytics.get_top_sessions_by_duration(5)
percentiles = analytics.get_session_duration_percentiles([50, 95, 99])
report = analytics.generate_analytics_report()
```

---

For more details, see the code and docstrings in [`sorting_algorithms.py`](./sorting_algorithms.py).
