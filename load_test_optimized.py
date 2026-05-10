import argparse
import json
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev

import requests

# Configuration
DEFAULT_API_URL = "http://localhost:5000/predict"
DEFAULT_HEALTH_URL = "http://localhost:5000/health"
DEFAULT_CACHE_STATS_URL = "http://localhost:5000/cache-stats"
DEFAULT_REQUESTS = 50
DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 30

PAYLOADS = [
    "<script>alert('XSS')</script>",
    "normal text without any attack",
    "javascript:alert(1)",
    "this is a benign payload",
    "<img src=x onerror=alert('xss')>",
    "SELECT * FROM users",
    "Union Select 1,2,3",
    "safe string here",
    "<iframe src='javascript:alert(1)'></iframe>",
    "hello world",
]

PRINT_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized load test for XSS Detection API")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Predict endpoint URL")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent workers")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--unique-payloads", action="store_true", help="Use unique payloads to test cold-path inference")
    parser.add_argument("--clear-cache", action="store_true", help="Clear API cache before running the test")
    parser.add_argument("--no-cache-report", action="store_true", help="Disable cache summary if API does not return cache_hit")
    return parser.parse_args()


def build_payloads(total_requests, unique=False):
    if unique:
        return [f"unique payload {i} {uuid.uuid4()}" for i in range(1, total_requests + 1)]
    return [PAYLOADS[i % len(PAYLOADS)] for i in range(total_requests)]


def fetch_cache_stats(session, cache_stats_url):
    try:
        response = session.get(cache_stats_url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def request_cache_clear(session, cache_clear_url):
    try:
        response = session.post(cache_clear_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def is_api_alive(session, health_url):
    try:
        response = session.get(health_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_request(session, api_url, payload, timeout, request_id):
    result = {
        "request_id": request_id,
        "payload": payload,
        "status_code": None,
        "latency_ms": None,
        "cache_hit": None,
        "timings": {},
        "error": None,
    }

    start = time.perf_counter()
    try:
        response = session.post(api_url, json={"text": payload}, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        result["status_code"] = response.status_code
        result["latency_ms"] = latency_ms

        try:
            body = response.json()
        except ValueError:
            body = {}

        if isinstance(body, dict):
            result["cache_hit"] = body.get("cache_hit")
            result["timings"] = body.get("timings", {}) if body.get("timings") else {}
        else:
            result["error"] = "Invalid JSON response"

        with PRINT_LOCK:
            status_emoji = "✅" if response.status_code == 200 else "⚠️"
            cache_indicator = "💾" if result["cache_hit"] else "🆕"
            print(
                f"[{request_id:03d}] {status_emoji} Status: {response.status_code} | "
                f"{cache_indicator if result['cache_hit'] is not None else '---'} | "
                f"Latency: {latency_ms:7.2f}ms"
            )

    except requests.exceptions.Timeout:
        result["status_code"] = 504
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        result["error"] = "timeout"
        with PRINT_LOCK:
            print(f"[{request_id:03d}] ❌ Timeout after {timeout}s")
    except Exception as exc:
        result["status_code"] = 0
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        result["error"] = str(exc)
        with PRINT_LOCK:
            print(f"[{request_id:03d}] ❌ Request failed: {exc}")

    return result


def summarize(results, total_wall_time, no_cache_report):
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    successes = [r for r in results if r["status_code"] == 200]
    failures = [r for r in results if r["status_code"] != 200]
    cache_hits = [r for r in successes if r["cache_hit"] is True]
    cache_misses = [r for r in successes if r["cache_hit"] is False]
    component_timings = defaultdict(list)

    for r in successes:
        for component, timing in r["timings"].items():
            if isinstance(timing, (int, float)):
                component_timings[component].append(timing)

    print("\n" + "=" * 100)
    print("📊 OVERALL LOAD TEST SUMMARY")
    print("=" * 100)
    print(f"Total requests:      {len(results)}")
    print(f"Successful:          {len(successes)}")
    print(f"Failed:              {len(failures)}")
    print(f"Wall time:           {total_wall_time:.2f}s")
    print(f"Average throughput:  {len(results) / total_wall_time:.2f} req/s")

    if latencies:
        print(f"\nLatency (ms):")
        print(f"  Min:    {min(latencies):.2f}")
        print(f"  Max:    {max(latencies):.2f}")
        print(f"  Median: {median(latencies):.2f}")
        print(f"  Avg:    {mean(latencies):.2f}")
        if len(latencies) > 1:
            print(f"  StdDev: {stdev(latencies):.2f}")

    if not no_cache_report and successes:
        total_hits = len(cache_hits)
        total_misses = len(cache_misses)
        if total_hits + total_misses > 0:
            hit_rate = total_hits / (total_hits + total_misses) * 100
            print(f"\nCache hit rate:      {total_hits}/{total_hits + total_misses} = {hit_rate:.1f}%")
            print(f"Cache misses:        {total_misses}/{total_hits + total_misses}")

            if cache_hits and cache_misses:
                avg_hit = mean(r["timings"].get("bert", 0) for r in cache_hits)
                avg_miss = mean(r["timings"].get("bert", 0) for r in cache_misses)
                print(f"\nBERT latency: ")
                print(f"  Cache hit avg:     {avg_hit:.2f}ms")
                print(f"  Cache miss avg:    {avg_miss:.2f}ms")
                print(f"  Speedup:           {avg_miss / avg_hit:.2f}x")

    if component_timings:
        print("\nComponent latency breakdown:")
        print(f"{'Component':<20} {'Avg (ms)':<12} {'Min':<12} {'Max':<12} {'Samples':<8}")
        print("-" * 70)
        for component in ["preprocessing", "tfidf", "bert", "prediction", "total"]:
            if component_timings.get(component):
                values = component_timings[component]
                print(
                    f"{component:<20} {mean(values):<12.2f} {min(values):<12.2f} {max(values):<12.2f} {len(values):<8}"
                )

    if failures:
        failure_reasons = Counter(r["status_code"] if r["status_code"] else r["error"] for r in failures)
        print("\nFailure breakdown:")
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count}")

    print("=" * 100)


def main():
    args = parse_args()
    payloads = build_payloads(args.requests, unique=args.unique_payloads)
    cache_clear_url = args.url.replace("/predict", "/cache-clear")
    health_url = args.url.replace("/predict", "/health")
    cache_stats_url = args.url.replace("/predict", "/cache-stats")

    print("\n🔍 Optimized Load Test untuk XSS Detection API\n")
    print(f"Target URL:       {args.url}")
    print(f"Total requests:   {args.requests}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Timeout:          {args.timeout}s")
    print(f"Unique payloads:  {args.unique_payloads}")
    print(f"Clear cache:      {args.clear_cache}\n")

    with requests.Session() as session:
        if not is_api_alive(session, health_url):
            print("❌ API tidak dapat dijangkau. Pastikan server berjalan di alamat yang benar.")
            return

        if args.clear_cache:
            if request_cache_clear(session, cache_clear_url):
                print("✅ Cache berhasil dibersihkan sebelum pengujian.\n")
            else:
                print("⚠️  Gagal membersihkan cache. Lanjut pengujian tanpa reset cache.\n")

        cache_stats_before = fetch_cache_stats(session, cache_stats_url)
        if cache_stats_before:
            print("✅ Cache stats endpoint ditemukan. Mengumpulkan data awal...\n")

        start_wall = time.perf_counter()
        results = []

        if args.concurrency <= 1:
            for idx, payload in enumerate(payloads, start=1):
                results.append(send_request(session, args.url, payload, args.timeout, idx))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_to_idx = {
                    executor.submit(send_request, session, args.url, payload, args.timeout, idx): idx
                    for idx, payload in enumerate(payloads, start=1)
                }
                for future in as_completed(future_to_idx):
                    results.append(future.result())

        end_wall = time.perf_counter()
        total_wall_time = end_wall - start_wall

        cache_stats_after = fetch_cache_stats(session, cache_stats_url)
        if cache_stats_before and cache_stats_after:
            print("\n✅ Cache stats summary:")
            print(json.dumps({
                "before": cache_stats_before,
                "after": cache_stats_after,
            }, indent=2, ensure_ascii=False))

        summarize(results, total_wall_time, args.no_cache_report)


if __name__ == "__main__":
    main()
