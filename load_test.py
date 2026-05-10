import requests
import time
import json
from statistics import mean, stdev, median
from collections import defaultdict

# Configuration
API_URL = "http://localhost:5000/predict"
NUM_REQUESTS = 50

# Sample payloads to test
test_payloads = [
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

def load_test():
    """Send 50 requests and measure latency with component breakdown"""
    
    print(f"🚀 Starting load test with {NUM_REQUESTS} requests...")
    print(f"📍 Target: {API_URL}")
    print("-" * 80)
    
    latencies = []
    successful = 0
    failed = 0
    
    # Collect component timings
    component_timings = defaultdict(list)
    
    # Send requests
    for i in range(NUM_REQUESTS):
        payload = test_payloads[i % len(test_payloads)]
        
        try:
            # Measure request time
            start_time = time.time()
            response = requests.post(
                API_URL,
                json={"text": payload},
                timeout=30
            )
            end_time = time.time()
            
            # Calculate latency in milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if response.status_code == 200:
                resp_data = response.json()
                # Collect component timings from response
                if "timings" in resp_data:
                    for component, timing in resp_data["timings"].items():
                        component_timings[component].append(timing)
                successful += 1
            else:
                failed += 1
            
            status = "✅" if response.status_code == 200 else "⚠️"
            print(f"[{i+1:2d}/50] {status} Status: {response.status_code} | Total Latency: {latency_ms:.2f}ms")
                
        except requests.exceptions.Timeout:
            print(f"[{i+1:2d}/50] ❌ Request timeout!")
            failed += 1
        except Exception as e:
            print(f"[{i+1:2d}/50] ❌ Error: {str(e)}")
            failed += 1
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("📊 COMPONENT LATENCY BREAKDOWN")
    print("=" * 80)
    
    # Display component timing table
    if component_timings:
        print(f"\n{'Komponen':<30} {'Rata-rata':<15} {'Min':<12} {'Max':<12} {'Keterangan':<25}")
        print("-" * 80)
        
        descriptions = {
            "preprocessing": "URL decode, HTML entity decode",
            "tfidf": "Transform satu sampel TF-IDF",
            "bert": "Tokenisasi + forward pass BERT",
            "prediction": "LR predict + probabilitas",
            "total": "Total end-to-end latency"
        }
        
        for component in ["preprocessing", "tfidf", "bert", "prediction", "total"]:
            if component in component_timings and component_timings[component]:
                timings = component_timings[component]
                avg = mean(timings)
                min_t = min(timings)
                max_t = max(timings)
                desc = descriptions.get(component, "")
                
                print(f"{component:<30} {avg:>6.2f} ms        {min_t:>6.2f} ms   {max_t:>6.2f} ms   {desc:<25}")
        
        print("-" * 80)
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("📊 OVERALL LATENCY STATISTICS")
    print("=" * 80)
    
    if latencies:
        print(f"✅ Successful requests: {successful}/{NUM_REQUESTS}")
        print(f"❌ Failed requests:     {failed}/{NUM_REQUESTS}")
        print(f"\n⏱️  Min Latency:      {min(latencies):.2f}ms")
        print(f"⏱️  Max Latency:      {max(latencies):.2f}ms")
        print(f"⏱️  Median Latency:   {median(latencies):.2f}ms")
        print(f"⏱️  Average Latency:  {mean(latencies):.2f}ms")
        
        if len(latencies) > 1:
            print(f"⏱️  Std Dev:          {stdev(latencies):.2f}ms")
        
        total_time = sum(latencies)
        print(f"\n⏱️  Total Time:       {total_time:.2f}ms ({total_time/1000:.2f}s)")
        print(f"📈 Throughput:       {NUM_REQUESTS / (total_time/1000):.2f} req/s")
        
        # Performance assessment
        avg_latency = mean(latencies)
        print(f"\n{'='*80}")
        if avg_latency < 200:
            print(f"🟢 EXCELLENT: Average latency {avg_latency:.2f}ms - Very fast for real-time detection")
        elif avg_latency < 350:
            print(f"🟡 GOOD: Average latency {avg_latency:.2f}ms - Suitable for real-time detection")
        elif avg_latency < 500:
            print(f"🟠 ACCEPTABLE: Average latency {avg_latency:.2f}ms - Close to target threshold")
        else:
            print(f"🔴 SLOW: Average latency {avg_latency:.2f}ms - Needs optimization")
    else:
        print("❌ No successful requests!")
    
    print("=" * 80)

if __name__ == "__main__":
    print("\n🔍 Load Test untuk XSS Detection API\n")
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running!\n")
            load_test()
        else:
            print(f"❌ API returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("⚠️  Make sure the API is running on http://localhost:5000")
