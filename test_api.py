#!/usr/bin/env python3
"""
Test the Changi Virtual Assistant API
"""

import requests
import json

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        print("Health Check:")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction(query):
    """Test prediction endpoint"""
    try:
        data = {"query": query}
        response = requests.post('http://localhost:5000/api/predict', 
                               headers={'Content-Type': 'application/json'},
                               json=data)
        print(f"Query: '{query}'")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Intent: {result.get('top_intent', 'Unknown')}")
            print(f"Confidence: {result.get('predictions', [{}])[0].get('confidence', 0):.1f}%")
            print(f"Response: {result.get('response', 'No response')[:100]}...")
        else:
            print(f"Error: {response.text}")
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"Prediction failed for '{query}': {e}")
        return False

def main():
    print("🧪 Testing Changi Virtual Assistant API")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("❌ Health check failed!")
        return
    
    # Test various queries
    test_queries = [
        "Where is gate A15?",
        "Flight to Bangkok",
        "How to get to city center?",
        "Singapore Airlines information",
        "What's the cheapest flight?",
        "Flight departure time"
    ]
    
    print("🎯 Testing Predictions:")
    print("=" * 50)
    
    success_count = 0
    for query in test_queries:
        if test_prediction(query):
            success_count += 1
    
    print(f"✅ Results: {success_count}/{len(test_queries)} tests passed")
    
    if success_count == len(test_queries):
        print("🎉 All tests passed! Backend is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the server logs.")

if __name__ == "__main__":
    main()