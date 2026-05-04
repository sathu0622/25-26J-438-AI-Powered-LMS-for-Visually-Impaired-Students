import requests
import json
import time

# Give the server a second to be ready
time.sleep(2)

# Test the audio generation endpoint
url = "http://localhost:8000/api/topic/10/ch1/tp1/generate-audio"
payload = {
    "emotion_intensity": 1.0,
    "include_effects": True,
    "effects_only": False
}

try:
    print("Sending request to:", url)
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print("Raw Response:", response.text[:500])
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

