#!/usr/bin/env python3
"""Test audio generation"""
import requests
import json
import time

# Wait a bit for server to be ready
time.sleep(2)

url = 'http://localhost:8000/api/topic/10/1/10_ch1_1/generate-audio'
data = {
    'emotion_intensity': 1.0,
    'include_effects': False,
    'effects_only': False
}

try:
    print('Sending request to audio generation endpoint...')
    print(f'URL: {url}')
    print(f'Data: {data}')
    
    response = requests.post(url, json=data, timeout=60)
    print(f'Status Code: {response.status_code}')
    print(f'Response Headers: {dict(response.headers)}')
    print(f'Response Body: {response.text[:1000]}')
    
    if response.status_code == 200:
        print('\nAudio generation successful!')
        result = response.json()
        print(f'Result: {json.dumps(result, indent=2)}')
    else:
        print(f'\nError: {response.status_code}')
        
except Exception as e:
    import traceback
    print(f'Exception: {e}')
    traceback.print_exc()
