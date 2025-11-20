"""
Check Gemini API Quota Status
Uses the Google Cloud API to check current quota usage
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GEMINI_API_KEY
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import time

def check_quota():
    """Check quota by making a test API call"""
    print("="*80)
    print("GEMINI API QUOTA CHECK")
    print("="*80)
    print(f"\nAPI Key: {GEMINI_API_KEY[:20]}...{GEMINI_API_KEY[-10:]}")
    print(f"Model: gemini-2.0-flash")
    print("\nAttempting test API call to check quota status...")
    print("-"*80)
    
    try:
        # Configure API
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Make a simple test call
        test_prompt = "Say 'Hello' in one word."
        print(f"\nTest Prompt: {test_prompt}")
        print("Making API call...")
        
        start_time = time.time()
        response = model.generate_content(test_prompt)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ API Call Successful!")
        print(f"   Response: {response.text}")
        print(f"   Time taken: {elapsed_time:.2f} seconds")
        print(f"\n✅ Quota Status: ACTIVE - You have available quota")
        print(f"   The API key is working and has quota available.")
        
        return True
        
    except google_exceptions.ResourceExhausted as e:
        error_str = str(e)
        print(f"\n❌ QUOTA EXHAUSTED (429 Resource Exhausted)")
        print(f"   Error: {error_str}")
        
        # Try to extract quota information from error
        if "quota" in error_str.lower():
            print(f"\n📊 Quota Information:")
            if "RPM" in error_str or "requests per minute" in error_str.lower():
                print(f"   - RPM (Requests Per Minute) limit reached")
            if "RPD" in error_str or "requests per day" in error_str.lower():
                print(f"   - RPD (Requests Per Day) limit reached")
            if "TPM" in error_str or "tokens per minute" in error_str.lower():
                print(f"   - TPM (Tokens Per Minute) limit reached")
        
        # Check for retry delay
        import re
        delay_match = re.search(r'retry.*?in.*?(\d+(?:\.\d+)?).*?seconds?', error_str.lower())
        if delay_match:
            retry_delay = float(delay_match.group(1))
            print(f"\n⏳ Retry Delay: {retry_delay:.0f} seconds ({retry_delay/60:.1f} minutes)")
        
        print(f"\n💡 Solutions:")
        print(f"   1. Wait for quota reset (usually at midnight Pacific time)")
        print(f"   2. Enable billing to unlock Tier 1 and request quota increases")
        print(f"   3. Check quota dashboard: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
        
        return False
        
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ ERROR: {error_str}")
        
        if "429" in error_str or "resource exhausted" in error_str.lower():
            print(f"   This is a quota/rate limit error")
        elif "401" in error_str or "unauthorized" in error_str.lower():
            print(f"   This is an authentication error - check your API key")
        elif "403" in error_str or "forbidden" in error_str.lower():
            print(f"   This is a permission error - check API access")
        else:
            print(f"   Unknown error type")
        
        return False

def check_rate_limits():
    """Check rate limits by making multiple test calls"""
    print("\n" + "="*80)
    print("RATE LIMIT TEST")
    print("="*80)
    print("\nTesting rate limits with multiple calls...")
    print("-"*80)
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        success_count = 0
        error_count = 0
        
        for i in range(3):
            try:
                print(f"\nTest call {i+1}/3...")
                response = model.generate_content("Say 'Test' in one word.")
                print(f"   ✅ Success: {response.text}")
                success_count += 1
                time.sleep(5)  # Wait 5 seconds between calls
            except google_exceptions.ResourceExhausted as e:
                print(f"   ❌ Rate limit hit: {str(e)[:100]}")
                error_count += 1
                break
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}")
                error_count += 1
                break
        
        print(f"\n📊 Results:")
        print(f"   Successful calls: {success_count}/3")
        print(f"   Errors: {error_count}/3")
        
        if success_count > 0:
            print(f"\n✅ Rate limits: OK - You can make API calls")
        else:
            print(f"\n❌ Rate limits: EXCEEDED - Cannot make API calls")
        
    except Exception as e:
        print(f"\n❌ Error during rate limit test: {e}")

if __name__ == "__main__":
    # Check quota
    quota_ok = check_quota()
    
    # If quota is OK, test rate limits
    if quota_ok:
        check_rate_limits()
    
    print("\n" + "="*80)
    print("QUOTA CHECK COMPLETE")
    print("="*80)
    print("\n💡 To check detailed quota information:")
    print("   1. Visit: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
    print("   2. Sign in with your Google account")
    print("   3. Select your project")
    print("   4. View quota limits and usage")
    print("\n💡 To enable billing and unlock Tier 1:")
    print("   1. Visit: https://console.cloud.google.com/billing")
    print("   2. Link a billing account to your project")
    print("   3. This unlocks Tier 1 and allows quota increases")
    print("="*80)




