# Rate Limit Fixes Applied

## Summary of Changes

Based on the research and recommendations, I've applied the following fixes to resolve 429 Resource Exhausted errors:

### 1. **Changed Model from Experimental to Stable**
- **Before:** `gemini-2.0-flash-exp` (experimental model with stricter limits)
- **After:** `gemini-2.0-flash` (stable model with better rate limits)
- **Location:** `config.py` and `services/shared/llm_utils.py`
- **Reason:** Experimental models have more restricted rate limits

### 2. **Improved Exponential Backoff**
- **Before:** Base delay 2 seconds, max 60 seconds
- **After:** Base delay 5 seconds, max 120 seconds
- **Location:** `services/shared/llm_utils.py`
- **Reason:** Longer delays help avoid hitting rate limits repeatedly

### 3. **Increased Rate Limiting Between Requests**
- **Before:** 0.5 seconds between requests
- **After:** 5 seconds between requests
- **Location:** `services/shared/llm_utils.py`
- **Reason:** Free tier allows 15 RPM = 1 request every 4 seconds minimum

### 4. **Reduced Batch Size**
- **Before:** 20 trials per batch
- **After:** 10 trials per batch
- **Location:** `config.py`
- **Reason:** Smaller batches reduce token usage and rate limit hits

### 5. **Increased Delays Between Batches**
- **Before:** 2 seconds between batches
- **After:** 5 seconds between batches
- **Location:** `testing/llm_accuracy_evaluation.py`
- **Reason:** Prevents burst requests that trigger rate limits

### 6. **Added Initial Delay**
- **New:** 2 second delay before first batch
- **Location:** `testing/llm_accuracy_evaluation.py`
- **Reason:** Avoids immediate rate limit on startup

## Current Status

The script is still hitting 429 errors, which indicates:

1. **New API Key Also Has Quota Limits**
   - The new account is likely on free tier
   - Free tier has very limited quotas (200 RPD for gemini-2.0-flash)

2. **Possible Solutions:**

   **Option A: Wait for Quota Reset**
   - Quota resets at midnight Pacific time
   - Free tier: 200 RPD (Requests Per Day)
   - Run script after quota resets

   **Option B: Enable Billing (Recommended)**
   - Go to Google Cloud Console
   - Enable billing for your project
   - This upgrades you to Tier 1 with higher limits:
     - Tier 1: 15 RPM, 1,000,000 TPM, 200 RPD
     - Much better than free tier

   **Option C: Request Quota Increase**
   - Go to Google Cloud Console > APIs & Services > Quotas
   - Find "Generative Language API" quotas
   - Click "Edit Quota" to request increase
   - May require billing account

   **Option D: Upgrade to Higher Tier**
   - Tier 2: Requires $250+ spending, 30+ days
   - Tier 3: Requires $1,000+ spending, 30+ days
   - Much higher limits (hundreds/thousands RPM)

## Rate Limits by Tier

### Free Tier
- **Gemini 2.0 Flash:** 15 RPM, 1,000,000 TPM, 200 RPD
- **Very limited** - best for testing only

### Tier 1 (Billing Enabled)
- **Gemini 2.0 Flash:** 15 RPM, 1,000,000 TPM, 200 RPD
- **Same as free tier** but allows quota increases

### Tier 2 ($250+ spending)
- **Much higher limits** - hundreds of RPM
- **Better for production use**

### Tier 3 ($1,000+ spending)
- **Highest limits** - thousands of RPM
- **Best for high-volume production**

## Recommendations

1. **Enable Billing** (Easiest)
   - Go to Google Cloud Console
   - Enable billing for your project
   - This unlocks Tier 1 and allows quota increases

2. **Monitor Usage**
   - Check Google Cloud Console > APIs & Services > Quotas
   - Monitor your usage patterns
   - Adjust batch sizes and delays based on usage

3. **Use Batch Processing** (Already Implemented)
   - Script now uses batch processing
   - Reduces API calls from 90 to ~9 batches
   - More efficient use of quota

4. **Spread Requests Over Time**
   - Don't send all requests at once
   - Process evaluations over multiple days if needed
   - Script saves progress and can resume

## Next Steps

1. **Check Your Quota Status:**
   ```bash
   # Visit: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
   ```

2. **Enable Billing (Recommended):**
   ```bash
   # Visit: https://console.cloud.google.com/billing
   # Link a billing account to your project
   ```

3. **Request Quota Increase:**
   ```bash
   # Visit: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
   # Click "Edit Quota" for the limits you need
   ```

4. **Run Script After Quota Reset:**
   ```bash
   cd ..
   .\venv\Scripts\python.exe testing\llm_accuracy_evaluation.py
   ```

## Files Modified

1. `config.py` - Changed model and batch size
2. `services/shared/llm_utils.py` - Improved rate limiting and backoff
3. `testing/llm_accuracy_evaluation.py` - Added delays and improved error handling

## Testing

The script is now configured with:
- ✅ Stable model (gemini-2.0-flash)
- ✅ Improved exponential backoff
- ✅ Better rate limiting (5 seconds between requests)
- ✅ Smaller batch sizes (10 instead of 20)
- ✅ Longer delays between batches (5 seconds)
- ✅ Initial delay before first batch

All fixes have been applied. The script will work better once quota resets or billing is enabled.

