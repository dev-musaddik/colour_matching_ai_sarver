# Fix 502 Bad Gateway Error on Render

## Problem Diagnosis

You're getting **two errors**:

### 1. 502 Bad Gateway
```
GET https://colour-matching-ai-sarver-wb7d.onrender.com/colors net::ERR_FAILED 502
```

**This means**: Your Render backend is either:
- ❌ Not running
- ❌ Failed to deploy
- ❌ Crashed during startup
- ❌ Out of memory

### 2. CORS Error
```
Access to fetch at 'https://colour-matching-ai-sarver-wb7d.onrender.com/colors' 
from origin 'http://localhost:5173' has been blocked by CORS policy
```

**This means**: Backend needs to allow your frontend URLs.

---

## ✅ CORS Fix Applied

I've updated `main.py` to allow:
- ✅ `http://localhost:5173` (local development)
- ✅ `http://localhost:3000` (alternative local)
- ✅ `https://hair-color-analyzer-frontend.vercel.app` (production)

---

## 🔧 Fix the 502 Error

### Step 1: Check Render Deployment Status

1. Go to https://render.com/dashboard
2. Find your service: `hair-color-analyzer-backend`
3. Check the status:
   - 🟢 **Live** = Good
   - 🔴 **Deploy failed** = Need to fix
   - 🟡 **Building** = Wait for it to finish

### Step 2: Check Render Logs

Click your service → **"Logs"** tab

**Look for these errors**:

#### Error 1: Out of Memory
```
==> Out of memory (used over 512Mi)
```

**Solution**: This shouldn't happen with our optimizations, but if it does:
- Verify ResNet50 is lazy loading (not loading at startup)
- Check if models are pre-downloaded in Docker build

#### Error 2: Port Binding Failed
```
==> No open ports detected
```

**Solution**: 
- Verify `/health` endpoint exists (it does ✓)
- Check PORT environment variable is used (it is ✓)

#### Error 3: Build Failed
```
ERROR: Failed to build
```

**Solution**: Check build logs for specific error

### Step 3: Redeploy with Fixes

```bash
cd backend
git add .
git commit -m "Fix CORS for Vercel frontend"
git push origin main
```

Render will auto-deploy.

---

## 🧪 Test After Deployment

### 1. Test Health Endpoint
```bash
curl https://colour-matching-ai-sarver-wb7d.onrender.com/health
```

**Expected**:
```json
{"status": "healthy", "service": "Hair Color Analyzer API"}
```

**If you get 502**: Backend is not running. Check Render logs.

### 2. Test CORS from Browser Console

Open your frontend in browser, then in console:
```javascript
fetch('https://colour-matching-ai-sarver-wb7d.onrender.com/health')
  .then(r => r.json())
  .then(console.log)
```

**Expected**: `{status: "healthy", ...}`

**If CORS error**: Backend needs to be redeployed with CORS fix.

---

## 🔍 Common Render Issues

### Issue 1: Free Tier Sleeping
**Symptom**: First request takes 30-60 seconds  
**Cause**: Render spins down free tier services after 15 min of inactivity  
**Solution**: Wait for service to wake up, or upgrade to paid tier

### Issue 2: Build Timeout
**Symptom**: Build fails after 15 minutes  
**Cause**: Model downloads taking too long  
**Solution**: Models should be cached after first build

### Issue 3: Memory Limit
**Symptom**: Service crashes with OOM error  
**Cause**: Exceeding 512MB RAM  
**Solution**: Our lazy loading should prevent this

---

## 📊 Expected Render Logs (Healthy)

```
Building Docker image...
[+] Building 300.5s
 => [1/6] Installing build dependencies
 => [2/6] Installing Python packages
 => [3/6] Copying application code
 => [4/6] Pre-downloading YOLO model ✓
 => [5/6] Pre-downloading ResNet50 model ✓
 => [6/6] Creating user
Build complete ✓

Deploying...
Starting uvicorn on port 10000
No trained models found. Skipping model loading to save memory.
Application startup complete.
Uvicorn running on http://0.0.0.0:10000

==> Port detected on 10000 ✓
==> Health check passed ✓
==> Service is live ✓
```

---

## 🚨 If Still Getting 502

### Option 1: Check Service Status
```bash
# Test if service is running
curl -I https://colour-matching-ai-sarver-wb7d.onrender.com/health
```

If you get nothing or timeout → Service is down

### Option 2: Manual Redeploy
1. Go to Render dashboard
2. Click your service
3. Click **"Manual Deploy"** → **"Deploy latest commit"**
4. Wait for build to complete

### Option 3: Check Environment Variables
In Render dashboard:
- Go to your service
- Click **"Environment"** tab
- Verify no conflicting variables

### Option 4: View Full Logs
1. Render dashboard → Your service
2. **"Logs"** tab
3. Look for the exact error message
4. Share the error if you need help

---

## ✅ Checklist

- [x] CORS fixed in main.py
- [ ] Code committed and pushed to GitHub
- [ ] Render auto-deploys successfully
- [ ] Health endpoint responds (no 502)
- [ ] CORS allows Vercel frontend
- [ ] Frontend can fetch data

---

## 🆘 Quick Debug Commands

```bash
# 1. Test if backend is alive
curl https://colour-matching-ai-sarver-wb7d.onrender.com/health

# 2. Test with verbose output
curl -v https://colour-matching-ai-sarver-wb7d.onrender.com/health

# 3. Test colors endpoint
curl https://colour-matching-ai-sarver-wb7d.onrender.com/colors

# 4. Check response headers
curl -I https://colour-matching-ai-sarver-wb7d.onrender.com/health
```

---

## 📝 Next Steps

1. **Commit CORS fix**:
   ```bash
   git add backend/main.py
   git commit -m "Fix CORS for Vercel frontend"
   git push origin main
   ```

2. **Wait for Render to deploy** (~5-10 min)

3. **Check Render logs** for deployment status

4. **Test health endpoint** to verify service is running

5. **Refresh frontend** to test connection

---

## 💡 Pro Tip

If Render keeps failing, you can:
1. Check if you're on the free tier limit
2. Try deploying to Railway instead (better free tier)
3. Upgrade to Render Starter ($7/month) for more reliability
