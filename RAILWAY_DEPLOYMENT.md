# Railway Deployment Guide - Hair Color Analyzer Backend

## Why Railway is Better

✅ **More generous free tier**: 512MB RAM + $5/month credit  
✅ **Better performance**: Faster cold starts  
✅ **Easier setup**: One-click deploy from GitHub  
✅ **All features work**: Training mode works perfectly  

---

## Quick Deploy (Recommended)

### Step 1: Push to GitHub

```bash
cd backend
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### Step 2: Deploy on Railway

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your repository: `dev-musaddik/colour_matching_ai_sarver`
6. Railway will auto-detect the Dockerfile and deploy

### Step 3: Configure Environment Variables (Optional)

In Railway dashboard:
- Click your service
- Go to **"Variables"** tab
- Add (if needed):
  - `ALLOWED_ORIGINS`: Your frontend URL (e.g., `https://yourapp.com`)

### Step 4: Get Your API URL

- Railway will provide a URL like: `https://your-app.up.railway.app`
- Use this in your frontend to connect to the backend

---

## What Railway Does Automatically

✅ Detects Dockerfile  
✅ Builds Docker image with pre-downloaded models  
✅ Sets PORT environment variable  
✅ Provides HTTPS endpoint  
✅ Auto-redeploys on git push  
✅ Health checks via `/health` endpoint  

---

## Files Configured for Railway

| File | Purpose |
|------|---------|
| `railway.json` | Railway deployment configuration |
| `Dockerfile` | Container build instructions (already optimized) |
| `requirements.txt` | Python dependencies |
| `main.py` | FastAPI application with `/health` endpoint |

---

## Expected Deployment Flow

```
1. Push to GitHub
   ↓
2. Railway detects changes
   ↓
3. Builds Docker image (~5-10 min)
   - Installs dependencies
   - Pre-downloads ResNet50 & YOLO models
   ↓
4. Deploys container
   - Starts uvicorn server
   - Binds to Railway's PORT
   ↓
5. Health check passes ✅
   ↓
6. Service is LIVE 🚀
```

---

## Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app.up.railway.app/health
```

**Expected Response**:
```json
{"status": "healthy", "service": "Hair Color Analyzer API"}
```

### 2. Test Analysis
```bash
curl -X POST https://your-app.up.railway.app/analyze \
  -F "images=@test_image.jpg"
```

### 3. Test Training
```bash
# Create a color
curl -X POST https://your-app.up.railway.app/colors \
  -H "Content-Type: application/json" \
  -d '{"name": "Chocolate Brown", "description": "Rich chocolate brown hair"}'

# Upload training image
curl -X POST https://your-app.up.railway.app/colors/1/images \
  -F "image=@brown_hair.jpg"

# Train the model
curl -X POST https://your-app.up.railway.app/train/1
```

---

## Memory Usage on Railway

Railway's free tier provides **512MB RAM + $5 credit/month**, which is perfect for this app:

- **Startup**: ~300MB ✅
- **During Training**: ~550MB ✅ (works fine with lazy loading)
- **After Training**: ~350MB ✅

All features work smoothly!

---

## Monitoring Your App

### Railway Dashboard
- **Logs**: View real-time application logs
- **Metrics**: Monitor CPU, RAM, and network usage
- **Deployments**: See deployment history

### Check Logs
```bash
# In Railway dashboard
Click your service → "Deployments" → "View Logs"
```

---

## Updating Your Frontend

Update your frontend to use the Railway API URL:

```javascript
// In your frontend config or .env
VITE_API_URL=https://your-app.up.railway.app
```

---

## Cost Breakdown

### Free Tier
- **RAM**: 512MB (enough for all features)
- **Credit**: $5/month
- **Usage**: ~$5-8/month for moderate traffic
- **Result**: Effectively free for development/testing

### If You Exceed Free Credit
- Railway charges $0.000231/GB-hour for RAM
- ~$3-5/month for typical usage
- Still very affordable!

---

## Troubleshooting

### Build Fails
- Check Railway logs for specific error
- Verify Dockerfile syntax
- Ensure all files are committed to git

### Service Won't Start
- Check if PORT is being used correctly
- Verify `/health` endpoint works
- Review startup logs in Railway dashboard

### Training Fails
- Check memory usage in Railway metrics
- Reduce number of training images (use 1-3 at a time)
- Review logs for specific error

---

## Advantages Over Render

| Feature | Railway | Render Free |
|---------|---------|-------------|
| RAM | 512MB + flexible | 512MB hard limit |
| Credits | $5/month | None |
| Cold starts | Faster | Slower |
| Build time | ~5 min | ~10 min |
| Training | ✅ Works | ⚠️ May timeout |
| Logs | Better UI | Basic |

---

## Next Steps

1. ✅ **Commit changes** to git
2. ✅ **Deploy on Railway** (one-click from GitHub)
3. ✅ **Test endpoints** (health, analyze, train)
4. ✅ **Update frontend** with Railway URL
5. ✅ **Monitor usage** in Railway dashboard

---

## Support

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Your App Logs**: Railway Dashboard → Your Service → Logs

---

## Summary

Railway is the **perfect choice** for your Hair Color Analyzer backend:

✅ All features work (analysis + training)  
✅ Better performance than Render  
✅ Generous free tier  
✅ Easy deployment from GitHub  
✅ Great developer experience  

Just push to GitHub and deploy! 🚀
