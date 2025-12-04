# Railway Deployment Checklist

## ✅ Pre-Deployment Checklist

- [x] Dockerfile optimized (lazy loading, pre-download models)
- [x] railway.json configuration created
- [x] .dockerignore created (faster builds)
- [x] /health endpoint exists in main.py
- [x] PORT environment variable handled dynamically
- [x] All features enabled (analysis + training)
- [ ] Code committed to git
- [ ] Code pushed to GitHub

## 🚀 Deployment Steps

### 1. Commit and Push
```bash
cd backend
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### 2. Deploy on Railway

1. Go to https://railway.app
2. Click **"Login"** → Sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Authorize Railway to access your repos
6. Select: `dev-musaddik/colour_matching_ai_sarver`
7. Railway auto-detects Dockerfile and starts building
8. Wait ~5-10 minutes for build to complete

### 3. Get Your URL

- Railway provides: `https://your-app.up.railway.app`
- Copy this URL for your frontend

### 4. Test Deployment

```bash
# Replace with your Railway URL
export API_URL="https://your-app.up.railway.app"

# Test health
curl $API_URL/health

# Test analysis
curl -X POST $API_URL/analyze -F "images=@test.jpg"
```

## 📊 Expected Build Output

```
Building...
[+] Building Docker image
 => [1/6] Installing build dependencies
 => [2/6] Installing Python packages
 => [3/6] Copying application code
 => [4/6] Pre-downloading YOLO model ✓
 => [5/6] Pre-downloading ResNet50 model ✓
 => [6/6] Creating user and setting permissions
Build complete ✓

Deploying...
Starting service on port 8000
No trained models found. Skipping model loading to save memory.
Health check passed ✓
Service is live at https://your-app.up.railway.app
```

## 🔧 Configuration

### Environment Variables (Optional)
Add in Railway dashboard if needed:

| Variable | Value | Purpose |
|----------|-------|---------|
| `ALLOWED_ORIGINS` | `https://yourfrontend.com` | CORS (if needed) |
| `PORT` | Auto-set by Railway | Don't change |

## 📝 Update Frontend

In your frontend `.env` or config:
```env
VITE_API_URL=https://your-app.up.railway.app
```

## ✅ Post-Deployment Verification

- [ ] Health endpoint responds
- [ ] Analysis endpoint works
- [ ] Can create colors
- [ ] Can upload training images
- [ ] Training works (test with 1-2 images)
- [ ] Frontend connects successfully

## 💰 Cost Estimate

- **Free tier**: $5 credit/month
- **Typical usage**: $3-5/month
- **Result**: Effectively free for development

## 🆘 Troubleshooting

### Build fails
- Check Railway logs
- Verify all files committed
- Check Dockerfile syntax

### Service won't start
- Check logs for errors
- Verify PORT is used correctly
- Test /health endpoint

### Out of memory
- Check Railway metrics
- Reduce training batch size
- Upgrade to higher tier if needed

## 📚 Resources

- Railway Docs: https://docs.railway.app
- Your Deployment Guide: RAILWAY_DEPLOYMENT.md
- Memory Optimization: MEMORY_OPTIMIZED_DEPLOYMENT.md
