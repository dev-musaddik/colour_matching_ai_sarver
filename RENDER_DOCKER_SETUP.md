# How to Configure Render for Docker Deployment

## Problem

Render is **still using Python 3.13 native buildpack** instead of Docker, causing:
- ❌ scikit-learn trying to build from source
- ❌ numpy version conflicts
- ❌ Build failures

## Solution: Force Render to Use Docker

Render might not be detecting `render.yaml` or you need to manually configure the service.

---

## Method 1: Manual Configuration in Render Dashboard (RECOMMENDED)

### Step 1: Go to Your Service Settings

1. Log in to https://dashboard.render.com/
2. Click on your **hair-color-analyzer-backend** service
3. Click **"Settings"** in the left sidebar

### Step 2: Change Environment to Docker

Scroll down and update these settings:

| Setting | Current Value | Change To |
|---------|---------------|-----------|
| **Environment** | `Python 3` or `Python` | **`Docker`** |
| **Root Directory** | (empty or `/`) | **`backend`** |
| **Dockerfile Path** | (empty) | **`./Dockerfile`** |

### Step 3: Configure Health Check

| Setting | Value |
|---------|-------|
| **Health Check Path** | `/health` |

### Step 4: Verify Environment Variables

Make sure these are set:
- `CORS_ORIGINS` = Your frontend URL (e.g., `https://your-app.vercel.app`)

### Step 5: Save and Redeploy

1. Click **"Save Changes"** at the bottom
2. Go to **"Manual Deploy"** → **"Deploy latest commit"**
3. Watch the build logs - you should see Docker building instead of pip installing

---

## Method 2: Delete and Recreate Service

If Method 1 doesn't work, create a fresh service:

### Step 1: Delete Old Service

1. Go to your service settings
2. Scroll to bottom → **"Delete Web Service"**
3. Confirm deletion

### Step 2: Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your repository
3. Configure:
   - **Name**: `hair-color-analyzer-backend`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: **`backend`**
   - **Environment**: **`Docker`** ⚠️ IMPORTANT!
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: `Free`

### Step 3: Add Environment Variables

Click **"Advanced"** → Add:
- `CORS_ORIGINS` = Your Vercel frontend URL

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will build using Docker with Python 3.11
3. Build should succeed in ~5-10 minutes

---

## Method 3: Use render.yaml (Auto-detection)

If Render supports `render.yaml` auto-detection:

### Step 1: Move render.yaml to Root

```powershell
# Move render.yaml to project root
cd d:\hair-color-analyzer
move backend\render.yaml render.yaml
```

### Step 2: Update render.yaml

```yaml
services:
  - type: web
    name: hair-color-analyzer-backend
    env: docker
    plan: free
    rootDir: backend
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: CORS_ORIGINS
        fromGroup: frontend-urls
```

### Step 3: Commit and Push

```powershell
git add render.yaml
git commit -m "Move render.yaml to root for auto-detection"
git push origin main
```

---

## How to Verify Docker is Being Used

### ✅ Correct Docker Build Logs:

```
==> Building with Docker
==> Pulling Docker image
==> Building Docker image
Step 1/15 : FROM python:3.11-slim as builder
Step 2/15 : WORKDIR /app
...
==> Docker build completed
==> Starting container
```

### ❌ Wrong Native Python Build Logs:

```
==> Installing Python 3.13
==> Installing dependencies from requirements.txt
Collecting scikit-learn==1.4.2
  Downloading scikit-learn-1.4.2.tar.gz
  Installing build dependencies
ERROR: Could not find numpy==2.0.0rc1
```

---

## Quick Checklist

Before deploying, verify:

- [ ] Render service **Environment** is set to **`Docker`**
- [ ] **Root Directory** is set to **`backend`**
- [ ] **Dockerfile Path** is set to **`./Dockerfile`**
- [ ] **Health Check Path** is set to **`/health`**
- [ ] `CORS_ORIGINS` environment variable is set
- [ ] `backend/Dockerfile` exists in your repository
- [ ] Latest code is pushed to GitHub

---

## Expected Build Time

- **First Docker build**: 5-10 minutes (building all layers)
- **Subsequent builds**: 2-3 minutes (cached layers)

---

## Troubleshooting

### "Could not find Dockerfile"

- Check **Root Directory** is `backend`
- Check **Dockerfile Path** is `./Dockerfile` (relative to root directory)
- Verify `backend/Dockerfile` exists in your repo

### Still Using Python 3.13

- Render is not using Docker
- Double-check **Environment** setting is `Docker`, not `Python`
- Try Method 2 (delete and recreate service)

### Build Timeout

- Free tier has 15-minute build limit
- Docker build should complete in ~10 minutes
- If timeout, try deploying again (cached layers will speed it up)

---

## Next Steps

1. **Configure Render** using Method 1 (dashboard settings)
2. **Redeploy** and watch build logs
3. **Verify** you see "Building with Docker" in logs
4. **Test** the `/health` endpoint after deployment

Once you see Docker being used, the build will succeed! 🚀
