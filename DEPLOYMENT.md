# Hair Color Analyzer - Deployment Guide

## Overview

This guide covers deploying the Hair Color Analyzer backend to Render using Docker.

## Prerequisites

- Docker installed locally (for testing)
- Render account
- Git repository connected to Render

## Local Testing with Docker

### Option 1: Using Docker Compose (Recommended)

```powershell
# Navigate to project root
cd d:\hair-color-analyzer

# Build and start the backend
docker-compose up --build

# The API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

To stop the service:
```powershell
docker-compose down
```

### Option 2: Using Docker Directly

```powershell
# Navigate to backend directory
cd d:\hair-color-analyzer\backend

# Build the Docker image
docker build -t hair-color-analyzer-backend .

# Run the container
docker run -p 8000:8000 --env-file .env hair-color-analyzer-backend

# Test the health endpoint
curl http://localhost:8000/health
```

## Deploying to Render

### Step 1: Prepare Your Repository

1. Ensure all Docker files are committed:
   - `backend/Dockerfile`
   - `backend/.dockerignore`
   - `backend/render.yaml`
   - `backend/requirements.txt` (with updated versions)

2. Remove `runtime.txt` (no longer needed):
   ```powershell
   git rm backend/runtime.txt
   ```

3. Commit and push changes:
   ```powershell
   git add .
   git commit -m "Add Docker deployment configuration"
   git push origin main
   ```

### Step 2: Configure Render Service

1. **Create New Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your Git repository

2. **Service Configuration**:
   - **Name**: `hair-color-analyzer-backend`
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `backend`
   - **Dockerfile Path**: `./Dockerfile`

3. **Environment Variables**:
   Add the following environment variables in Render dashboard:
   - `PORT`: `8000` (automatically set by Render)
   - `CORS_ORIGINS`: Your frontend URL (e.g., `https://your-frontend.com`)
   - Any other variables from your `.env` file

4. **Health Check**:
   - **Health Check Path**: `/health`
   - **Health Check Interval**: 30 seconds

5. **Instance Type**:
   - Select "Free" tier for testing
   - Upgrade to paid tier for production

### Step 3: Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Build the Docker image
   - Deploy the container
   - Run health checks

3. Monitor the build logs for any errors

### Step 4: Verify Deployment

Once deployed, test your endpoints:

```powershell
# Health check
curl https://your-service.onrender.com/health

# API documentation
# Visit: https://your-service.onrender.com/docs
```

## Troubleshooting

### Build Fails with "No space left on device"

Render's free tier has limited disk space. To fix:
1. Remove unnecessary files via `.dockerignore`
2. Use multi-stage builds (already implemented)
3. Upgrade to paid tier for more resources

### Container Crashes on Startup

Check logs in Render dashboard:
1. Go to your service
2. Click "Logs" tab
3. Look for Python errors or missing dependencies

Common issues:
- Missing environment variables
- Database connection errors
- Port conflicts

### Health Check Failing

Ensure:
1. The `/health` endpoint returns 200 status
2. The container is listening on `0.0.0.0:8000`
3. No firewall blocking the health check

### Slow Cold Starts

Render's free tier spins down after inactivity:
- First request after inactivity takes 30-60 seconds
- Upgrade to paid tier for always-on instances
- Use a cron job to ping your service every 10 minutes

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `PORT` | Server port (auto-set by Render) | `8000` |
| `CORS_ORIGINS` | Allowed frontend origins | `https://example.com,http://localhost:3000` |

## Updating the Deployment

To deploy updates:

```powershell
# Make your changes
git add .
git commit -m "Your update message"
git push origin main
```

Render will automatically detect the push and redeploy.

## Rollback

If a deployment fails:

1. Go to Render dashboard
2. Click your service
3. Go to "Events" tab
4. Click "Rollback" on a previous successful deployment

## Performance Optimization

### Reduce Build Time

- Dependencies are cached between builds
- Only rebuild when `requirements.txt` changes
- Multi-stage build reduces final image size

### Reduce Memory Usage

Current configuration uses:
- Python 3.11 slim base image
- Minimal runtime dependencies
- Non-root user for security

### Scale Up

For production:
1. Upgrade to paid tier
2. Enable auto-scaling
3. Add Redis for caching
4. Use CDN for static assets

## Monitoring

Render provides:
- Real-time logs
- Metrics dashboard
- Health check status
- Deployment history

Access via: Dashboard → Your Service → Logs/Metrics

## Support

For issues:
1. Check Render status page
2. Review deployment logs
3. Test locally with Docker
4. Contact Render support

## Additional Resources

- [Render Docker Deployment Docs](https://render.com/docs/docker)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
