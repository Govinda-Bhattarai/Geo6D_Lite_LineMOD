# Production Deployment Checklist

## Pre-Deployment Verification ✓

- [ ] **Model Checkpoint**
  - [ ] Best model exists: `checkpoints/epoch_39.pth` (78.43% accuracy)
  - [ ] Backup created: `checkpoints/best_model.pth`
  - [ ] Training log available: `checkpoints/train_log.json`

- [ ] **Environment**
  - [ ] Python 3.8+ installed
  - [ ] PyTorch 1.9+ installed
  - [ ] CUDA compatible (if using GPU)
  - [ ] Dependencies installed: `pip install -r requirements.txt`

- [ ] **Code Quality**
  - [ ] All scripts tested and working
  - [ ] No broken imports
  - [ ] Device fallback to CPU if no GPU

- [ ] **Documentation**
  - [ ] `DEPLOYMENT_GUIDE.md` reviewed
  - [ ] Model metrics confirmed: 78.43% overall accuracy
  - [ ] API documentation available

## Deployment Options

### Option 1: Standalone Python Script (Recommended for Testing)
```bash
# Evaluate model
python3 evaluate.py --object_ids 05 --checkpoint checkpoints/epoch_39.pth

# Run inference
python3 infer.py --object_ids 05 --checkpoint checkpoints/epoch_39.pth
```

### Option 2: REST API (Recommended for Production)
```bash
# Start API server
python3 api_server.py \
  --checkpoint checkpoints/epoch_39.pth \
  --port 5000 \
  --host 0.0.0.0

# Test endpoint
curl http://localhost:5000/health

# Send prediction request
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.jpg"
```

### Option 3: Docker Container (Recommended for Deployment)
```bash
# Build image
docker build -t geo6d-lite .

# Run container
docker run -p 5000:5000 geo6d-lite

# Or use docker-compose
docker-compose up -d
```

### Option 4: Python Module (Recommended for Integration)
```python
from inference import PoseEstimator
import numpy as np

# Initialize
estimator = PoseEstimator('checkpoints/epoch_39.pth', device='cuda')

# Predict
image = np.random.randn(256, 256, 3)
rotation, translation = estimator.predict(image)
rotation_matrix = estimator.rot6d_to_matrix(rotation)

print(f"Rotation (6D): {rotation}")
print(f"Translation: {translation}")
print(f"Rotation Matrix:\n{rotation_matrix}")
```

## Post-Deployment Verification

- [ ] **API Health Check**
  ```bash
  curl http://localhost:5000/health
  # Expected: {"status": "healthy", "model": "Geo6D-Lite", ...}
  ```

- [ ] **Model Info Endpoint**
  ```bash
  curl http://localhost:5000/info
  # Verify accuracy metrics are correct
  ```

- [ ] **Inference Test**
  ```bash
  python3 evaluate.py --object_ids 05 --checkpoint checkpoints/epoch_39.pth
  # Expected output:
  # Overall Accuracy: 78.43%
  # Rotation Accuracy: 81.77%
  # Translation Accuracy: 95.90%
  ```

- [ ] **Performance Monitoring**
  - Monitor CPU/GPU usage
  - Track inference latency (should be <100ms per image on GPU)
  - Monitor memory usage (should be <2GB)

## System Requirements

### Minimum (CPU-only)
- CPU: 4-core processor
- RAM: 8 GB
- Storage: 500 MB (model + dependencies)
- Network: Optional

### Recommended (GPU)
- GPU: NVIDIA with 4GB+ VRAM
- CPU: 4-core processor
- RAM: 16 GB
- Storage: 1 GB
- Network: Optional

### Enterprise (High-throughput)
- GPU: NVIDIA A100 or better
- CPU: 16-core processor
- RAM: 64 GB
- Storage: Fast SSD, 10 GB
- Network: High-bandwidth connection for distributed inference

## Performance Expectations

| Metric | Value |
|--------|-------|
| Inference Time (GPU) | ~20-50 ms per image |
| Inference Time (CPU) | ~200-500 ms per image |
| Memory Usage (GPU) | ~1-2 GB |
| Memory Usage (CPU) | ~500-800 MB |
| Throughput (GPU) | ~20-50 images/sec |
| Throughput (CPU) | ~2-5 images/sec |

## Scaling Considerations

### Horizontal Scaling
- Run multiple API instances behind load balancer
- Use shared filesystem for checkpoints
- Each instance: 1-2 GB memory, 50-100 images/sec throughput

### Vertical Scaling
- Increase batch size for higher throughput
- Use multi-GPU setup for faster inference
- Optimize model with quantization/pruning if needed

## Monitoring and Logging

### Key Metrics to Track
1. **Inference Latency**
   - P50, P95, P99 latencies
   - Alert if >500ms

2. **Throughput**
   - Images processed per second
   - Alert if <5 images/sec on GPU

3. **Accuracy**
   - Track prediction confidence
   - Monitor for distribution shift

4. **Resource Usage**
   - CPU/GPU utilization
   - Memory usage
   - Disk I/O

5. **Availability**
   - API uptime
   - Error rates
   - Request success rate

### Log Locations
- API logs: stdout/stderr
- Errors: Application error log
- Performance: Application performance log

## Troubleshooting

### Issue: Out of Memory
**Solution:**
- Reduce batch size in API requests
- Use CPU-only inference
- Restart API server to clear cache

### Issue: Slow Inference
**Solution:**
- Verify GPU is being used: `nvidia-smi`
- Check batch size (larger = faster)
- Profile with `torch.profiler`

### Issue: Low Accuracy
**Solution:**
- Verify correct checkpoint is loaded
- Check input image preprocessing
- Verify image normalization (should be in [-1, 1])

### Issue: API Connection Refused
**Solution:**
- Check port is not in use: `lsof -i :5000`
- Verify host binding is correct
- Check firewall settings

## Security Considerations

- [ ] Restrict API access with authentication/token
- [ ] Use HTTPS/TLS in production
- [ ] Validate input image format and size
- [ ] Rate limit API requests
- [ ] Monitor for anomalous usage patterns
- [ ] Keep dependencies updated
- [ ] Run in isolated container/environment

## Backup and Recovery

- [ ] Store checkpoint in multiple locations
- [ ] Version control configuration files
- [ ] Regular model performance audits
- [ ] Rollback plan for model updates
- [ ] Log all predictions for auditing

## Cleanup Commands

After verification, you can remove unnecessary training files:
```bash
bash CLEANUP.sh
```

This removes:
- Training scripts
- Diagnostic files
- Old configurations
- Temporary logs
- Backup directories

## Success Criteria

✅ Deployment is successful if:
1. API responds to `/health` endpoint
2. `/predict` endpoint returns valid pose predictions
3. Accuracy metrics match: 78.43% overall
4. Inference latency is acceptable (<100ms per image on GPU)
5. No errors in logs during normal operation
6. Memory usage is within limits

## Next Steps

1. **Test locally first**
   ```bash
   python3 evaluate.py --object_ids 05 --checkpoint checkpoints/epoch_39.pth
   ```

2. **Start API server**
   ```bash
   python3 api_server.py --port 5000
   ```

3. **Verify with curl**
   ```bash
   curl http://localhost:5000/health
   ```

4. **Deploy to Docker (optional)**
   ```bash
   docker-compose up -d
   ```

5. **Monitor and log**
   - Set up monitoring dashboard
   - Configure log aggregation
   - Set up alerts for failures

---
**Last Updated:** 2025-11-17
**Model:** Geo6D-Lite
**Best Checkpoint:** epoch_39.pth (78.43% accuracy)
**Status:** ✅ Production Ready
