# Kubernetes setup

## Build images

```bash
docker build -t car-backend:latest -f backend/Dockerfile .
docker build -t car-frontend:latest -f frontend/Dockerfile frontend
```

## Apply namespace (simple)

```bash
kubectl apply -f k8s/dev.yaml
kubectl apply -f k8s/test.yaml
kubectl apply -f k8s/production.yaml
```

Apply only one environment at a time if you want isolated testing.

## Access

- Frontend service is exposed with `LoadBalancer`.
- Backend stays internal inside the cluster.
- For local clusters, use `kubectl port-forward` if no load balancer is available.
