# GKE Deployment Guide (Step by Step)

This guide explains what to do before deployment and how to deploy this project to GCP with a good architecture.

## 1) Target architecture

- Frontend is public through a Kubernetes Service of type LoadBalancer.
- Backend is internal (ClusterIP) and only reachable inside the cluster.
- Environments are isolated by namespace: dev, test, production.
- Promote in order: dev -> test -> production.

Why this is good architecture:

- Better security: backend is not exposed publicly.
- Better reliability: you validate changes in dev/test before production.
- Better operations: namespace isolation simplifies debugging and rollback.

## 2) Prerequisites (run these first)

Run from project root in PowerShell.

```powershell
docker --version
gcloud --version
kubectl version --client
```

If any command fails, install that tool before continuing.

## 3) Configure GCP project and region

Set your own values first:

```powershell
$PROJECT_ID = "your-gcp-project-id"
$REGION = "us-central1"
$ZONE = "us-central1-a"
$CLUSTER_NAME = "car-app-gke"
$REPO_NAME = "car-app-repo"
$VERSION = "v1.0.0"
```

Authenticate and select project:

```powershell
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE
```

Enable required APIs:

```powershell
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## 4) Create Artifact Registry and enable Docker auth

```powershell
gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=$REGION --description="Docker repo for car project"
gcloud auth configure-docker "$REGION-docker.pkg.dev"
```

If the repository already exists, continue to the next step.

## 5) Create GKE cluster

```powershell
gcloud container clusters create $CLUSTER_NAME --zone $ZONE --num-nodes 2 --machine-type e2-standard-4
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE
kubectl cluster-info
```

## 6) Build and push images with version tags

Set image names:

```powershell
$BACKEND_IMAGE = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/car-backend:$VERSION"
$FRONTEND_IMAGE = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/car-frontend:$VERSION"
```

Build images:

```powershell
docker build -t $BACKEND_IMAGE -f backend/Dockerfile .
docker build -t $FRONTEND_IMAGE -f frontend/Dockerfile frontend
```

Push images:

```powershell
docker push $BACKEND_IMAGE
docker push $FRONTEND_IMAGE
```

Optional reproducibility evidence for report:

```powershell
gcloud artifacts docker images list "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
```

## 7) Deploy to dev first (recommended)

Apply base manifest:

```powershell
kubectl apply -f k8s/dev.yaml
```

Update deployments to use pushed GCP images:

```powershell
kubectl set image deployment/backend backend=$BACKEND_IMAGE -n dev
kubectl set image deployment/frontend frontend=$FRONTEND_IMAGE -n dev
```

Wait for rollout:

```powershell
kubectl rollout status deployment/backend -n dev
kubectl rollout status deployment/frontend -n dev
kubectl get pods -n dev
kubectl get svc -n dev
```

Get frontend public IP:

```powershell
$DEV_IP = kubectl get svc frontend -n dev -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
Write-Host "Dev frontend URL: http://$DEV_IP"
```

Note: LoadBalancer IP assignment can take several minutes.

## 8) Promote to test, then production

Deploy test:

```powershell
kubectl apply -f k8s/test.yaml
kubectl set image deployment/backend backend=$BACKEND_IMAGE -n test
kubectl set image deployment/frontend frontend=$FRONTEND_IMAGE -n test
kubectl rollout status deployment/backend -n test
kubectl rollout status deployment/frontend -n test
```

Deploy production:

```powershell
kubectl apply -f k8s/production.yaml
kubectl set image deployment/backend backend=$BACKEND_IMAGE -n production
kubectl set image deployment/frontend frontend=$FRONTEND_IMAGE -n production
kubectl rollout status deployment/backend -n production
kubectl rollout status deployment/frontend -n production
```

## 9) Validation commands

Check workload state:

```powershell
kubectl get all -n dev
kubectl get all -n test
kubectl get all -n production
```

Check logs:

```powershell
kubectl logs -n dev deployment/backend --tail=100
kubectl logs -n dev deployment/frontend --tail=100
```

Backend internal access for debugging:

```powershell
kubectl port-forward -n dev service/backend 7860:7860
```

## 10) Good architecture checklist before final production sign-off

Before final report submission or real production usage, confirm these:

- Use versioned image tags (already done with $VERSION).
- Keep backend internal (ClusterIP), frontend public (LoadBalancer).
- Roll out progressively (dev -> test -> production).
- Increase replicas in production for higher availability.
- Add readiness and liveness probes in deployments.
- Add CPU and memory requests/limits.
- Store secrets outside code (Kubernetes Secret or GCP Secret Manager).

Example scaling command for production:

```powershell
kubectl scale deployment backend -n production --replicas=2
kubectl scale deployment frontend -n production --replicas=2
```

## 11) Troubleshooting quick commands

```powershell
kubectl describe pod -n dev <pod-name>
kubectl get events -n dev --sort-by=.metadata.creationTimestamp
kubectl rollout restart deployment/backend -n dev
kubectl rollout restart deployment/frontend -n dev
```

## 12) What to capture in your academic report

- GCP project, region, and cluster configuration.
- Exact image tags and Artifact Registry references.
- Kubernetes rollout evidence (kubectl rollout status output).
- Service endpoint verification (LoadBalancer IP access).
- Logs showing successful backend startup and request flow.
- Discussion of architecture decisions and limitations.
