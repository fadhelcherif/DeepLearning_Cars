# Step-by-Step Cloud Deployment Guide for Vehicle Classification App

This guide documents every step taken to deploy the deep learning vehicle classification application on Google Cloud Platform (GCP) using Docker, Kubernetes (GKE), and all supporting tools. It includes commands, screenshots, and explanations for each phase.

---

## 1. Project Structure Preparation
- Organized code into `backend/` (Flask + PyTorch) and `frontend/` (Nginx + static UI).
- Ensured `car_model.pth` is present in `backend/`.
- Created Dockerfiles for both backend and frontend.

```
project_dl/
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   └── nginx.conf
├── k8s/
│   ├── dev.yaml
│   ├── test.yaml
│   └── production.yaml
└── ...
```

---

## 2. Docker Image Build & Push to Artifact Registry
- Authenticate Docker with GCP:
  ```sh
  gcloud auth configure-docker
  ```
- Build backend image:
  ```sh
  docker build -t car-backend:latest -f backend/Dockerfile .
  ```
- Build frontend image:
  ```sh
  docker build -t car-frontend:latest -f frontend/Dockerfile frontend
  ```
- Tag and push images to Artifact Registry:
  ```sh
  docker tag car-backend:latest <region>-docker.pkg.dev/<project-id>/repo/car-backend:latest
  docker tag car-frontend:latest <region>-docker.pkg.dev/<project-id>/repo/car-frontend:latest
  docker push <region>-docker.pkg.dev/<project-id>/repo/car-backend:latest
  docker push <region>-docker.pkg.dev/<project-id>/repo/car-frontend:latest
  ```

---

## 3. GKE Cluster Creation & Authentication
- Create a GKE cluster:
  ```sh
  gcloud container clusters create dl-cluster --num-nodes=3 --region=<region>
  ```
- Authenticate kubectl:
  ```sh
  gcloud container clusters get-credentials dl-cluster --region=<region>
  ```
- (Screenshot: GCP Console showing cluster creation)

---

## 4. Kubernetes Namespace & Manifest Preparation
- Created `k8s/dev.yaml`, `k8s/test.yaml`, `k8s/production.yaml` for each environment.
- Each manifest defines:
  - Namespace
  - Backend Deployment & Service (ClusterIP)
  - Frontend Deployment & Service (LoadBalancer)
- Example snippet:
  ```yaml
  apiVersion: v1
  kind: Namespace
  metadata:
    name: dev
  ---
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: backend
    namespace: dev
  ...
  ```

---

## 5. Apply Manifests & Deploy
- Apply manifests for all environments:
  ```sh
  kubectl apply -f k8s/dev.yaml
  kubectl apply -f k8s/test.yaml
  kubectl apply -f k8s/production.yaml
  ```
- Check pod and service status:
  ```sh
  kubectl get pods -n dev
  kubectl get svc -n dev
  kubectl get pods -n test
  kubectl get svc -n test
  kubectl get pods -n production
  kubectl get svc -n production
  ```
- (Screenshot: kubectl output showing running pods and services)

---

## 6. Accessing the Application
- Each frontend service exposes an external IP (see `kubectl get svc`).
- Access the app in browser:
  - `http://<dev-external-ip>`
  - `http://<test-external-ip>`
  - `http://<prod-external-ip>`
- (Screenshot: Application running in browser)

---

## 7. Testing & Validation
- Uploaded images, validated predictions in all environments.
- Checked logs for errors and fixed issues (e.g., missing model file, image errors).
- (Screenshot: App UI, prediction results)

---

## 8. Iteration & Improvements
- Refined manifests for clarity and resource limits.
- Used namespaces for isolation and reproducibility.
- (Screenshot: Namespace isolation in GCP Console)

---

## 9. Summary Table (Example)
| Environment | Namespace   | External IP         | Status   |
|-------------|-------------|---------------------|----------|
| Development | dev         | <dev-external-ip>   | Running  |
| Testing     | test        | <test-external-ip>  | Running  |
| Production  | production  | <prod-external-ip>  | Running  |

---

## 10. Useful GCP Console Screenshots
- Cluster creation
- Artifact Registry images
- Workload and service dashboards
- Namespace isolation
- Application in browser

---

## 11. Additional Notes
- All YAML files are in `k8s/`.
- Images are in Artifact Registry: `<region>-docker.pkg.dev/<project-id>/repo/`
- Cluster name: `dl-cluster`
- All steps are reproducible with the commands above.

---

**Replace `<region>`, `<project-id>`, and `<external-ip>` with your actual values.**

**Add your screenshots at each step for a complete, illustrated deployment guide.**
