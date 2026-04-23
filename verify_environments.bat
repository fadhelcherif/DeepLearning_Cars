@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo Verifying Kubernetes environments in this project...
echo.

set "NAMESPACES=dev test production"

for %%N in (%NAMESPACES%) do (
    echo ================================
    echo Checking namespace: %%N
    echo ================================

    kubectl get namespace %%N >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] Namespace %%N does not exist.
        echo.
    ) else (
        echo [OK] Namespace exists.
        echo Deployments:
        kubectl get deployments -n %%N
        echo.
        echo Pods:
        kubectl get pods -n %%N
        echo.
        echo Services:
        kubectl get svc -n %%N
        echo.
        echo Rollout status:
        kubectl rollout status deployment/backend -n %%N
        kubectl rollout status deployment/frontend -n %%N
        echo.
        echo Frontend external IP:
        for /f "usebackq delims=" %%I in (`kubectl get svc frontend -n %%N -o jsonpath="{.status.loadBalancer.ingress[0].ip}" 2^>nul`) do set "FRONTEND_IP=%%I"
        if defined FRONTEND_IP (
            echo !FRONTEND_IP!
        ) else (
            echo Pending or not assigned yet.
        )
        set "FRONTEND_IP="
        echo.
    )
)

echo Verification finished.
pause