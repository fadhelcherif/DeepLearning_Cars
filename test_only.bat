@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

title Deployment Results

echo.
echo DEPLOYMENT RESULTS
echo ===============================================================
echo Namespace: dev
kubectl get svc -n dev
echo.
echo Namespace: test
kubectl get svc -n test
echo.
echo Namespace: prod
kubectl get svc -n production
echo.
echo ===============================================================
echo Summary: frontend IP is shown in the EXTERNAL-IP column for each namespace.
pause
