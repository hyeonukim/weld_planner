@echo off
echo ============================================
echo  Weld Path Planner - Windows Setup
echo ============================================
echo.

echo [1/2] Installing PyTorch (CPU)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo ERROR: PyTorch failed & pause & exit /b 1 )

echo.
echo [2/2] Installing remaining dependencies...
pip install flask flask-cors numpy matplotlib opencv-contrib-python Pillow huggingface_hub safetensors einops timm
if errorlevel 1 ( echo ERROR: Dependencies failed & pause & exit /b 1 )

echo.
echo Done! Run start.bat to launch the app.
pause