@echo off
cd /d "%~dp0"
set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PY%" set "PY=py -3.10"
echo === Reconstructing t-joint_1..3.jpg with the updated detector ===
%PY% validate_photos.py t-joint_1.jpg t-joint_2.jpg t-joint_3.jpg --output tjoint_output > tjoint_run.log 2>&1
type tjoint_run.log
if errorlevel 1 ( echo RUN FAILED - see tjoint_run.log & pause & exit /b 1 )
echo === Opening viewer at http://127.0.0.1:5003/?job=saved ===
%PY% preview_saved.py tjoint_output --port 5003
pause
