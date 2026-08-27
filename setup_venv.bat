@echo off
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo Virtual environment ready.
echo Run: venv\Scripts\activate.bat
