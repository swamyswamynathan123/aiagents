@echo off
REM Launch the Streamlit app: the 16 agent patterns and the three capstone projects.
REM First run creates the venv, installs requirements, and creates .env from .env.example.
REM The server only listens on localhost (also set in .streamlit\config.toml).
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || goto :error
)
call venv\Scripts\activate.bat

python -c "import streamlit, langgraph, langchain_tavily, faiss, pandas, matplotlib, pypdf" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt || goto :error
)

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo Created .env from .env.example - fill in OPENAI_API_KEY and TAVILY_API_KEY, then re-run start.bat.
    notepad .env
    exit /b 1
)

streamlit run main.py --server.address localhost
exit /b %errorlevel%

:error
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
