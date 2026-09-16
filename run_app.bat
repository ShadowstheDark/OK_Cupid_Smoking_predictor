@echo off
REM Launch the Cupid 2012 Streamlit app.
REM
REM Works from any working directory and uses whatever "python" is on your PATH,
REM so it is portable across machines. Install the dependencies first:
REM     pip install -r requirements.txt
REM
REM Any equivalent command works too:
REM     python -m streamlit run app.py

cd /d "%~dp0"

echo Starting the Cupid 2012 Streamlit app...
echo Open http://localhost:8501 in your browser.
echo Press Ctrl+C in this window to stop.

python -m streamlit run app.py --server.port 8501

pause
