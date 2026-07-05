@echo off
echo ==============================================
echo Running Data Science Pipeline
echo ==============================================
echo.
echo Running EDA and Preprocessing...
.venv\Scripts\python src\eda_and_preprocess.py
if %ERRORLEVEL% neq 0 (
    echo Preprocessing failed!
    exit /b %ERRORLEVEL%
)
echo.
echo Running Model Training and Evaluation...
.venv\Scripts\python src\train_and_evaluate.py
if %ERRORLEVEL% neq 0 (
    echo Model training failed!
    exit /b %ERRORLEVEL%
)
echo.
echo Pipeline completed successfully!
