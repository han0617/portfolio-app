@echo off
chcp 65001 >nul
title 포트폴리오 대시보드
echo ============================================
echo   리스크 패리티 + CAPM 포트폴리오 대시보드
echo ============================================
echo.
echo  잠시 후 웹 브라우저가 자동으로 열립니다.
echo  (이 검은 창은 닫지 마세요. 닫으면 종료됩니다.)
echo.
echo  끄려면: 이 창을 닫거나 Ctrl + C 를 누르세요.
echo ============================================
echo.
cd /d "%~dp0"
python -m streamlit run portfolio_app.py
pause
