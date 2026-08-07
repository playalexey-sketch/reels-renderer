@echo off
chcp 65001 >nul
echo ============================================
echo  Reels Renderer — генератор видео (одним кликом)
echo ============================================
where python >nul 2>nul || (
  echo НЕ НАЙДЕН Python! Скачайте python.org ^>=3.10 и поставьте галочку "Add to PATH".
  pause
  exit /b 1
)
echo [1/3] Установка зависимостей (первый запуск ~5 минут)...
python -m pip install -q -r "%~dp0requirements.txt"
echo [2/3] Загрузка весов (первый запуск, далее кэш)...
echo [3/3] Рендер видео...
python "%~dp0main.py" %*
echo.
echo ГОТОВО! Файл: result.mp4 в этой папке.
pause
