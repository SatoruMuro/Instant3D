@echo off
REM 📂 このバッチファイルを実行するフォルダ内の拡張子なしファイルに .dcm を追加します

cd /d "%~dp0"

for %%f in (*) do (
    if "%%~xf"=="" (
        ren "%%f" "%%~nf.dcm"
        echo Renamed: %%f → %%~nf.dcm
    ) else (
        echo Skipping: %%f (already has extension)
    )
)

echo.
echo ✅ すべての処理が完了しました。
pause
