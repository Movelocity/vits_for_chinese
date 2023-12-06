@echo off

:: 这是一行注释
:: 启用延迟变量扩展
setlocal enabledelayedexpansion

:: 提示用户拖拽包含.wav文件的文件夹到命令行窗口
echo 请拖拽包含.wav文件的文件夹到这个窗口，然后按回车键。
set /p folder="文件夹路径: "

:: 检查用户是否真的输入了路径
if "!folder!"=="" (
    echo 没有输入文件夹路径，程序将退出。
    pause
    exit /b
)

:: 去除路径字符串末尾的反斜杠（如果存在）
if "!folder:~-1!"=="\" set "folder=!folder:~0,-1!"

:: 统计指定文件夹中.wav文件的数量
for /f %%i in ('dir /b /a-d "!folder!\*.wav" ^| find /c /v ""') do set total_files=%%i

:: 如果没有找到文件，通知用户并退出
if "!total_files!"=="0" (
    echo 指定的文件夹中没有找到.wav文件。
    pause
    exit /b
)

set counter=0

:: 处理文件夹中的每个.wav文件
for /r "!folder!" %%i in (*.wav) do (
    cls
    set /a counter+=1
    set /a percentage=!counter! * 100 / !total_files!
    echo [!percentage!%%] 正在处理: "%%i"

    :: 调用test.exe处理文件并忽略输出
    :: 这个文件在新的 vgmstream 中为 stream-cli.exe
    D:\dataset\vgmstream-win\test.exe "%%i" > NUL

    :: 删除原始的.wav文件
    del "%%i"
)

echo [100%%] 处理完成！

:: 等待用户按键后退出
pause
