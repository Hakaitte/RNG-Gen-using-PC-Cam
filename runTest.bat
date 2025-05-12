@echo off

for /f "usebackq tokens=*" %%i in (".env") do (
    for /f "tokens=1,2 delims==" %%a in ("%%i") do (
        if "%%a"=="pythonDir" set pythonDir=%%b
        if "%%a"=="projectDir" set projectDir=%%b
    )
)

%pythonDir% %projectDir%
cd ./NIST-Statistical-Test-Suite/sts
start assess 1000000
cd ../../