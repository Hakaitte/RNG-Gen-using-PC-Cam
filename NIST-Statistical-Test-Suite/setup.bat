@echo off
setlocal enabledelayedexpansion

REM Module: setup.bat
REM Function:
REM     Create additional directories and prepare for building after checkout.

REM Variables
set PNAME=%~n0
set ROOTDIR=%~dp0sts


REM Functions
:help
echo %PNAME% is used after checking out a clean repository, to make sure that
echo all the required directories are present. sts/assess and sts/makefile
echo do not have the ability to create missing directories, and git doesn't
echo record empty directories. I'd like to keep the NIST distribution pristine
echo (at least for now), so you need to run this script after a clean checkout,
echo before doing your first build. It won't hurt to re-run it.
echo.
echo For testing, %PNAME% --clean will remove all the subdirectories that this
echo script creates.
exit /b 0

:error
echo %PNAME%: %~1 1>&2
exit /b 1

REM Check for --help argument
if "%~1"=="--help" (
    call :help
)

REM Check if ROOTDIR exists
if not exist "%ROOTDIR%" (
    call :error "STS distribution directory not found: %ROOTDIR%"
)

REM Create subdirectories
set SUBDIRS=
set SUBDIRS=%ROOTDIR%\obj
for %%i in (AlgorithmTesting BBS CCG G-SHA1 LCG MODEXP MS QCG1 QCG2 XOR) do (
    set SUBDIRS=!SUBDIRS!;%ROOTDIR%\experiments\%%i
)

REM Handle --clean argument
if "%~1"=="--clean" (
    echo Cleaning up directory tree.
    for %%i in (!SUBDIRS!) do (
        if exist "%%i" (
            rmdir /s /q "%%i"
            echo Removed %%i
        )
    )
    exit /b 0
)

REM Check for unrecognized arguments
if not "%~1"=="" (
    call :error "Unrecognized arguments; use --help to get help"
)

echo Setting up directories in %ROOTDIR%\experiments.

REM Create directories
for %%i in (!SUBDIRS!) do (
    if not exist "%%i" (
        mkdir "%%i" || call :error "Can't create dir: %%i"
        echo Created %%i.
    ) else (
        echo %%i already exists.
    )
)


echo ROOTDIR is set to: %ROOTDIR%
echo SUBDIRS are: !SUBDIRS!

REM Run create-dir-script if necessary
if not exist "%ROOTDIR%\experiments\AlgorithmTesting\Frequency" (
    echo Running create-dir-script...
    pushd "%ROOTDIR%\experiments" || call :error "Can't cd: %ROOTDIR%\experiments"
    if not exist create-dir-script.bat (
        echo create-dir-script.bat not found.
    ) else (
        call create-dir-script.bat || echo %PNAME%: some directory creations failed; this probably isn't a problem, but please check!
    )
    popd
) else (
    echo Skipping create-dir-script. It appears that it was already run.
)

echo Directories are set up. Change directory to %ROOTDIR% and say 'make'!

exit /b 0