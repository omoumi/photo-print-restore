@echo off
REM photo-print-restore -- reproducible, verifiable restoration of flatbed scans
REM Copyright (C) 2026 omoumi
REM
REM This program is free software: you can redistribute it and/or modify it
REM under the terms of the GNU General Public License as published by the
REM Free Software Foundation, either version 3 of the License, or (at your
REM option) any later version.
REM
REM This program is distributed in the hope that it will be useful, but
REM WITHOUT ANY WARRANTY; without even the implied warranty of
REM MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
REM General Public License for more details.
REM
REM You should have received a copy of the GNU General Public License along
REM with this program.  If not, see <https://www.gnu.org/licenses/>.

REM ==========================================================================
REM  run_pipeline.bat  --  STAGE 5: colorizes every B&W print, unattended
REM
REM  STAGE 5 colorizes the B&W prints: a model paints each of them, and only
REM  the model's colour is kept, laid on the print's own lightness:
REM
REM      colorize_model.py        step 1 of 2   colour from the model
REM      transfer_color.py        step 2 of 2   the colour on the print
REM      calibrate_color.py       sets step 2's colour numbers (not a step)
REM      common.py                shared code, not a step
REM      parallel.py              shared worker-count policy, not a step
REM      colorize_by_hand.json    step 1's graph, for ComfyUI's own page
REM
REM  Keep this file beside the scripts above.  Run it from a command prompt
REM  or just double-click it; it asks where the prints are, and where to
REM  create the output folder and what to call it (Enter keeps
REM  ROOT\colorized; never the prints folder itself), then runs on its own:
REM
REM      1. step 2: writes the results whose guides an earlier run made;
REM      2. step 1 counts the guides still to make; when there are none,
REM         it goes on with 5 and ComfyUI is not needed;
REM      3. starts ComfyUI in a window of its own, unless it is running
REM         already (COMFY_VENV and COMFY_DIR below say where it is), and
REM         waits for it;
REM      4. step 1: makes the guides of every B&W print still without one
REM         -- the long part -- and writes each print's result into the
REM         output folder as soon as its guide is made (--apply), while the
REM         model works on the next;
REM      5. step 2: writes any result still missing (one that failed, say).
REM
REM  It can be stopped at any time -- close the window, or Ctrl+C -- and run
REM  again later: it goes on from the first print not done.  A print that
REM  failed is tried again by the next run, given the same output folder.
REM  The guides go to ROOT\colorize_guides.
REM
REM  The ComfyUI window stays open when this is done; close it then.
REM ==========================================================================
setlocal

REM --- where ComfyUI is (change these if it is somewhere else) --------------
set "COMFY_VENV=C:\venvs\comfy"
set "COMFY_DIR=C:\tools\ComfyUI"
REM the same address as COMFY_URL in colorize_model.py: change both together
set "COMFY_URL=http://127.0.0.1:8188"
set "COMFY_WAIT=60"
REM     COMFY_WAIT  times 5 seconds: how long to wait for ComfyUI to start
REM the seed of the guides: the same as SEED in common.py; change both together
set "SEED=220476035819595"

REM --- colours, the same meanings as in the Python steps (common.py) ------
REM     HEAD  bold        headings
REM     GOOD  green       success
REM     WARN  yellow      warning: the run goes on
REM     BAD   bold red    error: the run stops or a result failed
REM ESC holds the escape character (Windows 10 and later).
for /F "delims=#" %%E in ('"prompt #$E# & for %%E in (1) do rem"') do set "ESC=%%E"
set "HEAD=%ESC%[1m"
set "GOOD=%ESC%[32m"
set "WARN=%ESC%[33m"
set "BAD=%ESC%[1;31m"
set "OFF=%ESC%[0m"

REM --- where the scripts are, without the trailing backslash ----------------
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" if not "%HERE:~-2%"==":\" set "HERE=%HERE:~0,-1%"

echo.
echo %HEAD%==========================================================================
echo  STAGE 5  --  colorizing the B^&W prints
echo ==========================================================================%OFF%

REM parallel.py is optional: without it step 2 runs in one process
for %%F in (colorize_model.py transfer_color.py common.py) do (
    if not exist "%HERE%\%%F" (
        echo.
        echo %BAD%%%F is not beside this file.  The scripts must sit together.%OFF%
        goto :end
    )
)

REM --- ask for the folder holding the prints ---------------------------------
:askroot
echo.
echo  Where are the prints?
echo.
echo    Press Enter to use the folder this file is in:
echo        "%HERE%"
echo.
echo    Or type/paste another folder.  Quotes are fine.
echo.
set "ROOT="
set /p "ROOT=  folder: "

if not defined ROOT set "ROOT=%HERE%"

REM strip any quotes the user pasted, then a trailing backslash, which would
REM otherwise escape the closing quote when the path is passed on
set "ROOT=%ROOT:"=%"
if not defined ROOT set "ROOT=%HERE%"
if "%ROOT:~-1%"=="\" if not "%ROOT:~-2%"==":\" set "ROOT=%ROOT:~0,-1%"

REM a bare drive letter means "wherever I last was on that drive", not its
REM root, so make it explicit
if "%ROOT:~-1%"==":" set "ROOT=%ROOT%\"

REM no unquoted path inside a ( ) block: a ")" in it, as in
REM "Program Files (x86)", would end the block early
if exist "%ROOT%\" goto :rootok
echo.
echo  %WARN%no such folder:%OFF%
echo      "%ROOT%"
goto :askroot

:rootok
REM The path handed to Python never ends in a backslash: in "D:\" the
REM backslash would escape the closing quote.  "D:\." names the same folder.
set "ROOTARG=%ROOT%"
if "%ROOT:~-1%"=="\" set "ROOTARG=%ROOT%."
set "GUIDES=%ROOT%\colorize_guides"
if "%ROOT:~-1%"=="\" set "GUIDES=%ROOT%colorize_guides"
echo.
echo  using: "%ROOT%"

REM --- Python itself, then the packages the steps cannot run without ---------
python --version 1>nul 2>nul
if not errorlevel 1 goto :packages
echo.
echo  %BAD%Python was not found.%OFF%  Install Python 3 from python.org, tick
echo  "Add python.exe to PATH" during the install, then run this file again.
goto :end

:packages
python -c "import numpy, tifffile, cv2" 1>nul 2>nul
if not errorlevel 1 goto :packagesok
echo.
echo  %BAD%a package the steps need is not installed:%OFF%
echo      pip install numpy tifffile opencv-python imagecodecs psutil
goto :end

REM --- where the results go ---------------------------------------------
REM Asked the same way as the prints folder above, and for the same reason
REM no unquoted path is echoed inside a ( ) block.
:packagesok
:askout
echo.
echo  Where should the output folder be created?
echo.
echo    Press Enter to use the prints folder:
echo        "%ROOT%"
echo.
echo    Or type/paste another folder.  Quotes are fine.  To go on with an
echo    earlier run, give the same folder and name as then.
echo.
set "OUTLOC="
set /p "OUTLOC=  location: "

if not defined OUTLOC set "OUTLOC=%ROOT%"
set "OUTLOC=%OUTLOC:"=%"
if not defined OUTLOC set "OUTLOC=%ROOT%"
if "%OUTLOC:~-1%"=="\" if not "%OUTLOC:~-2%"==":\" set "OUTLOC=%OUTLOC:~0,-1%"
if "%OUTLOC:~-1%"==":" set "OUTLOC=%OUTLOC%\"

if exist "%OUTLOC%\" goto :askname
echo.
echo  %WARN%no such folder:%OFF%
echo      "%OUTLOC%"
goto :askout

:askname
echo.
echo  Name of the folder the colorized prints go into?
echo.
echo    Press Enter for:  colorized
echo.
set "OUTNAME="
set /p "OUTNAME=  folder name: "

if not defined OUTNAME set "OUTNAME=colorized"
set "OUTNAME=%OUTNAME:"=%"
if not defined OUTNAME set "OUTNAME=colorized"

REM a bare drive root already ends in a backslash
set "OUT=%OUTLOC%\%OUTNAME%"
if "%OUTLOC:~-1%"=="\" set "OUT=%OUTLOC%%OUTNAME%"

REM the prints folder itself: the results take the prints' names
if /i not "%OUT%"=="%ROOT%" goto :notroot
echo.
echo  %WARN%that is the prints folder itself - choose another name%OFF%
goto :askname

:notroot
REM create it now, so a name Windows refuses (: * ? < > | and the like) is
REM caught here and asked again, not after the run has started
if not exist "%OUT%\" mkdir "%OUT%" 2>nul
if exist "%OUT%\" goto :outok
echo.
echo  %WARN%could not create a folder with that name:%OFF%
echo      "%OUT%"
goto :askname

:outok
echo.
echo  output: "%OUT%"

REM --- step 2: results of guides an earlier run made ------------------------
set "S0="
set "S1="
set "S2="
dir /b "%GUIDES%\*_seed%SEED%.png" 1>nul 2>nul
if errorlevel 1 goto :count
echo.
echo %HEAD%[step 2 of 2]  results of the guides made before...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\transfer_color.py" "%ROOTARG%" --apply --out "%OUT%"
set "S0=%errorlevel%"
if "%S0%"=="2" echo %WARN%some results could not be written - see above%OFF%
if "%S0%"=="1" echo %WARN%step 2 stopped - see above; step 1 goes on%OFF%

REM --- step 1: is there anything to make?  3 = nothing, 0 = some ----------
:count
echo.
echo %HEAD%[step 1 of 2]  the guides still to make...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\colorize_model.py" "%ROOTARG%" --count
set "C=%errorlevel%"
if "%C%"=="3" goto :nothing
if "%C%"=="0" goto :comfy
echo %BAD%*** step 1 could not count the guides - see above ***%OFF%
goto :end

:nothing
set "S1=0"
goto :afterstep1

REM --- ComfyUI: running already, or started here -----------------------------
:comfy
echo.
echo %HEAD%ComfyUI%OFF%
echo --------------------------------------------------------------------
call :comfyup
if not errorlevel 1 goto :comfyok
if exist "%COMFY_DIR%\main.py" goto :venvchk
echo  %BAD%ComfyUI is not running, and there is no main.py in:%OFF%
echo      "%COMFY_DIR%"
echo  Set COMFY_DIR at the top of this file.
goto :end

:venvchk
if exist "%COMFY_VENV%\Scripts\python.exe" goto :comfystart
echo  %BAD%ComfyUI is not running, and there is no Python in:%OFF%
echo      "%COMFY_VENV%\Scripts"
echo  Set COMFY_VENV at the top of this file.
goto :end

:comfystart
echo  starting ComfyUI in a window of its own...
start "ComfyUI" /d "%COMFY_DIR%" cmd /k ""%COMFY_VENV%\Scripts\python.exe" main.py"
set /a "TRIES=0"
:comfywait
timeout /t 5 /nobreak >nul
call :comfyup
if not errorlevel 1 goto :comfyok
set /a "TRIES+=1"
if %TRIES% LSS %COMFY_WAIT% goto :comfywait
echo  %BAD%ComfyUI did not answer at %COMFY_URL% - see its window%OFF%
goto :end

:comfyok
echo  %GOOD%ComfyUI is running at %COMFY_URL%%OFF%

REM --- step 1: the guides ------------------------------------------------------
echo.
echo %HEAD%[step 1 of 2]  the guides, each result as its guide is made...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\colorize_model.py" "%ROOTARG%" --apply --out "%OUT%"
REM 2 = some prints failed or could not be read, the others are done;
REM 1 = it stopped
set "S1=%errorlevel%"
if "%S1%"=="2" echo %WARN%some prints failed or could not be read - see above%OFF%
if "%S1%"=="1" echo %WARN%step 1 stopped - run this file again to go on%OFF%

REM --- step 2: the results -----------------------------------------------------
:afterstep1
dir /b "%GUIDES%\*_seed%SEED%.png" 1>nul 2>nul
if not errorlevel 1 goto :haveguides
echo.
if "%S1%"=="0" echo  %GOOD%no B^&W print to colorize in this folder%OFF%
if not "%S1%"=="0" echo  %WARN%no guide was made yet - run this file again to go on%OFF%
goto :end

:haveguides
REM nothing made now: the results were written above already
if not "%C%"=="3" goto :results
if not defined S0 goto :results
set "S2=%S0%"
goto :final

:results
echo.
echo %HEAD%[step 2 of 2]  any result still missing...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\transfer_color.py" "%ROOTARG%" --apply --out "%OUT%"
set "S2=%errorlevel%"
if "%S2%"=="2" echo %WARN%some results could not be written - see above%OFF%
if "%S2%"=="1" echo %BAD%*** step 2 stopped with an error - see above ***%OFF%

:final
echo.
if "%S1%"=="0" if "%S2%"=="0" (
    echo %GOOD%==========================================================================
    echo  done: every B^&W print is colorized%OFF%
    goto :summary
)
echo %WARN%==========================================================================
echo  not all done yet: run this file again to go on%OFF%

:summary
echo.
echo    "%OUT%\"
echo        the colorized prints, under their own names
echo    "%GUIDES%\"
echo        the model's colour pictures, one per print
echo.
if not "%C%"=="3" echo  The ComfyUI window can be closed now.
echo ==========================================================================
goto :end

REM --- does ComfyUI answer?  errorlevel 0: yes ------------------------------
:comfyup
python -c "import urllib.request; urllib.request.urlopen('%COMFY_URL%/system_stats', timeout=3)" 1>nul 2>nul
exit /b %errorlevel%

:end
echo.
pause
endlocal
