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
REM  run_pipeline.bat  --  STAGE 1: runs steps 1 to 3 in order
REM
REM  STAGE 1 repairs the vertical seams a CIS flatbed leaves in every scan
REM  it makes:
REM
REM      calibrate_scanner.py     step 1 of 3   measure pitch and phase
REM      detect_seams.py          step 2 of 3   locate the seams in each scan
REM      repair_seams.py          step 3 of 3   put the dropped column back
REM      common.py                shared code, not a step
REM      parallel.py              shared worker-count policy, not a step
REM      pixel_ruler.py           helper for step 2, not a step
REM
REM  Keep this file beside the five scripts above (pixel_ruler.py is not
REM  needed here).  Run it from a command prompt or just double-click it; it
REM  asks where the scans are.
REM
REM  The scans do not have to be beside the scripts.  Press Enter at the
REM  prompt if they are, or give any other folder.
REM
REM  Step 1 is skipped when the scanner is already measured: the pitch and
REM  phase belong to the machine, not to the batch.  When it runs, it
REM  measures up to 200 scans.  Before step 3 it asks where to create the
REM  output folder and what to call it (Enter keeps ROOT\seam-repaired;
REM  never the scans folder itself), and whether to verify the repairs
REM  afterwards.
REM ==========================================================================
setlocal

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
echo  STAGE 1  --  repair the CIS scanner seams
echo ==========================================================================%OFF%

for %%F in (calibrate_scanner.py detect_seams.py repair_seams.py common.py parallel.py) do (
    if not exist "%HERE%\%%F" (
        echo.
        echo %BAD%%%F is not beside this file.  All five scripts must sit together.%OFF%
        goto :end
    )
)

REM --- ask for the folder holding the scans ---------------------------------
:askroot
echo.
echo  Where are the scans?
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
echo.
echo  using: "%ROOT%"

REM --- Python itself, then the packages the steps cannot run without --------
python --version 1>nul 2>nul
if not errorlevel 1 goto :packages
echo.
echo  %BAD%Python was not found.%OFF%  Install Python 3 from python.org, tick
echo  "Add python.exe to PATH" during the install, then run this file again.
goto :end

:packages
python -c "import numpy, tifffile" 1>nul 2>nul
if not errorlevel 1 goto :step1
echo.
echo  %BAD%a package the steps need is not installed:%OFF%
echo      pip install numpy tifffile pillow imagecodecs psutil
goto :end

REM --- step 1 ---------------------------------------------------------------
:step1
echo.
if not exist "%ROOT%\json files\boxes_calibrate_scanner.json" goto :calibrate
echo %HEAD%[step 1 of 3]%OFF%  the scanner is already measured - skipping.
echo                delete "json files\boxes_calibrate_scanner.json" to measure again.
goto :step2

:calibrate
echo %HEAD%[step 1 of 3]  measuring the scanner...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\calibrate_scanner.py" "%ROOTARG%" --max-images 200
REM 2 = measured, but some scans could not be read; 1 = no profile written
if errorlevel 2 goto :calwarn
if errorlevel 1 goto :calfail
goto :step2

:calwarn
echo %WARN%some scans could not be read; the measurement used the rest%OFF%
goto :step2

:calfail
echo.
echo %BAD%*** step 1 wrote no profile - see the messages above ***%OFF%
echo  measure more scans:
echo      python "%HERE%\calibrate_scanner.py" "%ROOTARG%" --max-images 400
echo  or, if you know the scanner's pitch, skip this step:
echo      python "%HERE%\detect_seams.py" "%ROOTARG%" --pitch N
goto :end

REM --- step 2 ---------------------------------------------------------------
:step2
echo.
echo %HEAD%[step 2 of 3]  locating the seams in each scan...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\detect_seams.py" "%ROOTARG%"
REM 2 = some scans could not be read and are left out; 1 = the step stopped
if errorlevel 2 goto :detwarn
if errorlevel 1 goto :detfail
goto :step3

:detwarn
echo %WARN%some scans could not be read and are left out of step 3%OFF%
goto :step3

:detfail
echo.
echo %BAD%*** step 2 stopped with an error - see the messages above ***%OFF%
goto :end

REM --- step 3: where the repaired scans go --------------------------------
REM Asked the same way as the scans folder above, and for the same reason
REM no unquoted path is echoed inside a ( ) block.
:step3
echo.
echo %HEAD%[step 3 of 3]  putting the dropped columns back%OFF%
echo --------------------------------------------------------------------
:askout
echo.
echo  Where should the output folder be created?
echo.
echo    Press Enter to use the scans folder:
echo        "%ROOT%"
echo.
echo    Or type/paste another folder.  Quotes are fine.
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
echo  Name of the folder the repaired scans go into?
echo.
echo    Press Enter for:  seam-repaired
echo.
set "OUTNAME="
set /p "OUTNAME=  folder name: "

if not defined OUTNAME set "OUTNAME=seam-repaired"
set "OUTNAME=%OUTNAME:"=%"
if not defined OUTNAME set "OUTNAME=seam-repaired"

REM a bare drive root already ends in a backslash
set "OUT=%OUTLOC%\%OUTNAME%"
if "%OUTLOC:~-1%"=="\" set "OUT=%OUTLOC%%OUTNAME%"

REM the scans folder itself would put the repairs over the scans
if /i not "%OUT%"=="%ROOT%" goto :notroot
echo.
echo  %WARN%that is the scans folder itself - choose another name%OFF%
goto :askname

:notroot
REM create it now, so a name Windows refuses (: * ? < > | and the like) is
REM caught here and asked again, not after the repair has started
if not exist "%OUT%\" mkdir "%OUT%" 2>nul
if exist "%OUT%\" goto :outok
echo.
echo  %WARN%could not create a folder with that name:%OFF%
echo      "%OUT%"
goto :askname

:outok
echo.
echo  output: "%OUT%"

REM --- step 3: optional verification --------------------------------------
echo.
echo  Verify the repairs afterwards?
echo.
echo    Verification rebuilds every repair in the manifest from its raw scan
echo    and checks that every scanned sample is still there, unchanged
echo    (SHA-256), and that the inserted columns are exactly what the
echo    recorded method gives.  Reading and taking the inserted columns out
echo    again are done by separate code; the fill of the new columns itself
echo    is shared with step 3.  It also checks that the bit depth, colour
echo    model, ICC colour profile, resolution (dpi) and the other TIFF tags
echo    were carried over unchanged.  It reads every scan and every repair
echo    once more, so step 3 takes roughly twice as long.
echo.
set "VER="
set "VERIFYARG="
set /p "VER=  verify? [y/N]: "
if defined VER set "VER=%VER:"=%"
if /i "%VER%"=="y"   set "VERIFYARG=--verify"
if /i "%VER%"=="yes" set "VERIFYARG=--verify"

echo.
if defined VERIFYARG echo  putting the dropped columns back, then verifying...
if not defined VERIFYARG echo  putting the dropped columns back, no verification...
echo --------------------------------------------------------------------
python "%HERE%\repair_seams.py" "%ROOTARG%" --out "%OUT%" %VERIFYARG%
REM 3 = verification failed, 2 = a scan could not be repaired, 1 = other error
if errorlevel 3 goto :verfail
if errorlevel 2 goto :repfail
if errorlevel 1 goto :othfail

echo.
echo %GOOD%==========================================================================
echo  done%OFF%
echo.
echo    "%ROOT%\json files\"
echo        boxes_calibrate_scanner.json, boxes_detect_seams.json,
echo        boxes_repair_seams.json
echo    "%OUT%\"
if defined VERIFYARG echo        the repaired scans, verified: your input to Stage 2
if not defined VERIFYARG echo        the repaired scans: your input to Stage 2, NOT verified
echo.
echo  next, Stage 2: when its run_pipeline.bat asks where the scans are,
echo  give it "%OUT%".
if defined VERIFYARG goto :tips
echo.
echo  to verify them later, run:
echo      python "%HERE%\repair_seams.py" "%ROOTARG%" --out "%OUT%" --verify
:tips
echo.
echo  Every step runs on several cores at once, as many as the free memory
echo  allows; add --workers 1 to any of them to run in a single process.
echo.
echo  to look at the seams step 2 found, run:
echo      python "%HERE%\detect_seams.py" "%ROOTARG%" --preview
echo ==========================================================================
goto :end

:verfail
echo.
echo %BAD%*** verification failed - do not use those files ***%OFF%
goto :end

:repfail
echo.
echo %BAD%*** some scans could not be repaired - see the messages above ***%OFF%
goto :end

:othfail
echo.
echo %BAD%*** step 3 stopped with an error - see the messages above ***%OFF%
goto :end

:end
echo.
pause
endlocal
