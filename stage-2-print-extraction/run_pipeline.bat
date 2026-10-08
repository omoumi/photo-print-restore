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
REM  run_pipeline.bat  --  STAGE 2: runs steps 1 to 4 in order
REM
REM  STAGE 2 turns a folder of flatbed scans (Stage 1's repaired scans)
REM  into straightened, cropped image files:
REM
REM      detect_borders.py        step 1 of 4   find the print
REM      suggest_orientation.py   step 2 of 4   suggest the way up
REM      review.py                step 3 of 4   approve the box, turn and way up
REM      deskew_crop.py           step 4 of 4   straighten, crop and orient
REM      common.py                shared code, not a step
REM      parallel.py              shared worker-count policy, not a step
REM      anchor_report.py         diagnostic for step 1, not a step
REM
REM  Keep this file beside the six scripts above (anchor_report.py is not
REM  needed here).  Run it from a command prompt or just double-click it; it
REM  asks where the scans are.
REM
REM  The scans do not have to be beside the scripts.  Press Enter at the
REM  prompt if they are, or give any other folder.
REM
REM  Step 1 measures only the scans it has not measured yet (--new-only), so
REM  scans added to the folder later are picked up and a folder already
REM  measured passes in a moment.  Delete json files\boxes_detect_borders.json
REM  to measure every scan again.
REM
REM  Step 2, suggest_orientation.py, suggests the way up of each print from
REM  the faces in it; review.py then opens those scans already turned, and
REM  Enter confirms as usual.  A print without a face gets no suggestion:
REM  turn it by hand in review.py.  It is asked every time, like the flip,
REM  and Enter means yes.  When suggestions already exist it offers to keep
REM  them, make them again, or not use them this time; keeping them needs
REM  nothing more.  Making them needs onnxruntime (pip install onnxruntime),
REM  which is checked first, and downloads its face model once, into
REM  models\ beside the scripts.
REM
REM  Step 3 opens a window and waits for you. This is not an unattended run.
REM  It asks first whether the whole archive is upside down, because --flip
REM  is a state rather than a toggle: not passing it turns the flip off.
REM  Before step 4 it asks where to create the output folder and what to
REM  call it (Enter keeps ROOT\deskewed-cropped; never the scans folder
REM  itself), and whether to verify.
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
echo  STAGE 2  --  find, straighten and crop the prints
echo ==========================================================================%OFF%

for %%F in (detect_borders.py suggest_orientation.py review.py deskew_crop.py common.py parallel.py) do (
    if not exist "%HERE%\%%F" (
        echo.
        echo %BAD%%%F is not beside this file.  All six scripts must sit together.%OFF%
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
python -c "import numpy, tifffile, cv2, pyqtgraph" 1>nul 2>nul
if not errorlevel 1 goto :step1
echo.
echo  %BAD%a package the steps need is not installed:%OFF%
echo      pip install numpy tifffile opencv-python imagecodecs pyqtgraph PyQt5 psutil
goto :end

REM --- step 1 ---------------------------------------------------------------
:step1
echo.
echo %HEAD%[step 1 of 4]  finding the print in each scan not measured yet...%OFF%
echo --------------------------------------------------------------------
echo  delete "json files\boxes_detect_borders.json" to measure every scan again.
python "%HERE%\detect_borders.py" "%ROOTARG%" --new-only
REM 2 = some scans could not be processed and are left out; 1 = it stopped
if errorlevel 2 goto :detwarn
if errorlevel 1 goto :detfail
goto :step2

:detwarn
echo %WARN%some scans could not be processed and are left out%OFF%
goto :step2

:detfail
echo.
echo %BAD%*** step 1 stopped with an error - see the messages above ***%OFF%
goto :end

REM --- step 2: suggest the way up --------------------------------------------
:step2
echo.
echo %HEAD%[step 2 of 4]  suggest the way up of each print%OFF%
echo --------------------------------------------------------------------
echo  The faces in each print suggest its way up.  Nothing is applied:
echo  review.py opens those scans turned, and Enter confirms as usual.
echo  A print without a face gets no suggestion - turn it by hand there.
echo.
set "NOSUG="
set "SUGGEST="
if exist "%ROOT%\json files\orientation_suggestions.json" goto :sugexist
python -c "import onnxruntime" 1>nul 2>nul
if errorlevel 1 goto :noonnx
set /p "SUGGEST=  suggest the way up? [Y/n]: "
if defined SUGGEST set "SUGGEST=%SUGGEST:"=%"
if /i "%SUGGEST%"=="n"  goto :sugoff
if /i "%SUGGEST%"=="no" goto :sugoff
goto :sugrun

:sugexist
echo  Suggestions from an earlier run already exist.
echo.
echo    Enter or K   keep them and use them
echo    R            make them again
echo    N            do not use them this time
echo.
set /p "SUGGEST=  [K/r/n]: "
if defined SUGGEST set "SUGGEST=%SUGGEST:"=%"
if /i "%SUGGEST%"=="n"  goto :sugoff
if /i "%SUGGEST%"=="no" goto :sugoff
if /i "%SUGGEST%"=="r"  goto :sugremake
echo  keeping the existing suggestions.
goto :step3

:sugremake
python -c "import onnxruntime" 1>nul 2>nul
if errorlevel 1 goto :noonnx

:sugrun
python "%HERE%\suggest_orientation.py" "%ROOTARG%"
REM 2 = some scans could not be processed and got no suggestion; 1 = none
if errorlevel 2 goto :sugwarn
if not errorlevel 1 goto :step3
echo %WARN%the suggestions failed - continuing without them%OFF%
goto :sugoff

:sugwarn
echo %WARN%some scans could not be processed and got no suggestion%OFF%
goto :step3

:noonnx
echo  %WARN%onnxruntime is not installed%OFF% - this step needs it:
echo      pip install onnxruntime
echo.
set /p "SUGGEST=  continue without suggestions? [Y/n]: "
if defined SUGGEST set "SUGGEST=%SUGGEST:"=%"
if /i "%SUGGEST%"=="n"  goto :end
if /i "%SUGGEST%"=="no" goto :end

:sugoff
REM review.py would otherwise still read a suggestions file left on disk
set "NOSUG=--no-suggestions"
echo  no suggestions this time.

:step3
REM --- step 3 ---------------------------------------------------------------
echo.
echo %HEAD%[step 3 of 4]  the review window%OFF%
echo --------------------------------------------------------------------
echo  Is the whole archive upside down?
echo.
echo    This SETS the archive-wide flip, it does not toggle it: answering
echo    n turns the flip off even if an earlier run turned it on.  Inside
echo    the window, V still flips one scan at a time on top of this.
echo.
set "FLIP="
set "REVIEWFLIP="
set /p "FLIP=  flip the whole archive? [y/N]: "
if defined FLIP set "FLIP=%FLIP:"=%"
if /i "%FLIP%"=="y"   set "REVIEWFLIP=--flip"
if /i "%FLIP%"=="yes" set "REVIEWFLIP=--flip"
if defined REVIEWFLIP (
    echo.
    echo  opening the review window, archive flip ON...
) else (
    echo.
    echo  opening the review window, archive flip off...
)
echo.
echo  Enter confirms a scan, S skips it, Esc saves and quits.
echo  Ctrl+Enter confirms this scan and every later one not yet decided,
echo  each as it opens, then closes the window - it asks first.
echo  Your decisions are saved after every confirmation, so quitting part
echo  way through is safe - the next run opens where you left off.
echo.
python "%HERE%\review.py" "%ROOTARG%" %REVIEWFLIP% %NOSUG%
if errorlevel 1 goto :revfail

REM --- step 4: where the finished images go -----------------------------
REM Asked the same way as the scans folder above, and for the same reason
REM no unquoted path is echoed inside a ( ) block.
echo.
echo %HEAD%[step 4 of 4]  straightening, cropping and orienting%OFF%
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
echo  Name of the folder the finished images go into?
echo.
echo    Press Enter for:  deskewed-cropped
echo.
set "OUTNAME="
set /p "OUTNAME=  folder name: "

if not defined OUTNAME set "OUTNAME=deskewed-cropped"
set "OUTNAME=%OUTNAME:"=%"
if not defined OUTNAME set "OUTNAME=deskewed-cropped"

REM a bare drive root already ends in a backslash
set "OUT=%OUTLOC%\%OUTNAME%"
if "%OUTLOC:~-1%"=="\" set "OUT=%OUTLOC%%OUTNAME%"

REM the scans folder itself: the crops are named after the scans
if /i not "%OUT%"=="%ROOT%" goto :notroot
echo.
echo  %WARN%that is the scans folder itself - choose another name%OFF%
goto :askname

:notroot
REM create it now, so a name Windows refuses (: * ? < > | and the like) is
REM caught here and asked again, not after the crop has started
if not exist "%OUT%\" mkdir "%OUT%" 2>nul
if exist "%OUT%\" goto :outok
echo.
echo  %WARN%could not create a folder with that name:%OFF%
echo      "%OUT%"
goto :askname

:outok
echo.
echo  output: "%OUT%"

REM --- step 4: optional verification ------------------------------------
echo.
echo  Verify the crops afterwards?
echo.
echo    Verification rebuilds every crop in the manifest from its raw scan
echo    and checks that the two are identical pixel for pixel (SHA-256).
echo    Reading, turning and cropping are rebuilt by separate code; the
echo    Lanczos resampling itself is shared with step 4.  It also checks
echo    that the bit depth, colour model, ICC colour profile, resolution
echo    (dpi) and the other TIFF tags were carried over unchanged.  A pass
echo    proves every crop is exactly what the recorded flip, turn,
echo    straightening and crop produce from its raw scan.  A print too
echo    straight to need straightening keeps every sample unchanged.  It
echo    costs about one more rotation per scan, so step 4 takes roughly
echo    twice as long.
echo.
set "VER="
set "VERIFYARG="
set /p "VER=  verify? [y/N]: "
if defined VER set "VER=%VER:"=%"
if /i "%VER%"=="y"   set "VERIFYARG=--verify"
if /i "%VER%"=="yes" set "VERIFYARG=--verify"

echo.
if defined VERIFYARG echo  straightening, cropping, orienting, then verifying...
if not defined VERIFYARG echo  straightening, cropping and orienting, no verification...
echo --------------------------------------------------------------------
python "%HERE%\deskew_crop.py" "%ROOTARG%" --out "%OUT%" %VERIFYARG%
REM 3 = verification failed, 2 = a scan could not be cropped, 1 = other error
if errorlevel 3 goto :verfail
if errorlevel 2 goto :cropfail
if errorlevel 1 goto :othfail

echo.
echo %GOOD%==========================================================================
echo  done%OFF%
echo.
echo    "%ROOT%\json files\"
echo        boxes_detect_borders.json, boxes_review.json, boxes_deskew_crop.json
if exist "%ROOT%\json files\orientation_suggestions.json" echo        orientation_suggestions.json
echo    "%OUT%\"
if defined VERIFYARG echo        the straightened crops, verified: the input of Stage 3
if not defined VERIFYARG echo        the straightened crops: the input of Stage 3, NOT verified
echo.
echo  next, Stage 3: when its run_pipeline.bat asks where Stage 2's crops are,
echo  give it "%OUT%".
if defined VERIFYARG goto :tips
echo.
echo  to verify them later, run:
echo      python "%HERE%\deskew_crop.py" "%ROOTARG%" --out "%OUT%" --verify
:tips
echo.
echo  Steps 1, 2 and 4 run on several cores at once, as many as the free
echo  memory allows; add --workers 1 to any of them to run in a single
echo  process.
echo.
echo  to see what step 1 found as preview pictures (this measures every
echo  scan again), run:
echo      python "%HERE%\detect_borders.py" "%ROOTARG%" --review
echo ==========================================================================
goto :end

:revfail
echo.
echo %BAD%*** step 3 stopped with an error - see the messages above ***%OFF%
goto :end

:verfail
echo.
echo %BAD%*** verification failed - do not use those files ***%OFF%
goto :end

:cropfail
echo.
echo %BAD%*** some scans could not be cropped - see the messages above ***%OFF%
goto :end

:othfail
echo.
echo %BAD%*** step 4 stopped with an error - see the messages above ***%OFF%
goto :end

:end
echo.
pause
endlocal
