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
REM  run_pipeline.bat  --  STAGE 4: runs the step, then applies its plan
REM
REM  STAGE 4 corrects the colour of the prints: a faded colour print gets
REM  its colour back, a B&W print (toned, yellowed or not) becomes neutral
REM  grey:
REM
REM      VueScan                  step 1 of 2   levels and curves
REM      restore_colors.py        step 2 of 2   colour and B&W correction
REM      common.py                shared code, not a step
REM      parallel.py              shared worker-count policy, not a step
REM
REM  Step 1 runs in VueScan, with the settings in README.md; this file runs
REM  step 2 on the TIFFs VueScan wrote.
REM
REM  Keep this file beside the three scripts above.  Run it from a command
REM  prompt or just double-click it; it asks where VueScan's TIFFs are.
REM
REM  The prints do not have to be beside the scripts.  Press Enter at the
REM  prompt if they are, or give any other folder.
REM
REM  It first analyses the prints and writes the plan: every print's class
REM  (back, colour or B&W) and contact sheets, before and after, in
REM  restore_review\.  Nothing is written yet.  Then it waits for you: look
REM  at the sheets, put corrections in restore_overrides.txt beside the
REM  prints and analyse again (fast: the measurements are cached), or apply
REM  the plan.
REM
REM  Before applying it asks where to create the output folder and what to
REM  call it (Enter keeps ROOT\restored).  Applying writes the corrected
REM  prints there under their own names.  When an earlier plan was applied,
REM  it offers to undo that first.
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
echo  STAGE 4  --  colour and B^&W correction
echo ==========================================================================%OFF%

for %%F in (restore_colors.py common.py parallel.py) do (
    if not exist "%HERE%\%%F" (
        echo.
        echo %BAD%%%F is not beside this file.  All three scripts must sit together.%OFF%
        goto :end
    )
)

REM --- ask for the folder holding the prints ---------------------------------
:askroot
echo.
echo  Where are the TIFFs VueScan wrote (step 1)?
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
set "REVIEW=%ROOT%\restore_review"
if "%ROOT:~-1%"=="\" set "REVIEW=%ROOT%restore_review"
echo.
echo  using: "%ROOT%"

REM --- Python itself, then the packages the step cannot run without ---------
python --version 1>nul 2>nul
if not errorlevel 1 goto :packages
echo.
echo  %BAD%Python was not found.%OFF%  Install Python 3 from python.org, tick
echo  "Add python.exe to PATH" during the install, then run this file again.
goto :end

:packages
python -c "import numpy, scipy, tifffile, cv2" 1>nul 2>nul
if not errorlevel 1 goto :aicheck
echo.
echo  %BAD%a package the step needs is not installed:%OFF%
echo      pip install numpy scipy tifffile opencv-python imagecodecs psutil
goto :end

REM --- the AI model that tells backs needs torch and transformers ---------
:aicheck
python -c "import torch, transformers" 1>nul 2>nul
if not errorlevel 1 goto :analyse
echo.
echo  %WARN%torch or transformers is not installed%OFF% - telling backs from
echo  fronts needs them:
echo      pip install torch transformers
echo.
echo  Without them backs are told by the pixel test only, which misses some.
echo.
set "GOON="
set /p "GOON=  continue without the AI? [y/N]: "
if defined GOON set "GOON=%GOON:"=%"
if /i "%GOON%"=="y"   goto :analyse
if /i "%GOON%"=="yes" goto :analyse
goto :end

REM --- step 2: analyse --------------------------------------------------------
:analyse
echo.
echo %HEAD%[step 2 of 2]  analysing the prints and writing the plan...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\restore_colors.py" "%ROOTARG%"
REM 2 = some prints could not be read and are left out; 1 = it stopped
if errorlevel 2 goto :anawarn
if errorlevel 1 goto :anafail
goto :review

:anawarn
echo %WARN%some prints could not be read - they are left out of the plan%OFF%
goto :review

:anafail
echo.
echo %BAD%*** the analysis stopped with an error - see the messages above ***%OFF%
goto :end

REM --- look at the plan -------------------------------------------------------
:review
echo.
echo %HEAD%the plan%OFF%
echo --------------------------------------------------------------------
echo  Nothing is written yet.  The plan is in:
echo      "%REVIEW%"
echo          restore_sheet_NN.jpg   every print before and after, with its
echo                                 class; red when it is to be checked
echo          restore_plan.csv       every print's class, tests and notes
echo.
echo  To correct a print, add a line to restore_overrides.txt beside the
echo  prints, one print per line, then analyse again:
echo      raw0012 bw              a B^&W print, whatever the tests say
echo      raw0013 colour          a colour print, whatever the tests say
echo      raw0014 back            a back: copied as it is
echo      raw0015 front           not a back
echo      raw0020 skip            not written
echo.
echo    Enter or A   apply the plan
echo    R            analyse again, after editing restore_overrides.txt
echo    O            open the plan's folder
echo    Q            stop here; the plan stays for later
echo.
set "ACT="
set /p "ACT=  [A/r/o/q]: "
if defined ACT set "ACT=%ACT:"=%"
if /i "%ACT%"=="r" goto :analyse
if /i "%ACT%"=="q" goto :stopped
if /i "%ACT%"=="o" goto :openrev
goto :apply

:openrev
start "" "%REVIEW%"
goto :review

:stopped
echo.
echo  stopped.  To apply the plan later, run this file again, or:
echo      python "%HERE%\restore_colors.py" "%ROOTARG%" --apply
goto :end

REM --- an earlier plan that was applied ------------------------------------
:apply
if not exist "%REVIEW%\restore_log.csv" goto :askout
echo.
echo  An earlier plan was applied already.  Its files must be removed
echo  before this plan can be applied.
echo.
set "UNDO="
set /p "UNDO=  remove them now? [Y/n]: "
if defined UNDO set "UNDO=%UNDO:"=%"
if /i "%UNDO%"=="n"  goto :undooff
if /i "%UNDO%"=="no" goto :undooff
python "%HERE%\restore_colors.py" "%ROOTARG%" --undo
REM 2 = some files changed since and were left in place; 1 = it stopped
if errorlevel 2 goto :undowarn
if errorlevel 1 goto :undofail
goto :askout

:undowarn
echo %WARN%some files had changed since and were left in place - see above%OFF%
goto :askout

:undooff
echo.
echo  nothing applied.  The earlier files are where they were.
goto :end

:undofail
echo.
echo %BAD%*** removing the earlier files failed - see the messages above ***%OFF%
goto :end

REM --- where the corrected prints go ----------------------------------------
REM Asked the same way as the prints folder above, and for the same reason
REM no unquoted path is echoed inside a ( ) block.
:askout
echo.
echo %HEAD%applying the plan%OFF%
echo --------------------------------------------------------------------
echo.
echo  Where should the output folder be created?
echo.
echo    Press Enter to use the prints folder:
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
echo  Name of the folder the corrected prints go into?
echo.
echo    Press Enter for:  restored
echo.
set "OUTNAME="
set /p "OUTNAME=  folder name: "

if not defined OUTNAME set "OUTNAME=restored"
set "OUTNAME=%OUTNAME:"=%"
if not defined OUTNAME set "OUTNAME=restored"

REM a bare drive root already ends in a backslash
set "OUT=%OUTLOC%\%OUTNAME%"
if "%OUTLOC:~-1%"=="\" set "OUT=%OUTLOC%%OUTNAME%"

REM the prints folder itself would put the corrected prints over the prints
if /i not "%OUT%"=="%ROOT%" goto :notroot
echo.
echo  %WARN%that is the prints folder itself - choose another name%OFF%
goto :askname

:notroot
REM create it now, so a name Windows refuses (: * ? < > | and the like) is
REM caught here and asked again, not after the writing has started
if not exist "%OUT%\" mkdir "%OUT%" 2>nul
if exist "%OUT%\" goto :outok
echo.
echo  %WARN%could not create a folder with that name:%OFF%
echo      "%OUT%"
goto :askname

:outok
echo.
echo  output: "%OUT%"
echo.
echo  writing the corrected prints...
echo --------------------------------------------------------------------
python "%HERE%\restore_colors.py" "%ROOTARG%" --apply --out "%OUT%"
REM 2 = some prints failed and the others were written; 1 = nothing written
if errorlevel 2 goto :applywarn
if errorlevel 1 goto :applyfail
goto :done

:applywarn
echo %WARN%some prints could not be written - see the messages above%OFF%

:done
echo.
echo %GOOD%==========================================================================
echo  done%OFF%
echo.
echo    "%REVIEW%\"
echo        restore_plan.csv, restore_sheet_NN.jpg, restore_log.csv
echo    "%OUT%\"
echo        the corrected prints, under their own names
echo.
echo  To remove what was just written, run:
echo      python "%HERE%\restore_colors.py" "%ROOTARG%" --undo
echo.
echo  next, Stage 5: give its run_pipeline.bat the folder
echo      "%OUT%"
echo.
echo  Measuring runs on several cores at once, as many as the free memory
echo  allows; add --workers 1 to run it in a single process.
echo ==========================================================================
goto :end

:applyfail
echo.
echo %BAD%*** nothing was applied - see the messages above ***%OFF%
echo  If the prints changed since the analysis, run this file again.
goto :end

:end
echo.
pause
endlocal
