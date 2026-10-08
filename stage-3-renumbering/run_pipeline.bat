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
REM  run_pipeline.bat  --  STAGE 3: runs the step, then applies its plan
REM
REM  STAGE 3 puts the prints that belong together side by side, from the
REM  straightened, cropped files Stage 2 writes:
REM
REM      renumber_by_similarity.py  step 1 of 1   put related prints side by side
REM      common.py                  shared code, not a step
REM      parallel.py                shared worker-count policy, not a step
REM
REM  Keep this file beside the three scripts above.  Run it from a command
REM  prompt or just double-click it; it asks where Stage 2's crops are.
REM
REM  The crops do not have to be beside the scripts.  Press Enter at the
REM  prompt if they are, or give any other folder.
REM
REM  It first analyses the crops and writes the plan: contact sheets and a
REM  summary in organize_review\.  Nothing is renamed yet.  Then it waits
REM  for you: look at the sheets, put corrections in
REM  organize_overrides.txt beside the crops and analyse again (fast: the
REM  measurements are cached), or apply the plan.
REM
REM  Before applying it asks where to create the output folder and what to
REM  call it (Enter keeps ROOT\organized).  Applying links each crop into
REM  that folder under its new name; the crops keep their own names.
REM  When an earlier plan was applied, it offers to undo that first.
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
echo  STAGE 3  --  put related prints side by side
echo ==========================================================================%OFF%

for %%F in (renumber_by_similarity.py common.py parallel.py) do (
    if not exist "%HERE%\%%F" (
        echo.
        echo %BAD%%%F is not beside this file.  All three scripts must sit together.%OFF%
        goto :end
    )
)

REM --- ask for the folder holding Stage 2's crops ---------------------------
:askroot
echo.
echo  Where are Stage 2's crops?
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
set "REVIEW=%ROOT%\organize_review"
if "%ROOT:~-1%"=="\" set "REVIEW=%ROOT%organize_review"
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
python -c "import numpy, tifffile, cv2, scipy" 1>nul 2>nul
if not errorlevel 1 goto :aicheck
echo.
echo  %BAD%a package the step needs is not installed:%OFF%
echo      pip install numpy tifffile opencv-python scipy imagecodecs psutil
goto :end

REM --- the AI model needs torch and transformers ----------------------------
:aicheck
python -c "import torch, transformers" 1>nul 2>nul
if not errorlevel 1 goto :analyse
echo.
echo  %WARN%torch or transformers is not installed%OFF% - the grouping by picture
echo  content needs them:
echo      pip install torch transformers
echo.
echo  Without them the prints are grouped by size, cast and corners only.
echo.
set "GOON="
set /p "GOON=  continue without the AI? [y/N]: "
if defined GOON set "GOON=%GOON:"=%"
if /i "%GOON%"=="y"   goto :analyse
if /i "%GOON%"=="yes" goto :analyse
goto :end

REM --- step 1: analyse --------------------------------------------------------
:analyse
echo.
echo %HEAD%[step 1 of 1]  analysing the prints and writing the plan...%OFF%
echo --------------------------------------------------------------------
python "%HERE%\renumber_by_similarity.py" "%ROOTARG%"
REM 2 = some prints could not be read and are filed at the end; 1 = it stopped
if errorlevel 2 goto :anawarn
if errorlevel 1 goto :anafail
goto :review

:anawarn
echo %WARN%some prints could not be read - they are filed at the end of the plan%OFF%
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
echo  Nothing is renamed yet.  The plan is in:
echo      "%REVIEW%"
echo          sheet_NN.jpg           contact sheets of the groups
echo          organize_summary.txt   the groups as a tree, with notes
echo.
echo  To correct a print, add a line to organize_overrides.txt beside the
echo  crops, one print per line, then analyse again:
echo      raw0012.tif back        raw0020.tif bw
echo      raw0013.tif front       raw0021.tif colour
echo.
echo    Enter or A   apply the plan
echo    R            analyse again, after editing organize_overrides.txt
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
echo      python "%HERE%\renumber_by_similarity.py" "%ROOTARG%" --apply
goto :end

REM --- an earlier plan that was applied ------------------------------------
:apply
if not exist "%REVIEW%\organize_log.csv" goto :askout
echo.
echo  An earlier plan was applied already.  Its files must be removed
echo  before this plan can be applied.
echo.
set "UNDO="
set /p "UNDO=  remove them now? [Y/n]: "
if defined UNDO set "UNDO=%UNDO:"=%"
if /i "%UNDO%"=="n"  goto :undooff
if /i "%UNDO%"=="no" goto :undooff
python "%HERE%\renumber_by_similarity.py" "%ROOTARG%" --undo
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

REM --- where the prints under their new names go ---------------------------
REM Asked the same way as the crops folder above, and for the same reason
REM no unquoted path is echoed inside a ( ) block.
:askout
echo.
echo %HEAD%applying the plan%OFF%
echo --------------------------------------------------------------------
echo.
echo  Where should the output folder be created?
echo.
echo    Press Enter to use the crops folder:
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
echo  Name of the folder the prints go into, under their new names?
echo.
echo    Press Enter for:  organized
echo.
set "OUTNAME="
set /p "OUTNAME=  folder name: "

if not defined OUTNAME set "OUTNAME=organized"
set "OUTNAME=%OUTNAME:"=%"
if not defined OUTNAME set "OUTNAME=organized"

REM a bare drive root already ends in a backslash
set "OUT=%OUTLOC%\%OUTNAME%"
if "%OUTLOC:~-1%"=="\" set "OUT=%OUTLOC%%OUTNAME%"

REM the crops folder itself would mix old and new names
if /i not "%OUT%"=="%ROOT%" goto :notroot
echo.
echo  %WARN%that is the crops folder itself - choose another name%OFF%
goto :askname

:notroot
REM create it now, so a name Windows refuses (: * ? < > | and the like) is
REM caught here and asked again, not after the linking has started
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
echo  linking the prints under their new names...
echo --------------------------------------------------------------------
python "%HERE%\renumber_by_similarity.py" "%ROOTARG%" --apply --out "%OUT%"
if errorlevel 1 goto :applyfail

echo.
echo %GOOD%==========================================================================
echo  done%OFF%
echo.
echo    "%REVIEW%\"
echo        organize_plan.csv, organize_summary.txt, sheet_NN.jpg,
echo        organize_log.csv
echo    "%OUT%\"
echo        the prints under their new names
echo.
echo  A linked print and its crop are one file on disk: edit neither in
echo  place.  To remove what was just made, run:
echo      python "%HERE%\renumber_by_similarity.py" "%ROOTARG%" --undo
echo.
echo  next, Stage 4: run VueScan on the prints in "%OUT%"
echo  with the settings in Stage 4's README.
echo.
echo  Measuring runs on several cores at once, as many as the free memory
echo  allows; add --workers 1 to run it in a single process.
echo ==========================================================================
goto :end

:applyfail
echo.
echo %BAD%*** applying the plan stopped - see the messages above ***%OFF%
echo  If the crops changed since the analysis, run this file again.  If it
echo  stopped part way, the files written so far are removed with:
echo      python "%HERE%\renumber_by_similarity.py" "%ROOTARG%" --undo
goto :end

:end
echo.
pause
endlocal
