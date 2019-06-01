@echo off

set CODE_FOLDER=C:\Users\%USERNAME%\iscat_gui\code
set ANACONDA_FOLDER=C:\Users\%USERNAME%\Anaconda3


call %ANACONDA_FOLDER%\Scripts\activate.bat %root%
call activate iscat

cd %CODE_FOLDER%

python gui_iscat.py
