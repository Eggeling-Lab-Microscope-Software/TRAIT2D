@echo off

set ANACONDA_FOLDER=C:\Users\%USERNAME%\Anaconda3

call %ANACONDA_FOLDER%\Scripts\activate.bat %root%
call conda env create -f environment.yml

ECHO Environment installed

pip install -e .

ECHO Installations done 
