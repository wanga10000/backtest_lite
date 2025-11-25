Remove-Item -R -Force -Path "build" -ErrorAction SilentlyContinue
Remove-Item -R -Path "./c_utils/*.c" -ErrorAction SilentlyContinue
Remove-Item -R -Path "./c_utils/*.pyd" -ErrorAction SilentlyContinue
python setup.py build_ext --inplace
Get-ChildItem -Path . -Filter *.cpp | Move-Item -Destination ./c_utils -Force
Get-ChildItem -Path . -Filter *.pyd | Move-Item -Destination ./c_utils -Force
Remove-Item -R -Force -Path "build" -ErrorAction SilentlyContinue