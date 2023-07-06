mkdir build
cd build
cmake -G "Visual Studio 16 2019"  -DOpenCV_DIR="D:\tools\OpenCV\opencv\build" ..
cmake --build . --config Release -j8
pause
