make clean

if [ -d "CMakeFiles" ]; then
    rm -rf CMakeFiles
    rm ./cmake_install.cmake
    rm ./CMakeCache.txt
    rm Makefile
fi

# Default:lib
# cmake ..
cmake -DCMAKE_BUILD_TYPE=Debug_test ..

make