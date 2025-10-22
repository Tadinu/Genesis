# NOTE: MAKE SURE WE ARE IN A CONDA ENV AS INSTRUCTED BY https://spirimirror.github.io/libuipc-doc/build_install/linux
mkdir -p CMakeBuild
pushd CMakeBuild
cmake -S ../libuipc -DUIPC_BUILD_PYBIND=1 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j16
pushd python
pip install .
popd
popd

