# lab_glcuda
Some code examples for OpenGL [+CUDA interoperability]

These code examples show how to use OpenGL/GLSL/GLFW/CUDA within a cmake project.

# Usage
Clone the repository including the submodules glm and glfw.
```
git clone --recursive https://github.com/tdd11235813/lab_glcuda.git
```
Then run cmake+make from your build directory:
```
mkdir release && cd release
cmake ..
make -j 4
```
If you run the applications on a laptop with an integrated and Nvidia GPU, use `optirun`, to ensure CUDA and OpenGL interoperability:
```
optirun ./lab03
```

# Applications
- lab00: basic setup including animated background color
- lab01: rotating triangle
- lab02: like lab01, but here you should implement a smooth triangle coloring (see lab02-solution)
- lab03: mouse-attracting particles using CUDA (todo: use flake texture for lab03-flake)
- lab04: animated heightfield using CUDA
- lab05: animated colored heightfield using CUDA and texture

