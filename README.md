# Install requirements
1. Install GLUT on Ubuntu
```bash
sudo apt update  
sudo apt-get install freeglut3-dev -y
```
2. Install glfw3
```bash
sudo apt update  
sudo apt-get install libglfw3 -y
sudo apt-get install libglfw3-dev -y
```
3. Install glm
```bash
sudo apt update
sudo apt install libglm-dev -y
```
4. Install glew
```bash
sudo apt update
sudo apt-get install -y libglew-dev
```
5. Install CMake
```bash
sudo apt update
sudo apt install make cmake -y
```

# Run the code
```bash
git submodule update --init --recursive
```
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
../bin/RALab3
```


# Simulation
Particle sources are hardcoded on start, but can be deleted with `Clear Particle Sources` and manually added with right click anywhere in simulation.
- to start simulation press `Start particle sources`, this will start creating objects
- parameters that can be changed
- 1. Particle size - size of the particle (only effects new particles that will be created from sources, to change existing particles check `Change global particle size`)
- 2. Particle segments - particle resoulution (higher resolution means lower performance, doesn't effect physics)
- 3. Grid size - change number of grid cells (more grid cells usually means higher performance, but to many can decrese performance)
::warning:: Be careful when decreasing grid size becuase on lower grid sizes simulation becomes unstable ::warning::
- 4. Thread count - change number of threads that will solve physics (more is better until maximum number of threads on cpu is reached)
- 5. Physics substeps - changes how accurate the physiscs simulation will be (drastically changes performance, more steps means better results for physiscs simulation but worse for performance)
- 6. Gravity slider - change gravity (setting gravity to 0 enables simulation to be stable even on 1 physiscs substep)
- 7. Start particle sources - all sources (red rectangles) will start creating particles
- 8. Stop particle sources - all sources will stop creating particles
- 9. Clear particle sources - remove all particle sources, and stop all particle creation
- 10. Show grid - displays/hides grid (doesn't effect physiscs)
- 11. Render game objects - displays/hides all particles (doesn't effect physiscs, but increases performance)
- 12. Show particle grid color - color of the particle is based on cell that particle is in (doesn't effect physiscs)
- 13. Change global particle size - if togged changing particle size with `particle size` slider will effect all particles
- 14. Clear particles - removes all existing particles, doesn't stop particle sources from creating new ones

At the bottom of the GUI some useful debugging infromation can be found.
- FPS
- mouse position, moouse grid coordinates
- ratio between cell and particle size (if ratio is to big simulation can become unstable)
- cell capacity - number of particles each cell can hold (if there are more particles in the cell simulation becomes unstable)

# Creating Graphs
You will need python venv with matplotlib installed
Make sure to run this code from base directory.
```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib
mkdir graphs
```
To create graph from last simulation run
```bash
python3 graph_data.py
```
