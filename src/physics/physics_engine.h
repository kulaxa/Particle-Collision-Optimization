#include <vector>
#include <glm/glm.hpp>
#include <atomic>
#include <omp.h>
#include <iostream>


// extern "C" void cuda_solve_collisions(float *currPositionsX, float *currPositionsY, float *radii,
//                                       float *lastPositionsX, float *lastPositionsY, float *accelerationX,
//                                       float *accelerationY,  float xGravity,  float yGravity,
//                                       float dt, int substeps, int numElements);

class PhysicsEngine {
public:

    PhysicsEngine(glm::vec2 d, int gridSize, int sub_steps) : sub_steps(sub_steps){
        gravity = d;
        gridWidth = gridSize;
        gridHeight = gridSize;
    }

    ~PhysicsEngine() {}
    void update();

    void addGameObject(int index, float xPos, float yPos, float xAcc, float yAcc, float radius, glm::vec3 color){
        gameObjectsXPositions[index] = xPos;
        gameObjectsYPositions[index] = yPos;
        gameObjectsXLastPosition[index] = xPos;
        gameObjectsYLastPosition[index] = yPos;
        gameObjectsXAcceleration[index] = xAcc;
        gameObjectsYAcceleration[index] = yAcc;
        gameObjectsRadius[index] = radius;
        numberOfGameObjects++;
        colors[index * 3] = color.x;
        colors[index * 3 + 1] = color.y;
        colors[index * 3 + 2] = color.z;
    }

    glm::ivec2 getGridSize() { return glm::ivec2(gridHeight, gridWidth); }
    glm::ivec2 mapWorldToGrid(const glm::vec2 &worldCoord, glm::ivec2 gridSize);

    float getCellSize() {return 2.0f / gridHeight; }
    void setThreadCount(int threadCount) { this->threadCount = threadCount; }
    void update(float dt);
    void setSubSteps(int sub_steps) { this->sub_steps = sub_steps; }


    int getCollisionCount() { return collisionCount; }
    int getCollisionTestCount() { return collisionTestCount; }
    int getNumberOfGameObjects() { return numberOfGameObjects; }
    double getGameObjectXPosition(int index) { return gameObjectsXPositions[index]; }
    double getGameObjectYPosition(int index) { return gameObjectsYPositions[index]; }
    double getGameObjectRadius(int index) { return gameObjectsRadius[index]; }
    double setGameObjectXPosition(int index, double x) { gameObjectsXPositions[index] = x; }
    void setGridSize(int width, int height) { gridWidth = width; gridHeight = height; }

    void clearGameObjects(){ numberOfGameObjects = 0;}
    void setUseGPU(bool useGPU) { this->useGPU = useGPU; }
    void getColor(int index, float &r, float &g, float &b) {
        r = colors[index * 3];
        g = colors[index * 3 + 1];
        b = colors[index * 3 + 2];
    }

    void setOptimizeCPU(bool optimizeCPU) { this->optimizeCPU = optimizeCPU; }

private:
    const static int maxNumberOfGameObjects = 400000;
    const static int maxGridSize = 400 * 400;
    const static int maxNumberOfGameObjectsPerCell = 30;
    glm::vec2 gravity;  // Gravity force
    uint32_t numberOfGameObjects = 0;
    float gameObjectsXPositions[maxNumberOfGameObjects];
    float gameObjectsXPositionsTmp[maxNumberOfGameObjects];
    float gameObjectsYPositions[maxNumberOfGameObjects];
    float gameObjectsYPositionsTmp[maxNumberOfGameObjects];
    float gameObjectsRadius[maxNumberOfGameObjects];
    float gameObjectsXLastPosition[maxNumberOfGameObjects];
    float gameObjectsYLastPosition[maxNumberOfGameObjects];
    float gameObjectsXAcceleration[maxNumberOfGameObjects];
    float gameObjectsYAcceleration[maxNumberOfGameObjects];
    bool gameObjectsCollided[maxNumberOfGameObjects];
    float colors[maxNumberOfGameObjects * 3];
    uint32_t grid_centers[maxGridSize * maxNumberOfGameObjectsPerCell];
    float debug[maxNumberOfGameObjects];
    uint32_t gridWidth = 200;
    uint32_t gridHeight = 200;
    int threadCount = 1;
    uint32_t sub_steps = 8;
    std::atomic<int>  collisionCount = 0;
    std::atomic<int>  collisionTestCount = 0;
    bool useGPU = false;
    bool optimizeCPU = false;

    void solveContact(uint32_t atom_1_idx, uint32_t atom_2_idx);

    void updateObject(int index, float dt);

    void resolveCollisionsWithWalls(int index);

    void update(int index, float dt);


    void generateGridParticleCenters(uint32_t *gridParticleCenter, const float *gameObjectsXPositions,
                                     const float *gameObjectsYPositions, uint32_t numberOfGameObjects,
                                     uint32_t gridWidth,
                                     uint32_t gridHeight);

    void solveCollisionsNotOptimized();

    void solveCollisionsOptimized();
};


