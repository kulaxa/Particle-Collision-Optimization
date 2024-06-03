#include <valarray>
#include <glm/vec2.hpp>
#include <cstring>
#include "physics_engine.h"

void PhysicsEngine::updateObject(int index, float dt) {

        gameObjectsXAcceleration[index] += gravity.x;
        gameObjectsYAcceleration[index] += gravity.y;
        double VELOCITY_DAMPING = 40.0f;
        const float last_update_move_x = gameObjectsXPositions[index] - gameObjectsXLastPosition[index];
        const float last_update_move_y = gameObjectsYPositions[index] - gameObjectsYLastPosition[index];

        float new_position_x = gameObjectsXPositions[index] + last_update_move_x + (gameObjectsXAcceleration[index] - last_update_move_x * VELOCITY_DAMPING) * (dt * dt);
        float new_position_y = gameObjectsYPositions[index] + last_update_move_y + (gameObjectsYAcceleration[index] - last_update_move_y * VELOCITY_DAMPING) * (dt * dt);

        float movex = new_position_x - gameObjectsXPositions[index];
        float movey = new_position_y - gameObjectsYPositions[index];

        float max_speed = gameObjectsRadius[index];

        if (gameObjectsCollided[index]){
            max_speed = gameObjectsRadius[index] / 10.f;
        }
        if (movex >max_speed) {
            new_position_x = gameObjectsXPositions[index] +max_speed ;
        }
        if(movex < -max_speed) {
            new_position_x = gameObjectsXPositions[index] - max_speed;
        }
        if(movey > max_speed) {
            new_position_y = gameObjectsYPositions[index] + max_speed;
        }
        if(movey < -max_speed) {
            new_position_y = gameObjectsYPositions[index] - max_speed;
        }


        gameObjectsXLastPosition[index] = gameObjectsXPositions[index];
        gameObjectsYLastPosition[index] = gameObjectsYPositions[index];
        gameObjectsXPositions[index] = new_position_x;
        gameObjectsYPositions[index] = new_position_y;
        gameObjectsXAcceleration[index] = 0.0f;
        gameObjectsYAcceleration[index] = 0.0f;

        resolveCollisionsWithWalls(index);
}


void PhysicsEngine::generateGridParticleCenters(uint32_t gridParticleCenter[], const float gameObjectsXPositions[],
                                 const float gameObjectsYPositions[], uint32_t numberOfGameObjects, uint32_t gridWidth,
                                 uint32_t gridHeight) {
    # pragma omp parallel for num_threads(threadCount)
    for(uint32_t i =0; i < gridHeight * gridWidth * maxNumberOfGameObjectsPerCell; i++) {
        gridParticleCenter[i] = -1;
    }
    # pragma omp parallel for num_threads(threadCount)
    for(uint32_t i = 0; i < numberOfGameObjects; i++) {
        int cellX = (int)((gameObjectsXPositions[i] + 1.0f) / 2.0f * gridWidth);
        int cellY = (int)((gameObjectsYPositions[i] + 1.0f) / 2.0f * gridHeight);
        if (cellX >= gridWidth) {
            cellX = gridWidth - 1;
        }
        if (cellY >= gridHeight) {
            cellY = gridHeight - 1;
        }
        if (cellX < 0) {
            cellX = 0;
        }
        if (cellY < 0) {
            cellY = 0;
        }
        uint32_t start_index = cellX * gridHeight * maxNumberOfGameObjectsPerCell + cellY * maxNumberOfGameObjectsPerCell;
        for (uint32_t j = 0; j < maxNumberOfGameObjectsPerCell; j++) {
            if (gridParticleCenter[start_index + j] == -1) {
                gridParticleCenter[start_index + j] = i;
                break;
            }
        }
    }
}

void PhysicsEngine::solveCollisionsNotOptimized(){
    std::memcpy(gameObjectsXPositionsTmp, gameObjectsXPositions, numberOfGameObjects * sizeof(float));
    std::memcpy(gameObjectsYPositionsTmp, gameObjectsYPositions, numberOfGameObjects * sizeof(float));

    # pragma omp parallel for num_threads(threadCount)
    for (int i1 = 0; i1 < numberOfGameObjects; i1++) {
        for (int i2 = 0; i2 < numberOfGameObjects; i2++) {
            solveContact(i1, i2);
        }
    }
    std::memcpy(gameObjectsXPositions, gameObjectsXPositionsTmp, numberOfGameObjects * sizeof(float));
    std::memcpy(gameObjectsYPositions, gameObjectsYPositionsTmp, numberOfGameObjects * sizeof(float));
}

void PhysicsEngine::solveCollisionsOptimized() {
    std::memcpy(gameObjectsXPositionsTmp, gameObjectsXPositions, numberOfGameObjects * sizeof(float));
    std::memcpy(gameObjectsYPositionsTmp, gameObjectsYPositions, numberOfGameObjects * sizeof(float));

    generateGridParticleCenters(grid_centers, gameObjectsXPositions, gameObjectsYPositions, numberOfGameObjects,
                                gridWidth, gridHeight);

    # pragma omp parallel for num_threads(threadCount)
    for (uint32_t i = 0; i < gridWidth * gridHeight; i++) {
        if (grid_centers[i * maxNumberOfGameObjectsPerCell] == -1) {
            continue;
        }
        for (uint32_t j = 0; j < maxNumberOfGameObjectsPerCell; j++) {
            if (grid_centers[i * maxNumberOfGameObjectsPerCell + j] == -1) {
                break;
            }
            for (int cell_row = -1; cell_row < 2; cell_row++) {
                for (int cell_col = -1; cell_col < 2; cell_col++) {
                    int index = i + cell_row * gridHeight + cell_col;
                    if (index >= 0 && index < gridWidth * gridHeight) {
                        for (uint32_t k = 0; k < maxNumberOfGameObjectsPerCell; k++) {
                            if (grid_centers[index * maxNumberOfGameObjectsPerCell + k] == -1) {
                                break;
                            }
                            uint32_t index1 = grid_centers[i * maxNumberOfGameObjectsPerCell + j];
                            uint32_t index2 = grid_centers[index * maxNumberOfGameObjectsPerCell + k];
                            if (index1 != index2) {
                                solveContact(index1, index2);
                            }
                        }
                    }

                }
            }
        }
    }
    std::memcpy(gameObjectsXPositions, gameObjectsXPositionsTmp, numberOfGameObjects * sizeof(float));
    std::memcpy(gameObjectsYPositions, gameObjectsYPositionsTmp, numberOfGameObjects * sizeof(float));
}
void PhysicsEngine::update(float dt) {
    if (useGPU){
           cuda_solve_collisions(gameObjectsXPositions, gameObjectsYPositions, gameObjectsRadius,
                                 gameObjectsXLastPosition, gameObjectsYLastPosition,
                                 gameObjectsXAcceleration, gameObjectsYAcceleration,
                                 gravity.x, gravity.y, dt, sub_steps,
                                 numberOfGameObjects);
    }
    else{
        float sub_dt = dt / (float)sub_steps;
        memset(gameObjectsCollided, 0, numberOfGameObjects * sizeof(bool));
        for (uint32_t j = 0; j < sub_steps; j++) {
            if(optimizeCPU) {
                solveCollisionsOptimized();
            }
            else{
                solveCollisionsNotOptimized();
            }

            # pragma omp parallel for num_threads(threadCount)
            for (int i = 0; i < numberOfGameObjects; i++) {
                updateObject(i, sub_dt);
            }
        }
    }
}

glm::ivec2 PhysicsEngine::mapWorldToGrid(const glm::vec2& worldCoord, glm::ivec2 gridSize) {

    float normalizedX = (worldCoord.x + 1.0f) / 2.0f;
    float normalizedY = (worldCoord.y + 1.0f) / 2.0f;

    // Mapping normalized coordinates to grid coordinates
    int gridX = static_cast<int>(normalizedX * gridSize.x);
    int gridY = static_cast<int>(normalizedY * gridSize.y);

    // Make sure the grid coordinates are within bounds
    gridX = std::max(0, std::min(gridX, gridSize.x - 1));
    gridY = std::max(0, std::min(gridY, gridSize.y - 1));

    return {gridX, gridY};
}

// Checks if two atoms are colliding and if so create a new contact
void PhysicsEngine::solveContact(uint32_t atom_1_idx, uint32_t atom_2_idx)
{
    if(atom_1_idx == atom_2_idx) {
        return;
    }
    constexpr float response_coef = 1.0f;
    constexpr float eps           = 0.00001f;

    double pos1_x = gameObjectsXPositions[atom_1_idx];
    double pos1_y = gameObjectsYPositions[atom_1_idx];
    double pos2_x = gameObjectsXPositions[atom_2_idx];
    double pos2_y = gameObjectsYPositions[atom_2_idx];
    double radius1 = gameObjectsRadius[atom_1_idx];
    double radius2 = gameObjectsRadius[atom_2_idx];
    const double dist2 = (pos2_x - pos1_x) * (pos2_x - pos1_x) + (pos2_y - pos1_y) * (pos2_y - pos1_y);
    if (dist2 > eps && dist2 < (radius1 + radius2) * (radius1 + radius2)) {
        const float dist          = sqrt(dist2);
        // Radius are all equal to 1.0f
        const float delta  = response_coef * 0.5f * (radius1 + radius2 - dist);
        double col_vec_x = (pos2_x - pos1_x) / dist * delta;
        double col_vec_y = (pos2_y - pos1_y) / dist * delta;

        gameObjectsXPositionsTmp[atom_1_idx] -= col_vec_x;
        gameObjectsYPositionsTmp[atom_1_idx] -= col_vec_y;
        gameObjectsCollided[atom_1_idx] = true;
    }
}


void PhysicsEngine::resolveCollisionsWithWalls(int index) {
    constexpr float eps           = 0.000f;
    double pos_x = gameObjectsXPositions[index];
    double pos_y = gameObjectsYPositions[index];
    double radius = gameObjectsRadius[index];

    if (pos_x - radius < -1.0) {
        gameObjectsXPositions[index] = -1.0 + radius + eps;  // Adjust position to be just outside the left wall
    } else if (pos_x + radius > 1.0) {
        gameObjectsXPositions[index] = 1.0 - radius - eps;  // Adjust position to be just outside the right wall
    }

    // Vertical walls
    if (pos_y - radius < -1.0) {
        gameObjectsYPositions[index] = -1.0 + radius + eps;  // Adjust position to be just outside the bottom wall
    } else if (pos_y + radius > 1.0) {
        gameObjectsYPositions[index] = 1.0 - radius - eps;  // Adjust position to be just outside the top wall
    }
}
