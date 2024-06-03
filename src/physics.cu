#include <cuda_runtime.h>
#include <stdio.h>


__global__ void update(float *xCurrPos, float *yCurrPos, float *xLastPos, float *yLastPos, float *xAcc, float *yAcc, float xGravity, float yGravity, const float dt, int numElements, const float *radii, bool *collided) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        xAcc[i] += xGravity;
        yAcc[i] += yGravity;


        float VELOCITY_DAMPING = 40.0f;

        const float curr_pos_x = xCurrPos[i];
        const float curr_pos_y = yCurrPos[i];
        float last_update_move_x = curr_pos_x - xLastPos[i];

        float last_update_move_y = curr_pos_y - yLastPos[i];

         float new_position_x =
                curr_pos_x + last_update_move_x + (xAcc[i] - last_update_move_x * VELOCITY_DAMPING) * (dt * dt);
         float new_position_y =
                curr_pos_y + last_update_move_y + (yAcc[i] - last_update_move_y * VELOCITY_DAMPING) * (dt * dt);
        float movex = new_position_x - curr_pos_x;
        float movey = new_position_y - curr_pos_y;
        float max_speed = radii[i];

        if (collided[i]){
            max_speed = radii[i] / 10.f;
        }
        if (movex >max_speed) {
            new_position_x = curr_pos_x +max_speed ;
        }
        if(movex < -max_speed) {
            new_position_x = curr_pos_x - max_speed;
        }
        if(movey > max_speed) {
            new_position_y = curr_pos_y + max_speed;
        }
        if(movey < -max_speed) {
            new_position_y = curr_pos_y - max_speed;
        }

        xLastPos[i] = curr_pos_x;
        yLastPos[i] = curr_pos_y;
        xCurrPos[i] = new_position_x;
        yCurrPos[i] = new_position_y;

        xAcc[i] = 0.0f;
        yAcc[i] = 0.0f;

        float pos_x = xCurrPos[i];
        float pos_y = yCurrPos[i];
        float radius = radii[i];
        if (pos_x - radius < -1.0) {
            xCurrPos[i] = -1.0 + radius;
        } else if (pos_x + radius > 1.0) {
            xCurrPos[i] = 1.0 - radius;
        }

        // Vertical walls
        if (pos_y - radius < -1.0) {
            yCurrPos[i] = -1.0 + radius;
        } else if (pos_y + radius > 1.0) {
            yCurrPos[i] = 1.0 - radius;
        }
    }


}

__global__ void solveContact(float *xPos, float *yPos, float *xPosRes, float *yPosRes, float *radii, int numElements, bool *collided) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


if (i < numElements && j < numElements && i != j) {
        constexpr float response_coef = 1.10f;
        constexpr float eps           = 0.00001f;
        float pos1_x = xPos[i];
        float pos1_y = yPos[i];
        float pos2_x = xPos[j];
        float pos2_y = yPos[j];
        float radius1 = radii[i];
        float radius2 = radii[j];


        const float dist2 = (pos2_x - pos1_x) * (pos2_x - pos1_x) + (pos2_y - pos1_y) * (pos2_y - pos1_y);
        if (dist2 > eps && dist2 < (radius1 + radius2) * (radius1 + radius2)) {
            const float dist          = sqrt(dist2);
            const float delta  = response_coef * 0.5f * (radius1 + radius2 - dist);
            float col_vec_x = (pos2_x - pos1_x) / dist * delta;
            float col_vec_y = (pos2_y - pos1_y) / dist * delta;

            atomicAdd(xPosRes + i, -col_vec_x);
            atomicAdd(yPosRes + i, -col_vec_y);
            collided[i] = true;
        }



    }
}

extern "C" void cuda_solve_collisions(float *currPositionsX, float *currPositionsY, float *radii,
                                      float *lastPositionsX, float *lastPositionsY, float *accelerationX,
                                      float *accelerationY,  float xGravity,  float yGravity,
                                      float dt, int substeps, int numElements){
    float *d_curr_positions_x = NULL; // input
    float *d_curr_positions_y= NULL; // input
    float *d_radii = NULL; // input
    float *d_last_positions_x = NULL; // input
    float *d_last_positions_y = NULL; // input
    float *d_acceleration_x = NULL; // input
    float *d_acceleration_y = NULL; // input
    float *d_result_x = NULL; // output
    float *d_result_y = NULL; // output
    bool *collided = NULL;


    cudaMalloc((void **)&d_curr_positions_x, numElements * sizeof(float));
    cudaMalloc((void **)&d_curr_positions_y, numElements * sizeof(float));
    cudaMalloc((void **)&d_radii, numElements * sizeof(float));
    cudaMalloc((void **)&d_result_x, numElements * sizeof(float));
    cudaMalloc((void **)&d_result_y, numElements * sizeof(float));
    cudaMalloc((void **)&d_last_positions_x, numElements * sizeof(float));
    cudaMalloc((void **)&d_last_positions_y, numElements * sizeof(float));
    cudaMalloc((void **)&d_acceleration_x, numElements * sizeof(float));
    cudaMalloc((void **)&d_acceleration_y, numElements * sizeof(float));
    cudaMalloc((void **)&collided, numElements * sizeof(bool));

    cudaMemcpy(d_curr_positions_x, currPositionsX, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr_positions_y, currPositionsY, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_radii, radii, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_last_positions_x, lastPositionsX, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_last_positions_y, lastPositionsY, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceleration_x, accelerationX, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceleration_y, accelerationY, numElements * sizeof(float), cudaMemcpyHostToDevice);



    cudaMemset(collided, 0, numElements * sizeof(bool));

    dim3 blockSize(16, 16);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x, (numElements + blockSize.y - 1) / blockSize.y);

    float blockSize2 = 256;
    int gridSize2 = (numElements + blockSize2 - 1) / blockSize2;
    float sub_dt = dt / (float)substeps;

    for (int j = 0; j < substeps; j++) {
        cudaMemcpy(d_result_x, d_curr_positions_x, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_result_y, d_curr_positions_y, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
        solveContact<<<gridSize, blockSize>>>(d_curr_positions_x, d_curr_positions_y, d_result_x, d_result_y,
                                              d_radii, numElements, collided);
        cudaDeviceSynchronize();

        cudaMemcpy(d_curr_positions_x, d_result_x, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_curr_positions_y, d_result_y, numElements * sizeof(float), cudaMemcpyDeviceToDevice);

        update<<<gridSize2, blockSize2>>>(d_curr_positions_x, d_curr_positions_y, d_last_positions_x,
                                          d_last_positions_y, d_acceleration_x, d_acceleration_y,
                                          xGravity, yGravity, sub_dt,
                                          numElements, d_radii, collided);
        cudaDeviceSynchronize();
        cudaMemset(collided, 0, numElements * sizeof(bool));
    }



    cudaMemcpy(currPositionsX, d_curr_positions_x, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(currPositionsY, d_curr_positions_y, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(lastPositionsX, d_last_positions_x, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(lastPositionsY, d_last_positions_y, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(accelerationX, d_acceleration_x, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(accelerationY, d_acceleration_y, numElements * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_curr_positions_x);
    cudaFree(d_curr_positions_y);
    cudaFree(d_radii);
    cudaFree(d_result_x);
    cudaFree(d_result_y);

    cudaFree(d_last_positions_x);
    cudaFree(d_last_positions_y);

    cudaFree(d_acceleration_x);
    cudaFree(d_acceleration_y);
}
