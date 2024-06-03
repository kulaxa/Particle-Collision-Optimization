#include <GL/glut.h>
#include <cmath>
#include <iostream>
#include "imgui.h"
#include "backends/imgui_impl_opengl2.h"
#include "backends/imgui_impl_glut.h"
#include "physics/physics_engine.h"
#include "imgui_internal.h"
#include <glm/glm.hpp>
#include <fstream>
#include <omp.h>

#include <chrono>


GLuint window;
GLuint width = 1000, height = 1000;

int particle_counter = 0;
float particle_velocity_x = 300.f;
float particle_velocity_y = -300.f;
int particle_time_delta = 2;
float particle_size = 0.0030;
int particle_segments = 3;
int balls_to_add = 350000;

bool showGrid = true;
bool global_particle_size = false;
bool renderGameObjects = true;
bool use_gpu = false;
bool optimize_cpu = true;
float gravity_y = 0.5f;

glm::vec2 gravity = glm::vec2(0.0, -gravity_y);
int grid_size = 300;
int thread_count = 1;
int physics_step = 8;

std::ofstream outputFile;

//glm::vec2 lastSourcePosition = glm::vec2(0.0, 0.0);
struct ParticleSource{
    glm::vec2 position;
    glm::vec2 velocity;
    float radius;
    int number_of_balls;
    int particle_time_delta;
    bool active = false;
    int particle_add_counter = 0;
};



std::vector<ParticleSource> particleSources = {
};

void MainLoopStep();

void displaySourcePoints();

PhysicsEngine physicsEngine = PhysicsEngine(gravity, grid_size, physics_step);


void drawGrid(glm::ivec2 gridSize){
    int gridHeight = gridSize.y;
    int gridWidth = gridSize.x;
    glBegin(GL_LINES);

    glColor3f(0.0f, 0.0f, 0.0f);
    // Draw horizontal lines
    for (int i = 0; i <= gridHeight; ++i) {
        glVertex2f(-1.0, 2.0 * i / gridHeight - 1.0);
        glVertex2f(1.0, 2.0 * i / gridHeight - 1.0);
    }

    // Draw vertical lines
    for (int i = 0; i <= gridWidth; ++i) {
        glVertex2f(2.0 * i / gridWidth - 1.0, -1.0);
        glVertex2f(2.0 * i / gridWidth - 1.0, 1.0);
    }

    glEnd();
}

//void getColor(int index, float& red, float& green, float& blue) {
//    // Example: Coloring cells based on a pattern (you can modify this logic)
//        const float frequency = 0.5;
//        red = sin(frequency * index + 0) * 0.5 + 0.5;
//        green = sin(frequency * index + 2) * 0.5 + 0.5;
//        blue = sin(frequency * index + 4) * 0.5 + 0.5;
//
//}



void drawCircle(glm::vec2 position, float radius, int num_segments, glm::vec3 color) {
    glBegin(GL_TRIANGLE_FAN );
    for (int i = 0; i < num_segments; i++) {
        float theta = 2.0f * 3.1415926f * float(i) / float(num_segments);
        float x = radius * cosf(theta);
        float y = radius * sinf(theta);
        glColor3f(color.x, color.y, color.z);
        glVertex2f(x + position.x, y + position.y);
    }
    glEnd();
}

// Display callback function
void display() {
    displaySourcePoints();

    if(showGrid){
        drawGrid(physicsEngine.getGridSize());

    }
    // Draw balls
    float red = 0.0, green = 0.0, blue = 0.0;

    for (size_t i = 0; i < physicsEngine.getNumberOfGameObjects(); ++i) {
        double xPos = physicsEngine.getGameObjectXPosition(i);
        double yPos = physicsEngine.getGameObjectYPosition(i);
        double radius = physicsEngine.getGameObjectRadius(i);
        float r, g, b;
        physicsEngine.getColor(i, r, g, b);
        glm::vec3 color = glm::vec3(r, g, b);
        if(renderGameObjects) {
            drawCircle({xPos, yPos}, radius, particle_segments, color);
        }
    }


}

void displaySourcePoints() {
    glColor3f(1.0, 0.0, 0.0);
    for (auto& source: particleSources) {
        glPointSize(width * source.radius);

        glBegin(GL_POINTS);
        glVertex2f(source.position.x, source.position.y);
        glEnd();    }


}

// Timer callback function for animation
void timer(int) {


    // imgui position in world
    glutPostRedisplay();
    glutTimerFunc(16, timer, 0);  // 60 frames per second
        for (auto& source: particleSources) {
            if(source.active){
                if (source.number_of_balls > 0 && source.particle_add_counter % source.particle_time_delta == 0) {
                    glm::vec2 particlePosition = source.position;
                    // check if particle is inside another particle
                    bool inside = false;
                    for (int i = 0; i < physicsEngine.getNumberOfGameObjects(); i++) {
                        double xPos = physicsEngine.getGameObjectXPosition(i);
                        double yPos = physicsEngine.getGameObjectYPosition(i);
                        double radius = physicsEngine.getGameObjectRadius(i);
                        if (glm::distance({xPos, yPos}, particlePosition) < (radius+ source.radius)) {
                            inside = true;
                            break;
                        }
                    }
                    if (inside) {
                        continue;
                    }
                    float red = random() % 100 / 100.0;
                    float green = random() % 100 / 100.0;
                    float blue = random() % 100 / 100.0;
                    glm::vec3 color = glm::vec3(red, green, blue);
                    physicsEngine.addGameObject(physicsEngine.getNumberOfGameObjects(), particlePosition.x, particlePosition.y, source.velocity.x, source.velocity.y, source.radius, color);
                    source.number_of_balls--;
                    particle_counter++;
                }
                source.particle_add_counter++;
            }
        }
}

int main(int argc, char** argv) {


    putenv( (char *) "__GL_SYNC_TO_VBLANK=0" );
    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutCreateWindow("2D Particle System DEMO");

    glutDisplayFunc(MainLoopStep);
    glutTimerFunc(0, timer, 0);
    glClearColor(0.0, 0.0, 0.0, 1.0);  // Black background
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);  // Set the coordinate system
//    physicsEngine.setGameObjects(balls);
    outputFile.open("../results.txt");


    // Check if the file is successfully opened
    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1; // Return an error code
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGLUT_Init();
    ImGui_ImplOpenGL2_Init();
    ImGui_ImplGLUT_InstallFuncs();

    for(auto &source: particleSources){
        source.active = true;
    }
    for(int j = 0; j < 3; j++) {
        for (int i = 0; i < 20; i++) {
                float x_pos = -0.9f + i * (1.8f / 20);
                float y_pos = 0.7 + (j * 0.1);
                particleSources.push_back(
                        ParticleSource{glm::vec2(x_pos, y_pos), glm::vec2(particle_velocity_x, particle_velocity_y),
                                       particle_size, balls_to_add, particle_time_delta, false});
            }
    }
    glutMainLoop();


    // Cleanup
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();
    outputFile.close();

    return 0;
}

void MainLoopStep()

{

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::NewFrame();


    ImGuiIO& io = ImGui::GetIO();

    {
        ImGui::Begin("Interactive GUI panel");
        ImGui::SliderInt("Number of particles to add", &balls_to_add,   1, 10000);

        ImGui::Separator();

        if(ImGui::SliderFloat("Particle size", &particle_size, 0.003, 0.05)){
            if(global_particle_size) {
                for (int i = 0; i < physicsEngine.getNumberOfGameObjects(); i++) {
                    if (physicsEngine.getGameObjectRadius(i) < particle_size) {
                        continue;
                    }
                    physicsEngine.setGameObjectXPosition(i, particle_size);
                }
            }

                for (auto &source: particleSources) {
                    source.radius = particle_size;


            }
        }
        if (ImGui::SliderInt("Physics substeps", &physics_step, 1, 32)){
            physicsEngine.setSubSteps(physics_step);
        }


        if(ImGui::Button("Start Particle Sources")){
            int mod = balls_to_add % particleSources.size();
            int div = balls_to_add / particleSources.size();

            for(int i = 0; i < particleSources.size(); i++){
                auto &source = particleSources[i];
                source.active = true;
                source.number_of_balls = div + (i < mod ? 1 : 0);
            }
        }
        if(ImGui::Button("Stop Particle Sources")){
            for(auto &source: particleSources){
                source.active = false;
            }
        }

        if(ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !ImGui::IsWindowHovered()){
//            lastSourcePosition = glm::vec2{io.MousePos.x / width * 2.0f - 1.0f, -(io.MousePos.y / height * 2.0f - 1.0f)};
//            particleSources.push_back(ParticleSource{lastSourcePosition, glm::vec2(particle_velocity_x, particle_velocity_y), particle_size, balls_to_add, particle_time_delta, false});
            float red = random() % 100 / 100.0;
            float green = random() % 100 / 100.0;
            float blue = random() % 100 / 100.0;
            glm::vec3 color = glm::vec3(red, green, blue);
physicsEngine.addGameObject(physicsEngine.getNumberOfGameObjects(), io.MousePos.x / width * 2.0f - 1.0f, -(io.MousePos.y / height * 2.0f - 1.0f), 0.f, 0.f, particle_size, color);
        }
//        ImGui::Checkbox("Show grid", &showGrid);
        ImGui::Checkbox("Render game objects", &renderGameObjects);
        optimize_cpu = false;
        use_gpu = false;
        static int e = 1;
        ImGui::RadioButton("Use Unoptimized CPU", &e, 1);
        ImGui::RadioButton("Use Optimized CPU", &e, 2);
        ImGui::RadioButton("Use GPU", &e, 3);

        switch (e) {
            case 1:
                physicsEngine.setUseGPU(false);
                physicsEngine.setOptimizeCPU(false);
                break;
            case 2:

                physicsEngine.setUseGPU(false);
                physicsEngine.setOptimizeCPU(true);
                optimize_cpu = true;
                break;
            case 3:

                physicsEngine.setUseGPU(true);
                physicsEngine.setOptimizeCPU(false);
                use_gpu = true;
                break;
            default:
                break;
        }

        if (e != 3) {
            if (ImGui::SliderInt("Thread count", &thread_count, 1, 16)) {
                physicsEngine.setThreadCount(thread_count);
                omp_set_num_threads(thread_count);
            }

            if (ImGui::SliderInt("Grid size", &grid_size, 50, 400)) {
                physicsEngine.setGridSize(grid_size, grid_size);
            }
        }


        ImGui::Text("Number of particles: %d", particle_counter);
        ImGui::Text("Number of cells: %d", grid_size*grid_size);

        if(ImGui::Button("Clear particles")){
            physicsEngine.clearGameObjects();
            particle_counter = 0;
        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        glm::vec2 mouseWorldPos = glm::vec2{io.MousePos.x / width * 2.0f - 1.0f, io.MousePos.y / height * 2.0f - 1.0f};
        glm::vec2 mousePosGrid = physicsEngine.mapWorldToGrid(mouseWorldPos, physicsEngine.getGridSize());
        ImGui::Text("Mouse position x= %.3f y= %.3f, Grid coordinates x=%d, y=%d (index = %d)", mouseWorldPos.x, mouseWorldPos.y, (int)mousePosGrid.x, (int)mousePosGrid.y, (int)(mousePosGrid.x * physicsEngine.getGridSize().x + mousePosGrid.y));

        ImGui::Text("Number of collisions: %d, number of tested particles: %d (%f)", physicsEngine.getCollisionCount(), physicsEngine.getCollisionTestCount(), (float)physicsEngine.getCollisionCount() / (float)physicsEngine.getCollisionTestCount());
        ImGui::End();
    }


    // Rendering
    ImGui::Render();
    glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);
    glClearColor(0.5f, 0.5f, 0.5f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    float dt = 1.0f / io.Framerate;
    auto start = std::chrono::high_resolution_clock::now();

    physicsEngine.update(dt);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
//    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
//    outputFile << "FPS:" <<io.Framerate << ";GRID_SIZE:" << physicsEngine.getGridSize().x <<";PARTICLE_COUNTER:" << particle_counter << ";COLLISION_COUNT:"<< physicsEngine.getCollisionCount() << ";COLLISION_TEST_COUNT:" << physicsEngine.getCollisionTestCount() << ";THREAD_COUNT:"<< thread_count<< std::endl;
    outputFile << "EXECUTION_TIME:" << duration.count() << ";PARTICLE_COUNTER:"<< particle_counter<<";GPU:"<<use_gpu<<";CPU_OPTIMIZED:"<<optimize_cpu<< ";THREAD_COUNT:"<< thread_count << ";GRID_SIZE:" <<grid_size<< std::endl;

    display();

    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());


    glutSwapBuffers();


}