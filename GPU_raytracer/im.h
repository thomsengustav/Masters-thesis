#pragma once
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm> // for std::copy




//std::vector<std::vector<float>> import_mesh(std::string fileName)
std::array<float4, VERTEX_COUNT>* import_mesh(std::string fileName){
//std::vector<std::vector<float>> import_mesh(std::string fileName) {
    std::ifstream inFile;
    inFile.open(fileName);
    std::istream_iterator<float> start(inFile), end;
    std::vector<float> numbers(start, end);
    std::cout << "Read " << numbers.size() << " numbers" << std::endl;

    std::vector<std::vector<float>> vector2D(3, std::vector<float>(numbers.size() / 3, 0));// 
    
    int counter = 0;
    const int g = numbers.size() / 3;

    std::array<float4, VERTEX_COUNT>* array2D;
    array2D = new std::array<float4, VERTEX_COUNT>;
    for (int j = 0; j < g; j++) {// 
        for (int i = 0; i < 3; i++) {
            vector2D[i][j] = numbers[counter++];
            
        }
        float4 as = { vector2D[0][j],vector2D[1][j],vector2D[2][j],0};
        (*array2D)[j] = as;
    }
    
    
    
    return array2D;//array2D;//vector2D;
}




std::array<uint32_t, TRIANGLE_COUNT>* import_mesh_mat_indices(std::string fileName_triangleIndex) {
    
    std::ifstream inFile;
    inFile.open(fileName_triangleIndex);
    std::istream_iterator<float> start(inFile), end;
    std::vector<float> numbers(start, end);
    std::cout << "Read " << numbers.size() << " numbers" << std::endl;

     

    int counter = 0;
  

    std::array<uint32_t, TRIANGLE_COUNT>* array1D;
    array1D = new std::array<uint32_t, TRIANGLE_COUNT>;
    for (int j = 0; j < TRIANGLE_COUNT; j++) {// 
       
        (*array1D)[j] = numbers[counter++];

        
        
    }



    return array1D;//array2D;//vector2D;
}



