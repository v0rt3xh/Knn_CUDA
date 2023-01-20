// We use this file for scaling analysis.
#include "../include/knnSearch.h"
#include<iostream>
#include <random>

int main(int argc, char* argv[]) 
{
    // Read in scaling parameters.
    int refNum = std::stoi(argv[1]);
    int queryNum = std::stoi(argv[2]);
    int dimension = std::stoi(argv[3]);
    int k = std::stoi(argv[4]);
    // Print the configurations
    std::cout << "===== Debug Configuration =====" << std::endl;
    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Number of Reference Points: " << refNum << std::endl;
    std::cout << "queryNum: " << queryNum << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "===============================" << std::endl;
    // Initialize arrays and generate synthetic data
    float* referencePoints = new float[refNum * dimension];
    float* queryPoints = new float[queryNum * dimension];
    // Generate random numbers
    std::random_device randomizer;
    std::mt19937 gen(randomizer());
    std::uniform_real_distribution<float> generator(-1.0, 1.0);
    for (int i = 0; i < refNum; i++) 
    {
        for (int j = 0; j < dimension; j++) 
        {
            referencePoints[j * refNum + i] = generator(gen);
        }
    }
    for (int i = 0; i < queryNum; i++) 
    {
        for (int j = 0; j < dimension; j++) 
        {
            queryPoints[j * queryNum + i] = generator(gen);
        }
    }
    // For better timing results, we will repeat the computation for 10 iterations.
    const int ITER = 10;
    float executionTime = 0;
    // To use knnSearch, we also need to initialize the following stuffs.
    // knnSimilarity array
    // knnIndex array
    float* resultSimilarity = new float[queryNum * k];
    int* resultIndex = new int[queryNum * k];
    for (int m = 0; m < ITER; m++) 
    {

        // Call the KNN Search function
        float knnTime = knnSearch(referencePoints, refNum, queryPoints, queryNum, dimension, k, resultSimilarity, resultIndex);
        if (knnTime < 0) 
        {
            std::cout << "Something went wrong, check logs" << std::endl;
            delete[] referencePoints;
            delete[] queryPoints;
            delete[] resultSimilarity;
            delete[] resultIndex;
            return -1;            
        }
        executionTime += knnTime;
    }
    executionTime /= ITER;
    std::cout << "With Average Execution Time:" << executionTime << std::endl;
    delete[] referencePoints;
    delete[] queryPoints;
    delete[] resultSimilarity;
    delete[] resultIndex;
    return 0;
}