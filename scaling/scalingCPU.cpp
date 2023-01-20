#include "../include/knnCPU.h"
#include<iostream>
#include <random>
#include <chrono>
#include <ratio>

// Credit, I using the scripts in timing.md on GitHub.
using std::chrono::high_resolution_clock;
using std::chrono::duration;

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

    // Output time
    float executionTime = 0;
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
    // Timer
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<float, std::milli> duration_sec;
    // To use knnSearch, we also need to initialize the following stuffs.
    // knnSimilarity array
    // knnIndex array
    float* resultSimilarity = new float[queryNum * k];
    int* resultIndex = new int[queryNum * k];
    // The helper array will be used in the loop in a later section. 
    float* similarityEach = new float[refNum];
    int* indexEach = new int[refNum];
    // First compute the L2 norms.
    float* queryNorm = new float[queryNum];
    float* refNorm = new float[refNum];
    for (int m = 0; m < ITER; m++) 
    {
        start = high_resolution_clock::now();
        // Record the time for computing norms & the knn process.
        computeL2Norm_CPU(queryPoints, queryNorm, dimension, queryNum);
        computeL2Norm_CPU(referencePoints, refNorm, dimension, refNum);
        // Then we compute the similarities and start the KNN search.
        for (int i = 0; i < queryNum; i++) 
        {
            for (int j = 0; j < refNum; j++) 
                {
                    similarityEach[j] = computeSimilarity_CPU(referencePoints, refNum, queryPoints, queryNum, dimension, j, i, refNorm, queryNorm);
                    indexEach[j] = j;
                }
            // Sort the distance and index.
            insertionSort_CPU(similarityEach, indexEach, refNum, k);
            // Copy the result to resultSimilarity and resultIndex (Global output)
            for (int l = 0; l < k; l++) 
            {
                resultSimilarity[l * queryNum + i] = similarityEach[l];
                resultIndex[l * queryNum + i] = indexEach[l];
            }
        }
        end = high_resolution_clock::now();
        duration_sec = std::chrono::duration_cast<duration<float, std::milli> >(end - start);
        executionTime += duration_sec.count();
    }
    executionTime /= ITER; 
    std::cout << "With Average Execution Time:" << executionTime << std::endl;
    // Deallocate memory
    delete[] similarityEach;
    delete[] indexEach;
    delete[] queryNorm;
    delete[] refNorm;
    delete[] resultSimilarity;
    delete[] resultIndex;
    delete[] referencePoints;
    delete[] queryPoints;
    return 0;
}