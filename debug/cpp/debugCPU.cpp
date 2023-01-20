#include "../../include/knnCPU.h"
#include<iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() 
{
    const int dimension = 25;
    const int referenceNum = 23;
    const int queryNum = 8;
    const int k = 3;
    // Print the debug information
    std::cout << "===== Debug Configuration =====" << std::endl;
    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Number of Reference Points: " << referenceNum << std::endl;
    std::cout << "queryNum: " << queryNum << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "ON CPU with cpp code " << std::endl;
    std::cout << "===============================" << std::endl;
    // Define a vector of strings, store the words.
    std::vector<std::string> refWords;
    std::vector<std::string> queryWords;
    // Define the data matrices.
    // We first define the reference data points,
    // The embeddings come from GloVe.
    // Each column is one point. 
    float* referencePoints = new float[referenceNum * dimension];
    std::ifstream referenceFile("data/refEmbeddings.txt");
    std::string line; 
    std::string word;
    // Read each line in the file 
    int wordIndex = 0; // store the index of the word that we are processing
    while (std::getline(referenceFile, line)) 
    {
        int lineIndex = 0; // Helper index for each line 
        std::stringstream lineStream(line);
        while (lineStream >> word) 
        {
            if (lineIndex) 
            {
                // If it's not the first element, it should be the embedding values
                referencePoints[referenceNum * (lineIndex - 1) + wordIndex] = std::stof(word);
            }
            else 
            {
                // The first element is a word
                refWords.push_back(word);
            }
            lineIndex++; 
        }
        wordIndex++;
    }
    // Then the query points
    // Query data
    float* queryPoints = new float[queryNum * dimension];
    std::ifstream queryFile("data/queryEmbeddings.txt");
    wordIndex = 0;
    while (std::getline(queryFile, line)) 
    {
        int lineIndex = 0; // Helper index for each line 
        std::stringstream lineStream(line);
        while (lineStream >> word) 
        {
            if (lineIndex) 
            {
                // If it's not the first element, it should be the embedding values
                queryPoints[queryNum * (lineIndex - 1) + wordIndex] = std::stof(word);
            }
            else 
            {
                // The first element is a word
                queryWords.push_back(word);
            }
            lineIndex++; 
        }
        wordIndex++;
    }
    // To use knnSearch, we also need to initialize the following stuffs.
    // knnSimilarity array
    // knnIndex array
    float* resultSimilarity = new float[queryNum * k];
    int* resultIndex = new int[queryNum * k];
    // The helper array will be used in the loop in a later section. 
    float* similarityEach = new float[referenceNum];
    int* indexEach = new int[referenceNum];
    // First compute the L2 norms.
    float* queryNorm = new float[queryNum];
    float* refNorm = new float[referenceNum];
    computeL2Norm_CPU(queryPoints, queryNorm, dimension, queryNum);
    computeL2Norm_CPU(referencePoints, refNorm, dimension, referenceNum);
    // Then we compute the similarities and start the KNN search.
    for (int i = 0; i < queryNum; i++) 
    {
        for (int j = 0; j < referenceNum; j++) 
        {
            similarityEach[j] = computeSimilarity_CPU(referencePoints, referenceNum, queryPoints, queryNum, dimension, j, i, refNorm, queryNorm);
            indexEach[j] = j;
        }
        // Sort the distance and index.
        insertionSort_CPU(similarityEach, indexEach, referenceNum, k);
        // Copy the result to resultSimilarity and resultIndex (Global output)
        for (int l = 0; l < k; l++) 
        {
            resultSimilarity[l * queryNum + i] = similarityEach[l];
            resultIndex[l * queryNum + i] = indexEach[l];
        }
    }
    for (int i = 0; i < queryNum; i++) 
    {
        std::cout << "The " << k << " closest words for " << queryWords[i] << " are " << std::endl;
        for (int j = 0; j < k; j++) 
        {
            std::cout << refWords[resultIndex[j * queryNum + i]] << " ";
        }
        std::cout << std::endl;
    }
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