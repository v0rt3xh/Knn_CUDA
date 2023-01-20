// We use this file for debugging.
#include "../../include/knnSearch.h"
#include<iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() 
{
    const int dimension = 100;
    const int referenceNum = 100;
    const int queryNum = 8;
    const int k = 10;
    // Print the debug information
    std::cout << "===== Debug Configuration =====" << std::endl;
    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Number of Reference Points: " << referenceNum << std::endl;
    std::cout << "queryNum: " << queryNum << std::endl;
    std::cout << "k: " << k << std::endl;
    // Define a vector of strings, store the words.
    std::vector<std::string> refWords;
    std::vector<std::string> queryWords;
    // Define the data matrices.
    // We first define the reference data points,
    // The embeddings come from GloVe.
    // Each column is one point. 
    float* referencePoints = new float[referenceNum * dimension];
    std::ifstream referenceFile("data/refEmbeddingsExtra.txt");
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
    std::ifstream queryFile("data/queryEmbeddingsExtra.txt");
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
    float knnTime = knnSearch(referencePoints, referenceNum, queryPoints, queryNum, dimension, k, resultSimilarity, resultIndex);
    if (knnTime < 0) 
    {
        std::cout << "Something went wrong, check logs" << std::endl;
    }
    else 
    {
        for (int i = 0; i < queryNum; i++) 
        {
            std::cout << "The " << k << " closest words for " << queryWords[i] << " are " << std::endl;
            for (int j = 0; j < k; j++) 
            {
                std::cout << refWords[resultIndex[j * queryNum + i]] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "With Execution Time:" << knnTime << std::endl;
    }
    delete[] resultSimilarity;
    delete[] resultIndex;
    delete[] referencePoints;
    delete[] queryPoints;
    return 0;
}