#include "../include/knnCPU.h"
#include <algorithm>
float computeSimilarity_CPU(float* reference,
                                  int referenceNum,
                                  float* query,
                                  int queryNum,
                                  int dimension,
                                  int refIndex,
                                  int queryIndex,
                                  float* refNorm,
                                  float* queryNorm) 
{
    float innerProduct = 0;
    for (int d = 0; d < dimension; d++) 
    {
        innerProduct += reference[referenceNum * d + refIndex] * query[queryNum * d + queryIndex];
    }
    innerProduct /= refNorm[refIndex] * queryNorm[queryIndex]; 
    return innerProduct;
}

void computeL2Norm_CPU(float* inputArray, 
                        float* outputArray,
                        int dimension,
                        int width) 
{
    float result = 0; 
    for (int i = 0; i < width; i++) 
    {
        for (int j = 0; j < dimension; j++) 
        {
            result += inputArray[j * width + i] * inputArray[j * width + i];
        }
        outputArray[i] = sqrt(result);
        result = 0;
    } 

}

void insertionSort_CPU(float* similarity, 
                       int* indices,
                       int width,
                       int k) 
{
    // Init the first index
    indices[0] = 0; 

    // Iterate through all the reference points
    for (int i = 1; i < width; i++) 
    {
        // Current similarity & index
        float curSimilarity = similarity[i];
        int curIndex = i; 
        // If current similarity is smaller than the k-th sorted smallest element,
        // And its index is smaller than k, skip. 
        if (i >= k && curSimilarity <= similarity[k - 1]) 
        {
            continue;
        }
        // Move similarities and indices smaller than current similarity to the right
        int j = std::min(i, k - 1);
        while (j > 0 && similarity[j - 1] < curSimilarity) 
        {
            similarity[j] = similarity[j - 1];
            indices[j] = indices[j - 1];
            j--;
        } 
        // Set current similarity and index to the right position.
        similarity[j] = curSimilarity;
        indices[j] = curIndex;
    }
}