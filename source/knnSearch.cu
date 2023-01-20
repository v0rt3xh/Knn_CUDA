#include<iostream>
#include <stdlib.h>
#include <math.h>
#include "../include/knnSearch.h" 

#define BLOCK_DIM 16

/**
 * @brief The function we use to compute the similarity 
 *        between query embeddings and reference embeddings.
 *        In my project, I choose to use cosine similarity.
 * 
 * @param reference Reference points, size of (dimension, referenceNum) 
 * Note that we are using 2D memory & pitch.
 * @param referenceWidth Number of reference Points
 * @param referencePitch Pitch of the reference points (in number of columns)
 * @param query Query points, size of (dimension, queryNum)
 * @param queryWidth Number of query points
 * @param queryPitch Pitch of the query points (in number of columns)
 * @param dimension Embedding dimensions
 * @param similarity output array, of size (referenceNum, queryNum), stored similarities
 * @param queryNorm 1D array, consists of L2 norm of query points
 * @param referenceNorm 1Dc array consists of L2 norm of reference points.
 * @return __global__ 
 */
__global__ void computeSimilarity(float* reference,
                                  int referenceWidth,
                                  int referencePitch,
                                  float* query,
                                  int queryWidth,
                                  int queryPitch,
                                  int dimension,
                                  float* similarity,
                                  float* queryNorm,
                                  float* referenceNorm) 
{
    // Define shared memory to store the submatrices of reference and query matrices.
    // A: reference matrix, B: query matrix.
    __shared__ float sharedA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sharedB[BLOCK_DIM][BLOCK_DIM];
    __shared__ float normA[BLOCK_DIM];
    __shared__ float normB[BLOCK_DIM];

    // Range for matrix A and matrix B.
    __shared__ int startA;
    __shared__ int startB;
    __shared__ int stepA;
    __shared__ int stepB;
    __shared__ int endA;

    // Thread index
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    
    // Initialize inner product result 
    float innerProd = 0;

    // Initialize range and step
    startA = BLOCK_DIM * blockIdx.y;
    startB = BLOCK_DIM * blockIdx.x;
    stepA = BLOCK_DIM * referencePitch;
    stepB = BLOCK_DIM * queryPitch;
    endA = startA + (dimension - 1) * referencePitch;

    // Conditions for loading shared memory / inner product computation / write results
    int condition1 = (startA + threadX < referenceWidth);
    int condition2 = (startB + threadX < queryWidth);
    int condition3  = (startA + threadY < referenceWidth);

    // Loop over the submatrices and compute the inner product
    for (int a = startA, b = startB; a <= endA; a += stepA, b += stepB) 
    {
        // Load the embedding values into the shared memory.
        if (a / referencePitch + threadY < dimension) 
        {
            sharedA[threadY][threadX] = (condition1) ? reference[a + referencePitch * threadY + threadX] : 0;
            sharedB[threadY][threadX] = (condition2) ? query[b + queryPitch * threadY + threadX] : 0;
        }
        else 
        {
            sharedA[threadY][threadX] = 0;
            sharedB[threadY][threadX] = 0;
        }
        
        // synchronize the embedding loading process;
        __syncthreads();

        // Load the norms
        normA[threadX] = (condition1) ? referenceNorm[startA + threadX] : 1;
        normB[threadY] = (condition2) ? queryNorm[startB + threadX] : 1;
        // synchronize the norm loading process;
        __syncthreads();

        // compute the inner product
        if (condition2 && condition3) 
        {
            for (int k = 0; k < BLOCK_DIM; k++) 
            {
                innerProd += sharedA[k][threadY] * sharedB[k][threadX];
            }
        }
        // synchronize the computation
        __syncthreads();
    }
    // Write the result if meet the constraints.
    if (condition2 && condition3) 
    {
        similarity[(startA + threadY) * queryPitch + startB + threadX] = innerProd / (normA[threadY] * normB[threadX]);
    }
}

/**
 * @brief For each of the reference point, we find the k largest similarities in the 
 *        similarity matrix. 
 *        
 *        We are finding k nearest embeddings, so we need to sort. 
 *        Notice that we can have some optimization by not sorting all the entries.
 * 
 * @param similarity Similarity matrix of size (referenceNum, queryNum)
 * @param simialrityPitch Pitch of the similarity matrix
 * @param indices Indices matrix
 * @param indicesPitch Pitch of the indices matrix
 * @param width Width of the similarity matrix and indices matrix
 * @param height Height of the similarity matrix
 * @param k Number of neighbors we want
 * @return 
 */
__global__ void insertionSort(float* similarity, 
                              int similarityPitch,
                              int* indices,
                              int indicesPitch,
                              int width,
                              int height,
                              int k) 
{
    // Compute index
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Keep in range
    if(index < width) 
    {
        // Add offset to similarity ptr & indices ptr
        float* similarityPtr = similarity + index;
        int* indicesPtr = indices + index; 
        // Initialize the first indices entry with 0
        indicesPtr[0] = 0;
        // Iterate along query points' dimension
        for (int i = 1; i < height; i++) 
        {
            // current similarity & correpsonding index
            float curSimilarity = similarityPtr[i * similarityPitch];
            int curIndex = i;
            // If current similarity is smaller than the k-th sorted largest element,
            // And current index is larger than k, skip.
            if (i >= k && curSimilarity <= similarityPtr[(k - 1) * similarityPitch]) 
            {
                continue;
            }
            // Move similarities and indices smaller than current similarity to the right
            int j = min(i, k - 1);
            while (j > 0 && similarityPtr[(j - 1) * similarityPitch] < curSimilarity) 
            {
                similarityPtr[j * similarityPitch] = similarityPtr[(j - 1) * similarityPitch];
                indicesPtr[j * indicesPitch] = indicesPtr[(j - 1) * indicesPitch];
                j--; 
            }
            // Set current similarity and index to the right position.
            similarityPtr[j * similarityPitch] = curSimilarity;
            indicesPtr[j * indicesPitch] = curIndex;
        }
    }

}
/**
 * @brief Since we are using cosine similarity, 
 *        computing the L2 norm of each point is necessary. 
 * 
 * @param inputArray Input array, we assume that one column stands for one point
 * @param width Number of columns in the array.
 * @param pitch Pitch of the array
 * @param height In our setting, it's the embedding dimension.
 * @param outputArray We store the l2 norms as a column vector
 * @return  
 */
__global__ void computeL2Norm(float* inputArray, 
                              int width, 
                              int pitch, 
                              int height,
                              float* outputArray)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width) 
    {   // Always ensure we are in the range.
        float sum = 0;
        for (int i = 0; i < height; i++) 
        {
            sum += inputArray[i * pitch + index] * inputArray[i * pitch + index];
        }
        sum = sqrt(sum);
        outputArray[index] = sum;
    }
}

float knnSearch(const float* reference,
              int          referenceNum, 
              const float* query, 
              int          queryNum, 
              int          dimension, 
              int          k,
              float*       knnSimilarity,
              int*       knnIndex) 
{
    // Reference: timing.md, record event time.
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float executionTime = 0;
    float runningTime;
    const unsigned int floatSize = sizeof(float);
    const unsigned int intSize = sizeof(int);

    // Cuda Error variables
    cudaError_t error1, error2, error3, error4, error5, error6;

    // Check devices
    int numOfDevices;
    error1  = cudaGetDeviceCount(&numOfDevices);
    if (error1 != cudaSuccess || numOfDevices == 0) 
    {
        std::cout << "No CUDA devices!" << std::endl; 
        return -1;
    } 
    // Set device
    error1 = cudaSetDevice(0);
    if (error1 != cudaSuccess) 
    {
        std::cout << "Cannot set CUDA device!" << std::endl;
        return -1; 
    }
    // Allocate global memory
    float* referenceDevice; 
    float* queryDevice; 
    float* similarityDevice; 
    float* refNormDevice; 
    float* queryNormDevice;
    int* indicesDevice;
    size_t referencePitchBytes, queryPitchBytes, similarityPitchBytes, indicesPitchBytes, refNormPitchBytes, queryNormPitchBytes;
    error1 = cudaMallocPitch((void**)&referenceDevice, &referencePitchBytes, referenceNum * floatSize, dimension);
    error2 = cudaMallocPitch((void**)&queryDevice, &queryPitchBytes, queryNum * floatSize, dimension);
    error3 = cudaMallocPitch((void**)&similarityDevice, &similarityPitchBytes, queryNum * floatSize, referenceNum);
    error4 = cudaMallocPitch((void**)&indicesDevice, &indicesPitchBytes, queryNum * intSize, k);
    error5 = cudaMallocPitch((void**)&refNormDevice, &refNormPitchBytes, referenceNum * floatSize, 1);
    error6 = cudaMallocPitch((void**)&queryNormDevice, &queryNormPitchBytes, queryNum * floatSize, 1);
    if (error1 != cudaSuccess || error2 != cudaSuccess || error3 != cudaSuccess || error4 != cudaSuccess || error5 != cudaSuccess || error6 != cudaSuccess) 
    {
        std::cout << "CUDA memory allocation failed!" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }
    // Compute pitch values
    size_t referencePitch = referencePitchBytes / floatSize;
    size_t queryPitch = queryPitchBytes / floatSize;
    size_t similarityPitch = similarityPitchBytes / floatSize;
    size_t indicesPitch = indicesPitchBytes / intSize;

    // Check if pitch values match
    if (queryPitch != indicesPitch || queryPitch != similarityPitch) 
    {
        std::cout << "Invalid pitch values!" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }

    // Copy data from host to device
    error1 = cudaMemcpy2D(referenceDevice, referencePitchBytes, reference, referenceNum * floatSize, referenceNum * floatSize, dimension, cudaMemcpyHostToDevice);
    error2 = cudaMemcpy2D(queryDevice, queryPitchBytes, query, queryNum * floatSize, queryNum * floatSize, dimension, cudaMemcpyHostToDevice);
    if (error1 != cudaSuccess || error2 != cudaSuccess) 
    {
        std::cout << "Fail to copy from host to device!" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }

    // Compute L2 norm
    // For reference and query norms
    dim3 blockNorm(256, 1, 1);
    dim3 gridRefNorm(referenceNum / 256, 1, 1);
    dim3 gridQueryNorm(queryNum / 256, 1, 1);
    if (referenceNum % 256 != 0) gridRefNorm.x += 1; 
    if (queryNum % 256 != 0) gridQueryNorm.x += 1;
    // For query
    cudaEventRecord(start);
    computeL2Norm<<<gridRefNorm, blockNorm>>>(referenceDevice, referenceNum, referencePitch, dimension, refNormDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runningTime, start, stop); 
    executionTime += runningTime;
    if (cudaGetLastError() != cudaSuccess) 
    {
        std::cout << "Fail to compute L2 norm for reference" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }
    cudaEventRecord(start);
    computeL2Norm<<<gridQueryNorm, blockNorm>>>(queryDevice, queryNum, queryPitch, dimension, queryNormDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runningTime, start, stop); 
    executionTime += runningTime;    
    if (cudaGetLastError() != cudaSuccess) 
    {
        std::cout << "Fail to compute L2 norm for query" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }

    // Compute the similarity
    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0(queryNum / BLOCK_DIM, referenceNum / BLOCK_DIM, 1);
    if (queryNum % BLOCK_DIM != 0) grid0.x += 1;
    if (referenceNum % BLOCK_DIM != 0) grid0.y += 1;
    cudaEventRecord(start);
    computeSimilarity<<<grid0, block0>>>(referenceDevice, referenceNum, referencePitch, queryDevice, queryNum, queryPitch, dimension, similarityDevice, queryNormDevice, refNormDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runningTime, start, stop); 
    executionTime += runningTime;   
    if (cudaGetLastError() != cudaSuccess) 
    {
        std::cout << "Fail to execute similarity computing kernel!" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }

    // Sort the similarity with corresponding indices
    dim3 block1(256, 1, 1);
    dim3 grid1(queryNum / 256, 1, 1);
    if (queryNum % 256 != 0) grid1.x += 1;
    cudaEventRecord(start);
    insertionSort<<<grid1, block1>>>(similarityDevice, similarityPitch, indicesDevice, indicesPitch, queryNum, referenceNum, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runningTime, start, stop); 
    executionTime += runningTime;       
    if (cudaGetLastError() != cudaSuccess) 
    {
        std::cout << "Fail to execute sorting kernel!" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }

    // Copy the k largest similarities and indices from device to host
    error1 = cudaMemcpy2D(knnSimilarity, queryNum * floatSize, similarityDevice, similarityPitchBytes, queryNum * floatSize, k, cudaMemcpyDeviceToHost);
    error2 = cudaMemcpy2D(knnIndex, queryNum * intSize, indicesDevice, indicesPitchBytes, queryNum * intSize, k, cudaMemcpyDeviceToHost);   
    if (error1 != cudaSuccess || error2 != cudaSuccess) 
    {
        std::cout << "Fail to copy from device to host!" << std::endl;
        cudaFree(referenceDevice);
        cudaFree(queryDevice);
        cudaFree(similarityDevice);
        cudaFree(indicesDevice);
        cudaFree(refNormDevice);
        cudaFree(queryNormDevice);
        return -1;
    }

    // Release allocated memory
    cudaFree(referenceDevice);
    cudaFree(queryDevice);
    cudaFree(similarityDevice);
    cudaFree(indicesDevice);
    cudaFree(refNormDevice);
    cudaFree(queryNormDevice);

    return executionTime;
}   