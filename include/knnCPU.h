#ifndef knnCPU_H    
#define knnCPU_H
#include<math.h>
// Implement KNN using CPU. 
/**
 * @brief We want to do some comparison of the knn GPU methods and the CPU methods,
 * The following function compute the cosine similarity between a given reference-
 * query pair.
 * @param reference the reference data points (on host)
 * @param referenceNum number of reference points 
 * @param query the query data points (on host)
 * @param queryNum number of query points
 * @param dimension embedding dimension
 * @param refIndex which reference point we are considering
 * @param queryIndex which query point we are considering
 * @param refNorm L2 norm of the reference points
 * @param queryNorm L2 norm of the queryNorm
 * @return float 
 */
float computeSimilarity_CPU(float* reference,
                                  int referenceNum,
                                  float* query,
                                  int queryNum,
                                  int dimension,
                                  int refIndex,
                                  int queryIndex,
                                  float* refNorm,
                                  float* queryNorm);

/**
 * @brief Compute L2 norm with CPU.
 * 
 * @param inputArray The input array, assuming we have size: dimension * width.
 * @param outputArray The output array, a vector of size (width, 1)
 * @param dimension Embedding dimension
 * @param width 
 */
void computeL2Norm_CPU(float* inputArray, 
                       float* outputArray,
                       int dimension,
                       int width);

/**
 * @brief This insertionSort_GPU is slightly different from the GPU version.
 *        We are computing the similarity for each query-embedding pair, 
 *        So, it's possible to include the insertionSort step into a loop.
 *        The parameter is fewer than the GPU version (Do not need to know the 
 *        number of queries.)
 * @param similarity 
 * @param indices 
 * @param width 
 * @param k 
 */
void insertionSort_CPU(float* similarity, 
                       int* indices,
                       int width,
                       int k);  


#endif