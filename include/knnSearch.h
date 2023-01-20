/**
 * @brief Header file for the function knnSearch.
 * 
 * @param reference     reference points, of size dimension * referenceNum.
 *                      e.g., we have N job positions in the form of d dimension embeddings,
 *                      The reference array will have the size d * N.
 * @param referenceNum  Number of reference points, using the example above, N.
 * @param query         The points of which you want to retrieve knn. e.g., Now we have
 *                      M job seekers with their embeddings of d dimensional.
 *                      In such case, the query array will have size d * M
 * @param queryNum      Number of query points, using the example above, M.
 * @param dimension     Dimension of the point vector (or embedding dimension), d.
 * @param k             Number of neighbors we want to retrieve.
 * @param knnSimilarity   Output array, with distances to each of the k neighbors, queryNum * k.
 * @param knnIndex      Output array, with index of the k neighbors, queryNum * k.
 * @return double         >= 0 : Execution completes successfully, return the execution time
 *                      -1: Meet some troubles in allocation memory / Other issues.
 *                      
 */
float knnSearch(const float* reference,
              int          referenceNum, 
              const float* query, 
              int          queryNum, 
              int          dimension, 
              int          k,
              float*       knnSimilarity,
              int*       knnIndex);