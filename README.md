# ME/CS/ECE759 Final Project
## Implement k Nearest Neighbor Algorithm with CUDA
Haitao Huang

### Repo Structure
We have seven folders and this README file. 
```
FinalProject759/
├── data, This folder stores the data for debugging. 
├── debug, This folder includes the debugging scripts for CUDA/C++ implementations.
├── include, This folder consists of header files.
├── output, This folder stores the results of our experiments.
├── scaling, This folder stores the scaling analysis scripts.
├── source, You can find the implementation of k-NN search here.
├── visualizations, Containing the resulting charts and a jupyter notebook that genereates them.
```

### Instructions for Running the Scripts
<span style="color:red">
Notice: Please stay in the directory <i>FinalProject759</i> and do not descend into its child directories.
</span>

#### Running the Debugging Scripts
To reproduce the results in section 4.1, please run the following scripts.
*(CUDA implementation)*  
```
nvcc -o debugExec source/knnSearch.cu debug/CUDA/debug.cu 
./debugExec
```
*(C++ implementation)*
```
g++ debug/cpp/debugCPU.cpp source/knnCPU.cpp -Wall -O3 -std=c++17 -o debugCPU
./debugCPU 
```

We also have extra debugging scripts and you can use the following commands to run them. 
*(CUDA implementation)*
```
nvcc -o debugExtra source/knnSearch.cu debug/CUDA/debugExtra.cu
./debugExtra
```
*(C++ implementation)*
```
g++ debug/cpp/debugCPU_extra.cpp source/knnCPU.cpp -Wall -O3 -std=c++17 -o debugCPUExtra
./debugCPUExtra 
```
#### Checking the Scaling Results
To check the scaling behavior, please use the following compile commands.
*(CUDA implementation)*
```
nvcc -o scalingGPU source/knnSearch.cu scaling/scaling.cu
```
*(C++ implementation)*
```
g++ scaling/scalingCPU.cpp source/knnCPU.cpp -Wall -O3 -std=c++17 -o scalingCPU
```
In both cases, the main functions take in the following command line arguments (in order): reference number, query number, embedding dimension, and k.

The commands below run the CUDA implementation of the k-NN search for reference number $1024$, query number $100$, embedding dimension $200$, and $k=10$. 
```
./scalingGPU 1024 100 200 10
```
Replace scalingGPU by scalingCPU, you can run the C++ implementation with the same setting. One last comment is that the code might not be able to execute when the number of reference points exceeds $2^{20}$.

### Relevant Notes

**1. Pitched pointer & padding in CUDA** 

[Reference_1](https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api), [Reference_2](https://nichijou.co/cuda5-coalesce/), 
[Reference_3](https://nichijou.co/cudaRandom-memAlign/),
[Reference_4](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
`
In the class, we have learned about commands like ```cudaMalloc()```. The result of ```cudaMalloc()``` is similar to that of ```malloc()```: We obtain a contiguous memory chunk. Things will be different if we treat the memory as a 2D array.

Memory access on GPU works much better if the data items are aligned. When allocating 2D arrays, if we can let every row starts at a memory boundary address, we may improve the performance. Pitch can help us with the alignment process.

When using pitch with 2D arrays, the rows are padded with extra space to make them multiples of 64 or 128. ```cudaMallocPitch()``` and ```cudaMemcpy2D()``` will be important for allocation. 
****

**2. Pitch and 2D memory copy in CUDA** 

```
cudaMallocPitch(void** devPtr, 
                size_t* pitch, 
                size_t width,
                size_t height)

devPtr: Pointer to allocated pitched device memory
pitch: Pitch for allocation
width: Requested pitched allocation width (in bytes)
height: Requested pitched allocation height
```
```cudaMallocPitch``` will allocate at least width (in bytes) * height bytes of linear memory on device. The value returned in ```*pitch``` gives the width in bytes of the allocation. Then, we can use it to compute addresses within the 2D array. Given the row and column of an array element of type T, the address is computed as:
```T* pElement = (T*) ((char*) baseAddress + Row * pitch) + Column;```
We can interpret the code above as follows. We use ```baseAddress``` and ```Row * pitch``` to get the starting address of the given row. Then, access the column element just like ```givenRow[Column]```.

Now, we proceed to the usage of ```cudaMemcpy2D()```。 Different from the ```cudaMemcpy()``` we have tried in the assignments, ```cudaMemcpy2D()``` takes pitch into the consideration when copying memory. 
```
cudaMemcpy2D (void* dst, 
              size_t dpitch, 
              const void* src, 
              size_t spitch, 
              size_t width, 
              size_t height, 
              cudaMemcpyKind kind)

dst: Destination memory address
dpitch: Pitch of destination memory
src: Source memory address
spitch: Pitch of source memory
width: Width of matrix transfer (columns in bytes)
height: Height of matrix transfer (rows)
kind: Type of transfer
```

**3. More comments on the functions in** ```knnSearch.cu```

**insertionSort( )**

```
insertionSort(float* similarity, 
                              int simialrityPitch,
                              int* indices,
                              int indicesPitch,
                              int width,
                              int height,
                              int k)
```

We need to know that the width refers to the number of query points and height refers to the number of reference points. The entry (i, j) in the similarity matrix stands for the similarity between the i-th reference point and j-th query point. Notice that we have pitches, so some of the entries are void.

When doing the insertion sort, we do a column-wise operation (Each thread in a block deals with one query point). We sort the column along the way. If the current row index is i such that i >= k, and that the similarity is lower than previous entry, we skip this element. 

Otherwise, we do the usual insertion sort steps. The important note is that we are computing cosine similarity, so there will be some adaptations for those signs.

**computeSimilarity()**

```
computeSimilarity(float* reference,
                                  int referenceWidth,
                                  int referencePitch,
                                  float* query,
                                  int queryWidth,
                                  int queryPitch,
                                  int dimension,
                                  float* similarity) 
```

This function is particularly interesting, as we need to apply the shared memory knowledge that we have learned in class. In the original code, the author chooses to use l2 distance. We use cosine similarity, so some adaptations are necessary. 

```reference``` and ```query``` are two important matrices. The columns in those matrices stand for one point (reference or query). In other words, the number of rows equal to the embedding size. In this implementation, the block and grid both have 2D structures. We are computing the similarity between points, so the grid size is defined as ```dim3 grid0(queryNum / BLOCK_DIM, referenceNum / BLOCK_DIM, 1);```. Each block has a size of ```block0(BLOCK_DIM, BLOCK_DIM, 1)```, where we set ```BLOCK_DIM``` to 16. Thus, there are 256 threads. Each thread computes one element in the similarity matrix.

We use shared memory to efficiently access the data. The two shared matrices ```sharedA, sharedB``` stores the elements in the reference matrix and the query matrix respectively. The starting positions of the sub-matrix computation are defined as: 

```startA = BLOCK_DIM * blockIdx.y;```
```startB = BLOCK_DIM * blockIdx.x;```

With step sizes as

```stepA = BLOCK_DIM * referencePitch;```
```stepB = BLOCK_DIM * queryPitch;```

To prevent out of range issue, we have the following two conditions for loading data into the shared array. Notice the indexing convenction for 2D block structure, to ensure the data do not exceed width, we use ```threadX = threadIdx.x```.

```int condition1 = (startA + threadX < referenceWidth);```
```int condition2 = (startB + threadX < queryWidth);```

```threadY = threadIdx.y``` is used for accessing the entries within a data point. The scripts below show the process of loading data into the shared memory. 

```
    // Loop over the submatrices and compute the inner product
    for (int a = startA, b = startB; a <= endA; a += stepA, b += stepB) 
    {
        // Load the values into the shared memory.
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
        
        // synchronize the loading process;
        __syncthreads();
    }
```

At last, we need to determine which entry in the output array should we write to. The output array has the shape (referenceNum, queryPitch). Each thread is computing the similarity between the (startB + threadX)-th element in the query array and the (startA + threadY)-th element in the reference array. Therefore, we also need to check if (startA + threadY) is outside of reference width. 
```int condition3  = (startA + threadY < referenceWidth);``` .

The result should be written into the right position in the following fashion:
```
// Write the result if meet the constraints.
if (condition2 && condition3) 
    {
        similarity[(startA + threadY) * queryPitch + startB + threadX] = innerProd;
    }
```

**Add cosine similarity scripts**

The discussion above is about the general computation of similarity. To compute the cosine similarity, we need to compute inner product and scale the result by L2 norm. In this case, we have to load the norm into shared memory as well.
```
// Load the norms
normA[threadX] = (condition1) ? referenceNorm[startA + threadX] : 1;
normB[threadY] = (condition2) ? queryNorm[startB + threadX] : 1;
// synchronize the norm loading process;
__syncthreads();
```

**4. Debug And Testing Log**

To ensure the implementation is correct, we use text embedding data in GloVe.

> GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

To simplify the case, we choose the embedding dimension as $25$. The query words are \{ person, male, bread, taco, chicken, at, on, you\} and the reference words are \{ people, children, adult, animal, dog, cat, duck, spicy, roast, over, the, cake, cheese, milk, cow, around, to, with, in, me, they, badger, without\}.

Compile command: ```nvcc -o debug knnSearch.cu debug.cu -lcuda -Wno-deprecated-gpu-targets```, ```(for pure cpp code) g++ debugCPU.cpp knnCPU.cpp -Wall -O3 -std=c++17 -o debugCPU```

**4.1 Reading .txt files in C++**

We created query embeddings file and reference embeddings file for debugging. The two files have the following structure.
```
queryEmbeddings.txt
person -0.079901 -0.091993 -0.46233 0.29856 0.78472 0.071502 1.5598 -0.36813 -0.8971 0.11984 -0.58358 0.56479 -4.8192 0.41785 -0.52116 1.0823 0.93162 -0.15663 0.28685 -0.84895 1.3194 0.36093 -0.38828 -0.17877 -0.88096
male -0.69132 -0.24751 -0.091082 -0.50218 1.1348 -0.84616 0.87484 0.26716 -0.11469 0.62137 -0.057546 -1.7072 -2.8744 0.45555 -0.73985 1.4007 0.08409 0.001865 -0.38978 -0.24283 0.29985 1.0122 -0.05706 -1.3766 -0.7072
(6 rows omitted)
```

We decide to use ```ifstream``` and ```sstream``` to read the data. Cpp seems to have a relatively complicated data I/O method. The following method only works for the data format of ```queryEmbeddings.txt / refEmbeddings.txt```.

```
    std::ifstream referenceFile("data/debug/refEmbeddings.txt"); 
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
```
**4.2 Debug and test results**
We compile and run the file on Euler. Below shows the result.

```
===== Debug Configuration =====
Dimension: 25
Number of Reference Points: 23
queryNum: 8
k: 3
The 3 closest words for "person" are 
"people they without" 

The 3 closest words for "male" are 
"animal cat dog" 

The 3 closest words for "bread" are 
"cheese milk roast" 

The 3 closest words for "taco" are 
"milk cheese dog" 

The 3 closest words for "chicken" are 
"cheese spicy milk" 

The 3 closest words for "at" are 
"the to with" 

The 3 closest words for "on" are 
"the in with"

The 3 closest words for "you" are 
"they with people" 
```
Recall that:
>The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence.

It seems that the above results are plausible. E.g. For "male", it might often occur with animal, cat, or dog. For "at", a preposition, can occur with the, to, or with. We also conduct KNN search using ```sklearn``` on Python. The results are the same. When checking the output log, we do not see cuda errors. 

**5. Scaling Analysis**

Scaling analysis will be an interesting part of this project. We can control the range of the following parameters. 

- Number of reference points
- Number of query points
- Value of k
- Value of embedding dimensions


For simplicity's sake, we generate random data instead of pre-trained embeddings. 

Compile commands:```nvcc -o scaling knnSearch.cu scaling.cu -lcuda -Wno-deprecated-gpu-targets```, ```g++ scalingCPU.cpp knnCPU.cpp -Wall -O3 -std=c++17 -o scalingCPU ```

**5.1 Number of Reference Points**

We will investigate the following range. Other parameters will be fixed. 
$$2^5, 2^6, ..., 2^{19}$$

**5.2 Number of Query Points**
The number of query points has the following range.
$$2^4, 2^5, ..., 2^{13}$$

**5.3 Value of k**

$$5, 10, 20, 40, 80, 160, 320$$

**5.4 Value of Embedding Dimensions**

$$25, 50, 100, 200, 400, 800$$

