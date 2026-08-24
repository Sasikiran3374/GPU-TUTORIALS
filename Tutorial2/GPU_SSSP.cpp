#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <cuda_runtime.h>

using namespace std;

#define INF INT_MAX

__global__ void ssspKernel(
    int V,
    const int *row_ptr,
    const int *col,
    const int *weight,
    int *distance,
    bool *changed)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V) return;

    if (distance[u] == INF) return;

    for (int i = row_ptr[u]; i < row_ptr[u + 1]; i++){
        int v = col[i];
        int w = weight[i];
        int newDistance = distance[u] + w;

        if (newDistance < distance[v]){
            atomicMin(&distance[v], newDistance);
            *changed = true;
        }
    }
}

int main()
{
    ifstream file("csr.txt");
    if (!file){
        cout << "Error opening csr.txt" << endl;
        return 1;
    }

    int V, E;
    file >> V >> E;

    vector<int> h_row_ptr(V + 1);
    vector<int> h_col(E);
    vector<int> h_weight(E);

    for (int i = 0; i <= V; i++){
        file >> h_row_ptr[i];
    }

    for (int i = 0; i < E; i++){
        file >> h_col[i];
    }

    for (int i = 0; i < E; i++){
        file >> h_weight[i];
    }

    file.close();

    int source;

    cout << "Enter source vertex: ";
    cin >> source;

    if (source < 0 || source >= V){
        cout << "Invalid source vertex" << endl;
        return 1;
    }

    int *d_row_ptr;
    int *d_col;
    int *d_weight;
    int *d_distance;
    bool *d_changed;

    cudaMalloc(&d_row_ptr, (V + 1) * sizeof(int));
    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_weight, E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));


    cudaMemcpy(d_row_ptr,h_row_ptr.data(),(V + 1) * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_col,h_col.data(),E * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_weight,h_weight.data(),E * sizeof(int),cudaMemcpyHostToDevice);

    vector<int> h_distance(V, INF);
    h_distance[source] = 0;

    cudaMemcpy(d_distance,h_distance.data(),V * sizeof(int),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (V + threadsPerBlock - 1)/ threadsPerBlock;

    bool changed = true;

    while (changed)
    {
        changed = false;

        cudaMemcpy(d_changed,&changed,sizeof(bool),cudaMemcpyHostToDevice);

        ssspKernel<<<blocks, threadsPerBlock>>>(V,d_row_ptr,d_col,d_weight,d_distance,d_changed);

        cudaDeviceSynchronize();
        cudaMemcpy(&changed,d_changed,sizeof(bool),cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(h_distance.data(),d_distance,V * sizeof(int),cudaMemcpyDeviceToHost);

    cout << "\nGPU SSSP Result" << endl;

    for (int i = 0; i < V; i++){
        cout << "Vertex " << i << " : ";

        if (h_distance[i] == INF){
            cout << "INF";
        }
        else{
            cout << h_distance[i];
        }

        cout << endl;
    }
    //free the allocated memory
    cudaFree(d_row_ptr);
    cudaFree(d_col);
    cudaFree(d_weight);
    cudaFree(d_distance);
    cudaFree(d_changed);

    return 0;
}
