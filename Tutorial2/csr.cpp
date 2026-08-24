#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

struct Edge
{
    int src;
    int dst;
    int weight;
};

int main()
{
    ifstream file("graph.txt");
    if (!file){
        cout << "Error opening graph.txt" << endl;
        return 1;
    }

    int V, E;
    file >> V >> E;

    vector<Edge> edges(E);

    for (int i = 0; i < E; i++){
        file >> edges[i].src
             >> edges[i].dst
             >> edges[i].weight;
    }

    //CSR format arrays
    vector<int> row_ptr(V + 1, 0);
    vector<int> col(E);
    vector<int> weight(E);

    //outgoing edges
    for (int i = 0; i < E; i++){
        row_ptr[edges[i].src + 1]++;
    }

    //row pointers
    for (int i = 1; i <= V; i++){
        row_ptr[i] += row_ptr[i - 1];
    }

    //temporary position array to keep track of the current position in col and weight arrays   
    vector<int> position = row_ptr;


    for (int i = 0; i < E; i++){
        int src = edges[i].src;

        col[position[src]] = edges[i].dst;
        weight[position[src]] = edges[i].weight;

        position[src]++;
    }
    ofstream output("csr.txt");

    if (!output){
        cout << "Error creating csr.txt" << endl;
        return 1;
    }

    output << V << " " << E << endl;

    for (int x : row_ptr){
        output << x << " ";
    }    

    output << endl;

    for (int x : col){
        output << x << " ";
    }

    output << endl;

    for (int x : weight){
        output << x << " ";
    }

    output << endl;
    output.close();

    // Display CSR
    cout << "CSR Format" << endl;

    cout << "row_ptr: ";
    for (int x : row_ptr){
        cout << x << " ";
    }

    cout << endl;

    cout << "col:";
    for (int x : col){
        cout << x << " ";
    }

    cout << endl;

    cout << "weight:";
    for (int x : weight){
        cout << x << " ";
    }

    cout << endl;
    return 0;
}
