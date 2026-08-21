#include <iostream>
#include <fstream>
#include <vector>
#include <climits>

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

    int V, E;
    file >> V >> E;

    vector<Edge> edges(E);

    
    for (int i = 0; i < E; i++)
    {
        file >> edges[i].src
             >> edges[i].dst
             >> edges[i].weight;
    }

    int source;

    cout << "Enter source vertex: ";
    cin >> source;

    //dijkstra
    vector<int> distance(V, INT_MAX);
    vector<bool> visited(V, false);

    distance[source] = 0;

    for (int i = 0; i < V; i++)
    {
        int u = -1;

        for (int v = 0; v < V; v++)
        {
            if (!visited[v] &&
                distance[v] != INT_MAX &&
                (u == -1 || distance[v] < distance[u]))
            {
                u = v;
            }
        }

        if (u == -1)
            break;

        visited[u] = true;

        //all edges checking
        for (int j = 0; j < E; j++)
        {
            if (edges[j].src == u)
            {
                int v = edges[j].dst;
                int w = edges[j].weight;

                if (distance[u] + w < distance[v])
                {
                    distance[v] = distance[u] + w;
                }
            }
        }
    }

    //result
    cout << "\nCPU SSSP Result\n";

    for (int i = 0; i < V; i++)
    {
        cout << "Vertex " << i << " : ";

        if (distance[i] == INT_MAX)
            cout << "INF";
        else
            cout << distance[i];

        cout << endl;
    }

    return 0;
}
