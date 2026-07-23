#include "semantic_graph.h"

// 语义相似度计算函数，使用余弦相似度
double calculateSimilarity(const vector<double> &vec1, const vector<double> &vec2)
{
    if (vec1.empty() || vec2.empty() || vec1.size() != vec2.size())
    {
        return -1.0;
    }

    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (size_t i = 0; i < vec1.size(); i++)
    {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    if (norm1 == 0.0 || norm2 == 0.0)
    {
        return 0.0;
    }

    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}

Graph::Graph(int vertices) : V(vertices), E(0), adj(vertices), adjSet(vertices)
{
}

Graph::Graph(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        V = 0;
        E = 0;
        return;
    }

    string line;
    getline(file, line);
    istringstream ss(line);
    ss >> V;

    int edgeCount;
    ss >> edgeCount;
    E = edgeCount;

    adj.resize(V);
    adjSet.resize(V);
    semanticVectors.resize(V);
    vectorDim = 0;

    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        istringstream edgeStream(line);
        int u, v;
        edgeStream >> u >> v;

        if (u >= V || v >= V)
        {
            cerr << "Warning: Vertex index out of range: " << u << " " << v << endl;
            continue;
        }

        addEdge(u, v);
    }

    file.close();
}

void Graph::addEdge(int u, int v)
{
    adj[u].push_back(v);
    adj[v].push_back(u);
    adjSet[u].insert(v);
    adjSet[v].insert(u);
    E++;
}

int Graph::getEdgeCount() const
{
    return E;
}

bool Graph::isAdjacent(int u, int v) const
{
    return adjSet[u].find(v) != adjSet[u].end();
}

bool Graph::loadVectorsFromFile(const string &vectorFile)
{
    ifstream file(vectorFile);
    if (!file.is_open())
    {
        cerr << "Error opening vector file: " << vectorFile << endl;
        return false;
    }

    vectorDim = 0;
    
    // 逐行读取并解析
    string line;
    bool inVector = false;
    vector<double> currentVec;
    int currentVertexId = -1;
    string numStr;
    
    while (getline(file, line))
    {
        // 查找顶点ID "xxx": [
        size_t keyPos = line.find('"');
        if (keyPos != string::npos)
        {
            size_t keyEnd = line.find('"', keyPos + 1);
            if (keyEnd != string::npos && keyEnd > keyPos + 1)
            {
                string key = line.substr(keyPos + 1, keyEnd - keyPos - 1);
                
                // 查找 [
                size_t bracketPos = line.find('[', keyEnd);
                if (bracketPos != string::npos)
                {
                    currentVertexId = stoi(key);
                    inVector = true;
                    currentVec.clear();
                    numStr.clear();
                    
                    // 解析该行的数字
                    size_t i = bracketPos + 1;
                    while (i < line.size())
                    {
                        char c = line[i];
                        if (c == ']' || c == ',')
                        {
                            if (!numStr.empty())
                            {
                                numStr.erase(0, numStr.find_first_not_of(" \t"));
                                numStr.erase(numStr.find_last_not_of(" \t") + 1);
                                if (!numStr.empty())
                                {
                                    currentVec.push_back(stod(numStr));
                                }
                                numStr.clear();
                            }
                            if (c == ']')
                            {
                                inVector = false;
                                break;
                            }
                        }
                        else if (!isspace(static_cast<unsigned char>(c)))
                        {
                            numStr += c;
                        }
                        i++;
                    }
                    
                    // 存储向量
                    if (currentVertexId >= 0 && currentVertexId < V)
                    {
                        semanticVectors[currentVertexId] = currentVec;
                        if (vectorDim == 0)
                        {
                            vectorDim = currentVec.size();
                        }
                    }
                }
            }
        }
    }
    
    file.close();
    cout << "Loaded vectors from " << vectorFile << endl;
    cout << "Vector dimension: " << vectorDim << endl;
    return true;
}

void Graph::setSemanticVector(int v, const vector<double> &vec)
{
    if (v >= 0 && v < V)
    {
        semanticVectors[v] = vec;
        vectorDim = vec.size();
    }
}

const vector<double>& Graph::getSemanticVector(int v) const
{
    static vector<double> empty;
    if (v >= 0 && v < V)
    {
        return semanticVectors[v];
    }
    return empty;
}

bool Graph::loadVectorsFromBinary(const string &binaryFile)
{
    ifstream file(binaryFile, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening binary vector file: " << binaryFile << endl;
        return false;
    }

    int V_in, dim_in;
    file.read(reinterpret_cast<char*>(&V_in), sizeof(int));
    file.read(reinterpret_cast<char*>(&dim_in), sizeof(int));

    if (V_in != V)
    {
        cerr << "Error: vertex count mismatch " << V_in << " vs " << V << endl;
        return false;
    }

    vectorDim = dim_in;
    semanticVectors.resize(V);

    size_t perVertex = (size_t)dim_in * sizeof(double);
    for (int i = 0; i < V; i++)
    {
        semanticVectors[i].resize(dim_in);
        file.read(reinterpret_cast<char*>(semanticVectors[i].data()), perVertex);
    }

    file.close();
    cout << "Loaded binary vectors from " << binaryFile << endl;
    cout << "Vector dimension: " << vectorDim << endl;
    return true;
}

bool Graph::loadVectorsFromChunks(const string &indexFile)
{
    ifstream index(indexFile);
    if (!index.is_open())
    {
        cerr << "Error opening index file: " << indexFile << endl;
        return false;
    }
    
    int totalVertices, vectorDim;
    index >> totalVertices >> vectorDim;
    
    if (totalVertices != V)
    {
        cerr << "Error: vertex count mismatch " << totalVertices << " vs " << V << endl;
        return false;
    }
    
    this->vectorDim = vectorDim;
    semanticVectors.resize(V);
    
    string chunkFile;
    int loadedVertices = 0;
    
    while (index >> chunkFile)
    {
        ifstream chunk(chunkFile, ios::binary);
        if (!chunk.is_open())
        {
            cerr << "Error opening chunk file: " << chunkFile << endl;
            return false;
        }
        
        // 检查文件大小
        chunk.seekg(0, ios::end);
        size_t fileSize = chunk.tellg();
        chunk.seekg(0, ios::beg);
        
        // 读取向量数量和维度（如果文件有头部）
        int chunkVertices, chunkDim;
        chunk.read(reinterpret_cast<char*>(&chunkVertices), sizeof(int));
        chunk.read(reinterpret_cast<char*>(&chunkDim), sizeof(int));
        
        if (chunkDim != vectorDim)
        {
            cerr << "Error: dimension mismatch in chunk " << chunkFile << endl;
            return false;
        }
        
        // 计算每个向量的大小
        size_t perVertex = (size_t)chunkDim * sizeof(double);
        
        // 读取所有向量
        for (int i = 0; i < chunkVertices; i++)
        {
            if (loadedVertices + i >= V)
            {
                cerr << "Error: trying to load more vertices than expected" << endl;
                return false;
            }
            semanticVectors[loadedVertices + i].resize(chunkDim);
            chunk.read(reinterpret_cast<char*>(semanticVectors[loadedVertices + i].data()), perVertex);
        }
        
        loadedVertices += chunkVertices;
        chunk.close();
        cout << "Loaded chunk: " << chunkFile << " (" << chunkVertices << " vertices)" << endl;
    }
    
    index.close();
    cout << "Loaded vectors from " << loadedVertices << " vertices across multiple chunks" << endl;
    cout << "Vector dimension: " << vectorDim << endl;
    return true;
}

void Graph::saveVectorsToBinary(const string &binaryFile) const
{
    ofstream file(binaryFile, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening binary file for write: " << binaryFile << endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&V), sizeof(int));
    file.write(reinterpret_cast<const char*>(&vectorDim), sizeof(int));

    size_t perVertex = (size_t)vectorDim * sizeof(double);
    for (int i = 0; i < V; i++)
    {
        file.write(reinterpret_cast<const char*>(semanticVectors[i].data()), perVertex);
    }

    file.close();
    cout << "Saved binary vectors to " << binaryFile << endl;
}

void Graph::normalizeVectors()
{
    cout << "Normalizing vectors...\n";
    
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < V; i++)
    {
        double norm = 0.0;
        for (double val : semanticVectors[i])
        {
            norm += val * val;
        }
        
        if (norm > 1e-9)
        {
            double invNorm = 1.0 / sqrt(norm);
            for (double &val : semanticVectors[i])
            {
                val *= invNorm;
            }
        }
    }
    
    cout << "Vectors normalized successfully\n";
}
