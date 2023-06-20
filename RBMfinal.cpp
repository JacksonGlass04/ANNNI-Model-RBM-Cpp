// Import Statements
#include <cmath>
#include <string>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;

// Global Constants
const int L = 40;
const int N = 1000;

const int corrSteps = 20;

const int I = L;
const int J = 100;

const double lr = 0.075;

const int epochs = 40;

int t = 0;

// RBM Layers (input, hidden, reconstructed)
int Configs[N][L];
int Hidden[N][J];
int Reconst[N][L];

// RBM Weights
double Weights[I][J];

// Correlation
double CorrFunc[corrSteps];

// Set Seed
void init_srand(void)
{
    time_t seconds;
    time(&seconds);
    srand48(seconds);
}

// Mod Function
int mod(int x, int m)
{
    if (x >= 0 && x < m)
        return x;
    else if (x < 0)
        return m - 1 - mod(-1 - x, m);
    else
        return x % m;
}

//
//
// Import Dataset / Prepare Data
// Read in configs, Create weights matrix
//
// Read in data
void importConfigs()
{
    ifstream inFile;
    inFile.open("FinalConfigurations.txt");

    string line;
    for (int i = 0; i < N; i++)
    {
        getline(inFile, line);
        stringstream ss(line);
        for (int j = 0; j < L; j++)
        {
            string data;
            getline(ss, data, ',');
            int data_numeric = stoi(data);
            Configs[i][j] = data_numeric;
        }
    }
}

// Print functions for testing
void printConfigs()
{
    for (int i = 0; i < 10; i++) // CHANGE BACK TO N
    {
        for (int j = 0; j < L; j++)
        {
            cout << Configs[i][j] << "\t";
        }
        cout << endl;
    }
}

void printRecon()
{
    for (int i = 0; i < 10; i++) // CHANGE BACK TO N
    {
        for (int j = 0; j < L; j++)
        {
            cout << Reconst[i][j] << "\t";
        }
        cout << endl;
    }
}

// Initialize Weights
void initWeights()
{
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < J; j++)
        {
            Weights[i][j] = drand48() - drand48();
        }
    }
}

// Convert data to binary 0,1
void makeBinary()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < L; j++)
        {
            Configs[i][j] = (Configs[i][j] == -1) ? 0 : 1;
        }
    }
}

// Reset Hidden Layer Values
void resetHidden()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < J; j++)
        {
            Hidden[i][j] = 0;
        }
    }
}

// Actication Function
double Sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

//
//
// Train RBM
// Energy, Forward prop, back propagation, loss function
//
// Energy of RBM given sigma, tau (theta = 0)
double Energy()
{
    double sum = 0;
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < J; j++)
        {
            sum += Weights[i][j] * Configs[t][i] * Hidden[t][j];
        }
    }
    return -sum;
}

// Returns delta Wij = lr*(<vh>_d - <vh>_m)
double ErrorFunction(int i, int j)
{
    double dataSum = 0;
    double modelSum = 0;
    for (int n = 0; n < N; n++)
    {
        dataSum += Configs[n][i] * Hidden[n][j];
        modelSum += Hidden[n][j] * Reconst[n][i];
    }
    double dataAvg = dataSum / N;
    double modelAvg = modelSum / N;
    double deltaWij = lr * (dataAvg - modelAvg);
    return deltaWij;
}

// Forward propagation
void ForwardPropagate()
{
    for (int j = 0; j < J; j++)
    {
        double sum = 0;
        for (int i = 0; i < L; i++)
        {
            sum += Weights[i][j] * Configs[t][i];
            double p = Sigmoid(sum);
            Hidden[t][j] = (p > drand48()) ? 1 : 0;
        }
    }
}

// Back propagation
void Reconstruct()
{
    // Reconstruct
    for (int i = 0; i < L; i++)
    {
        double sum = 0;
        for (int j = 0; j < J; j++)
        {
            sum += Weights[i][j] * Hidden[t][j];
            double p = Sigmoid(sum);
            Reconst[t][i] = (p > drand48()) ? 1 : 0;
        }
    }
}

// Gradient Descent
void GradDescent()
{
    // Gradient Descent
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < J; j++)
        {
            Weights[i][j] += ErrorFunction(i, j);
        }
    }
}

void TrainingLoop()
{
    // Train for e epochs
    for (int e = 0; e < epochs; e++)
    {
        // Loop over N datasets
        cout << "Epoch: " << e << endl;
        while (t < N)
        {
            if (t % 100 == 0)
            {
                cout << "\tTime Step: " << t << endl;
            }
            ForwardPropagate();
            Reconstruct();
            GradDescent();
            // Counter variable t
            t++;
        }
        t = 0;
        if (e != epochs - 1)
        {
            resetHidden();
        }
    }
}

//
//
// Benchmark RBM
// Feed in random data, W_ij histogram, Run ANNNI Simulations, Compute C(x)
//
// Output W_ij
void outputWeights()
{
    ofstream file;
    file.open("Weights.txt");
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < J; j++)
        {
            // CSV Format
            file << Weights[i][j];
            file << ",";
        }
    }
    file.close();
}

// Randomize X^mu
void randomizeConfigs()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < L; j++)
        {
            Configs[i][j] = (drand48() > 0.5) ? 0 : 1;
        }
    }
}
// Feed X^mu in to trained model
void feedModel()
{
    ForwardPropagate();
    Reconstruct();
}

// Unconvert from binary
void reverseBinary()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < L; j++)
        {
            Reconst[i][j] = (Reconst[i][j] == 0) ? -1 : 1;
        }
    }
}

// Find average enery across reconstruction

// C(x) for RBM
void RBMcorrelation()
{
    for (int x = 0; x <= corrSteps; x++)
    {
        double sum = 0;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < L; j++)
            {
                // Modulus to find correct index
                int jplusx = mod((j + x), L);
                sum += Reconst[i][j] * Reconst[i][jplusx];
            }
        }
        // Sum is normalized by NL^-1
        cout << sum / (N * L) << endl;
        CorrFunc[x] = sum / (N * L);
    }
}

void outputCorrelation()
{
    ofstream file;
    file.open("RBMCorrelation.txt");
    for (int x = 0; x < corrSteps; x++)
    {
        file << CorrFunc[x];
        file << ",";
    }
    file.close();
}
//
// Main Function
//
int main()
{
    // Preparing the data/model
    init_srand();
    initWeights();

    importConfigs();
    makeBinary();

    // Running the model
    TrainingLoop();

    // Benchmarking model
    randomizeConfigs();
    feedModel();
    reverseBinary();
    RBMcorrelation();

    // Outputting data
    outputCorrelation();
    outputWeights();

    return 0;
}