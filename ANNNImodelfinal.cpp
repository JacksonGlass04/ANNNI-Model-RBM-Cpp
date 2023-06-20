// Import Statements
#include <cmath>
#include <string>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <array>
#include <vector>

using namespace std;

// Global Constants
const int L = 40;
const int N = 1000;

const double J1 = 1;
const double J2 = J1 / 2;
const int N_nn1 = 2;
const int N_nn2 = 2;

const double temp = 0.1;
const int corrSteps = 20;

// Lattice Class
class Site
{
public:
    int Sz;

    Site *nn1[N_nn1];
    Site *nn2[N_nn2];
};

Site lattice[L];

int finalConfigs[N][L];
int initConfigs[N][L];

// Correlation
double CorrFunc[corrSteps];

// Set Seed
void init_srand(void)
{
    time_t seconds;
    time(&seconds);
    srand48(seconds);
}

//
//
// Lattice Functions
// Randomize Spins, Modulus, Set Neighbors, Energy, Magnetization
//
// Randomize Spins
void randomizeSpins()
{
    for (int i = 0; i < L; i++)
    {
        double rand = drand48();
        lattice[i].Sz = (rand > 0.5) ? -1 : 1;
    }
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

// Set Nearest Neighbors
void set_nn()
{
    // Convention of left = 0, right = 1
    for (int i = 0; i < L; i++)
    {
        int h = 0;
        // Left 1 nn
        h = mod((i - 1), L);
        lattice[i].nn1[0] = &lattice[h];

        // Right 1 nn
        h = mod((i + 1), L);
        lattice[i].nn1[1] = &lattice[h];

        // Left 2 nn
        h = mod((i - 2), L);
        lattice[i].nn2[0] = &lattice[h];

        // Right 2 nn
        h = mod((i + 2), L);
        lattice[i].nn2[1] = &lattice[h];
    }
}

// Compute Magnetization
int magnetization()
{
    int mag = 0;
    for (int i = 0; i < L; i++)
    {
        mag += lattice[i].Sz;
    }
    return mag;
}

// Compute Energy
double energy()
{
    double J1sum = 0;
    for (int i = 0; i < L; i++)
    {
        J1sum += lattice[i].Sz * (lattice[i].nn1[1]->Sz);
    }

    double J2sum = 0;
    for (int i = 0; i < L; i++)
    {
        J2sum += lattice[i].Sz * (lattice[i].nn2[1]->Sz);
    }

    return (-J1 * J1sum + J2 * J2sum);
}

//
//
// Configuration Functions
// Clear, Store Init, Print Init, Store Final, Print Final
//
// Store initial configuration
void storeInitConfig(int i)
{
    for (int j = 0; j < L; j++)
    {
        initConfigs[i][j] = lattice[j].Sz;
    }
}

// Print initial configurations
void printInitConfigs()
{
    for (int i = 0; i < N; i++)
    {
        cout << "\n";
        for (int j = 0; j < L; j++)
        {
            cout << initConfigs[i][j] << "\t";
        }
    }
}

// Store final configuration
void storeFinalConfig(int i)
{
    for (int j = 0; j < L; j++)
    {
        finalConfigs[i][j] = lattice[j].Sz;
    }
}

// Print final configuration
void printFinalConfigs()
{
    for (int i = 0; i < N; i++)
    {
        cout << "\n";
        for (int j = 0; j < L; j++)
        {
            cout << finalConfigs[i][j] << "\t";
        }
    }
}

//
//
// Markov Chain Monte Carlo
// updateSpin, Monte Carlo Algorithm
//
// Update Spin
bool updateSpin(int i, double temp)
{
    // Energy difference
    double deltaE = 2 * lattice[i].Sz * (J1 * (lattice[i].nn1[0]->Sz + lattice[i].nn1[1]->Sz) - J2 * (lattice[i].nn2[0]->Sz + lattice[i].nn2[1]->Sz));

    double p_rand = drand48();
    if (deltaE == 0)
    {
        return p_rand <= 0.5 ? true : false;
    }
    else
    {
        double p_flip = exp(-deltaE / temp);
        return p_rand < p_flip ? true : false;
    }

    return false;
}

// Metropolis Algorithm
void MonteCarlo()
{
    // Number of sweeps
    int nData = 200000;

    // Loop N times to create N Ising configurations
    for (int n = 0; n < N; n++)
    {
        // Set global counter variable
        // storeInitConfig(n);

        // Output to console to monitor progress
        cout << n << endl;

        // Sweep
        for (int b = 0; b < nData; b++)
        {
            for (int i = 0; i < L; i++)
            {
                updateSpin(i, temp) ? lattice[i].Sz *= -1 : lattice[i].Sz *= 1;
            }
        }
        // Save the final Ising configuration to a datatset
        storeFinalConfig(n);

        // Randomize spins to reset the Ising chain
        randomizeSpins();
        // set_nn();
    }
}

//
//
// Output data & Correlation
// Write final to CSV, Write initial to CSV, Find Correlation, Output Correlation
//
// outputs inital configs to CSV
void outputInitalConfigs()
{
    ofstream file;
    file.open("InitialConfigurations.txt");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < L; j++)
        {
            // CSV Format
            file << initConfigs[i][j];
            file << ",";
        }
        file << "\n";
    }
    file.close();
}

// outputs final configs to CSV
void outputFinalConfigs()
{
    ofstream file;
    file.open("FinalConfigurations.txt");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < L; j++)
        {
            // CSV Format
            file << finalConfigs[i][j];
            file << ",";
        }
        file << "\n";
    }
    file.close();
}

// outputs C(x) correlation
void correlation()
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
                sum += finalConfigs[i][j] * finalConfigs[i][jplusx];
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
    file.open("IsingCorrelation.txt");
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
    // Input Validation
    if (L < corrSteps)
    {
        cout << "L must be > corrSteps";
        exit(0);
    }

    init_srand();
    randomizeSpins();
    set_nn();

    MonteCarlo();

    outputFinalConfigs();

    correlation();
    outputCorrelation();

    return 0;
}