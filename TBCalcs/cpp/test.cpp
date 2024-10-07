#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

int main()
{
   string fname = "cpptest.csv";

   ifstream file(fname);

   if (!file.is_open()){
        cerr << "Error: Could not open file!" << endl;
        return 1;
   }

   // create matrix
   vector<vector<double>> matrix;

   string line, word;

   // Read file
   while (getline(file, line)){
    // Vector to store row of matrix
    vector<double> row;

    stringstream s(line);

    while (getline(s, word, ',')){
        row.push_back(stod(word));
    }

    // Add row to matrix
    matrix.push_back(row);
   }

   // Close the file
   file.close();

   //Print matrix
   for (const auto& row: matrix){
    for (const auto& val : row){
        cout << val << "";
    }
        cout << endl;
   }

   return 0;
}