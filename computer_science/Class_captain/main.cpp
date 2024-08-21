#include <iostream>
#include <algorithm>

using namespace std;

int main() {
    char nm[4];
    int vt[4];
    int sv[4];
    cout << "Input the four names" << endl;
    for (int i = 0; i < 4; i++) {
        cin >> nm[i];
    }
    cout << "Votes Each person recieved";
    for (int i = 0; i < 4; i++) {
        cin >> vt[i];
        sv[i] = vt[i];
    }
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (sv[i] < sv[j]) {
                swap(sv[i], sv[j]);
                swap(nm[i], nm[j]);
            }
        }
    }
    cout << "Ranking and votes:" << endl;
    for (int i = 0; i < 4; i++) {
        cout << i+1 << ". " << nm[i] << " with " << sv[i] << " votes" << endl;
    }
    return 0;
}