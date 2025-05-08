#include <iostream>
using namespace std;

void change(int* ptr) {
	*ptr = 100;
}

int main() {
	int x = 5;
	change(&x);
	cout << x <<endl;
}