#include <iostream>
#include <string>
using namespace std;

void swap(int& a, int& b) {
	int tmp;
	tmp = a;
	a = b;
	b = tmp;
}

void startree(int a) {
	const int h =5;

	if (a % 2) {
		for (int i = 1; i <= h; i++) {
			cout << string((a/2+1)*(h - i), ' ');
			cout << string(2*i-1, '*');	
			cout << endl;
		}
	} else {
		a -= 1;
		for (int i = h; i >= 1; i--) {
			cout << string((a / 2 + 1)*(h - i), ' ');
			cout << string(2*i-1, '*');
			cout << endl;
		}
	}
}

void agegp(int a) {
	if (cin.fail() || a < 0) {
		cout << "Invalid input. Please enter a natural number." << endl;
	}

		
	if (a>=0) {
		if (a < 13) {
			cout << "Children" << endl;
		}
		else if (a < 20) {
			cout << "Teenager" << endl;
		}
		else if (a < 65) {
			cout << "Adult" << endl;
		}
		else {
			cout << "Senior" << endl;
		}

	}
}

int main() {
	int a, b;
	
	cout << "Enter the two values you want to swap:";
	cin >> a >> b;
	
	swap(a, b);
	cout << "Swapped values: " << a << " " << b << endl;
	cout << "\n";

	cout << "Choose a tree type number:";
	cin >> a;
	startree(a);
	cout << "\n";

	cout << "Please enter the age: ";
	cin >> a;
	agegp(a);
}