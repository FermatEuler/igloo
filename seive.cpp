#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <algorithm>
using namespace std;

int calc(int a, int b, char op) {
	switch (op) {
	case '+': return a + b;
	case '-': return a - b;
	default: return 0;
	}
}

pair<int, int> dpMin[21][21]; 
pair<int, int> dpMax[21][21]; 
bool visited[21][21];         

pair<int,int> getMinMax(const vector<int>& nums, const vector<char>& ops, int l, int r) {
	if (l == r) return{ nums[l],nums[l] };
	if (visited[l][r]) return{ dpMin[l][r].first, dpMax[l][r].second };
	
	int maxVal = INT_MIN, minVal = INT_MAX;
	
	for (int i = l; i < r; ++i) {
		pair<int, int> left = getMinMax(nums, ops, l, i);
		pair<int, int> right = getMinMax(nums, ops, i + 1, r);

		int lmin = left.first, lmax = left.second;
		int rmin = right.first, rmax = right.second;
		
		char op = ops[i];
		int valm, valM;

		if (op == '+') {
			valm = lmin + rmin;
			valM = lmax + rmax;
		}
		else if (op == '-') {
			valm = lmin - rmax;
			valM = lmax - rmin;
		}
		maxVal = max(maxVal, valM);
		minVal = min(minVal, valm);
	}

	return{ minVal,maxVal };
}

int solution(vector<string> arr)
{
	vector<int> nums;
	vector<char> ops;

	for (int i = 0; i < arr.size(); ++i) {
		if (i % 2 == 0) {
			nums.push_back(stoi(arr[i]));
		}
		else {
			ops.push_back(arr[i][0]);
		}
	}

	return getMinMax(nums, ops, 0, nums.size() - 1).second;
}

int main() {
	vector<string> arr = { "5", "-", "3", "+", "1", "+", "2", "-", "4" };

	cout << solution(arr) << endl;
	return 0;
}