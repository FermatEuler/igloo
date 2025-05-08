#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <algorithm>
using namespace std;

const int MAX_N = 101                                      ;

int dpMin[MAX_N][MAX_N];     // 구간 [l, r]의 최소값
int dpMax[MAX_N][MAX_N];     // 구간 [l, r]의 최대값
bool visited[MAX_N][MAX_N];  // 이미 계산했는지 확인

int calc(int a, int b, char op) {
	switch (op) {
	case '+': return a + b;
	case '-': return a - b;
	default: return 0;
	}
}

// 구간 [l, r]에 대해 (최솟값, 최댓값) 반환
pair<int, int> getMinMax(const vector<int>& nums, const vector<char>& ops, int l, int r) {
	if (l == r) return{ nums[l], nums[l] };

	if (visited[l][r]) return{ dpMin[l][r], dpMax[l][r] };

	int maxVal = INT_MIN;
	int minVal = INT_MAX;

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

	visited[l][r] = true;
	dpMin[l][r] = minVal;
	dpMax[l][r] = maxVal;
	return{ minVal, maxVal };
}

int solution(vector<string> arr) {
	vector<int> nums;
	vector<char> ops;

	// 초기화
	for (int i = 0; i < MAX_N; ++i) {
		for (int j = 0; j < MAX_N; ++j) {
			visited[i][j] = false;
			dpMin[i][j] = 0;
			dpMax[i][j] = 0;
		}
	}

	// 입력 파싱
	for (int i = 0; i < arr.size(); ++i) {
		if (i % 2 == 0) nums.push_back(stoi(arr[i]));
		else ops.push_back(arr[i][0]);
	}

	return getMinMax(nums, ops, 0, nums.size() - 1).second;  // 최댓값 반환
}

int main() {
	vector<string> arr = { "1", "-", "3", "+", "5", "-", "8" };
	cout << solution(arr) << endl;
	return 0;
}
