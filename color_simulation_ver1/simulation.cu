#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>

#define D65_ROW 531		// D65の行数
#define D65_COL 2		// D65の列数
#define OBS_ROW 441		// 標準観測者の行数
#define OBS_COL 4		// 標準観測者の列数
#define DATA_ROW 391	// 計算で使用するデータの行数 (390 - 780 nm)
#define DATA_MIN 390	// 使用する周波数の最小値
#define DATA_MAX 780	// 使用する周波数の最大値
#define PI 3.141592		// 円周率

#define BLOCKSIZE 371		// 1ブロック当たりのスレッド数
#define DATANUM 10			// 計算する数
#define CALCNUM 10000		// べき乗する数

/* 出力ファイルパス */
#define F_PATH "C:/Users/ryoin/source/repos/color_simulation_cuda/color_simulation_cuda"

using namespace std;

/* CUDAエラーチェック */
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

/* ファウルからデータを読み込む関数 */
int getFileData(vector<vector<double> >& d65_data, vector<vector<double> >& obs_data) {
	/* ファイルポインタ */
	FILE* fp_d65, * fp_obs;
	/* EOFを検出する変数 */
	int ret;
	/* カウンター */
	int count = 0;

	/* D65の読み込み */
	/* ファイルオープン */
	fp_d65 = fopen("./d65.csv", "r");
	/* 正しく開けているかをチェック */
	if (fp_d65 == NULL) {
		cout << "File open error" << endl;
		return -1;
	}

	/* ファイル読み込み */
	for (int i = 0; i < D65_ROW; i++) {
		/* 1行ずつ読み込む */
		ret = fscanf(fp_d65, "%lf, %lf", &(d65_data[count][0]), &(d65_data[count][1]));
		/* 終了判定 */
		if (d65_data[count][0] == DATA_MAX) {
			count = 0;
			break;
		}
		/* カウンタの更新 */
		if (d65_data[count][0] >= DATA_MIN) {
			count++;
		}
		/* エラーを検出した際の処理 */
		if (ret == EOF) {
			cout << "error" << endl;
			return -1;
		}
	}
	fclose(fp_d65);


	/* 標準観測者の読み込み */
	/* ファイルオープン */
	fp_obs = fopen("./std_obs_10deg.csv", "r");
	/* 正しく開けているかをチェック */
	if (fp_obs == NULL) {
		cout << "File open error" << endl;
		return -1;
	}

	/* ファイル読み込み */
	for (int i = 0; i < OBS_ROW; i++) {
		/* 1行ずつ読み込む */
		ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &(obs_data[i][0]), &(obs_data[i][1]), &(obs_data[i][2]), &(obs_data[i][3]));
		/* 終了判定 */
		if (obs_data[count][0] == DATA_MAX) {
			count = 0;
			break;
		}
		/* カウンタの更新 */
		if (obs_data[count][0] >= DATA_MIN) {
			count++;
		}
		/* エラーを検出した際の処理 */
		if (ret == EOF) {
			cout << "error" << endl;
			return -1;
		}
	}
	fclose(fp_d65);

	return 0;
}

/* ガウシアンのシフトを計算する関数 */
void makeGaussShift(vector<vector<double> >& shift_data) {
	double mu = 0;			// 計算で使用するミュー
	double sigma = 0;		// 計算で使用するシグマ
	double d_max = 0;		// 生成したガウシアンの中の最大値
	double w_length = 0;	// 振幅を0-1の間でランダムにするために使用する

	/* 乱数のシード生成 */
	srand((unsigned int)time(NULL));

	/* 波形は10パターン生成するので10回でループする */
	for (int i = 0; i < 10; i++) {
		mu = (double)DATA_MIN + ((double)DATA_MAX - (double)DATA_MIN) / 10 * i;
		sigma = 20 + (80 * (double)rand() / RAND_MAX);

		/* データ数だけ計算する */
		for (int j = 0; j < DATA_ROW; j++) {
			shift_data[j][i] = 1 / (sqrt(2 * PI) * sigma) * exp(-pow(((double)(DATA_MIN + j) - mu), 2) / (2 * sigma * sigma));
			/* 最大値を変数に格納する(更新する) */
			if (d_max < shift_data[j][i]) {
				d_max = shift_data[j][i];
			}
		}

		/* 生成したガウシアンを正規化し、振幅を0-1の間でランダムにする */
		w_length = (double)rand() / RAND_MAX;	// 0-1の間で乱数生成
		for (int j = 0; j < DATA_ROW; j++) {
			shift_data[j][i] = shift_data[j][i] / d_max * w_length;
		}
		/* 最大値初期化 */
		d_max = 0;
	}
}

/* vector型から配列へデータをコピーする関数 */
void cpyVecToArray(vector<vector<double> >& d65_data,
vector<vector<double> >& obs_data,
vector<vector<double> >& shift_data,
double* d65, double* obs_x, double* obs_y, double* obs_z, double* gauss_data) {
	for (int i = 0; i < DATA_ROW; i++) {
		d65[i] = d65_data[i][1];
		obs_x[i] = obs_data[i][1];
		obs_y[i] = obs_data[i][2];
		obs_z[i] = obs_data[i][3];
		for (int j = 0; j < 10; j++) {
			int aPos = DATA_ROW * j + i;
			gauss_data[aPos] = shift_data[i][j];
		}
	}
}


/* 積分計算カーネル */
template<int BLOCK_SIZE> __global__ void colorSim() {

}

int main(void) {
	/* D65のデータを格納する配列 */
	vector<vector<double> > d65_data(DATA_ROW, vector<double>(D65_COL, 0));
	/*標準観測者のデータを格納する配列 */
	vector<vector<double> > obs_data(DATA_ROW, vector<double>(OBS_COL, 0));
	/*ガウシアンを10個格納する配列 */
	vector<vector<double> > gauss_shift(DATA_ROW, vector<double>(10, 0));

	/* データを入れる１次元配列 */
	double* d65, * obs_x, * obs_y, * obs_z, * gauss_data;
	d65 = new double[DATA_ROW];
	obs_x= new double[DATA_ROW];
	obs_y = new double[DATA_ROW];
	obs_z = new double[DATA_ROW];
	gauss_data = new double[DATA_ROW * 10];

	/* ファイル読み込み関数実行 */
	int f_result = getFileData(d65_data, obs_data);

	/* ガウシアン計算 */
	makeGaussShift(gauss_shift);

	/* vectorを1次元配列へ変換 */
	cpyVecToArray(d65_data, obs_data, gauss_shift,d65,obs_x,obs_y,obs_z,gauss_data);

}