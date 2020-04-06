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
#define DATANUM 1			// 計算する数
#define CALCNUM 10000		// べき乗する数

/* 出力ファイルパス */
//#define F_PATH "C:/Users/ryoin/source/repos/color_simulation_cuda/color_simulation_cuda"

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

/* 総和計算の時に使用する変数を計算 */
int getRemain(void) {
	/* 余り */
	int remain = 0;

	/* 余り計算 */
	for (int i = 1; i < BLOCKSIZE; i *= 2) {
		remain = BLOCKSIZE - i;
	}

	/* 余り出力 */
	return remain;
}

/* 積分計算カーネル */
template<int BLOCK_SIZE> __global__ void colorSim(double simNum,double *g_data,double *d65,double *obs_x,double *obs_y,double *obs_z,double *result,int remain) {
	/* CUDAアクセス用変数 */
	int ix = threadIdx.x;
	int aPos = 0;
	/* どのガウシアンを決めるための変数 */
	__shared__ int sim_order[10];
	/* ガウシアン組み合わせの番号 */
	__shared__ double sim_num;
	/* 結果を格納するシェアードメモリ */
	__shared__ double calc_data[BLOCK_SIZE][3];
	/* 足し合わせたガウシアンの最大値 */
	__shared__ double g_max = 0;
	/* 足し合わせたガウシアンを格納する */
	double gaussian = 0;
	/* 足し合わせたガウシアンを格納(最大値比較用) */
	__shared__ double g_comp[BLOCK_SIZE];
	/* 比較用シェアードメモリ初期化 */
	g_comp[ix] = 0;

	/* sim_orderヘ値を入れる */
	if (ix == 0) {
		sim_num = blockIdx.x + simNum;
		int count = 512;	// カウンタ
		for (int i = 0; i < 10; i++) {
			if (sim_num >= count) {
				sim_num -= count;
				sim_order[i] = 1;
			}
			else { 
				sim_order[i] = 0;
			}
			count = count / 2;
		}
		/*printf("%d %d %d %d %d %d %d %d %d %d\n", 
			sim_order[0], sim_order[1], sim_order[2], sim_order[3], sim_order[4],
			sim_order[5], sim_order[6], sim_order[7], sim_order[8], sim_order[9] );*/
	}

	/* ブロック内のスレッド同期 */
	__syncthreads();

	/* ガウシアンを足し合わせる */
	for (int i = 0; i < 10; i++) {
		aPos = i * BLOCK_SIZE + ix;
		if (sim_order[i] == 1) {
			gaussian += g_data[aPos];
			g_comp[ix] += g_data[aPos];
		}
	}

	/* ブロック内のスレッド同期 */
	__syncthreads();

	/* 足し合わせたガウシアンの最大値を求める */
	if (ix == 0) {
		for (int i = 0; i < BLOCK_SIZE; i++) {
			if (g_max < g_comp[i]) {
				g_max = g_comp[i];
			}
		}
	}

	/* ブロック内のスレッド同期 */
	__syncthreads();

	/* g_max が1以上の場合、最大値が0.99になるように正規化 */
	if (g_max >= 1) {
		gaussian = gaussian / g_max * 0.99;
	}

	for (int i = 0; i < CALCNUM; i++) {
		/* シェアードメモリにデータ格納 */
		calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(gaussian, (0.01 * i));
		calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(gaussian, (0.01 * i));
		calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(gaussian, (0.01 * i));

		/* ブロック同期 */
		__syncthreads();

		/* ブロックごとにリダクション処理(総和計算) */
		/* 余りが0出ない場合 */
		if (remain != 0) {
			/* 余った要素のシェアードメモリを加算する */
			if (ix < remain) {
				calc_data[ix][0] += calc_data[BLOCK_SIZE - ix - 1][0];
				calc_data[ix][1] += calc_data[BLOCK_SIZE - ix - 1][1];
				calc_data[ix][2] += calc_data[BLOCK_SIZE - ix - 1][2];
			}
		}

		/* 総和計算する */
		if (BLOCK_SIZE >= 256) { if (ix < 128) { calc_data[ix][0] += calc_data[ix + 128][0];
												 calc_data[ix][1] += calc_data[ix + 128][1];
												 calc_data[ix][2] += calc_data[ix + 128][2];
												}__syncthreads(); }
		if (BLOCK_SIZE >= 128) { if (ix < 64) { calc_data[ix][0] += calc_data[ix + 64][0];
												calc_data[ix][1] += calc_data[ix + 64][1];
												calc_data[ix][2] += calc_data[ix + 64][2];
												}__syncthreads(); }
		if (BLOCK_SIZE >= 64) { if (ix < 32) { calc_data[ix][0] += calc_data[ix + 32][0];
											   calc_data[ix][1] += calc_data[ix + 32][1];
											   calc_data[ix][2] += calc_data[ix + 32][2];
											 } }
		if (BLOCK_SIZE >= 32) { if (ix < 16) { calc_data[ix][0] += calc_data[ix + 16][0];
											   calc_data[ix][1] += calc_data[ix + 16][1];
											   calc_data[ix][2] += calc_data[ix + 16][2];
											 } }
		if (BLOCK_SIZE >= 16) { if (ix < 8) { calc_data[ix][0] += calc_data[ix + 8][0];
											  calc_data[ix][1] += calc_data[ix + 8][1];
											  calc_data[ix][2] += calc_data[ix + 8][2];
											} }
		if (BLOCK_SIZE >= 8) { if (ix < 4) { calc_data[ix][0] += calc_data[ix + 4][0];
											 calc_data[ix][1] += calc_data[ix + 4][1];
											 calc_data[ix][2] += calc_data[ix + 4][2];
											} }
		if (BLOCK_SIZE >= 4) { if (ix < 2) { calc_data[ix][0] += calc_data[ix + 2][0];
											 calc_data[ix][1] += calc_data[ix + 2][1];
											 calc_data[ix][2] += calc_data[ix + 2][2];
											} }
		if (BLOCK_SIZE >= 2) { if (ix < 1) { calc_data[ix][0] += calc_data[ix + 1][0];
											 calc_data[ix][1] += calc_data[ix + 1][1];
											 calc_data[ix][2] += calc_data[ix + 1][2];
											} }

		/* 値出力 */
		if (ix == 0) {
			/* aPos更新 */
			aPos = blockIdx.x * 3 * CALCNUM + i;
			//printf("%d %d\n", blockIdx.x,calc_data[ix]);
			result[aPos] = calc_data[0][0];

			/* aPos更新 */
			aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
			//printf("%d %d\n", blockIdx.x,calc_data[ix]);
			result[aPos] = calc_data[0][1];

			/* aPos更新 */
			aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
			//printf("%d %d\n", blockIdx.x,calc_data[ix]);
			result[aPos] = calc_data[0][2];
		}

		/* ブロック同期 */
		__syncthreads();
	}
}

int main(void) {
	/* D65のデータを格納する配列 */
	vector<vector<double> > d65_data(DATA_ROW, vector<double>(D65_COL, 0));
	/*標準観測者のデータを格納する配列 */
	vector<vector<double> > obs_data(DATA_ROW, vector<double>(OBS_COL, 0));
	/*ガウシアンを10個格納する配列 */
	vector<vector<double> > gauss_shift(DATA_ROW, vector<double>(10, 0));

	/* 余り計算 */
	int remain = getRemain();

	/* データを入れる１次元配列 */
	double* d65, * obs_x, * obs_y, * obs_z, * gauss_data, * result;
	d65 = new double[DATA_ROW];
	obs_x= new double[DATA_ROW];
	obs_y = new double[DATA_ROW];
	obs_z = new double[DATA_ROW];
	gauss_data = new double[DATA_ROW * 10];
	result = new double[3 * DATANUM * CALCNUM];

	/* CUDA用の変数 */
	double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, *d_result;
	char* d_sim_order;

	/* GPUメモリ確保 */
	cudaMalloc((void**)&d_d65, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_x, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_y, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_z, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_gauss_data, DATA_ROW * 10 * sizeof(double));
	cudaMalloc((void**)&d_result, 3 * DATANUM * CALCNUM * sizeof(double));

	/* ファイル読み込み関数実行 */
	int f_result = getFileData(d65_data, obs_data);

	/* ガウシアン計算 */
	makeGaussShift(gauss_shift);

	/* vectorを1次元配列へ変換 */
	cpyVecToArray(d65_data, obs_data, gauss_shift,d65,obs_x,obs_y,obs_z,gauss_data);


	/* CUDAへのメモリコピー */
	cudaMemcpy(d_d65, d65, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_x, obs_x, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_y, obs_y, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_z, obs_z, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);

	colorSim<DATA_ROW> << <1, DATA_ROW >> > (0,d_gauss_data,d_d65,d_obs_x,d_obs_y,d_obs_z,d_result,remain);
	cudaDeviceSynchronize();

	/* 結果のコピー */
	cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);

	/* デバイスメモリ解放 */
	cudaFree(d_d65);
	cudaFree(d_gauss_data);
	cudaFree(d_obs_x);
	cudaFree(d_obs_y);
	cudaFree(d_obs_z);
	cudaFree(d_result);
}