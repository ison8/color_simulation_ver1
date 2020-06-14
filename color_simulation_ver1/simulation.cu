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
#define DATANUM 50			// 計算する数
#define CALCNUM 100		// べき乗する数
#define SIMNUM 1023			// シミュレーションする回数
#define LOOPNUM 2			// SIMNUM回のシミュレーション繰り返す回数

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

/* ファイルからデータを読み込む関数 */
int getFileData(vector<vector<double> >& d65_data, vector<vector<double> >& obs_data, vector<vector<double> > & xyz_data) {
	/* ファイルポインタ */
	FILE* fp_d65, * fp_obs, * fp_xyz;
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
		ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &(obs_data[count][0]), &(obs_data[count][1]), &(obs_data[count][2]), &(obs_data[count][3]));
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

	/* 標準観測者の読み込み */
	/* ファイルオープン */
	fp_xyz = fopen("./ciexyz31.csv", "r");
	/* 正しく開けているかをチェック */
	if (fp_xyz == NULL) {
		cout << "File open error" << endl;
		return -1;
	}

	/* ファイル読み込み */
	for (int i = 0; i < OBS_ROW; i++) {
		/* 1行ずつ読み込む */
		ret = fscanf(fp_xyz, "%lf, %lf, %lf, %lf", &(xyz_data[count][0]), &(xyz_data[count][1]), &(xyz_data[count][2]), &(xyz_data[count][3]));
		/* 終了判定 */
		if (xyz_data[count][0] == DATA_MAX) {
			count = 0;
			break;
		}
		/* カウンタの更新 */
		if (xyz_data[count][0] >= DATA_MIN) {
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
		//mu = (double)DATA_MIN + ((double)DATA_MAX - (double)DATA_MIN) / 10 * i;
		mu = (double)DATA_MIN + ( (double)rand() / RAND_MAX * ((double)DATA_MAX - (double)DATA_MIN ) );
		sigma = 5 + (95 * (double)rand() / RAND_MAX);

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
			//shift_data[j][i] = shift_data[j][i] / d_max * w_length;
			shift_data[j][i] = shift_data[j][i] / d_max;
		}
		/* 最大値初期化 */
		d_max = 0;
	}
}

/* vector型から配列へデータをコピーする関数 */
void cpyVecToArray(vector<vector<double> >& d65_data,
vector<vector<double> >& obs_data,
vector<vector<double> >& shift_data,
vector<vector<double> >& xyz_data,
double* d65, double* obs_l, double* obs_m, double* obs_s, double* obs_x, double* obs_y, double* obs_z, double* gauss_data) {
	for (int i = 0; i < DATA_ROW; i++) {
		d65[i] = d65_data[i][1];
		obs_l[i] = obs_data[i][1];
		obs_m[i] = obs_data[i][2];
		obs_s[i] = obs_data[i][3];
		obs_x[i] = xyz_data[i][1];
		obs_y[i] = xyz_data[i][2];
		obs_z[i] = xyz_data[i][3];
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
template<int BLOCK_SIZE> __global__ void colorSim(double simNum,double *g_data,double *d65,double *obs_x,double *obs_y,double *obs_z,double *result,int remain,int *d_mesh)  {
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
	__shared__ double g_max;
	g_max = 0;
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

	/* ブロック内のスレッド同期 */
	__syncthreads();

	for (int i = 0; i < CALCNUM; i++) {
		/* シェアードメモリにデータ格納 */
		calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(gaussian, (0.001 * i));

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
											 } __syncthreads();}
		if (BLOCK_SIZE >= 32) { if (ix < 16) { calc_data[ix][0] += calc_data[ix + 16][0];
											   calc_data[ix][1] += calc_data[ix + 16][1];
											   calc_data[ix][2] += calc_data[ix + 16][2];
											 } __syncthreads();
		}
		if (BLOCK_SIZE >= 16) { if (ix < 8) { calc_data[ix][0] += calc_data[ix + 8][0];
											  calc_data[ix][1] += calc_data[ix + 8][1];
											  calc_data[ix][2] += calc_data[ix + 8][2];
											}__syncthreads();
		}
		if (BLOCK_SIZE >= 8) { if (ix < 4) { calc_data[ix][0] += calc_data[ix + 4][0];
											 calc_data[ix][1] += calc_data[ix + 4][1];
											 calc_data[ix][2] += calc_data[ix + 4][2];
											} __syncthreads();
		}
		if (BLOCK_SIZE >= 4) { if (ix < 2) { calc_data[ix][0] += calc_data[ix + 2][0];
											 calc_data[ix][1] += calc_data[ix + 2][1];
											 calc_data[ix][2] += calc_data[ix + 2][2];
											} __syncthreads();
		}
		if (BLOCK_SIZE >= 2) { if (ix < 1) { calc_data[ix][0] += calc_data[ix + 1][0];
											 calc_data[ix][1] += calc_data[ix + 1][1];
											 calc_data[ix][2] += calc_data[ix + 1][2];
											} __syncthreads();
		}

		/*if (ix == 0) {
			for (int j = 1; j < BLOCK_SIZE; j++) {
				calc_data[ix][0] += calc_data[i][0];
				calc_data[ix][1] += calc_data[i][1];
				calc_data[ix][2] += calc_data[i][2];
			}
		}*/
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

			//printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
		}

		/* ブロック同期 */
		__syncthreads();

		/* メッシュの番号をふる */
		/* メッシュ用の変数 */
		double x, y;
		/* フラグxy */
		int f_x, f_y;
		/* メッシュ判定用の変数 */
		double m_x, m_y;

		/* f_x,f_yの初期化 */
		f_x = 0;
		f_y = 0;

		/* x方向 */
		if (ix <= 36) {
			/* xyの計算 */
			x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
			/* メッシュの判定 */
			m_x = (double)ix * 0.02;
			if (m_x <= x && (m_x + 0.02) > x) {
				f_x = 1;
			}
		}
		/* y方向 */
		if (ix >= 64 && ix <= 106) {
			/* xyの計算 */
			y = calc_data[0][1] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
			/* メッシュの判定 */
			m_y = (double)(ix - 64) * 0.02;
			if (m_y <= y && (m_y + 0.02) > y) {
				f_y = 1;
			}
		}

		/* ブロック同期 */
		__syncthreads();

		/* メッシュの位置計算 */
		if (ix <= 36){
			if (f_x == 1) {
				aPos = blockIdx.x * CALCNUM + i;
				d_mesh[aPos] = ix;
			}
		}

		/* ブロック同期 */
		__syncthreads();

		/* メッシュの位置計算 */
		if (ix >= 64 && ix <= 106) {
			if (f_y == 1) {
				aPos = blockIdx.x * CALCNUM + i;
				d_mesh[aPos] += (ix - 64) * 37;
			}
		}

		/* ブロック同期 */
		__syncthreads();
	}
}

/* LMS計算カーネル */
template<int BLOCK_SIZE> __global__ void lmsSim(double simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain) {
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
	__shared__ double g_max;
	g_max = 0;
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

	/* ブロック内のスレッド同期 */
	__syncthreads();

	for (int i = 0; i < CALCNUM; i++) {
		/* シェアードメモリにデータ格納 */
		calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(gaussian, (0.001 * i));

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
		if (BLOCK_SIZE >= 256) {
			if (ix < 128) {
				calc_data[ix][0] += calc_data[ix + 128][0];
				calc_data[ix][1] += calc_data[ix + 128][1];
				calc_data[ix][2] += calc_data[ix + 128][2];
			}__syncthreads();
		}
		if (BLOCK_SIZE >= 128) {
			if (ix < 64) {
				calc_data[ix][0] += calc_data[ix + 64][0];
				calc_data[ix][1] += calc_data[ix + 64][1];
				calc_data[ix][2] += calc_data[ix + 64][2];
			}__syncthreads();
		}
		if (BLOCK_SIZE >= 64) {
			if (ix < 32) {
				calc_data[ix][0] += calc_data[ix + 32][0];
				calc_data[ix][1] += calc_data[ix + 32][1];
				calc_data[ix][2] += calc_data[ix + 32][2];
			} __syncthreads();
		}
		if (BLOCK_SIZE >= 32) {
			if (ix < 16) {
				calc_data[ix][0] += calc_data[ix + 16][0];
				calc_data[ix][1] += calc_data[ix + 16][1];
				calc_data[ix][2] += calc_data[ix + 16][2];
			} __syncthreads();
		}
		if (BLOCK_SIZE >= 16) {
			if (ix < 8) {
				calc_data[ix][0] += calc_data[ix + 8][0];
				calc_data[ix][1] += calc_data[ix + 8][1];
				calc_data[ix][2] += calc_data[ix + 8][2];
			}__syncthreads();
		}
		if (BLOCK_SIZE >= 8) {
			if (ix < 4) {
				calc_data[ix][0] += calc_data[ix + 4][0];
				calc_data[ix][1] += calc_data[ix + 4][1];
				calc_data[ix][2] += calc_data[ix + 4][2];
			} __syncthreads();
		}
		if (BLOCK_SIZE >= 4) {
			if (ix < 2) {
				calc_data[ix][0] += calc_data[ix + 2][0];
				calc_data[ix][1] += calc_data[ix + 2][1];
				calc_data[ix][2] += calc_data[ix + 2][2];
			} __syncthreads();
		}
		if (BLOCK_SIZE >= 2) {
			if (ix < 1) {
				calc_data[ix][0] += calc_data[ix + 1][0];
				calc_data[ix][1] += calc_data[ix + 1][1];
				calc_data[ix][2] += calc_data[ix + 1][2];
			} __syncthreads();
		}

		/*if (ix == 0) {
			for (int j = 1; j < BLOCK_SIZE; j++) {
				calc_data[ix][0] += calc_data[i][0];
				calc_data[ix][1] += calc_data[i][1];
				calc_data[ix][2] += calc_data[i][2];
			}
		}*/
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

			//printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
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
	/*標準観測者(cie1931xyz)のデータを格納する配列 */
	vector<vector<double> > xyz_data(DATA_ROW, vector<double>(OBS_COL, 0));
	/*ガウシアンを10個格納する配列 */
	vector<vector<double> > gauss_shift(DATA_ROW, vector<double>(10, 0));

	/* 余り計算 */
	int remain = getRemain();

	/* データを入れる１次元配列 */
	double* d65, * obs_x, * obs_y, * obs_z, * obs_l, * obs_m, * obs_s, * gauss_data, * result, * fin_result, * lms_result, * lms_fin, * gauss_com;
	int* mesh_result, * mesh_f_result;
	d65 = new double[DATA_ROW];
	obs_l = new double[DATA_ROW];
	obs_m = new double[DATA_ROW];
	obs_s = new double[DATA_ROW];
	obs_x= new double[DATA_ROW];
	obs_y = new double[DATA_ROW];
	obs_z = new double[DATA_ROW];
	gauss_data = new double[DATA_ROW * 10];
	result = new double[3 * DATANUM * CALCNUM];
	fin_result = new double[3 * SIMNUM * CALCNUM * LOOPNUM];
	mesh_result = new int[DATANUM * CALCNUM];
	mesh_f_result = new int[SIMNUM * CALCNUM * LOOPNUM];
	lms_result = new double[3 * DATANUM * CALCNUM];
	lms_fin = new double[3 * SIMNUM * CALCNUM * LOOPNUM];
	gauss_com = new double[DATA_ROW * 10 * LOOPNUM];

	/* CUDA用の変数 */
	double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result, * d_lms;
	int* d_mesh;
	char* d_sim_order;

	/* GPUメモリ確保 */
	cudaMalloc((void**)&d_d65, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_x, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_y, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_z, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_gauss_data, DATA_ROW * 10 * sizeof(double));
	cudaMalloc((void**)&d_result, 3 * DATANUM * CALCNUM * sizeof(double));
	cudaMalloc((void**)&d_mesh, DATANUM * CALCNUM * sizeof(int));

	/* ファイル読み込み関数実行 */
	int f_result = getFileData(d65_data, obs_data, xyz_data);

	/* vectorを1次元配列へ変換 */
	cpyVecToArray(d65_data, obs_data, gauss_shift, xyz_data, d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z, gauss_data);

	/* CUDAへのメモリコピー */
	cudaMemcpy(d_d65, d65, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_x, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_y, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_z, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);


	for (int i = 0; i < LOOPNUM; i++) {
		/* ガウシアン計算 */
		makeGaussShift(gauss_shift);
		/* vectorを1次元配列へ変換 */
		cpyVecToArray(d65_data, obs_data, gauss_shift, xyz_data, d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z, gauss_data);
		/* ガウシアンを別の配列へ格納 */
		for (int j = 0; j < DATA_ROW * 10; j++) {
			gauss_com[(i * DATA_ROW * 10) + j] = gauss_data[j];
		}
		/* CUDAへのメモリコピー */
		cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);

		for(int j = 0; j < (SIMNUM - DATANUM); j += DATANUM) {
			colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > ((j + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh);
			cudaDeviceSynchronize();

			/* 結果のコピー */
			cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);

			for (int k = 0; k < (3 * DATANUM * CALCNUM); k++) {
				int aPos = (i * 3 * CALCNUM * SIMNUM) + (3 * CALCNUM * j) + k;
				fin_result[aPos] = result[k];
			}
			for (int k = 0; k < DATANUM * CALCNUM; k++) {
				int aPos = (i * CALCNUM * SIMNUM) + (CALCNUM * j) + k;
				mesh_f_result[aPos] = mesh_result[k];
			}
		}

		/* ループで余った残りの数をシミュレーション */
		int r_num = SIMNUM % DATANUM - 1;
		int sim_num = SIMNUM - r_num - 1;
		colorSim<DATA_ROW> << <r_num, DATA_ROW >> > ((sim_num + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh);
		cudaDeviceSynchronize();

		/* 結果のコピー */
		cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(mesh_result, d_mesh, DATANUM * CALCNUM * sizeof(int), cudaMemcpyDeviceToHost);

		for (int k = 0; k < (3 * r_num * CALCNUM); k++) { 
			int aPos = (i * 3 * CALCNUM * SIMNUM) + (3 * CALCNUM * sim_num) + k;
			fin_result[aPos] = result[k];
		}
		for (int k = 0; k < (r_num * CALCNUM); k++) {
			int aPos = (i * CALCNUM * SIMNUM) + (CALCNUM * sim_num) + k;
			mesh_f_result[aPos] = mesh_result[k];
		}
	}
	/* メモリ解放 */
	cudaFree(d_result);
	cudaMalloc((void**)&d_lms, 3 * DATANUM * CALCNUM * sizeof(double));
	cudaMemcpy(d_obs_x, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_y, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_z, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);

	/* LMS計算 */
	for (int i = 0; i < LOOPNUM; i++) {
		/* ガウシアン計算 */
		makeGaussShift(gauss_shift);
		/* vectorを1次元配列へ変換 */
		cpyVecToArray(d65_data, obs_data, gauss_shift, xyz_data, d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z, gauss_data);
		/* ガウシアンを別の配列へ格納 */
		for (int j = 0; j < DATA_ROW * 10; j++) {
			gauss_data[j] = gauss_com[(i * DATA_ROW * 10) + j];
		}
		/* CUDAへのメモリコピー */
		cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);

		for (int j = 0; j < (SIMNUM - DATANUM); j += DATANUM) {
			lmsSim<DATA_ROW> << <DATANUM, DATA_ROW >> > ((j + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_lms, remain);
			cudaDeviceSynchronize();

			/* 結果のコピー */
			cudaMemcpy(lms_result, d_lms, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);

			for (int k = 0; k < (3 * DATANUM * CALCNUM); k++) {
				int aPos = (i * 3 * CALCNUM * SIMNUM) + (3 * CALCNUM * j) + k;
				lms_fin[aPos] = lms_result[k];
			}
		}

		/* ループで余った残りの数をシミュレーション */
		int r_num = SIMNUM % DATANUM - 1;
		int sim_num = SIMNUM - r_num - 1;
		lmsSim<DATA_ROW> << <r_num, DATA_ROW >> > ((sim_num + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_lms, remain);
		cudaDeviceSynchronize();

		/* 結果のコピー */
		cudaMemcpy(lms_result, d_lms, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);

		for (int k = 0; k < (3 * r_num * CALCNUM); k++) {
			int aPos = (i * 3 * CALCNUM * SIMNUM) + (3 * CALCNUM * sim_num) + k;
			lms_fin[aPos] = lms_result[k];
		}
	}

	/* 結果が終了条件を満たしているときに値を0にする */
	for (int i = 0; i < LOOPNUM; i++) {
		for (int j = 0; j < SIMNUM; j++) {
			for (int k = 0; k < CALCNUM; k++) {
				int aPos = (i * 3 * SIMNUM * CALCNUM) + (j * 3 * CALCNUM) + k;
				if ((fin_result[0] * 0.005) > fin_result[aPos] &&
					(fin_result[CALCNUM] * 0.005) > fin_result[aPos + CALCNUM] && 
					(fin_result[CALCNUM * 2] * 0.005) > fin_result[aPos + (CALCNUM * 2)]) {
					fin_result[aPos] = 0;
					fin_result[aPos + CALCNUM] = 0;
					fin_result[aPos + (CALCNUM * 2)] = 0;

					aPos = (i * SIMNUM * CALCNUM) + (j * CALCNUM) + k;
					mesh_f_result[aPos] = -1;
				}
			}
		}
	}

	/* 出力ディレクトリ */
	//string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1023_15000_10_v2/";
	string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_test_1023_100_2/";

	/* 出力したファイルの情報を記録するファイル */
	string f_info = "sim_file_info.txt";
	f_info = directory + f_info;
	ofstream o_f_info(f_info);

	/* ファイル書き込み */
	for (int i = 0; i < LOOPNUM; i++) {
		/* 出力ファイル名 */
		string fname1 = "sim_result_L_xyz_1023_";
		string fname2 = "sim_result_S_xyz_1023_";
		string fname3 = "sim_result_lms_1023_";
		string fend = ".csv";
		fname1 = directory + fname1 + to_string(i + 1) + fend;
		fname2 = directory + fname2 + to_string(i + 1) + fend;
		fname3 = directory + fname3 + to_string(i + 1) + fend;

		/* ファイル出力ストリーム */
		ofstream o_file1(fname1);
		ofstream o_file2(fname2);
		ofstream o_file3(fname3);

		/* 出力したファイルの情報を記録するファイルにファイル名を出力 */
		o_f_info << fname1 << endl;
		o_f_info << fname2 << endl;
		o_f_info << fname3 << endl;

		/* ファイルへの出力桁数指定 */
		o_file1 << fixed << setprecision(3);
		o_file2 << fixed << setprecision(3);
		o_file3 << fixed << setprecision(3);
		for (int j = 0; j < CALCNUM; j++) {
			for (int k = 0; k < (SIMNUM - 1); k++) {
				int apos = j + ((3 * k) * CALCNUM) + (3 * SIMNUM * CALCNUM * i);

				double X = fin_result[apos];
				double Y = fin_result[apos + CALCNUM];
				double Z = fin_result[apos + (2 * CALCNUM)];

				double l = lms_fin[apos];
				double m = lms_fin[apos + CALCNUM];
				double s = lms_fin[apos + (2 * CALCNUM)];


				/* XYZ == 0のとき */
				if (X == 0 && Y == 0 && Z == 0) {
					apos = j + (k * CALCNUM) + (SIMNUM * CALCNUM * i);
					o_file1 << ",,,";
					o_file2 << ",,," << mesh_f_result[apos] << ",";
					o_file3 << ",,,,,";
				}

				/* それ以外のとき */
				else {
					apos = j + (k * CALCNUM) + (SIMNUM * CALCNUM * i);

					double x = X / (X + Y + Z);
					double y = Y / (X + Y + Z);
					double z = Z / (X + Y + Z);

					double lms_x = l / (l + m);
					double lms_y = s / (l + m);

					o_file1 << X << "," << Y << "," << Z << ",";
					o_file2 << x << "," << y << "," << z << "," << mesh_f_result[apos] << ",";
					o_file3 << l << "," << m << "," << s << "," << lms_x << "," << lms_y << ",";
				}
			}
			int apos = j + (3 * (SIMNUM - 1)) * CALCNUM + (3 * SIMNUM * CALCNUM * i);

			double X = fin_result[apos];
			double Y = fin_result[apos + CALCNUM];
			double Z = fin_result[apos + (2 * CALCNUM)];

			double l = lms_fin[apos];
			double m = lms_fin[apos + CALCNUM];
			double s = lms_fin[apos + (2 * CALCNUM)];

			/* XYZ == 0のとき */
			if (X == 0 && Y == 0 && Z == 0) {
				apos = j + (SIMNUM - 1) * CALCNUM + (SIMNUM * CALCNUM * i);
				o_file1 << ",,";
				o_file2 << ",,," << mesh_f_result[apos];
				o_file3 << ",,,,";
			}

			/* それ以外のとき */
			else {
				apos = j + (SIMNUM - 1) * CALCNUM + (SIMNUM * CALCNUM * i);

				double x = X / (X + Y + Z);
				double y = Y / (X + Y + Z);
				double z = Z / (X + Y + Z);

				double lms_x = l / (l + m);
				double lms_y = s / (l + m);

				o_file1 << X << "," << Y << "," << Z;
				o_file2 << x << "," << y << "," << z << mesh_f_result[apos] << ",";
				o_file3 << l << "," << m << "," << s << "," << lms_x << "," << lms_y;
			}

			o_file1 << endl << flush;
			o_file2 << endl << flush;
			o_file3 << endl << flush;
		}
		/* ファイルクローズ */
		o_file1.close();
		o_file2.close();
		o_file3.close();
	}

	/* デバイスメモリ解放 */
	cudaFree(d_d65);
	cudaFree(d_gauss_data);
	cudaFree(d_obs_x);
	cudaFree(d_obs_y);
	cudaFree(d_obs_z);
	cudaFree(d_lms); 

	/* ホストメモリ解放 */
	delete[] d65;
	delete[] obs_x;
	delete[] obs_y;
	delete[] obs_z;
	delete[] gauss_data;
	delete[] result;
	delete[] fin_result;
	delete[] lms_result;
	delete[] lms_fin;

	return 0;
}