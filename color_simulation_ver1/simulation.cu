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

#define D65_ROW 531		// D65�̍s��
#define D65_COL 2		// D65�̗�
#define OBS_ROW 441		// �W���ϑ��҂̍s��
#define OBS_COL 4		// �W���ϑ��҂̗�
#define DATA_ROW 391	// �v�Z�Ŏg�p����f�[�^�̍s�� (390 - 780 nm)
#define DATA_MIN 390	// �g�p������g���̍ŏ��l
#define DATA_MAX 780	// �g�p������g���̍ő�l
#define PI 3.141592		// �~����

#define BLOCKSIZE 371		// 1�u���b�N������̃X���b�h��
#define DATANUM 1			// �v�Z���鐔
#define CALCNUM 10000		// �ׂ��悷�鐔

/* �o�̓t�@�C���p�X */
//#define F_PATH "C:/Users/ryoin/source/repos/color_simulation_cuda/color_simulation_cuda"

using namespace std;

/* CUDA�G���[�`�F�b�N */
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

/* �t�@�E������f�[�^��ǂݍ��ފ֐� */
int getFileData(vector<vector<double> >& d65_data, vector<vector<double> >& obs_data) {
	/* �t�@�C���|�C���^ */
	FILE* fp_d65, * fp_obs;
	/* EOF�����o����ϐ� */
	int ret;
	/* �J�E���^�[ */
	int count = 0;

	/* D65�̓ǂݍ��� */
	/* �t�@�C���I�[�v�� */
	fp_d65 = fopen("./d65.csv", "r");
	/* �������J���Ă��邩���`�F�b�N */
	if (fp_d65 == NULL) {
		cout << "File open error" << endl;
		return -1;
	}

	/* �t�@�C���ǂݍ��� */
	for (int i = 0; i < D65_ROW; i++) {
		/* 1�s���ǂݍ��� */
		ret = fscanf(fp_d65, "%lf, %lf", &(d65_data[count][0]), &(d65_data[count][1]));
		/* �I������ */
		if (d65_data[count][0] == DATA_MAX) {
			count = 0;
			break;
		}
		/* �J�E���^�̍X�V */
		if (d65_data[count][0] >= DATA_MIN) {
			count++;
		}
		/* �G���[�����o�����ۂ̏��� */
		if (ret == EOF) {
			cout << "error" << endl;
			return -1;
		}
	}
	fclose(fp_d65);


	/* �W���ϑ��҂̓ǂݍ��� */
	/* �t�@�C���I�[�v�� */
	fp_obs = fopen("./std_obs_10deg.csv", "r");
	/* �������J���Ă��邩���`�F�b�N */
	if (fp_obs == NULL) {
		cout << "File open error" << endl;
		return -1;
	}

	/* �t�@�C���ǂݍ��� */
	for (int i = 0; i < OBS_ROW; i++) {
		/* 1�s���ǂݍ��� */
		ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &(obs_data[i][0]), &(obs_data[i][1]), &(obs_data[i][2]), &(obs_data[i][3]));
		/* �I������ */
		if (obs_data[count][0] == DATA_MAX) {
			count = 0;
			break;
		}
		/* �J�E���^�̍X�V */
		if (obs_data[count][0] >= DATA_MIN) {
			count++;
		}
		/* �G���[�����o�����ۂ̏��� */
		if (ret == EOF) {
			cout << "error" << endl;
			return -1;
		}
	}
	fclose(fp_d65);

	return 0;
}

/* �K�E�V�A���̃V�t�g���v�Z����֐� */
void makeGaussShift(vector<vector<double> >& shift_data) {
	double mu = 0;			// �v�Z�Ŏg�p����~���[
	double sigma = 0;		// �v�Z�Ŏg�p����V�O�}
	double d_max = 0;		// ���������K�E�V�A���̒��̍ő�l
	double w_length = 0;	// �U����0-1�̊ԂŃ����_���ɂ��邽�߂Ɏg�p����

	/* �����̃V�[�h���� */
	srand((unsigned int)time(NULL));

	/* �g�`��10�p�^�[����������̂�10��Ń��[�v���� */
	for (int i = 0; i < 10; i++) {
		mu = (double)DATA_MIN + ((double)DATA_MAX - (double)DATA_MIN) / 10 * i;
		sigma = 20 + (80 * (double)rand() / RAND_MAX);

		/* �f�[�^�������v�Z���� */
		for (int j = 0; j < DATA_ROW; j++) {
			shift_data[j][i] = 1 / (sqrt(2 * PI) * sigma) * exp(-pow(((double)(DATA_MIN + j) - mu), 2) / (2 * sigma * sigma));
			/* �ő�l��ϐ��Ɋi�[����(�X�V����) */
			if (d_max < shift_data[j][i]) {
				d_max = shift_data[j][i];
			}
		}

		/* ���������K�E�V�A���𐳋K�����A�U����0-1�̊ԂŃ����_���ɂ��� */
		w_length = (double)rand() / RAND_MAX;	// 0-1�̊Ԃŗ�������
		for (int j = 0; j < DATA_ROW; j++) {
			shift_data[j][i] = shift_data[j][i] / d_max * w_length;
		}
		/* �ő�l������ */
		d_max = 0;
	}
}

/* vector�^����z��փf�[�^���R�s�[����֐� */
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

/* ���a�v�Z�̎��Ɏg�p����ϐ����v�Z */
int getRemain(void) {
	/* �]�� */
	int remain = 0;

	/* �]��v�Z */
	for (int i = 1; i < BLOCKSIZE; i *= 2) {
		remain = BLOCKSIZE - i;
	}

	/* �]��o�� */
	return remain;
}

/* �ϕ��v�Z�J�[�l�� */
template<int BLOCK_SIZE> __global__ void colorSim(double simNum,double *g_data,double *d65,double *obs_x,double *obs_y,double *obs_z,double *result,int remain) {
	/* CUDA�A�N�Z�X�p�ϐ� */
	int ix = threadIdx.x;
	int aPos = 0;
	/* �ǂ̃K�E�V�A�������߂邽�߂̕ϐ� */
	__shared__ int sim_order[10];
	/* �K�E�V�A���g�ݍ��킹�̔ԍ� */
	__shared__ double sim_num;
	/* ���ʂ��i�[����V�F�A�[�h������ */
	__shared__ double calc_data[BLOCK_SIZE][3];
	/* �������킹���K�E�V�A���̍ő�l */
	__shared__ double g_max = 0;
	/* �������킹���K�E�V�A�����i�[���� */
	double gaussian = 0;
	/* �������킹���K�E�V�A�����i�[(�ő�l��r�p) */
	__shared__ double g_comp[BLOCK_SIZE];
	/* ��r�p�V�F�A�[�h������������ */
	g_comp[ix] = 0;

	/* sim_order�w�l������ */
	if (ix == 0) {
		sim_num = blockIdx.x + simNum;
		int count = 512;	// �J�E���^
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

	/* �u���b�N���̃X���b�h���� */
	__syncthreads();

	/* �K�E�V�A���𑫂����킹�� */
	for (int i = 0; i < 10; i++) {
		aPos = i * BLOCK_SIZE + ix;
		if (sim_order[i] == 1) {
			gaussian += g_data[aPos];
			g_comp[ix] += g_data[aPos];
		}
	}

	/* �u���b�N���̃X���b�h���� */
	__syncthreads();

	/* �������킹���K�E�V�A���̍ő�l�����߂� */
	if (ix == 0) {
		for (int i = 0; i < BLOCK_SIZE; i++) {
			if (g_max < g_comp[i]) {
				g_max = g_comp[i];
			}
		}
	}

	/* �u���b�N���̃X���b�h���� */
	__syncthreads();

	/* g_max ��1�ȏ�̏ꍇ�A�ő�l��0.99�ɂȂ�悤�ɐ��K�� */
	if (g_max >= 1) {
		gaussian = gaussian / g_max * 0.99;
	}

	for (int i = 0; i < CALCNUM; i++) {
		/* �V�F�A�[�h�������Ƀf�[�^�i�[ */
		calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(gaussian, (0.01 * i));
		calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(gaussian, (0.01 * i));
		calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(gaussian, (0.01 * i));

		/* �u���b�N���� */
		__syncthreads();

		/* �u���b�N���ƂɃ��_�N�V��������(���a�v�Z) */
		/* �]�肪0�o�Ȃ��ꍇ */
		if (remain != 0) {
			/* �]�����v�f�̃V�F�A�[�h�����������Z���� */
			if (ix < remain) {
				calc_data[ix][0] += calc_data[BLOCK_SIZE - ix - 1][0];
				calc_data[ix][1] += calc_data[BLOCK_SIZE - ix - 1][1];
				calc_data[ix][2] += calc_data[BLOCK_SIZE - ix - 1][2];
			}
		}

		/* ���a�v�Z���� */
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

		/* �l�o�� */
		if (ix == 0) {
			/* aPos�X�V */
			aPos = blockIdx.x * 3 * CALCNUM + i;
			//printf("%d %d\n", blockIdx.x,calc_data[ix]);
			result[aPos] = calc_data[0][0];

			/* aPos�X�V */
			aPos = blockIdx.x * 3 * CALCNUM + i + CALCNUM;
			//printf("%d %d\n", blockIdx.x,calc_data[ix]);
			result[aPos] = calc_data[0][1];

			/* aPos�X�V */
			aPos = blockIdx.x * 3 * CALCNUM + i + (2 * CALCNUM);
			//printf("%d %d\n", blockIdx.x,calc_data[ix]);
			result[aPos] = calc_data[0][2];
		}

		/* �u���b�N���� */
		__syncthreads();
	}
}

int main(void) {
	/* D65�̃f�[�^���i�[����z�� */
	vector<vector<double> > d65_data(DATA_ROW, vector<double>(D65_COL, 0));
	/*�W���ϑ��҂̃f�[�^���i�[����z�� */
	vector<vector<double> > obs_data(DATA_ROW, vector<double>(OBS_COL, 0));
	/*�K�E�V�A����10�i�[����z�� */
	vector<vector<double> > gauss_shift(DATA_ROW, vector<double>(10, 0));

	/* �]��v�Z */
	int remain = getRemain();

	/* �f�[�^������P�����z�� */
	double* d65, * obs_x, * obs_y, * obs_z, * gauss_data, * result;
	d65 = new double[DATA_ROW];
	obs_x= new double[DATA_ROW];
	obs_y = new double[DATA_ROW];
	obs_z = new double[DATA_ROW];
	gauss_data = new double[DATA_ROW * 10];
	result = new double[3 * DATANUM * CALCNUM];

	/* CUDA�p�̕ϐ� */
	double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, *d_result;
	char* d_sim_order;

	/* GPU�������m�� */
	cudaMalloc((void**)&d_d65, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_x, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_y, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_z, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_gauss_data, DATA_ROW * 10 * sizeof(double));
	cudaMalloc((void**)&d_result, 3 * DATANUM * CALCNUM * sizeof(double));

	/* �t�@�C���ǂݍ��݊֐����s */
	int f_result = getFileData(d65_data, obs_data);

	/* �K�E�V�A���v�Z */
	makeGaussShift(gauss_shift);

	/* vector��1�����z��֕ϊ� */
	cpyVecToArray(d65_data, obs_data, gauss_shift,d65,obs_x,obs_y,obs_z,gauss_data);


	/* CUDA�ւ̃������R�s�[ */
	cudaMemcpy(d_d65, d65, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_x, obs_x, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_y, obs_y, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_z, obs_z, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);

	colorSim<DATA_ROW> << <1, DATA_ROW >> > (0,d_gauss_data,d_d65,d_obs_x,d_obs_y,d_obs_z,d_result,remain);
	cudaDeviceSynchronize();

	/* ���ʂ̃R�s�[ */
	cudaMemcpy(result, d_result, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);

	/* �f�o�C�X��������� */
	cudaFree(d_d65);
	cudaFree(d_gauss_data);
	cudaFree(d_obs_x);
	cudaFree(d_obs_y);
	cudaFree(d_obs_z);
	cudaFree(d_result);
}