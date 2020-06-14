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
#define DATANUM 50			// �v�Z���鐔
#define CALCNUM 100		// �ׂ��悷�鐔
#define SIMNUM 1023			// �V�~�����[�V���������
#define LOOPNUM 2			// SIMNUM��̃V�~�����[�V�����J��Ԃ���

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

/* �t�@�C������f�[�^��ǂݍ��ފ֐� */
int getFileData(vector<vector<double> >& d65_data, vector<vector<double> >& obs_data, vector<vector<double> > & xyz_data) {
	/* �t�@�C���|�C���^ */
	FILE* fp_d65, * fp_obs, * fp_xyz;
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
		ret = fscanf(fp_obs, "%lf, %lf, %lf, %lf", &(obs_data[count][0]), &(obs_data[count][1]), &(obs_data[count][2]), &(obs_data[count][3]));
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

	/* �W���ϑ��҂̓ǂݍ��� */
	/* �t�@�C���I�[�v�� */
	fp_xyz = fopen("./ciexyz31.csv", "r");
	/* �������J���Ă��邩���`�F�b�N */
	if (fp_xyz == NULL) {
		cout << "File open error" << endl;
		return -1;
	}

	/* �t�@�C���ǂݍ��� */
	for (int i = 0; i < OBS_ROW; i++) {
		/* 1�s���ǂݍ��� */
		ret = fscanf(fp_xyz, "%lf, %lf, %lf, %lf", &(xyz_data[count][0]), &(xyz_data[count][1]), &(xyz_data[count][2]), &(xyz_data[count][3]));
		/* �I������ */
		if (xyz_data[count][0] == DATA_MAX) {
			count = 0;
			break;
		}
		/* �J�E���^�̍X�V */
		if (xyz_data[count][0] >= DATA_MIN) {
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
		//mu = (double)DATA_MIN + ((double)DATA_MAX - (double)DATA_MIN) / 10 * i;
		mu = (double)DATA_MIN + ( (double)rand() / RAND_MAX * ((double)DATA_MAX - (double)DATA_MIN ) );
		sigma = 5 + (95 * (double)rand() / RAND_MAX);

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
			//shift_data[j][i] = shift_data[j][i] / d_max * w_length;
			shift_data[j][i] = shift_data[j][i] / d_max;
		}
		/* �ő�l������ */
		d_max = 0;
	}
}

/* vector�^����z��փf�[�^���R�s�[����֐� */
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
template<int BLOCK_SIZE> __global__ void colorSim(double simNum,double *g_data,double *d65,double *obs_x,double *obs_y,double *obs_z,double *result,int remain,int *d_mesh)  {
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
	__shared__ double g_max;
	g_max = 0;
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

	/* �u���b�N���̃X���b�h���� */
	__syncthreads();

	for (int i = 0; i < CALCNUM; i++) {
		/* �V�F�A�[�h�������Ƀf�[�^�i�[ */
		calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(gaussian, (0.001 * i));

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

			//printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
		}

		/* �u���b�N���� */
		__syncthreads();

		/* ���b�V���̔ԍ����ӂ� */
		/* ���b�V���p�̕ϐ� */
		double x, y;
		/* �t���Oxy */
		int f_x, f_y;
		/* ���b�V������p�̕ϐ� */
		double m_x, m_y;

		/* f_x,f_y�̏����� */
		f_x = 0;
		f_y = 0;

		/* x���� */
		if (ix <= 36) {
			/* xy�̌v�Z */
			x = calc_data[0][0] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
			/* ���b�V���̔��� */
			m_x = (double)ix * 0.02;
			if (m_x <= x && (m_x + 0.02) > x) {
				f_x = 1;
			}
		}
		/* y���� */
		if (ix >= 64 && ix <= 106) {
			/* xy�̌v�Z */
			y = calc_data[0][1] / (calc_data[0][0] + calc_data[0][1] + calc_data[0][2]);
			/* ���b�V���̔��� */
			m_y = (double)(ix - 64) * 0.02;
			if (m_y <= y && (m_y + 0.02) > y) {
				f_y = 1;
			}
		}

		/* �u���b�N���� */
		__syncthreads();

		/* ���b�V���̈ʒu�v�Z */
		if (ix <= 36){
			if (f_x == 1) {
				aPos = blockIdx.x * CALCNUM + i;
				d_mesh[aPos] = ix;
			}
		}

		/* �u���b�N���� */
		__syncthreads();

		/* ���b�V���̈ʒu�v�Z */
		if (ix >= 64 && ix <= 106) {
			if (f_y == 1) {
				aPos = blockIdx.x * CALCNUM + i;
				d_mesh[aPos] += (ix - 64) * 37;
			}
		}

		/* �u���b�N���� */
		__syncthreads();
	}
}

/* LMS�v�Z�J�[�l�� */
template<int BLOCK_SIZE> __global__ void lmsSim(double simNum, double* g_data, double* d65, double* obs_x, double* obs_y, double* obs_z, double* result, int remain) {
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
	__shared__ double g_max;
	g_max = 0;
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

	/* �u���b�N���̃X���b�h���� */
	__syncthreads();

	for (int i = 0; i < CALCNUM; i++) {
		/* �V�F�A�[�h�������Ƀf�[�^�i�[ */
		calc_data[ix][0] = d65[ix] * obs_x[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][1] = d65[ix] * obs_y[ix] * pow(gaussian, (0.001 * i));
		calc_data[ix][2] = d65[ix] * obs_z[ix] * pow(gaussian, (0.001 * i));

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

			//printf("%.3lf %.3lf %.3lf\n", calc_data[0][0], calc_data[0][1], calc_data[0][2]);
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
	/*�W���ϑ���(cie1931xyz)�̃f�[�^���i�[����z�� */
	vector<vector<double> > xyz_data(DATA_ROW, vector<double>(OBS_COL, 0));
	/*�K�E�V�A����10�i�[����z�� */
	vector<vector<double> > gauss_shift(DATA_ROW, vector<double>(10, 0));

	/* �]��v�Z */
	int remain = getRemain();

	/* �f�[�^������P�����z�� */
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

	/* CUDA�p�̕ϐ� */
	double* d_d65, * d_obs_x, * d_obs_y, * d_obs_z, * d_gauss_data, * d_result, * d_lms;
	int* d_mesh;
	char* d_sim_order;

	/* GPU�������m�� */
	cudaMalloc((void**)&d_d65, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_x, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_y, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_obs_z, DATA_ROW * sizeof(double));
	cudaMalloc((void**)&d_gauss_data, DATA_ROW * 10 * sizeof(double));
	cudaMalloc((void**)&d_result, 3 * DATANUM * CALCNUM * sizeof(double));
	cudaMalloc((void**)&d_mesh, DATANUM * CALCNUM * sizeof(int));

	/* �t�@�C���ǂݍ��݊֐����s */
	int f_result = getFileData(d65_data, obs_data, xyz_data);

	/* vector��1�����z��֕ϊ� */
	cpyVecToArray(d65_data, obs_data, gauss_shift, xyz_data, d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z, gauss_data);

	/* CUDA�ւ̃������R�s�[ */
	cudaMemcpy(d_d65, d65, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_x, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_y, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_z, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);


	for (int i = 0; i < LOOPNUM; i++) {
		/* �K�E�V�A���v�Z */
		makeGaussShift(gauss_shift);
		/* vector��1�����z��֕ϊ� */
		cpyVecToArray(d65_data, obs_data, gauss_shift, xyz_data, d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z, gauss_data);
		/* �K�E�V�A����ʂ̔z��֊i�[ */
		for (int j = 0; j < DATA_ROW * 10; j++) {
			gauss_com[(i * DATA_ROW * 10) + j] = gauss_data[j];
		}
		/* CUDA�ւ̃������R�s�[ */
		cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);

		for(int j = 0; j < (SIMNUM - DATANUM); j += DATANUM) {
			colorSim<DATA_ROW> << <DATANUM, DATA_ROW >> > ((j + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh);
			cudaDeviceSynchronize();

			/* ���ʂ̃R�s�[ */
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

		/* ���[�v�ŗ]�����c��̐����V�~�����[�V���� */
		int r_num = SIMNUM % DATANUM - 1;
		int sim_num = SIMNUM - r_num - 1;
		colorSim<DATA_ROW> << <r_num, DATA_ROW >> > ((sim_num + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_result, remain, d_mesh);
		cudaDeviceSynchronize();

		/* ���ʂ̃R�s�[ */
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
	/* ��������� */
	cudaFree(d_result);
	cudaMalloc((void**)&d_lms, 3 * DATANUM * CALCNUM * sizeof(double));
	cudaMemcpy(d_obs_x, obs_l, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_y, obs_m, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_obs_z, obs_s, DATA_ROW * sizeof(double), cudaMemcpyHostToDevice);

	/* LMS�v�Z */
	for (int i = 0; i < LOOPNUM; i++) {
		/* �K�E�V�A���v�Z */
		makeGaussShift(gauss_shift);
		/* vector��1�����z��֕ϊ� */
		cpyVecToArray(d65_data, obs_data, gauss_shift, xyz_data, d65, obs_l, obs_m, obs_s, obs_x, obs_y, obs_z, gauss_data);
		/* �K�E�V�A����ʂ̔z��֊i�[ */
		for (int j = 0; j < DATA_ROW * 10; j++) {
			gauss_data[j] = gauss_com[(i * DATA_ROW * 10) + j];
		}
		/* CUDA�ւ̃������R�s�[ */
		cudaMemcpy(d_gauss_data, gauss_data, DATA_ROW * 10 * sizeof(double), cudaMemcpyHostToDevice);

		for (int j = 0; j < (SIMNUM - DATANUM); j += DATANUM) {
			lmsSim<DATA_ROW> << <DATANUM, DATA_ROW >> > ((j + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_lms, remain);
			cudaDeviceSynchronize();

			/* ���ʂ̃R�s�[ */
			cudaMemcpy(lms_result, d_lms, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);

			for (int k = 0; k < (3 * DATANUM * CALCNUM); k++) {
				int aPos = (i * 3 * CALCNUM * SIMNUM) + (3 * CALCNUM * j) + k;
				lms_fin[aPos] = lms_result[k];
			}
		}

		/* ���[�v�ŗ]�����c��̐����V�~�����[�V���� */
		int r_num = SIMNUM % DATANUM - 1;
		int sim_num = SIMNUM - r_num - 1;
		lmsSim<DATA_ROW> << <r_num, DATA_ROW >> > ((sim_num + 1), d_gauss_data, d_d65, d_obs_x, d_obs_y, d_obs_z, d_lms, remain);
		cudaDeviceSynchronize();

		/* ���ʂ̃R�s�[ */
		cudaMemcpy(lms_result, d_lms, 3 * DATANUM * CALCNUM * sizeof(double), cudaMemcpyDeviceToHost);

		for (int k = 0; k < (3 * r_num * CALCNUM); k++) {
			int aPos = (i * 3 * CALCNUM * SIMNUM) + (3 * CALCNUM * sim_num) + k;
			lms_fin[aPos] = lms_result[k];
		}
	}

	/* ���ʂ��I�������𖞂����Ă���Ƃ��ɒl��0�ɂ��� */
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

	/* �o�̓f�B���N�g�� */
	//string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_1023_15000_10_v2/";
	string directory = "C:/Users/KoidaLab-WorkStation/Desktop/isomura_ws/color_simulation_result/sim_test_1023_100_2/";

	/* �o�͂����t�@�C���̏����L�^����t�@�C�� */
	string f_info = "sim_file_info.txt";
	f_info = directory + f_info;
	ofstream o_f_info(f_info);

	/* �t�@�C���������� */
	for (int i = 0; i < LOOPNUM; i++) {
		/* �o�̓t�@�C���� */
		string fname1 = "sim_result_L_xyz_1023_";
		string fname2 = "sim_result_S_xyz_1023_";
		string fname3 = "sim_result_lms_1023_";
		string fend = ".csv";
		fname1 = directory + fname1 + to_string(i + 1) + fend;
		fname2 = directory + fname2 + to_string(i + 1) + fend;
		fname3 = directory + fname3 + to_string(i + 1) + fend;

		/* �t�@�C���o�̓X�g���[�� */
		ofstream o_file1(fname1);
		ofstream o_file2(fname2);
		ofstream o_file3(fname3);

		/* �o�͂����t�@�C���̏����L�^����t�@�C���Ƀt�@�C�������o�� */
		o_f_info << fname1 << endl;
		o_f_info << fname2 << endl;
		o_f_info << fname3 << endl;

		/* �t�@�C���ւ̏o�͌����w�� */
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


				/* XYZ == 0�̂Ƃ� */
				if (X == 0 && Y == 0 && Z == 0) {
					apos = j + (k * CALCNUM) + (SIMNUM * CALCNUM * i);
					o_file1 << ",,,";
					o_file2 << ",,," << mesh_f_result[apos] << ",";
					o_file3 << ",,,,,";
				}

				/* ����ȊO�̂Ƃ� */
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

			/* XYZ == 0�̂Ƃ� */
			if (X == 0 && Y == 0 && Z == 0) {
				apos = j + (SIMNUM - 1) * CALCNUM + (SIMNUM * CALCNUM * i);
				o_file1 << ",,";
				o_file2 << ",,," << mesh_f_result[apos];
				o_file3 << ",,,,";
			}

			/* ����ȊO�̂Ƃ� */
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
		/* �t�@�C���N���[�Y */
		o_file1.close();
		o_file2.close();
		o_file3.close();
	}

	/* �f�o�C�X��������� */
	cudaFree(d_d65);
	cudaFree(d_gauss_data);
	cudaFree(d_obs_x);
	cudaFree(d_obs_y);
	cudaFree(d_obs_z);
	cudaFree(d_lms); 

	/* �z�X�g��������� */
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