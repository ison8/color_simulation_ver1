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
#define DATANUM 10			// �v�Z���鐔
#define CALCNUM 10000		// �ׂ��悷�鐔

/* �o�̓t�@�C���p�X */
#define F_PATH "C:/Users/ryoin/source/repos/color_simulation_cuda/color_simulation_cuda"

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


/* �ϕ��v�Z�J�[�l�� */
template<int BLOCK_SIZE> __global__ void colorSim() {

}

int main(void) {
	/* D65�̃f�[�^���i�[����z�� */
	vector<vector<double> > d65_data(DATA_ROW, vector<double>(D65_COL, 0));
	/*�W���ϑ��҂̃f�[�^���i�[����z�� */
	vector<vector<double> > obs_data(DATA_ROW, vector<double>(OBS_COL, 0));
	/*�K�E�V�A����10�i�[����z�� */
	vector<vector<double> > gauss_shift(DATA_ROW, vector<double>(10, 0));

	/* �f�[�^������P�����z�� */
	double* d65, * obs_x, * obs_y, * obs_z, * gauss_data;
	d65 = new double[DATA_ROW];
	obs_x= new double[DATA_ROW];
	obs_y = new double[DATA_ROW];
	obs_z = new double[DATA_ROW];
	gauss_data = new double[DATA_ROW * 10];

	/* �t�@�C���ǂݍ��݊֐����s */
	int f_result = getFileData(d65_data, obs_data);

	/* �K�E�V�A���v�Z */
	makeGaussShift(gauss_shift);

	/* vector��1�����z��֕ϊ� */
	cpyVecToArray(d65_data, obs_data, gauss_shift,d65,obs_x,obs_y,obs_z,gauss_data);

}