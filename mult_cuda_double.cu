#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>



#define TILE_WIDTH 32

void ler_Matlab(double *L, int ordem, char nome[50]) {
	FILE *arq;
	double valor;
	int i = 0;

	arq = fopen(nome, "r");
	if (arq == NULL) {
		printf("\nERRO AO ABRIR ARQUIVO!!!\n");
		return;
	}
	while (!feof(arq)) {
		fscanf(arq, "%lf", &valor);
		L[i] = valor;
		i++;
	}
	fclose(arq);
}

void escrever_Matriz(double *L, int ordem, char nome[20]) {
	FILE *arq;
	int i, j;
	arq = fopen(nome, "w");
	if (arq == NULL) {
		printf("\nERRO AO CRIAR ARQUIVO!!!\n");
		return;
	}
	for (i = 0; i < ordem; i++) {
		for (j = 0; j < ordem; j++) {
			fprintf(arq, "%.2lf,", L[i*ordem + j]);
		}
		fprintf(arq, "\n");
	}
	fclose(arq);
}

__global__ void mult_Matriz(double *A, double *B, double *C, int ordem) {
	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;
	
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	double value = 0;
	
	__shared__ double a_sub[TILE_WIDTH][TILE_WIDTH]; //Matrizes Compartilhadas
	__shared__ double b_sub[TILE_WIDTH][TILE_WIDTH]; //Matrizes Compartilhadas
	a_sub[ty][tx] = 0.0;
	b_sub[ty][tx] = 0.0;

	for (int m = 0; m < (ordem - 1) / TILE_WIDTH + 1; ++m){
	
		if (Row < ordem && m*TILE_WIDTH + tx < ordem) a_sub[ty][tx] = A[Row*ordem + (m*TILE_WIDTH + tx)];
		else a_sub[ty][tx] = 0;
	   
		if (Col < ordem && m*TILE_WIDTH + ty < ordem) b_sub[ty][tx] = B[(m*TILE_WIDTH + ty)*ordem + Col];
		else b_sub[ty][tx] = 0;

		__syncthreads();
		
		for (int k = 0; k < TILE_WIDTH; k++) 
			value += a_sub[ty][k] * b_sub[k][tx];
			
		__syncthreads();
	}

	if (Row < ordem && Col < ordem) C[Row * ordem + Col] = value;
}

void multiplica_Matriz_Cuda(double *matA, double *matB, double *matC, int ordem) {
	double *dev_A;
	double *dev_B;
	double *dev_C;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaMalloc((void**)&dev_A, ordem*ordem * sizeof(double));
	cudaMalloc((void**)&dev_B, ordem*ordem * sizeof(double));
	cudaMalloc((void**)&dev_C, ordem*ordem * sizeof(double));
	
	cudaMemcpy(dev_A, matA, ordem*ordem * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, matB, ordem*ordem * sizeof(double), cudaMemcpyHostToDevice);
		
	cudaEventRecord(start, 0);
	
	dim3 dimGrid((ordem - 1) / TILE_WIDTH + 1, (ordem - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	
	mult_Matriz << <dimGrid, dimBlock >> >(dev_A, dev_B, dev_C, ordem);
	
	cudaDeviceSynchronize();
	cudaMemcpy(matC, dev_C, ordem*ordem * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milisegundos = 0;
	
	cudaEventElapsedTime(&milisegundos, start, stop);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	printf("\nMultiplicacao realizada em: %.2f milliseconds", milisegundos);
	
	int device;
	cudaGetDevice(&device);	
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	printf("\nDevice name: %s\n", prop.name);
	
}


int  main(int argc, char** argv) {
	double *matA;
	double *matB;
	double *matC;
	int i, j;
	
	int ordem = atoi(argv[3]);

	matA = (double*)malloc(ordem*ordem * sizeof(double));
	matB = (double*)malloc(ordem*ordem * sizeof(double));
	matC = (double*)malloc(ordem*ordem * sizeof(double));
	for (i = 0; i < ordem; i++) {
		for (j = 0; j < ordem; j++) {
			matA[i*ordem + j] = 0;
			matB[i*ordem + j] = 0;
			matC[i*ordem + j] = 0;
		}
	}
	
	ler_Matlab(matA, ordem, argv[1]);
	ler_Matlab(matB, ordem, argv[2]);
	
	multiplica_Matriz_Cuda(matA, matB, matC, ordem);
	
	escrever_Matriz(matC, ordem, "Mult_em_CUDA.txt");
	printf("\nDimensao: %d", ordem);
	printf("\nFINALIZADO\n\n");
	return 0;
}
