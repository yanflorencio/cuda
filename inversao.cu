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

#define TILE_WIDTH 1024
#define TILE_WIDTH_No_Col 16

void ler_Matlab(double *L, double ordem, char nome[50]) {
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
			fprintf(arq, "%.2f,", L[i*ordem + j]);
		}
		fprintf(arq, "\n");
	}
	fclose(arq);
}

// FUNÇÃO PARA EVITAR A DIVISÃO POR 0, NO MOMENTO DE DIVIDIR OS ELEMENTOS DA MATRIZ
// PELOS ELEMENTOS QUE ESTÃO NA DIAGONAL PRINCIPAL
__global__ void troca_diagonal_zero(double *Matrix, double *Idn, int ordem, int idLinha){
	int T = blockIdx.x * blockDim.x + threadIdx.x;
	
	if((T < ordem)){
			if(Matrix[idLinha*ordem+idLinha] == 0){
				__shared__ int k;
				double value;
		
				for(int i = 0; i < ordem; i++){
					if(Matrix[i * ordem + idLinha] != 0){
						k = i;
					}
				}
				__syncthreads();
				
				value = Matrix[k*ordem + T];
				Matrix[k*ordem + T] = Matrix[idLinha*ordem + T];
				Matrix[idLinha*ordem + T] = value;
							
				value = Idn[k*ordem + T];
				Idn[k*ordem + T] = Idn[idLinha*ordem + T];
				Idn[idLinha*ordem + T] = value;
		}
	}
}

//APLICA AS DIVISÕES
__global__ void fix_linha(double *Matrix, double *Idn, int ordem, int idLinha){
	int T = blockIdx.x * TILE_WIDTH + threadIdx.x;
	
	if((T < ordem)){
		int TIndex = threadIdx.x;
			
			//Resgata o valor da diagonal principal da matriz "Matrix"
			__shared__ double D;
			D = Matrix[idLinha*ordem+idLinha];
			
			//Valores da linha "idLinha" nas matrizes "Matrix" e "Idn"
			__shared__ double SharedRow_A[TILE_WIDTH];
			__shared__ double SharedRow_B[TILE_WIDTH];
	
			SharedRow_A[TIndex] = Matrix[idLinha * ordem + T];
			SharedRow_B[TIndex] = Idn[idLinha * ordem + T];
	
			__syncthreads();
			
			//Realiza as divisões
			SharedRow_A[TIndex] /= D;
			SharedRow_B[TIndex] /= D;
			
			Idn[idLinha * ordem + T] = SharedRow_B[TIndex];
			Matrix[idLinha * ordem + T] = SharedRow_A[TIndex];
	}
}

//APLICA OPERAÇÕES ELEMENTARES
__global__ void fix_No_idColunas (double *Matrix, double *Idn, int ordem, int idColuna){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int TIndex = threadIdx.x;
	
		if((i < ordem) && (j < ordem)){
		
		//Elementos da coluna j, nas matrizes "Matrix" e "Idn"
		//Elemetos "Lj" na equação	
		__shared__ double AColIdj;
		__shared__ double BColIdj;
		
		//Elementos da coluna idColuna da matriz "Matrix"
		//Elementos "Cj,i" na equação
		__shared__ double col_M[TILE_WIDTH_No_Col];
		
		//Elementos da coluna j, nas matrizes "Matrix" e "Idn"
		//Elementos "Li" na equação 
		__shared__ double colj_M[TILE_WIDTH_No_Col];
		__shared__ double colj_Idn[TILE_WIDTH_No_Col];
	
		col_M[TIndex] = Matrix[i * ordem + idColuna];
		
		__syncthreads();
	
			if (col_M[TIndex] != 0){
				colj_M[TIndex] = Matrix[i * ordem + j]; //"Li"
				colj_Idn[TIndex] = Idn[i * ordem + j];	//"Li"
				AColIdj = Matrix[idColuna * ordem + j]; //"Lj"
				BColIdj = Idn[idColuna * ordem + j];	//"Lj"
					
					//Aplica operações Elementares nos valores, exceto na coluna "idColuna"
					if (i != idColuna){
						if (j != idColuna)colj_M[TIndex] -= col_M[TIndex] * AColIdj;
						colj_Idn[TIndex] -= col_M[TIndex] * BColIdj;	
					}
					
				Idn[i * ordem + j] = colj_Idn[TIndex];
				Matrix[i * ordem + j] = colj_M[TIndex];
			}
	}
}

//APLICA OPERAÇÕES ELEMENTARES
__global__ void fix_idColuna(double *Matrix, double *Idn, int ordem, int idColuna){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int TIndex = threadIdx.x;
	
		if((i < ordem)){
		//Elementos da coluna j, nas matrizes "Matrix"
		//Elementos "Lj" na equação				
		__shared__ double AColIdj;
		
		//Elemento da coluna idColuna da matriz "Matrix"
		//Elementos "Cj,i" na equação		
		__shared__ double col_M[TILE_WIDTH];
		
		//Elemento da coluna j, nas matrizes "Matrix" e "Idn"
		//Elementos "Li" na equação 
		__shared__ double colj_M[TILE_WIDTH];
		
		col_M[TIndex] = Matrix[i * ordem + idColuna];
		
		__syncthreads();
	
			if (col_M[TIndex] != 0){
				colj_M[TIndex] = Matrix[i * ordem + idColuna]; //"Li"
				AColIdj = Matrix[idColuna * ordem + idColuna]; //"Lj"
				//Aplica operações Elementares nos valores apenas na coluna "idColuna"
					if (i != idColuna){
						colj_M[TIndex] -= col_M[TIndex] * AColIdj;
					}
				Matrix[i * ordem + idColuna] = colj_M[TIndex];
			}
	}
}


void inverte_Matriz_Cuda(double *matA, double *matInversa, int ordem){
	double *d_A, *Identidade, *dIdent;
	int i, j;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	
	Identidade = (double*)malloc(ordem*ordem * sizeof(double));
	cudaMalloc((void**)&d_A, ordem*ordem * sizeof(double));
	cudaMalloc((void**)&dIdent, ordem*ordem * sizeof(double));
	
	for (i = 0; i < ordem; i++){
		for (j = 0; j < ordem; j++){
			if (i == j) Identidade[i*ordem + i] = 1.0;
			else Identidade[i*ordem + j] = 0.0;
		}
	}
	
	cudaMemcpy(d_A, matA, ordem*ordem * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dIdent, Identidade, ordem*ordem * sizeof(double), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	
	dim3 dimGrid((ordem - 1) / TILE_WIDTH + 1, 1, 1);
	dim3 dimBlock(TILE_WIDTH, 1, 1);
	
	dim3 dimGrid_No_Col((ordem - 1) / TILE_WIDTH_No_Col + 1, (ordem - 1) / TILE_WIDTH_No_Col + 1, 1);
	dim3 dimBlock_No_Col(TILE_WIDTH_No_Col, TILE_WIDTH_No_Col, 1);
	
	for (int i = 0; i < ordem; i++){
	//	troca_diagonal_zero << <dimGrid, dimBlock>> >(d_A, dIdent, ordem, i);
		fix_linha << <dimGrid, dimBlock>> >(d_A, dIdent, ordem, i);
		fix_No_idColunas << <dimGrid_No_Col, dimBlock_No_Col>> >(d_A, dIdent, ordem, i);
		fix_idColuna<< <dimGrid, dimBlock>> >(d_A, dIdent, ordem, i);
	}
	cudaMemcpy(matInversa, dIdent, ordem*ordem * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Identidade, d_A, ordem*ordem * sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milisegundos = 0;
	
	cudaEventElapsedTime(&milisegundos, start, stop);
	
	escrever_Matriz(Identidade, ordem, "Identidade.txt");
	
	cudaFree(d_A);
	cudaFree(dIdent);
	printf("\nInversão realizada em: %.2f milliseconds", milisegundos);
	
	int device;
	cudaGetDevice(&device);	
	
	escrever_Matriz(Identidade, ordem, "Identidade.txt");
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	printf("\nDevice name: %s\n", prop.name);	
}

int  main(int argc, char** argv) {
	double *matA;
	double *matInversa;
	int i, j;
	int ordem = atoi(argv[2]);

	matA = (double*)malloc(ordem*ordem * sizeof(double));
	matInversa = (double*)malloc(ordem*ordem * sizeof(double));
	for (i = 0; i < ordem; i++) {
		for (j = 0; j < ordem; j++) {
			matA[i*ordem + j] = 0;
			matInversa[i*ordem + j] = 0;
		}
	}
	
	ler_Matlab(matA, ordem, argv[1]);
	inverte_Matriz_Cuda(matA, matInversa, ordem);
	
	escrever_Matriz(matInversa, ordem, "Inversa.txt");
	printf("\nDimensao: %d", ordem);
	printf("\nFINALIZADO\n\n");
	

	return 0;
}
