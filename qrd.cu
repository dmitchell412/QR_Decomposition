#include <stdio.h>
#include <iostream>
#include <math.h>
using namespace std;

__device__
void matrix_print(double *mat, int nDim)
{
	for (int j = 0; j < nDim; ++j)
	{
		for (int i = 0; i < nDim; ++i)
		{
			printf(" %8.3f", mat[j + i * nDim]);
		}
		printf("\n");
	}
	printf("\n");
}

__device__
double dotprod(double *vec1, double *vec2, int nDim)
{
	double x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec1[i] * vec2[i];
	return x;
}

__device__
double l2norm(double *vec, int nDim)
{
	double x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec[i] * vec[i];
	x = sqrt(x);
	return x;
}

__device__
void transpose(double *mat, int nDim)
{
	double *x = new double[nDim * nDim];
	for (int i = 0; i < nDim * nDim; ++i)
		x[i] = mat[i];
	for (int j = 0; j < nDim; ++j)
		for (int i = 0; i < nDim; ++i)
			mat[i + j * nDim] = x[j + i * nDim];
	delete[] x;
}

__device__
void pixel_mat_select(
	double *a_image, 
	double *a, 
	int nDim_image, 
	int nDim_matrix, 
	int i_image)
{
	int offset_2d = i_image * nDim_matrix * nDim_matrix;
	for (int i = 0; i < nDim_matrix * nDim_matrix; ++i)
		a[i] = a_image[i + offset_2d];
}

__device__
void pixel_mat_write(
	double *Q_image,
	double *R_image,
	double *Q,
	double *R,
	int nDim_image,
	int nDim_matrix,
	int i_image)
{
	int offset_2d = i_image * nDim_matrix * nDim_matrix;
	for (int i = 0; i < nDim_matrix * nDim_matrix; ++i)
	{
		Q_image[i + offset_2d] = Q[i];
		R_image[i + offset_2d] = R[i];
	}
}

__global__
void gram_schmidt(
	double *a_image, 
	double *Q_image, 
	double *R_image,
	int const nDim_image, 
	int const nDim_matrix)
{
	// Assign image pixels to blocks and threads
	int i_image = blockDim.x * blockIdx.x + threadIdx.x;
	if (i_image > nDim_image * nDim_image) return;

	double *a = new double[nDim_matrix * nDim_matrix];
	double *Q = new double[nDim_matrix * nDim_matrix];
	double *R = new double[nDim_matrix * nDim_matrix];
	double *u = new double[nDim_matrix];
	double *v = new double[nDim_matrix];
	double l2 = 0;
	for (int i = 0; i < nDim_matrix * nDim_matrix; ++i)
		a[i] = Q[i] = R[i] = 0;
	for (int i = 0; i < nDim_matrix; ++i)
		u[i] = v[i] = 0;
	
	pixel_mat_select(a_image, a, nDim_image, nDim_matrix, i_image);

	for (int k = 0; k < nDim_matrix; ++k)
	{
		for (int i = 0; i < nDim_matrix; ++i)
			u[i] = a[i + k * nDim_matrix];
		for (int j = k - 1; j >= 0; --j)
			for (int i = 0; i < nDim_matrix; ++i)
				u[i] -= R[j + k * nDim_matrix] * Q[i + j * nDim_matrix];
		l2 = l2norm(u, nDim_matrix);
		for (int i = 0; i < nDim_matrix; ++i)
			Q[i + k * nDim_matrix] = u[i] / l2;
		for (int j = k; j < nDim_matrix; ++j)
		{
			for (int i = 0; i < nDim_matrix; ++i)
			{
				u[i] = a[i + j * nDim_matrix];
				v[i] = Q[i + k * nDim_matrix];
			}
			R[k + j * nDim_matrix] = dotprod(u, v, nDim_matrix);
		}
	}

	delete[] u;
	delete[] v;

	pixel_mat_write(Q_image, R_image, Q, R, nDim_image, nDim_matrix, i_image);
}

