#include <stdio.h>
#include <iostream>
#include <math.h>
using namespace std;

void matrix_write(double *mat, int nDim)
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

double dotprod(double *vec1, double *vec2, int nDim)
{
	double x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec1[i] * vec2[i];
	return x;
}

double l2norm(double *vec, int nDim)
{
	double x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec[i] * vec[i];
	x = sqrt(x);
	return x;
}

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

void gram_schmidt(double *a, double *Q, double *R, int nDim)
{
	double *u = new double[nDim];
	double *v = new double[nDim];
	double l2 = 0;
	*Q = { 0 };
	*R = { 0 };
	*u = { 0 };
	*v = { 0 };

	for (int k = 0; k < nDim; ++k)
	{
		for (int i = 0; i < nDim; ++i)
			u[i] = a[i + k * nDim];
		for (int j = k - 1; j >= 0; --j)
			for (int i = 0; i < nDim; ++i)
				u[i] -= R[j + k * nDim] * Q[i + j * nDim];
		l2 = l2norm(u, nDim);
		for (int i = 0; i < nDim; ++i)
			Q[i + k * nDim] = u[i] / l2;
		for (int j = k; j < nDim; ++j)
		{
			for (int i = 0; i < nDim; ++i)
			{
				u[i] = a[i + j * nDim];
				v[i] = Q[i + k * nDim];
			}
			R[k + j * nDim] = dotprod(u, v, nDim);
		}
	}

	delete[](u);
	delete[](v);
}

int main()
{
	int nDim = 3;
	double a[9] = { 1, 1, 0, 1, 0, 1, 0, 1, 1 };
	double Q[9] = { 0 };
	double R[9] = { 0 };
	matrix_write(a, nDim);
	matrix_write(Q, nDim);
	matrix_write(R, nDim);
	gram_schmidt(a, Q, R, nDim);
	matrix_write(Q, nDim);
	matrix_write(R, nDim);
	transpose(R, nDim);
	matrix_write(R, nDim);
	return 0;
}
