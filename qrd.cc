#include <stdio.h>
#include <iostream>
#include <math.h>
using namespace std;

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

double dotprod(double *vec1, double *vec2, int nDim)
{
	double x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec1[i] * vec2[i];
	return x;
}

double* matmult(double *mat1, double *mat2, int nDim)
{
	double *x = new double[nDim * nDim];
	for (int i = 0; i < nDim * nDim; ++i)
		x[i] = 0;
	for (int k = 0; k < nDim; ++k)
		for (int j = 0; j < nDim; ++j)
			for (int i = 0; i < nDim; ++i)
				x[j + k * nDim] += mat1[j + i * nDim] * mat2[i + k * nDim];
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

void make_comp_mat(double *polynomial, double *companion, int nDim)
{
	for (int i = 0; i < nDim * nDim; ++i)
		companion[i] = 0;
	for (int i = 0; i < nDim; ++i)
		companion[i * nDim] = -polynomial[i + 1] / polynomial[0];
		//companion[i + nDim * (nDim - 1)] = -polynomial[i + 1] / polynomial[0];
	for (int i = 0; i < nDim - 1; ++i)
		companion[i * nDim + i + 1] = 1;
}

void select_diag(double *vector, double *matrix, int nDim)
{
	for (int i = 0; i < nDim; ++i)
		vector[i] = matrix[i * nDim + i];
}

void gram_schmidt(double *a, double *Q, double *R, int nDim)
{
	double *u = new double[nDim];
	double *v = new double[nDim];
	double l2 = 0;
	for (int i = 0; i < nDim * nDim; ++i)
		Q[i] = R[i] = 0;
	for (int i = 0; i < nDim; ++i)
		u[i] = v[i] = 0;

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

	delete[] u;
	delete[] v;
}

void root_find(double *polynomial, double *root, int nDim, double tolerance, int upperbound)
{
	double *a = new double[nDim * nDim];
	double *Q = new double[nDim * nDim];
	double *R = new double[nDim * nDim];
	int nTol = 0;
	for (int i = 0; i < nDim; ++i)
		root[i] = 0;

	make_comp_mat(polynomial, a, nDim);

	for (int k = 0; k < upperbound; ++k)
	{
		gram_schmidt(a, Q, R, nDim);
		a = matmult(R, Q, nDim);
		nTol = 0;
		for (int j = 0; j < nDim; ++j)
			for (int i = 0; i < nDim; ++i) {
printf(" k=%i i=%i j=%i a[]=%f tol=%f nTol=%i \n",k,i,j,fabs(a[i+j*nDim]),tolerance,nTol);
				if (i > j && fabs(a[i + j * nDim]) > tolerance) ++nTol; }
		if (nTol == 0) break;
	}
matrix_print(a, nDim);
	select_diag(root, a, nDim);

	delete[] a;
	delete[] Q;
	delete[] R;
}

int main()
{
	int nDim = 3;
	double tolerance = 0.0001;
	double upperbound = 10000;
	double *polynomial = new double[nDim];
	double *root = new double[nDim];
	double *comp = new double[nDim * nDim];

	polynomial[0]=1;polynomial[1]=-6;polynomial[2]=-72;polynomial[3]=-27;
	make_comp_mat(polynomial, comp, nDim);
	matrix_print(comp, nDim);

	root_find(polynomial, root, nDim, tolerance, upperbound);
	for (int i = 0; i < nDim; ++i)
		cout << " " << root[i] << endl;
/*
	int nDim = 5;
	double a[6] = { 2, 2, 3, 4, 5, 6 };
	double mat[25] = { 0 };
	make_comp_mat(a, mat, nDim);
	matrix_print(mat, nDim);
	double b[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	double c[3] = { 0, 0, 0 };
	select_diag(c, b, 3);
	matrix_print(c, 3);
*/
/*
	int nDim = 3;
	double a[9] = { 1, 1, 0, 1, 0, 1, 0, 1, 1 };
	double Q[9] = { 0 };
	double R[9] = { 0 };
	matrix_print(a, nDim);
	matrix_print(Q, nDim);
	matrix_print(R, nDim);
	gram_schmidt(a, Q, R, nDim);
	matrix_print(Q, nDim);
	matrix_print(R, nDim);
	transpose(R, nDim);
	matrix_print(R, nDim);
*/
	return 0;
}
