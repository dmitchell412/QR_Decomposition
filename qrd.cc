#include <stdio.h>
#include <iostream>
#include <math.h>
#include <complex>
using namespace std;

typedef complex<double> cdouble;

void matrix_print(cdouble *mat, int nDim)
{
	for (int j = 0; j < nDim; ++j)
	{
		for (int i = 0; i < nDim; ++i)
		{
			printf(" %8.3f + %8.3fi", mat[j + i * nDim].real(), mat[j + i * nDim].imag());
		}
		printf("\n");
	}
	printf("\n");
}

cdouble dotprod(cdouble *vec1, cdouble *vec2, int nDim)
{
	cdouble x = 0;
	cdouble tmp = 0;
	for (int i = 0; i < nDim; ++i)
	{
		tmp = vec1[i];
		tmp.imag() = -vec1[i].imag();
		x += tmp * vec2[i];
//printf("%f+%fi\n", x.real(), x.imag());
	}
	return x;
}

cdouble* matmult(cdouble *mat1, cdouble *mat2, int nDim)
{
	cdouble *x = new cdouble[nDim * nDim];
	for (int i = 0; i < nDim * nDim; ++i)
		x[i] = 0;
	for (int k = 0; k < nDim; ++k)
		for (int j = 0; j < nDim; ++j)
			for (int i = 0; i < nDim; ++i)
				x[j + k * nDim] += mat1[j + i * nDim] * mat2[i + k * nDim];
	return x;
}

cdouble l2norm(cdouble *vec, int nDim)
{
	cdouble x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec[i].real() * vec[i].real() + vec[i].imag() * vec[i].imag();
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
/*
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
*//*

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
//printf(" k=%i i=%i j=%i a[]=%f tol=%f nTol=%i \n",k,i,j,fabs(a[i+j*nDim]),tolerance,nTol);
				if (i > j && fabs(a[i + j * nDim]) > tolerance) ++nTol; }
		if (nTol == 0) break;
	}
//matrix_print(a, nDim);
	select_diag(root, a, nDim);

	delete[] a;
	delete[] Q;
	delete[] R;
}
*/
int main()
{
	int nDim = 4;
	double tolerance = 0.0001;
	double upperbound = 10000;
	cdouble *polynomial = new cdouble[nDim];
	cdouble *root = new cdouble[nDim];
	cdouble *comp = new cdouble[nDim * nDim];
	cdouble *m1 = new cdouble[nDim * nDim];
	cdouble *m2 = new cdouble[nDim * nDim];
	cdouble *m3 = new cdouble[nDim * nDim];

	polynomial[0].real()=1;polynomial[1].real()=-6;polynomial[2].real()=-72;polynomial[3].real()=-27;
	polynomial[0].imag()=1;polynomial[1].imag()=-6;polynomial[2].imag()=-72;polynomial[3].imag()=-27;
	root[0].real()=1;root[1].real()=-6;root[2].real()=-72;root[3].real()=-27;
	root[0].imag()=1;root[1].imag()=-6;root[2].imag()=-72;root[3].imag()=-27;
	cdouble product = dotprod(polynomial, root, nDim);
	cdouble result = l2norm(polynomial, nDim);
	printf("%f+%fi\n", result.real(), result.imag());
	/*printf("%f+%fi\n", product.real(), product.imag());
	matrix_print(polynomial, nDim);*/
	/*for (int i = 0; i < nDim; ++i)
		for (int j = 0; j < nDim; ++j)
		{
			m1[i + j * nDim].real() = nDim * j + i;
			m1[i + j * nDim].imag() = nDim * j + i;
			m2[i + j * nDim].real() = nDim * j + i;
			m2[i + j * nDim].imag() = nDim * j + i;
		}
	m3 = matmult(m1, m2, nDim);
	matrix_print(m1, nDim);
	matrix_print(m2, nDim);
	matrix_print(m3, nDim);*/

	//make_comp_mat(polynomial, comp, nDim);
	//matrix_print(comp, nDim);

	//root_find(polynomial, root, nDim, tolerance, upperbound);
	//for (int i = 0; i < nDim; ++i)
	//	cout << " " << root[i] << endl;
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
