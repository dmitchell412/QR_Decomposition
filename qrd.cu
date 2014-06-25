#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cusp/complex.h>
using namespace std;

typedef cusp::complex<double> cdouble;

__device__
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

__device__
cdouble dotprod(cdouble *vec1, cdouble *vec2, int nDim)
{
	cdouble x = 0;
	cdouble tmp = 0;
	for (int i = 0; i < nDim; ++i)
	{
		tmp.real(vec1[i].real());
		tmp.imag(-vec1[i].imag());
		/*tmp = vec1[i];
		tmp.imag() = -vec1[i].imag();*/
		x += tmp * vec2[i];
	}
	return x;
}

__device__
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

__device__
cdouble l2norm(cdouble *vec, int nDim)
{
	cdouble x = 0;
	for (int i = 0; i < nDim; ++i)
		x += vec[i].real() * vec[i].real() + vec[i].imag() * vec[i].imag();
	x = sqrt(x);
	return x;
}

__device__
void transpose(cdouble *mat, int nDim)
{
	cdouble *x = new cdouble[nDim * nDim];
	for (int i = 0; i < nDim * nDim; ++i)
		x[i] = mat[i];
	for (int j = 0; j < nDim; ++j)
		for (int i = 0; i < nDim; ++i)
			mat[i + j * nDim] = x[j + i * nDim];
	delete[] x;
}

__device__
void make_comp_mat(cdouble *polynomial, cdouble *companion, int nDim)
{
	for (int i = 0; i < nDim * nDim; ++i)
		companion[i] = 0;
	for (int i = 0; i < nDim; ++i)
		companion[i * nDim] = -polynomial[i + 1] / polynomial[0];
	for (int i = 0; i < nDim - 1; ++i)
		companion[i * nDim + i + 1] = 1;
}

__device__
void select_diag(cdouble *vector, cdouble *matrix, int nDim)
{
	for (int i = 0; i < nDim; ++i)
		vector[i] = matrix[i * nDim + i];
}

__device__
void pixel_mat_select_1d(
	double *a_image_real,
	double *a_image_imag,
	cdouble *a,
	int nDim_matrix,
	int i_image)
{
	int offset_1d = i_image * nDim_matrix;
	for (int i = 0; i < nDim_matrix; ++i)
		a[i] = cdouble(a_image_real[i + offset_1d], a_image_imag[i + offset_1d]);
}

__device__
void pixel_mat_write_1d(
	double *a_image_real,
	double *a_image_imag,
	cdouble *a,
	int nDim_matrix,
	int i_image)
{
	int offset_1d = i_image * nDim_matrix;
	for (int i = 0; i < nDim_matrix; ++i)
	{
		a_image_real[i + offset_1d] = a[i].real();
		a_image_imag[i + offset_1d] = a[i].imag();
	}
}

__device__
void pixel_mat_select_2d(
	cdouble *a_image,
	cdouble *a,
	int nDim_matrix,
	int i_image)
{
	int offset_2d = i_image * nDim_matrix * nDim_matrix;
	for (int i = 0; i < nDim_matrix * nDim_matrix; ++i)
		a[i] = a_image[i + offset_2d];
}

__device__
void pixel_mat_write_2d(
	cdouble *Q_image,
	cdouble *R_image,
	cdouble *Q,
	cdouble *R,
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

__device__
void gram_schmidt(cdouble *a, cdouble *Q, cdouble *R, int nDim)
{
	cdouble *u = new cdouble[nDim];
	cdouble *v = new cdouble[nDim];
	cdouble l2 = 0;
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

__device__
void root_find(
	cdouble *polynomial,
	cdouble *root,
	int nDim_in,
	double tolerance,
	int upperbound)
{
	int nDim = nDim_in - 1;
	cdouble *a = new cdouble[nDim * nDim];
	cdouble *Q = new cdouble[nDim * nDim];
	cdouble *R = new cdouble[nDim * nDim];
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
				if (i > j && sqrt(a[i + j * nDim].real() * a[i + j * nDim].real() + 
					a[i + j * nDim].imag() * a[i + j * nDim].imag()) > tolerance) 
					++nTol; }
		if (nTol == 0) break;
	}

	select_diag(root, a, nDim);

	delete[] a;
	delete[] Q;
	delete[] R;
}

__global__
void QRDRoot(
	double *polynomial_image_real,
	double *polynomial_image_imag,
	double *root_image_real,
	double *root_image_imag,
	double const tolerance,
	int const upperbound,
	int const nDim_image,
	int const nDim_matrix)
{
	int i_image = blockDim.x * blockIdx.x + threadIdx.x;
	if (i_image > nDim_image * nDim_image) return;

	cdouble *polynomial = new cdouble[(nDim_matrix) * (nDim_matrix)];
	cdouble *root = new cdouble [(nDim_matrix - 1) * (nDim_matrix - 1)];

	pixel_mat_select_1d(polynomial_image_real, polynomial_image_imag, polynomial,
		nDim_matrix, i_image);
	root_find(polynomial, root, nDim_matrix, tolerance, upperbound);
	pixel_mat_write_1d(root_image_real, root_image_imag, root, nDim_matrix - 1, i_image);

	delete[] polynomial;
	delete[] root;
}

/*
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
	
	pixel_mat_select_2d(a_image, a, nDim_matrix, i_image);

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

	pixel_mat_write_2d(Q_image, R_image, Q, R, nDim_matrix, i_image);
}
*/

