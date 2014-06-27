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
			printf(" %8.3f + %8.3fi ", mat[j + i * nDim].real(), mat[j + i * nDim].imag());
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
		tmp.real(vec1[i].real());
		tmp.imag(-vec1[i].imag());
		/*tmp = vec1[i];
		tmp.imag() = -vec1[i].imag();*/
		x += tmp * vec2[i];
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
				//j=2,k=1
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

void make_comp_mat(cdouble *polynomial, cdouble *companion, int nDim)
{
	for (int i = 0; i < nDim * nDim; ++i)
		companion[i] = 0;
	for (int i = 0; i < nDim; ++i)
		companion[i * nDim] = -polynomial[i + 1] / polynomial[0];
	for (int i = 0; i < nDim - 1; ++i)
		companion[i * nDim + i + 1] = 1;
}

void select_diag(cdouble *vector, cdouble *matrix, int nDim)
{
	for (int i = 0; i < nDim; ++i)
		vector[i] = matrix[i * nDim + i];
}

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
		for (int i = 0; i < nDim; ++i) //{
			u[i] = a[i + k * nDim];
//printf(" u[%i]=%2.1f+%2.1fi; a[%i]=%2.1f+%2.1fi\n", i, u[i].real(), u[i].imag(), i+k*nDim, a[i+k*nDim].real(), a[i+k*nDim].imag()); }
//printf("\n");
		for (int j = k - 1; j >= 0; --j)
			for (int i = 0; i < nDim; ++i) //{
				u[i] -= R[j + k * nDim] * Q[i + j * nDim];
//printf(" u[%i]=%2.1f+%2.1fi; R[%i]=%2.1f+%2.1fi; Q[%i]=%2.1f+%2.1fi\n", i, u[i].real(), u[i].imag(), j+k*nDim, R[j+k*nDim].real(), R[j+k*nDim].imag(), i+j*nDim, Q[i+j*nDim].real(), Q[i+j*nDim].imag()); }
//printf("\n");
		l2 = l2norm(u, nDim);
//printf(" l2=%2.1f+%2.1fi\n", l2.real(), l2.imag());
//printf("\n");
		for (int i = 0; i < nDim; ++i) //{
			Q[i + k * nDim] = u[i] / l2;
//printf(" Q[%i]=%2.1f+%2.1fi; u[%i]=%2.1f+%2.1fi; l2=%2.1f+%2.1fi\n", i+k*nDim, Q[i+k*nDim].real(), Q[i+k*nDim].imag(), i, u[i].real(), u[i].imag(), l2.real(), l2.imag()); }
//printf("\n");
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

void modified_gram_schmidt(cdouble *a, cdouble *Q, cdouble *R, int nDim)
{
	cdouble *u = new cdouble[nDim];
	cdouble *v = new cdouble[nDim];
	cdouble l2 = 0;
	for (int i = 0; i < nDim * nDim; ++i)
		Q[i] = R[i] = 0;
	for (int i = 0; i < nDim; ++i)
		u[i] = v[i] = 0;
}

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
matrix_print(a, nDim);
	for (int k = 0; k < upperbound; ++k)
	{
		gram_schmidt(a, Q, R, nDim);
		a = matmult(R, Q, nDim);
		nTol = 0;
		for (int j = 0; j < nDim; ++j)
			for (int i = 0; i < nDim; ++i) {
printf(" k=%i i=%i j=%i a[]=%f tol=%f nTol=%i \n",k,i,j,sqrt(a[i + j * nDim].real() * a[i + j * nDim].real() + a[i + j * nDim].imag() * a[i + j * nDim].imag()),tolerance,nTol);
				if (i > j && sqrt(a[i + j * nDim].real() * a[i + j * nDim].real() + 
					a[i + j * nDim].imag() * a[i + j * nDim].imag()) > tolerance) 
					++nTol; }
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
	cdouble *a = new cdouble[nDim * nDim];
	cdouble *Q = new cdouble[nDim * nDim];
	cdouble *R = new cdouble[nDim * nDim];

	for (int i = 0; i < nDim * nDim; ++i)
	{
		if (i==0 || i==1 || i==3 || i==5 || i==7 || i==8) {
			a[i].real() = 1;
			a[i].imag() = 1;
		}
		else {
			a[i].real() = 0;
			a[i].imag() = 0;
		}
	}

	/*for (int i = 0; i < nDim * nDim; ++i)
	{
		Q[i].real() = i;
		Q[i].imag() = 4 - i;
		R[i].real() = i + 1;
		R[i].imag() = i - 3;
	}*/

	//for (int i = 0; i < nDim * nDim; ++i)
	//	printf(" B[%i] = %f+%fi; C[%i] = %f+%fi\n", i, Q[i].real(), Q[i].imag(), i, R[i].real(), R[i].imag());

	//cdouble tmp = dotprod(Q, R, nDim * nDim);
	//printf("%f+%fi", tmp.real(), tmp.imag());

	//cdouble tmp = l2norm(a, nDim * nDim);
	//printf("%f+%fi", tmp.real(), tmp.imag());

	//Q = matmult(a, a, nDim);
	//matrix_print(a, nDim);
	//matrix_print(Q, nDim);

	gram_schmidt(a, Q, R, nDim);
	matrix_print(Q, nDim);
	matrix_print(R, nDim);

	/*soln: Q:	-0.5-0.5i	0.2887+0.2887i	0.4082+0.4082i
			-0.5-0.5i	-0.2887-0.2887i	-0.4082-0.4082i
			0+0i		0.5774+0.5774i	-0.4082-0.4082i

		R:	-2+0i		-1+0i		-1+0i
			0+0i		1.7321+0i	0.5774+0i
			0+0i		0+0i		-1.6330+0i	*/

/*
	int nDim = 4;
	double tolerance = 0.0001;
	double upperbound = 20;
	cdouble *polynomial = new cdouble[nDim];
	cdouble *root = new cdouble[nDim - 1];

	polynomial[0].real()=-1.2141;polynomial[0].imag()=-0.7697;
	polynomial[1].real()=-1.1135;polynomial[1].imag()=0.3714;
	polynomial[2].real()=-0.0068;polynomial[2].imag()=-0.2256;
	polynomial[3].real()=1.5326;polynomial[3].imag()=1.1174;

	root_find(polynomial, root, nDim, tolerance, upperbound);
	printf("\n");
	for (int i = 0; i < nDim - 1; ++i)
		printf(" %f + %fi", root[i].real(), root[i].imag());
	printf("\n");
*/

	/* soln:-0.8234 + 1.2525i
		-0.5960 - 0.7567i
		 0.9035 + 0.1371i */

	return 0;
}

/*
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
*/

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

