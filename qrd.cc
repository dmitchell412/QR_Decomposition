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

cdouble* proj(cdouble *u, cdouble *v, int nDim)
{
	cdouble *x = new cdouble[nDim];
	for (int i = 0; i < nDim; ++i)
		x[i] = 0;
	cdouble dp = dotprod(u, v, nDim) / dotprod(u, u, nDim);
	for (int i = 0; i < nDim; ++i)
		x[i] = dp * u[i];
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

void make_comp_mat(cdouble *polynomial, cdouble *companion, int nDim)
{
	for (int i = 0; i < nDim * nDim; ++i)
		companion[i] = 0;
	for (int i = 0; i < nDim; ++i)
		companion[i * nDim] = -polynomial[i + 1] / polynomial[0];
		//companion[i + nDim * (nDim - 1)] = -polynomial[i + 1] / polynomial[0];
	for (int i = 0; i < nDim - 1; ++i)
		companion[i * nDim + i + 1] = 1;
}

void select_diag(cdouble *vector, cdouble *matrix, int nDim)
{
	for (int i = 0; i < nDim; ++i)
		vector[i] = matrix[i * nDim + i];
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

void modified_gram_schmidt(cdouble *a, cdouble *Q, cdouble *R, int nDim)
{
	cdouble *u = new cdouble[nDim];
	cdouble *v = new cdouble[nDim];
	cdouble prj = 0;
	cdouble l2 = 0;
	for (int i = 0; i < nDim * nDim; ++i)
		Q[i] = R[i] = 0;
	for (int i = 0; i < nDim; ++i)
		u[i] = v[i] = 0;

	for (int k = 0; k < nDim; ++k)
	{
		for (int i = 0; i < nDim; ++i)
			u[i] = a[i + k * nDim];
		for (int j = 0; j < k; ++j)
		{
			for (int i = 0; i < nDim; ++i)
				v[i] = Q[i + j * nDim];
			prj = dotprod(v, u, nDim);
			for (int i = 0; i < nDim; ++i)
				u[i] -= prj * v[i];
		}
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

void root_find(cdouble *polynomial, cdouble *root, int nDim_in, double tolerance, int upperbound)
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
		modified_gram_schmidt(a, Q, R, nDim);
		a = matmult(R, Q, nDim);
		nTol = 0;
		for (int j = 0; j < nDim; ++j)
			for (int i = 0; i < nDim; ++i) {
printf(" k=%i i=%i j=%i a[%i,%i]=%f+%fi |a|=%f tol=%f nTol=%i \n",k,i,j,i,j,a[i+j*nDim].real(),a[i+j*nDim].imag(),sqrt(a[i + j * nDim].real() * a[i + j * nDim].real() + a[i + j * nDim].imag() * a[i + j * nDim].imag()),tolerance,nTol);
				//if (i > j && fabs(a[i + j * nDim]) > tolerance) ++nTol; }
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
	//int nDim = 3;
	int nDim = 2;
	int nTol;
	double tolerance = 0.001;
	double upperbound = 1000;
	cdouble *a = new cdouble[nDim * nDim];
	cdouble *Q = new cdouble[nDim * nDim];
	cdouble *R = new cdouble[nDim * nDim];
	cdouble *soln = new cdouble[nDim];

	//a[0]=2;a[1]=1;a[2]=1;a[3]=1;a[4]=2;a[5]=1;a[6]=1;a[7]=1;a[8]=2;
	a[0]=3;a[1]=4;a[2]=-2;a[3]=-1;
matrix_print(a, nDim);
	modified_gram_schmidt(a, Q, R, nDim);
matrix_print(Q, nDim);
matrix_print(R, nDim);
a = matmult(Q, R, nDim);
matrix_print(a, nDim);
/*	for (int k = 0; k < upperbound; ++k)
	{
		modified_gram_schmidt(a, Q, R, nDim);
		a = matmult(R, Q, nDim);
		nTol = 0;
		for (int j = 0; j < nDim; ++j)
			for (int i = 0; i < nDim; ++i) {
printf(" k=%i i=%i j=%i a[%i,%i]=%f+%fi |a|=%f tol=%f nTol=%i \n",k,i,j,i,j,a[i+j*nDim].real(),a[i+j*nDim].imag(),sqrt(a[i + j * nDim].real() * a[i + j * nDim].real() + a[i + j * nDim].imag() * a[i + j * nDim].imag()),tolerance,nTol);
				//if (i > j && fabs(a[i + j * nDim]) > tolerance) ++nTol; }
				if (i > j && sqrt(a[i + j * nDim].real() * a[i + j * nDim].real() + 
					a[i + j * nDim].imag() * a[i + j * nDim].imag()) > tolerance) 
					++nTol; }
		if (nTol == 0) break;
	}
matrix_print(a, nDim);
	select_diag(soln, a, nDim);
matrix_print(soln, nDim);
*/
delete[] a;
delete[] Q;
delete[] R;

/*
	int nDim = 4;
	double tolerance = 0.0001;
	double upperbound = 10000;
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
	/*soln:	-0.8234 + 1.2525i
		-0.5960 - 0.7567i
		 0.9035 + 0.1371i */

	/*int nDim = 4;
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
	printf("%f+%fi\n", result.real(), result.imag());*/
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
	cdouble a[9];
	cdouble Q[9];
	cdouble R[9];

	for (int i = 0; i < 9; ++i)
	{
		if (i==0||i==1||i==3||i==5||i==7||i==8)
		{
			a[i].real() = 1;
			a[i].imag() = 1;
		}
		else
		{
			a[i].real() = 0;
			a[i].imag() = 0;
		}
		Q[i].real() = Q[i].imag() = R[i].real() = R[i].imag() = 0;
	}

	matrix_print(a, nDim);
	matrix_print(Q, nDim);
	matrix_print(R, nDim);
	gram_schmidt(a, Q, R, nDim);
	matrix_print(Q, nDim);
	matrix_print(R, nDim);

	for (int i = 0; i < 9; ++i)
	{
		if (i==0||i==1||i==3||i==5||i==7||i==8)
		{
			a[i].real() = 1;
			a[i].imag() = 1;
		}
		else
		{
			a[i].real() = 0;
			a[i].imag() = 0;
		}
		Q[i].real() = Q[i].imag() = R[i].real() = R[i].imag() = 0;
	}

	modified_gram_schmidt(a, Q, R, nDim);
	matrix_print(Q, nDim);
	matrix_print(R, nDim);
*/
}
