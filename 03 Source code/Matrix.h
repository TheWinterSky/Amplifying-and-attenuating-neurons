#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include "Main.h"

//////////////////////////////////////////////////////////////////////////////
template <typename T>
class Matrix
{
private:
	bool isConstructed;								// Flag for construction
	bool isAllocated;								// Flag for allocation

public:
	Matrix();										// Constructor
	~Matrix();										// Destructor

	T *A;											// Matrix: In one dimension. Column major
	int m, n;										// Size of the matrix

	//	Construct matrix
	bool Construct();
	bool Construct(int m_, int n_);
	bool Construct(T *A_);
	bool Construct(T *A_, int m_, int n_);
	//	Destruct matrix
	bool Destruct();

	//	Transpose matrix
	bool Transpose();

	//	Save matrix
	bool Save_Matrix(std::string path_);
	//	Load matrix
	bool Load_Matrix(std::string path_);

	//	Set the size of the matrix
	bool Set_Size(int m_, int n_);
	//	Set (or copy) matrix component from Matrix B
	bool Set_Matrix(Matrix *B);
	//	Set the entire components of the matrix
	bool Set_Value(T x_);
	//	Set the component of the matrix
	bool Set_Value(int i_, int j_, T x_);
	//	Get the value of constructed
	bool Get_isConstructed();

	//	Print the status of the matrix
	bool Print_Status();
	//	Print matrix
	bool Print_Matrix();
	//	Print transposed matrix
	bool Print_Matrix_T();
};


/***********************************************/
/***        Constructor and Desructor        ***/
/***********************************************/
template <typename T>
Matrix<T>::Matrix()
{
	isConstructed = false;
	isAllocated = false;

	A = NULL;
	m = 0;
	n = 0;
}



template <typename T>
Matrix<T>::~Matrix()
{
	Destruct();
}



/***********************************************/
/***         Public Member Functions         ***/
/***********************************************/

/***********************************************
	Construct matrix from pre-determined m and n
***********************************************/
template <typename T>
bool Matrix<T>::Construct()
{
	int i = 0;

	//	Check if it is already constructed
	if (isConstructed)
	{
		std::cout << "Matrix is already constructed" << std::endl;
		return false;
	}

	//	Check if size if valid
	if ((m <= 0) && (n <= 0))
	{
		std::cout << "Matrix size is invalid" << std::endl;
		std::cout << "\tm: " << m << std::endl;
		std::cout << "\tn: " << n << std::endl;
		return false;
	}

	//	Allocate and initialize
	A = new T[m * n];
	for (i = 0; i < m*n; i++)
		A[i] = (T)0.0;

	//	Check once again and set 'constructed' and 'allocated'
	if (A != NULL)
	{
		isConstructed = true;
		isAllocated = true;
		return true;
	}
	else
		return false;
}



/***********************************************
	Construct matrix from given m_ and n_
		m_: Size of column
		n_: Size of row
***********************************************/
template <typename T>
bool Matrix<T>::Construct(int m_, int n_)
{
	//	If it is not constructed to that size can be changed, construct it
	if (Set_Size(m_, n_))
		return Construct();
	else
		return false;
}



/***********************************************
	Construct matrix having array address A_
		A_: Array address
		Matrix size should be pre-determined
***********************************************/
template <typename T>
bool Matrix<T>::Construct(T *A_)
{
	return Construct(A_, m, n);
}



/***********************************************
	Construct matrix having array address A_
		A_: Array address
		m_: Size of column
		n_: Size of row
***********************************************/
template <typename T>
bool Matrix<T>::Construct(T *A_, int m_, int n_)
{
	if (isConstructed)
	{
		std::cout << "Matrix is already constructed" << std::endl;
		return false;
	}

	//	Check if A_ is valid
	if (A_ != NULL)
	{
		if (Set_Size(m_, n_))
		{
			A = A_;
			isConstructed = true;
			return true;
		}

		//	Invalid matrix size
		else
			return false;
	}
	else
	{
		std::cout << "A_ is NULL" << std::endl;
		return false;
	}
}



/***********************************************
	Destruct matrix
***********************************************/
template <typename T>
bool Matrix<T>::Destruct()
{
	if (isAllocated)
	{
		delete[] A;
		A = NULL;
		m = 0;
		n = 0;
		isConstructed = false;
		isAllocated = false;
		return true;
	}
	else
	{
		A = NULL;
		m = 0;
		n = 0;
		isConstructed = false;
		isAllocated = false;
		return true;
	}
}



/***********************************************
	Transpose the matrix
***********************************************/
template <typename T>
bool Matrix<T>::Transpose()
{
	int i = 0, j = 0;
	int idx_1 = 0, idx_2 = 0;
	T tmp;

	if (!isConstructed)
		return false;

	//	Transpose matrix
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < m; i++)
		{
			idx_1 = i + m * j;
			idx_2 = (idx_1 / n) + m * (idx_1 % n);

			//	if idx_1 > idx_2. find right idx_2
			while (idx_1 > idx_2)
				idx_2 = (idx_2 / n) + m * (idx_2 % n);

			//	Swap component
			tmp = A[idx_1];
			A[idx_1] = A[idx_2];
			A[idx_2] = tmp;
		}
	}
	Set_Size(n, m);

	return true;
}



/***********************************************
	Save the matrix
	path_: File path including file name
***********************************************/
template <typename T>
bool Matrix<T>::Save_Matrix(std::string path_)
{
	int i = 0;
	std::ofstream file_out;

	if (!isConstructed)
		return false;

	//	Open file
	file_out.open(path_, std::ios::out | std::ios::binary);
	//	In the case of file open failure
	if (!file_out.is_open())
		return false;

	//	Write size of the matrix
	file_out.write((char*)&m, sizeof(m));
	file_out.write((char*)&n, sizeof(n));

	//	Write components of the matrix
	for (i = 0; i < m*n; i++)
		file_out.write((char*)(A + i), sizeof(A[i]));

	//	Close file
	file_out.close();

	return true;
}



/***********************************************
	Load the matrix
	path_: File path including file name
***********************************************/
template <typename T>
bool Matrix<T>::Load_Matrix(std::string path_)
{
	int i = 0;
	int m_ = 0, n_ = 0;
	std::ifstream file_in;

	//	Open file
	file_in.open(path_, std::ios::in | std::ios::binary);
	//	In the case of file open failure
	if (!file_in.is_open())
		return false;

	//	Read size of the matrix
	file_in.read((char*)&m_, sizeof(m_));
	file_in.read((char*)&n_, sizeof(n_));

	//	If matrix was already constructed, check the size and set it
	if (isConstructed)
	{
		if (!Set_Size(m_, n_))
			return false;
	}
	else
		Construct(m_, n_);

	//	Read component of the matrix
	for (i = 0; i < m*n; i++)
		file_in.read((char*)(A + i), sizeof(A[i]));

	//	Close file
	file_in.close();

	return true;
}



/***********************************************
	Set size of the matrix
***********************************************/
template <typename T>
bool Matrix<T>::Set_Size(int m_, int n_)
{
	//	Except above exception, size cannot be changed if it is already constructed
	if (isConstructed)
	{
		//	For exception, size can be changed even matrix is contructed when m * n = m_ * n_
		if ((m*n) == (m_*n_))
		{
			m = m_;
			n = n_;
			return true;
		}

		std::cout << "Size of the constructed matrix cannot be changed" << std::endl;
		return false;
	}

	//	Set the size of the matrix
	if ((m_ > 0) && (n_ > 0))
	{
		m = m_;
		n = n_;
		return true;
	}
	else
		return false;
}



/***********************************************
	Set (or copy) matrix component from Matrix B
***********************************************/
template <typename T>
bool Matrix<T>::Set_Matrix(Matrix *B)
{
	int i = 0;
	bool flag = false;

	if (!isConstructed)
		return false;

	//	Check dimension
	flag = (m == B->m);
	flag = (n == B->n) && flag;
	if (!flag)
		return false;
	for (i = 0; i < m*n; i++)
		A[i] = B->A[i];

	return true;
}



/***********************************************
	Set the entire components of the matrix
***********************************************/
template <typename T>
bool Matrix<T>::Set_Value(T x_)
{
	int i = 0;

	if (!isConstructed)
		return false;

	for (i = 0; i < m*n; i++)
		A[i] = x_;

	return true;
}



/***********************************************
	Set the component of the matrix
***********************************************/
template <typename T>
bool Matrix<T>::Set_Value(int i_, int j_, T x_)
{
	int index = 0;

	if (!isConstructed)
		return false;

	//	Set index
	index = m * j_ + i_;

	//	Check if index is valid
	if ((index < 0) || (index >= m * n))
	{
		std::cout << "Matrix set value: Out of index" << std::endl;
		std::cout << "Index: " << index << std::endl;
		std::cout << "Allowable index: 0 ~ " << m * n << std::endl;
		return false;
	}

	A[index] = x_;

	return true;
}



/***********************************************
	Get the constructed value
***********************************************/
template <typename T>
bool Matrix<T>::Get_isConstructed()
{
	return isConstructed;
}



/***********************************************
	Print the status of the matrix
***********************************************/
template <typename T>
bool Matrix<T>::Print_Status()
{
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "\t\tMatrix Status" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "\tisConstructed: " << isConstructed << std::endl;
	std::cout << "\tisAllocated  : " << isAllocated << std::endl;
	std::cout << std::endl;
	std::cout << "\tAddress: " << A << std::endl;
	std::cout << "\tm: " << m << std::endl;
	std::cout << "\tn: " << n << std::endl;
	std::cout << "----------------------------------------------" << std::endl;

	return true;
}



/***********************************************
	Print the matrix
***********************************************/
template <typename T>
bool Matrix<T>::Print_Matrix()
{
	int i = 0, j = 0;
	int index = 0;

	//	Check if matrix is constructed
	if (!isConstructed)
	{
		std::cout << "Matrix is not constructed yet" << std::endl;
		return false;
	}

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			index = m * j + i;
			std::cout.width(7);
			std::cout << A[index] << "   ";
		}
		std::cout << std::endl;
	}
	return true;
}



/***********************************************
	Print the transposed matrix
***********************************************/
template <typename T>
bool Matrix<T>::Print_Matrix_T()
{
	int i = 0, j = 0;
	int index = 0;

	//	Check if matrix is constructed
	if (!isConstructed)
	{
		std::cout << "Matrix is not constructed yet" << std::endl;
		return false;
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			index = m * i + j;
			std::cout.width(7);
			std::cout << A[index] << "   ";
		}
		std::cout << std::endl;
	}
	return true;
}
//////////////////////////////////////////////////////////////////////////////



/***********************************************
	Add the vector: C = A + B
***********************************************/
template <typename T>
void Add_Vector(Matrix<T> *A_, Matrix<T> *B_, Matrix<T> *C_)
{
	int i = 0, n = 0;

	//	Calculation part
	n = A_->m * A_->n;
	for (i = 0; i < n; i++)
		C_->A[i] = A_->A[i] + B_->A[i];
}

/***********************************************
	Add the vector: C = A + lambda * B
***********************************************/
template <typename T>
void Add_Vector(Matrix<T> *A_, T lambda_, Matrix<T> *B_, Matrix<T> *C_)
{
	int i = 0, n = 0;

	n = A_->m * A_->n;
	for (i = 0; i < n; i++)
		C_->A[i] = A_->A[i] + lambda_ * B_->A[i];
}



/***********************************************
	Multiply Matrix and Vector: Y = AX
	Transpose_ indicates if the matrix A is transposed
***********************************************/
template <typename T>
void Mult_Mat_Vect(Matrix<T> *A_, Matrix<T> *X_, Matrix<T> *Y_, bool Transpose_)
{
	int idx = 0;
	int i = 0, j = 0;
	int m = 0, n = 0;
	T sum = (T)0;

	//	Calculate ATTX
	if (Transpose_)
	{
		m = A_->n;
		n = A_->m;

		for (i = 0; i < m; i++)
		{
			idx = i * n;
			sum = (T)0;
			for (j = 0; j < n; j++)
				sum += X_->A[j] * A_->A[idx + j];
			Y_->A[i] = sum;
		}
	}
	//	Calculate AX
	else
	{
		m = A_->m;
		n = A_->n;

		for (i = 0; i < m; i++)
		{
			sum = (T)0;
			for (j = 0; j < n; j++)
				sum += A_->A[i + m * j] * X_->A[j];
			Y_->A[i] = sum;
		}
	}
}



/***********************************************
	Multiply Matrix and Matrix: C = AB
	Transpose_ indicates if the matrix A is transposed
***********************************************/
template <typename T>
void Mult_Mat_Mat(Matrix<T> *A_, Matrix<T> *B_, Matrix<T> *C_, bool Transpose_)
{
	int idx_1 = 0, idx_2 = 0, idx_3 = 0;
	int i = 0, j = 0, k = 0;
	int m = 0, n = 0, p = 0;
	T sum = (T)0;

	//	Calculate ATTB
	if (Transpose_)
	{
		m = A_->n;					// A.m = C.m
		n = B_->n;					// B.n = C.n
		p = B_->m;					// A.n = B.m: Size of inner product

		for (j = 0; j < n; j++)
		{
			idx_1 = m * j;
			idx_3 = p * j;
			for (i = 0; i < m; i++)
			{
				idx_2 = p * i;
				sum = (T)0;
				for (k = 0; k < p; k++)
					sum += A_->A[idx_2 + k] * B_->A[idx_3 + k];
				C_->A[idx_1 + i] = sum;
			}
		}
	}
	//	Calculate AB
	else
	{
		m = A_->m;					// A.m = C.m
		n = B_->n;					// B.n = C.n
		p = B_->m;					// A.n = B.m: Size of inner product

		for (j = 0; j < n; j++)
		{
			idx_1 = m * j;
			idx_3 = p * j;
			for (i = 0; i < m; i++)
			{
				sum = (T)0;
				for (k = 0; k < p; k++)
					sum += A_->A[i + k * m] * B_->A[idx_3 + k];
				C_->A[idx_1 + i] = sum;
			}
		}
	}
}



#endif // !MATRIX_H_INCLUDED