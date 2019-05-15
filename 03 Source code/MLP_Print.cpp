#include "MLP.h"

/**********************************************************************

	MLP public functions that prints

**********************************************************************/

using namespace std;

/***********************************************
	Print the information of the MLP
***********************************************/
bool MLP::Print_Information()
{
	cout << "----------------------------------------------" << endl;
	cout << "\t\tMLP information" << endl;
	cout << "----------------------------------------------" << endl;

	//	Check if MLP is constructed
	if (!isConstructed)
	{
		cout << "MLP is not constructed" << endl;
		cout << "----------------------------------------------" << endl;
		return false;
	}

	//	Constructed status and operation mode
	cout << "isConstructed: " << isConstructed << endl;
	cout << "isTransposed : " << isTransposed << endl;
	cout << "Operation mode: " << status << endl;
	cout << endl;

	/*
	cout << "Kernel function: Activate" << endl;
	cout << "Grid  size: " << Grid_Act  << '\t';
	cout << "Block size: " << Block_Act << endl;
	cout << "Kernel function: Elementwise_Multiply" << endl;
	cout << "Grid  size: " << Grid_EwM  << '\t';
	cout << "Block size: " << Block_EwM << endl;
	*/

	//	Neural network size
	cout << "Input layer (Including bias neuron)" << endl;
	cout << "\t" << y[0].m << " X " << y[0].n << endl;
	cout << "Hidden layers (Including bias neuron)" << endl;
	cout << "\t(" << y[1].m << " X " << y[1].n << ") X " << n_depth << endl;
	cout << "Output layer" << endl;
	cout << "\t" << y[n_depth + 1].m << " X " << y[n_depth + 1].n << endl;
	cout << endl;

	//	Neural network options
	cout << "Method" << endl << "\t";
	if (i_method == 0)	cout << "Stochastic Gradient Descent Method" << endl;
	if (i_method == 1)	cout << "Stochastic Gradient Descent Method with Momentum" << endl;
	if (i_method == 2)	cout << "ADaptive Moment Estimation(ADAM) method" << endl;

	cout << "Cost function" << endl << "\t";
	if (i_cost == 1)	cout << "L2-norm square" << endl;
	if (i_cost == 2)	cout << "Linear-Square" << endl;

	cout << "Regularization" << endl << "\t";
	if (i_regular == 0)			cout << "No regularization" << endl;
	if (i_regular == 1)			cout << "L1 regularization" << endl;
	if (i_regular == 2)			cout << "L2 regularization" << endl;

	cout << endl;

	//	Neural network parameters
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Lambda for regularization: " << lambda << endl;
	cout << "Parameters for moment estimation method" << endl;
	cout << "\tbeta1: " << beta1 << endl;
	cout << "\tbeta2: " << beta2 << endl;
	cout << "\teps  : " << eps << endl;
	cout << endl;

	//	Training data size
	cout << "Training data size" << endl;
	cout << "\tx: " << x_t.m << " X " << x_t.n << endl;
	cout << "\ty: " << y_t.m << " X " << y_t.n << endl;
	cout << endl;

	cout << "----------------------------------------------" << endl;
	cout << endl;

	return true;
}



/***********************************************
	Print the status of the MLP
		w, dw, m, v,
		y, phi, mu, activation, type
		cost
		Training data
***********************************************/
bool MLP::Print_Status(string input_)
{
	int i = 0;
	int type_ = 0;									// Type of input

	//	Set integer flag and size of the matrix array
	if ((input_ == "w") || (input_ == "dw") || (input_ == "m") || (input_ == "v"))
		type_ = 1;
	else if ((input_ == "y") || (input_ == "phi") || (input_ == "mu") || (input_ == "activation") || (input_ == "type"))
		type_ = 2;
	else if (input_ == "cost")
		type_ = 3;
	else if (input_ == "Training data")
		type_ = 4;
	else
	{
		cout << "Undefined property: " << input_ << endl;
		return false;
	}

	cout << "----------------------------------------------" << endl;
	cout << "\t\tMLP Status: " << input_ << endl;
	cout << "----------------------------------------------" << endl;
	//	Check if MLP is constructed
	if (!isConstructed)
	{
		cout << "MLP is not constructed" << endl;
		cout << "----------------------------------------------" << endl;
		return false;
	}

	//	Matrix
	if (type_ == 1)
	{
		for (i = 0; i < n_depth + 1; i++)
		{
			printf("Layer %2d\n", i);

			if (input_ == "w")				w	[i].Print_Matrix();
			if (input_ == "dw")				dw	[i].Print_Matrix();
			if (input_ == "m")				m	[i].Print_Matrix();
			if (input_ == "v")				v	[i].Print_Matrix();
			cout << endl;
		}
	}

	//	Vector
	if (type_ == 2)
	{
		for (i = 0; i < n_depth + 2; i++)
		{
			printf("Layer %2d: ", i);

			if (input_ == "y")				y	[i].Print_Matrix_T();
			if (input_ == "phi")			phi	[i].Print_Matrix_T();
			if (input_ == "mu")				mu	[i].Print_Matrix_T();
			if (input_ == "activation")		activation	[i].Print_Matrix_T();
			if (input_ == "type")			type		[i].Print_Matrix_T();
		}
	}

	//	Cost vector
	if (type_ == 3)
		cost.Print_Matrix_T();

	//	Training data
	if (type_ == 4)
	{
		cout << "- x -" << endl;
		x_t.Print_Matrix();
		cout << "- y -" << endl;
		y_t.Print_Matrix();
	}

	cout << "----------------------------------------------" << endl;

	return true;
}