#include "MLP.h"

/**********************************************************************

	MLP public functions that set option, parameter, value

**********************************************************************/

using namespace std;

/***********************************************/
/***         Public Member Functions         ***/
/***********************************************/

/***********************************************
	Set size of the neural network
		Work only when it is not constructed
***********************************************/
bool MLP::Set_Size(int n_in_, int n_out_, int n_neuron_, int n_depth_)
{
	//	Size cannot be changed if MLP is already constructed
	if (isConstructed)
	{
		cout << "Size of MLP cannot be changed if it is already constructed" << endl;
		return false;
	}

	//	Change size
	n_in		= n_in_ + 1;				// 1 is for bias neuron
	n_out		= n_out_;					// Output layer does not have bias neuron
	n_neuron	= n_neuron_;				// This input includes bias neuron
	n_depth		= n_depth_;

	return true;
}



/***********************************************
	Set the training data set
		Work only when it is CPU mode
***********************************************/
bool MLP::Set_Training_Data(Matrix<double> *x_t_, Matrix<double> *y_t_)
{
	bool flag = false;

	//	Check if MLP is constructed
	if (!isConstructed)
	{
		cout << "Input cannot be copied when MLP is not constructed" << endl;
		return false;
	}

	//	Check if it is CPU mode now
	if (status == "GPU")
	{
		cout << "Training data can be set in the CPU mode only" << endl;
		return false;
	}

	//	Check if dimensions match
	flag = (x_t_->m == (n_in - 1));
	flag = flag && (y_t_->m == n_out);
	flag = flag && (x_t_->n == y_t_->n);
	if (!flag)
	{
		cout << "Training data set dimension mis-match" << endl;
		cout << "\tn_in - 1: " << n_in - 1 << endl;
		cout << "\tn_out   : " << n_out << endl;
		cout << "\tInput  matrix x: <" << x_t_->m << " X " << x_t_->n << ">" << endl;
		cout << "\tOutput matrix y: <" << y_t_->m << " X " << y_t_->n << ">" << endl;
		return false;
	}

	//	Change current matrix: Destruct, construct and set component: Since size is changed
	x_t.Destruct();
	y_t.Destruct();
	x_t.Construct(x_t_->m, x_t_->n);
	y_t.Construct(y_t_->m, y_t_->n);
	x_t.Set_Matrix(x_t_);
	y_t.Set_Matrix(y_t_);

	return true;
}



/***********************************************
	Set y[0] value: CPU mode only
		Copy i_th training data into the y[0]
***********************************************/
bool MLP::Set_Input_y(int i_)
{
	int i = 0, index = 0;

	//	Set matrix cannot be used because of the bias neuron
	index = i_ * (n_in - 1) - 1;
	for (i = 1; i < n_in; i++)
		y[0].A[i] = x_t.A[index + i];

	return true;
}



/***********************************************
	Set y[0] value: CPU mode only
		Copy y_ into the y[0]
***********************************************/
bool MLP::Set_Input_y(Matrix<double> *y_)
{
	int i = 0;

	//	Set_Matrix cannot be used since y_ does not contain bias value
	for (i = 1; i < n_in; i++)
		y[0].A[i] = y_->A[i - 1];

	return true;
}



/***********************************************
	Set i_method
***********************************************/
void MLP::Set_Method(int i_method_)
{
	i_method = i_method_;
}



/***********************************************
	Set i_cost
***********************************************/
void MLP::Set_Cost(int i_cost_)
{
	i_cost = i_cost_;
}



/***********************************************
	Set i_regular
***********************************************/
void MLP::Set_Regularization(int i_regular_)
{
	i_regular = i_regular_;
}



/***********************************************
	Set i_regular and lambda
***********************************************/
void MLP::Set_Regularization(int i_regular_, double lambda_)
{
	i_regular = i_regular_;
	lambda = lambda_;
}



/***********************************************
	Set learning rate
***********************************************/
void MLP::Set_Learning_Rate(double learning_rate_)
{
	learning_rate = learning_rate_;
}



/***********************************************
	Set activation function of the whole neuron
***********************************************/
bool MLP::Set_Activation_F(int activation_)
{
	int i = 0, j = 0;

	//	Check if MLP is constructed
	if (!isConstructed)
		return false;

	//	Set the activation function of the whole neuron except input layer, output layer and bias neurons
	for (i = 1; i < n_depth + 1; i++)
		for (j = 1; j < y[i].m; j++)
			activation[i].A[j] = activation_;

	return true;
}



/***********************************************
	Set activation function of the specific neuron
***********************************************/
bool MLP::Set_Activation_F(int i_depth_, int i_neuron_, int activation_)
{
	bool flag = false;

	//	Check if MLP is constructed
	if (!isConstructed)
		return false;

	//	Check if depth and neuron is valid
	flag = (i_depth_	>= 0);
	flag = (i_depth_	< (n_depth + 2)) && flag;
	flag = (i_neuron_	>= 0) && flag;
	flag = (i_neuron_	< (y[i_depth_].m)) && flag;
	if (!flag)
		return false;

	//	Set the activation function
	activation[i_depth_].A[i_neuron_] = activation_;

	return true;
}



/***********************************************
	Set type of the specific neuron
***********************************************/
bool MLP::Set_Type(int i_depth_, int i_neuron_, int type_)
{
	bool flag = false;

	//	Check if MLP is constructed
	if (!isConstructed)
		return false;

	//	Check if depth and neuron is valid
	flag = (i_depth_	>= 0);
	flag = (i_depth_	< (n_depth + 2)) && flag;
	flag = (i_neuron_	>= 0) && flag;
	flag = (i_neuron_	< (y[i_depth_].m)) && flag;
	if (!flag)
		return false;

	//	Set the activation function
	type[i_depth_].A[i_neuron_] = type_;

	return true;
}



/***********************************************
	Set mini batch size: n_mini_batch
***********************************************/
void MLP::Set_Mini_Batch_Size(int n_mini_batch_)
{
	n_mini_batch = n_mini_batch_;
}