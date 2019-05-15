#include "MLP.h"

/**********************************************************************

	MLP public functions related to the calculation using CPU

**********************************************************************/

using namespace std;

/***********************************************
	Feed forward operation of the MLP in the CPU mode
***********************************************/
bool MLP::Forward_CPU()
{
	int i = 0;

	for (i = 0; i < n_depth + 1; i++)
	{
		Mult_Mat_Vect(&w[i], &y[i], &y[i + 1], isTransposed);
		Activate_CPU(i + 1);
	}

	return true;
}



/***********************************************
	Error back propagation of the MLP in the CPU mode
	i_: Training data index
***********************************************/
bool MLP::Backward_CPU(int i_)
{
	int i = 0, j = 0, k = 0;
	int m = 0, n = 0;
	int idx = 0;

	//	Calculate cost and phi of the last layer
	Calculate_Cost_CPU(i_);
	for (i = n_depth; i >= 0; i--)
	{
		//	Use dw as a temporary array
		Mult_Mat_Vect(&w[i], &phi[i + 1], &dw[i], !isTransposed);

		//	Calculate phi
		for (j = 0; j < phi[i].m; j++)
			phi[i].A[j] *= dw[i].A[j];
	}

	//	Calculate dw
	if (isTransposed)
	{
		for (k = 0; k < n_depth + 1; k++)
		{
			m = y	[k + 0].m;
			n = phi	[k + 1].m;

			for (j = 0; j < n; j++)
			{
				idx = j * m;
				for (i = 0; i < m; i++)
					dw[k].A[idx + i] = y[k].A[i] * phi[k + 1].A[j];
			}
		}
	}

	else
	{
		for (k = 0; k < n_depth + 1; k++)
		{
			m = phi	[k + 1].m;
			n = y	[k + 0].m;

			for (j = 0; j < n; j++)
			{
				idx = j * m;
				for (i = 0; i < m; i++)
					dw[k].A[idx + i] = phi[k + 1].A[i] * y[k].A[j];
			}
		}
	}

	return true;
}



/***********************************************
	Train MLP in the CPU mode
	i_: Training data index
***********************************************/
bool MLP::Train_CPU(int i_)
{
	if (!isConstructed)
		return false;

	Set_Input_y(i_);
	Forward_CPU();
	Backward_CPU(i_);
	Update_w_CPU();

	return true;
}



/***********************************************
	Update weights of the MLP
	i_method =	0: SGD
				1: SGD with momentum
				2: ADAM
***********************************************/
bool MLP::Update_w_CPU()
{
	int i = 0, j = 0;
	int n_1 = 0;

	n_1 = n_neuron * (n_in + (n_depth - 1) * n_neuron + n_out);

	//	SGD
	if (i_method == 0)
	{
		for (i = 0; i < n_1; i++)
			w[0].A[i] -= learning_rate * dw[0].A[i];
	}

	//	SGD with momentum
	if (i_method == 1)
	{
		for (i = 0; i < n_1; i++)
		{
			m[0].A[i] = beta1 * m[0].A[i] + learning_rate * dw[0].A[i];
			w[0].A[i] -= m[0].A[i];
		}
	}

	//	ADAM
	if (i_method == 2)
	{
		for (i = 0; i < n_1; i++)
		{
			m[0].A[i] = beta1 * m[0].A[i] + (1.0 - beta1) * dw[0].A[i];
			v[0].A[i] = beta2 * v[0].A[i] + (1.0 - beta2) * dw[0].A[i] * dw[0].A[i];
			w[0].A[i] -= learning_rate * m[0].A[i] / sqrt(v[0].A[i] + eps);
		}
	}

	//	Regularization
	if (i_regular == 1)
		cout << "L1 regularization is not implemented yet" << endl;
	if (i_regular == 2)
	{
		for (i = 0; i < n_1; i++)
			w[0].A[i] -= learning_rate * lambda * w[0].A[i];
	}

	return true;
}



/***********************************************
	Calculate the cost and phi of the output layer in the CPU mode
	i_: Index of the training data
	i_cost = 1: Error L2 norm square
			 2: Smooth |error|
***********************************************/
bool MLP::Calculate_Cost_CPU(int i_)
{
	int i = 0;
	int index = 0;
	double tmp = 0.0;

	//	Offset of the training data
	index = i_ * n_out;

	//	Error L2 norm square
	if (i_cost == 1)
	{
		for (i = 0; i < n_out; i++)
		{
			tmp = y[n_depth + 1].A[i] - y_t.A[index + i];
			cost.A[i] = pow(tmp, 2.0) / (double)n_out;
			phi[n_depth + 1].A[i] *= tmp / (double)n_out;
		}
	}
	else if (i_cost == 2)
	{
		for (i = 0; i < n_out; i++)
		{
			tmp = fabs(y[n_depth + 1].A[i] - y_t.A[index + i]) + 1.0;
			cost.A[i] = ((tmp - 1.0) + (1.0 / tmp - 1.0)) / (double)n_out;
			phi[n_depth + 1].A[i] *= (1.0 - 1.0 / (tmp * tmp)) / (double)n_out;

			if (y[n_depth + 1].A[i] < y_t.A[index + i])
				phi[n_depth + 1].A[i] *= -1.0;
		}
	}

	cost.A[n_out] = 0.0;
	for (i = 0; i < n_out; i++)
		cost.A[n_out] += cost.A[i];

	return true;
}



/***********************************************
	Activate neuron output vectors and calculate
	the activation function derivative in the CPU mode.
	i_: Index of the layer
	type =	0: None
			1: Sigmoid
			2: ReLU
			3: PReLU
			4: Softplus
			5: PSoftplus
			6: ELU
***********************************************/
bool MLP::Activate_CPU(int i_)
{
	int i = 0, m = 0;
	double expy = 0.0;

	m = y[i_].m;
	for (i = 0; i < m; i++)
	{
		switch (activation[i_].A[i])
		{
		case 0:
			phi[i_].A[i] = 1.0;
			break;

		case 1:
			//	Sigmoid
			//	To prevent too large exp(x), separate function
			if (y[i_].A[i] > 0.0)
			{
				expy = exp(-y[i_].A[i]);
				phi	[i_].A[i] = expy / ((expy + 1.0)*(expy + 1.0));
				y	[i_].A[i] = 1.0 / (expy + 1.0);
			}
			else
			{
				expy = exp(y[i_].A[i]);
				phi	[i_].A[i] = expy / ((expy + 1.0)*(expy + 1.0));
				y	[i_].A[i] = expy / (expy + 1.0);
			}
			break;

		case 2:
			//	ReLU
			if (y[i_].A[i] > 0.0)
			{
				phi[i_].A[i] = 1.0;
				//y	[i_].A[i] = y[i_].A[i];
			}
			else
			{
				phi	[i_].A[i] = 0.0;
				y	[i_].A[i] = 0.0;
			}
			break;

		case 3:
			//	PReLU
			if (y[i_].A[i] > 0.0)
			{
				phi	[i_].A[i] = 1.0;
				//y	[i_].A[i] = y[i_].A[i];
			}
			else
			{
				phi	[i_].A[i] = 0.3;
				y	[i_].A[i] *= 0.3;
			}
			break;

		case 4:
			//	Softplus
			//	To prevent too large exp(x), separate function
			if (y[i_].A[i] > 0.0)
			{
				expy = exp(-y[i_].A[i]);
				phi	[i_].A[i] = 1.0 / (expy + 1.0);
				y	[i_].A[i] += log(expy + 1.0);
			}
			else
			{
				expy = exp(y[i_].A[i]);
				phi	[i_].A[i] = expy / (expy + 1.0);
				y	[i_].A[i] = log(expy + 1.0);
			}
			break;

		case 5:
			//	PSoftplus
			//	To prevent too large exp(x), separate function
			if (y[i_].A[i] > 0.0)
			{
				expy = exp(-y[i_].A[i]);
				phi	[i_].A[i] = 0.3 + 0.7 / (expy + 1.0);
				y	[i_].A[i] += 0.7 * log(expy + 1.0);
			}
			else
			{
				expy = exp(y[i_].A[i]);
				phi	[i_].A[i] = 0.3 + 0.7 * expy / (expy + 1.0);
				y	[i_].A[i] = 0.3 * y[i_].A[i] + 0.7 * log(expy + 1.0);
			}
			break;

		case 6:
			//	ELU
			//	To prevent too large exp(x), separate function
			if (y[i_].A[i] > 0.0)
			{
				phi	[i_].A[i] = 1.0;
			}
			else
			{
				y	[i_].A[i] = exp(y[i_].A[i]) - 1.0;
				phi	[i_].A[i] = y[i_].A[i] + 1.0;
			}
			break;

		default:
			return false;
		}

		//	Amplifying and attenuating neuron
		//		Amplifying neuron
		if (type[i_].A[i] == 2)
		{
			phi	[i_].A[i] *= 2.0 * y[i_].A[i];
			y	[i_].A[i] *= y[i_].A[i];
		}
		//		Attenuating neuron
		else if (type[i_].A[i] == 3)
		{
			expy = y[i_].A[i] * y[i_].A[i];
			phi	[i_].A[i] *= (1.0 - expy) / ((expy + 1.0) * (expy + 1.0));
			y	[i_].A[i] /= (expy + 1.0);

			//	TEST: attenuating function has pole
			/*if(y[i_].A[i] > 0.0)
				expy = y[i_].A[i] + 1.0e-8;
			else
				expy = y[i_].A[i] - 1.0e-8;
			phi	[i_].A[i] /= - expy * expy;
			y	[i_].A[i] = 1.0 / expy;*/
		}
	}

	return true;
}



/***********************************************
	Initialize momentum array: m, v
***********************************************/
bool MLP::Initialize_Momentum()
{
	int i = 0;

	for (i = 0; i < n_depth + 1; i++)
	{
		m[i].Set_Value(0.0);
		v[i].Set_Value(0.0);
	}

	return true;
}