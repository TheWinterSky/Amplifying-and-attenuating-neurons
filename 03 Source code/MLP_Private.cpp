#include "MLP.h"

/**********************************************************************

	Private member functions of the MLP class
		Allocate
		Deallocate
		Initialize_Parameters
		Initialize

**********************************************************************/

using namespace std;

/***********************************************
	Allocate arrays for the CPU
***********************************************/
bool MLP::Allocate()
{
	bool flag = true;

	//	Check if size is valid
	flag = flag && (n_in > 0);
	flag = flag && (n_out > 0);
	flag = flag && (n_neuron > 0);
	flag = flag && (n_depth > 0);
	if (!flag)
	{
		cout << "You have to specify valid neural network size first" << endl;
		cout << "\t" << "n_in    : " << n_in << endl;
		cout << "\t" << "n_out   : " << n_out << endl;
		cout << "\t" << "n_neuron: " << n_neuron << endl;
		cout << "\t" << "n_depth : " << n_depth << endl;
		return false;
	}

	//	Check if MLP is already constructed
	if (isConstructed)
	{
		cout << "Arrays are already allocated" << endl;
		return false;
	}

	//	Allocate CPU arrays
	w	= new Matrix<double>[n_depth + 1];
	dw	= new Matrix<double>[n_depth + 1];
	m	= new Matrix<double>[n_depth + 1];
	v	= new Matrix<double>[n_depth + 1];
	y	= new Matrix<double>[n_depth + 2];
	phi = new Matrix<double>[n_depth + 2];
	mu	= new Matrix<double>[n_depth + 2];
	activation	= new Matrix<int>[n_depth + 2];
	type		= new Matrix<int>[n_depth + 2];

	return true;
}



/***********************************************
	Deallocate arrays for the CPU and GPU
		If it is GPU mode, it will switch to the CPU mode
***********************************************/
bool MLP::Deallocate()
{
	int i = 0;

	//	GPU memory deallocation and initialization of the pointers
	//if (status == "GPU")
		//GPU_to_CPU();

	//	CPU memory deallocation
	if (isConstructed)
	{
		//	Destruct constructed matrixes
		for (i = 0; i < n_depth + 1; i++)
		{
			w	[i].Destruct();
			dw	[i].Destruct();
			m	[i].Destruct();
			v	[i].Destruct();
		}

		for (i = 0; i < n_depth + 2; i++)
		{
			y	[i].Destruct();
			phi	[i].Destruct();
			mu	[i].Destruct();
			activation	[i].Destruct();
			type		[i].Destruct();
		}
		Array_D	.Destruct();
		Array_I	.Destruct();
		cost	.Destruct();
		x_t		.Destruct();
		y_t		.Destruct();

		//	Deallocate allocated arrays
		delete[] w;
		delete[] dw;
		delete[] m;
		delete[] v;
		delete[] y;
		delete[] phi;
		delete[] mu;
		delete[] activation;
		delete[] type;
	}

	//	Set flag
	isConstructed = false;

	//	Initialize pointers
	Initialize_Pointers("CPU");

	return true;
}



/***********************************************
	Initialize MLP parameters
***********************************************/
bool MLP::Initialize_Parameters()
{
	//	Flag variables
	isConstructed = false;
	isTransposed = false;

	//	Network status
	status = "CPU";

	//	Network size
	n_in			= -1;
	n_out			= -1;
	n_neuron		= -1;
	n_depth			= -1;
	n_mini_batch	= 1;

	//	Network options
	i_method	= 2;				// Default: ADAM
	i_cost		= 2;				// Default: Linear-Square cost function
	i_regular	= 2;				// Default: L2 norm regularization

	//	Network parameters
	learning_rate = 0.001f;				// Default parameter: Learning
	lambda	= 0.001f;					// Default parameter: Regularization
	beta1	= 0.9f;						// Default parameter: ADAM
	beta2	= 0.99f;					// Default parameter: ADAM
	eps		= (float)1.e-8;				// Default parameter: ADAM

	//	CUDA grid and block size
	Grid_Act	= -1;
	Block_Act	= -1;
	Grid_EwM	= -1;
	Block_EwM	= -1;
	Grid_UpW	= -1;
	Block_UpW	= -1;

	return true;
}



/***********************************************
	Initialize pointers
***********************************************/
bool MLP::Initialize_Pointers(string mode_)
{
	//	Pointer cannot be initialized if MLP is constructed.
	if (isConstructed)
	{
		cout << "Pointers cannot be initialized when MLP is constructed" << endl;
		return false;
	}

	if (mode_ == "CPU")
	{
		w	= NULL;
		dw	= NULL;
		m	= NULL;
		v	= NULL;
		y	= NULL;
		phi = NULL;
		mu	= NULL;
		activation	= NULL;
		type		= NULL;
	}

	if (mode_ == "GPU")
	{
		Array_D_dev = NULL;
		Array_I_dev = NULL;

		w_dev	= NULL;
		dw_dev	= NULL;
		m_dev	= NULL;
		v_dev	= NULL;
		y_dev	= NULL;
		phi_dev = NULL;
		mu_dev	= NULL;
		cost_dev		= NULL;
		activation_dev	= NULL;
		type_dev		= NULL;
		i_mini_batch_dev = NULL;

		x_t_dev = NULL;
		y_t_dev = NULL;
	}

	return true;
}



/***********************************************
	Initialize neural network
		Weight initialization
		Neuron type initialization
		Activation function initialization
		Drop-out mask initialization
***********************************************/
bool MLP::Initialize()
{
	int i = 0, j = 0, k = 0;
	int tmp = 0;

	//	Matrix value initialization
	//		y: Set bias value: Except output layer
	for (k = 0; k < n_depth + 1; k++)
		y[k].Set_Value(0, 0, 1.0);

	//		phi: Default value is 1 and bias is 0
	for (k = 0; k < n_depth + 2; k++)
		phi[k].Set_Value(1.0);
	for (k = 0; k < n_depth + 1; k++)
		phi[k].Set_Value(0, 0, 0.0);

	//		Drop out mask: Default: 1.0
	for (k = 0; k < n_depth + 2; k++)
		mu[k].Set_Value(1.0);

	//		Activation function: Except input, output layer and bias neuron: Default PReLU
	for (k = 1; k < n_depth + 1; k++)
	{
		//	Overall activation function
		activation[k].Set_Value(3);
		//	Biase neuron: No activation function: This value will not be referred
		activation[k].Set_Value(0, 0, 0);
	}

	//		Type: Ordinary neuron is default
	for (k = 0; k < n_depth + 2; k++)
		type[k].Set_Value(1);

	//		Weight:	TMP now
	for (k = 0; k < n_depth; k++)
	{
		for (i = 1; i < w[k].m; i++)			//	To exclude bias
		{
			for (j = 0; j < w[k].n; j++)
			{
				tmp = (n_in + n_out + n_neuron * n_depth);

				if (rand() % 2)
					w[k].Set_Value(i, j, +(double)(rand() % tmp + 1) / sqrt((double)(tmp)) * 0.3);
				else
					w[k].Set_Value(i, j, -(double)(rand() % tmp + 1) / sqrt((double)(tmp)) * 0.3);

			}
		}
		w[k].Set_Value(0, 0, 1.0f);
	}
	k = n_depth;
	for (i = 0; i < w[k].m; i++)
	{
		for (j = 0; j < w[k].n; j++)
		{
			tmp = (n_in + n_out + n_neuron * n_depth);

			if (rand() % 2)
				w[k].Set_Value(i, j, +(double)(rand() % tmp + 1) / sqrt((double)(tmp)) * 1.0);
			else
				w[k].Set_Value(i, j, -(double)(rand() % tmp + 1) / sqrt((double)(tmp)) * 1.0);
		}
	}

	return true;
}