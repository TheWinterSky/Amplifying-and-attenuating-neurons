#include "MLP.h"

/**********************************************************************

	Functions that is related to constructing and destructing MLP
		Constructor, Destructor
		Allocate, Deallocate
		Initialize_Parameters, Initialize
		Construct, Destruct

**********************************************************************/

using namespace std;

/***********************************************/
/***        Constructor and Desructor        ***/
/***********************************************/
MLP::MLP()
{
	//	Parameter initialization
	Initialize_Parameters();

	//	Pointer initialization
	Initialize_Pointers("CPU");
	Initialize_Pointers("GPU");
}



MLP::~MLP()
{
	Destruct();
}



/***********************************************/
/***         Public Member Functions         ***/
/***********************************************/

/***********************************************
	Construct neural network
***********************************************/
bool MLP::Construct()
{
	int i = 0;
	int n_1 = 0, n_2;

	//	Allocate arrays: If there was problem during allocation
	if (!Allocate())
	{
		cout << "Failed to construct" << endl;
		return false;
	}

	//	Set working array matrix
	n_1 = n_neuron * (4 * (n_in + n_out - n_neuron) + n_depth * (3 + 4 * n_neuron)) + 3 * n_in + 4 * n_out + 1;
	n_2 = 2 * (n_in + n_depth * n_neuron + n_out);
	Array_D.Construct(n_1, 1);
	Array_I.Construct(n_2, 1);

	//	Set matrix size
	//		Input layer
	w	[0].Set_Size(n_neuron, n_in);
	dw	[0].Set_Size(n_neuron, n_in);
	m	[0].Set_Size(n_neuron, n_in);
	v	[0].Set_Size(n_neuron, n_in);
	y	[0].Set_Size(n_in, 1);
	phi	[0].Set_Size(n_in, 1);
	mu	[0].Set_Size(n_in, 1);
	activation	[0].Set_Size(n_in, 1);
	type		[0].Set_Size(n_in, 1);

	//		Hidden layer
	for (i = 1; i < n_depth; i++)
	{
		w	[i].Set_Size(n_neuron, n_neuron);
		dw	[i].Set_Size(n_neuron, n_neuron);
		m	[i].Set_Size(n_neuron, n_neuron);
		v	[i].Set_Size(n_neuron, n_neuron);
	}
	for (i = 1; i < n_depth + 1; i++)
	{
		y	[i].Set_Size(n_neuron, 1);
		phi	[i].Set_Size(n_neuron, 1);
		mu	[i].Set_Size(n_neuron, 1);
		activation	[i].Set_Size(n_neuron, 1);
		type		[i].Set_Size(n_neuron, 1);
	}

	//		Output layer
	w	[n_depth + 0].Set_Size(n_out, n_neuron);
	dw	[n_depth + 0].Set_Size(n_out, n_neuron);
	m	[n_depth + 0].Set_Size(n_out, n_neuron);
	v	[n_depth + 0].Set_Size(n_out, n_neuron);
	y	[n_depth + 1].Set_Size(n_out, 1);
	phi	[n_depth + 1].Set_Size(n_out, 1);
	mu	[n_depth + 1].Set_Size(n_out, 1);
	cost.Set_Size(n_out + 1, 1);
	activation	[n_depth + 1].Set_Size(n_out, 1);
	type		[n_depth + 1].Set_Size(n_out, 1);

	//	Assign address of the matrix
	//		Input layer
	n_1 = n_neuron * (n_in + (n_depth - 1) * n_neuron + n_out);
	n_2 = n_in + n_depth * n_neuron + n_out;
	w	[0].Construct(Array_D.A + 0 * n_1);
	dw	[0].Construct(Array_D.A + 1 * n_1);
	m	[0].Construct(Array_D.A + 2 * n_1);
	v	[0].Construct(Array_D.A + 3 * n_1);
	y	[0].Construct(Array_D.A + 4 * n_1);
	phi	[0].Construct(y[0].A + 1 * n_2);
	mu	[0].Construct(y[0].A + 2 * n_2);
	cost.Construct(y[0].A + 3 * n_2);

	n_1 = n_in + n_depth * n_neuron + n_out;
	activation	[0].Construct(Array_I.A + 0 * n_1);
	type		[0].Construct(Array_I.A + 1 * n_1);

	//		Rest layers
	for (i = 1; i < n_depth + 1; i++)
	{
		n_1 = w[i - 1].m * w[i - 1].n;
		w	[i].Construct(w[i - 1].A + n_1);
		dw	[i].Construct(dw[i - 1].A + n_1);
		m	[i].Construct(m[i - 1].A + n_1);
		v	[i].Construct(v[i - 1].A + n_1);
	}
	for (i = 1; i < n_depth + 2; i++)
	{
		n_1 = y[i - 1].m * y[i - 1].n;
		y	[i].Construct(y[i - 1].A + n_1);
		phi	[i].Construct(phi[i - 1].A + n_1);
		mu	[i].Construct(mu[i - 1].A + n_1);
		activation	[i].Construct(activation[i - 1].A + n_1);
		type		[i].Construct(type[i - 1].A + n_1);
	}

	//	Set dummy training data matrix
	x_t.Construct(n_in - 1, 3);
	y_t.Construct(n_out, 3);

	//	Initialize network
	Initialize();

	//	Set flag
	isConstructed = true;

	return true;
}



/***********************************************
	Construct neural network for given size
		n_in_ : Size of input  layer excluding bias neuron
		n_out_: Size of output layer
		n_neuron_: Number of neuron per hidden layer
		n_depth_ : Number of hidden layer
***********************************************/
bool MLP::Construct(int n_in_, int n_out_, int n_neuron_, int n_depth_)
{
	if (Set_Size(n_in_, n_out_, n_neuron_, n_depth_))
		Construct();

	return true;
}



/***********************************************
	Destruct neural network
		Deallocate and initialize parameters
***********************************************/
bool MLP::Destruct()
{
	//	Deallocate CPU and GPU memory
	Deallocate();

	//	Initialize parameters
	Initialize_Parameters();

	//	Announce that MLP is destructed
	cout << "MLP is Destructed" << endl;

	return true;
}



/***********************************************
	Transpose the w matrixes of the MLP
***********************************************/
bool MLP::Transpose()
{
	int i = 0;

	if (!isConstructed)
		return false;

	isTransposed = !isTransposed;
	for (i = 0; i < n_depth + 1; i++)
	{
		w	[i].Transpose();
		dw	[i].Transpose();
	}

	return true;
}