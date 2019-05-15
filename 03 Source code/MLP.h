#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include "Main.h"
#include "Matrix.h"						// Matrix type variables


/**********************************************************************

	MLP code: Version 2.0
		Continuous memory version
		Column major
		Real number type: double

		2019.02.21

**********************************************************************/


class MLP
{
private:
	//	Flag variables
	bool isConstructed;					// Construction flag
	bool isTransposed;					// Flag if w matrix is transposed

	//	Neural network dimension
	int n_in;							// Input size
	int n_out;							// Output size
	int n_neuron;						// Number of the neuron per hidden layer including bias neuron
	int n_depth;						// Number of the hidden layer
	int n_mini_batch;					// Size of mini batch

	//	Options
	int i_method;						// Solving method option
	int i_cost;							// Cost function option
	int i_regular;						// Regularization option

	//	Parameters
	double learning_rate;				// Learning rate
	double lambda;						// Parameter for regularization
	double beta1, beta2, eps;			// Parameters for the ADAM

	//	CUDA and cuBLAS parameters
	std::string status;					// Network status string: CPU or GPU
	//cublasHandle_t cublasHandle;		// cuBLAS handle
	int Grid_Act, Block_Act;			// Grid and Block size for kernel function: Activate
	int Grid_EwM, Block_EwM;			// Grid and Block size for kernel function: Elementwise_Multiply
	int Grid_UpW, Block_UpW;			// Grid and Block size for kernel function: Update_Weight

	//	Working array
	Matrix<double>  Array_D;			// Double  type array of the host
	Matrix<int>		Array_I;			// Integer type array of the host
	double* Array_D_dev;				// Double  type array of the device
	int*	Array_I_dev;				// Integer type array of the device
	//	Matrix array: required as many layers exist - CPU
	//		#: Connection
	Matrix<double> *w;					// Connectivity
	Matrix<double> *dw;					// dE/dw
	Matrix<double> *m;					// ADAM: Biased first moment estimation
	Matrix<double> *v;					// ADAM: Biased second raw moment estimation
	//		#: Neuron
	Matrix<double> *y;					// Output vectors of the layers
	Matrix<double> *phi;				// dE/dz
	Matrix<double> *mu;					// Drop-out mask
	Matrix<double> cost;				// E array and its sum
	Matrix<int> *activation;			// Activation function option: sigmoid, ReLU, ...
	Matrix<int> *type;					// Type of neuron: bias, ordinary, amplifier, attenuator

	//	Matrix array: required as many layers exist - GPU device
	//		#: Connection
	double **w_dev;						// Connectivity
	double **dw_dev;					// dE/dw
	double **m_dev;						// ADAM: Biased first moment estimation
	double **v_dev;						// ADAM: Biased second raw moment estimation
	//		#: Neuron
	double **y_dev;						// Output vectors of the layers
	double **phi_dev;					// dE/dz
	double **mu_dev;					// Drop-out mask
	double *cost_dev;					// Cost function array: components and total: n_out + 1
	int **activation_dev;				// Activation function option: sigmoid, ReLU, ...
	int **type_dev;						// Type of neuron: bias, ordinary, amplifier, attenuator
	int *i_mini_batch_dev;				// Index array of the training data mini batch

	//	Training data
	Matrix<double> x_t, y_t;			// MLP Training data - Host
	double *x_t_dev, *y_t_dev;			// MLP Training data - Device

	/***********************************************/
	/***         Private Member Functions        ***/
	/***********************************************/
	// Allocate arrays
	bool Allocate();
	// Deallocate arrays
	bool Deallocate();
	// Initialize parameters
	bool Initialize_Parameters();
	// Initialize pointers
	bool Initialize_Pointers(std::string mode_);
	// Initialize neural network
	bool Initialize();

protected:

public:
	/***********************************************/
	/***        Constructor and Desructor        ***/
	/***********************************************/
	//	Constructor
	MLP();
	//	Destructor
	~MLP();

	/***********************************************/
	/***         Public Member Functions         ***/
	/***********************************************/
	/*	Functions related to construction and destruction of the MLP	*/
	//	Construct MLP
	bool Construct();
	//	Construct MLP with given size
	bool Construct(int n_in_, int n_out_, int n_neuron_, int n_depth_);
	//	Destruct MLP
	bool Destruct();
	//	Transpose w matrixes
	bool Transpose();

	/*	Functions related to kernel	*/
	/*
	//	Change mode: From CPU to GPU
	//	Start CUDA and cuBLAS environment
	//	Allocate device arrays
	//	Copy host arrays to device array
	int CPU_to_GPU();
	//	Change mode: From GPU to CPU
	//	Finish CUDA and cuBLAS environment
	//	Copy device arrays to host array
	//	Deallocate device arrays
	int GPU_to_CPU();
	*/

	/*	Functions related to calculation	*/
	//	MLP feed forward operation in the CPU mode
	bool Forward_CPU();
	//	MLP error backward propagation operation to get dw in the CPU mode
	bool Backward_CPU(int i_);
	//	Train the MLP using training data index i_ in the CPU mode
	bool Train_CPU(int i_);
	//	Update weight of the MLP
	bool Update_w_CPU();
	//	Calculate the cost and phi of the output layer in the CPU mode
	//	i_: Training data index
	bool Calculate_Cost_CPU(int i_);
	//	Activate neuron ouput vectors in the CPU mode
	bool Activate_CPU(int i_);
	//	Initialize momentum array: m, v
	bool Initialize_Momentum();

	/*	Functions related to set value	*/
	//	Set the size of the MLP
	bool Set_Size(int n_in_, int n_out_, int n_neuron_, int n_depth_);
	//	Set training data set
	bool Set_Training_Data(Matrix<double> *x_t_, Matrix<double> *y_t_);
	//	Set the input layer y vector of the MLP: CPU mode only
	//	Copy i_th training data to y[0]
	bool Set_Input_y(int i_);
	//	Set the input layer y vector of the MLP: CPU mode only
	//	Copy y_ into the y[0] in the host
	bool Set_Input_y(Matrix<double> *y_);
	//	Set i_method: CPU and GPU mode
	void Set_Method(int i_method_);
	//	Set i_cost: CPU and GPU mode
	void Set_Cost(int i_cost_);
	//	Set i_regular: CPU and GPU mode
	void Set_Regularization(int i_regular_);
	//	Set i_regular and lambda: CPU and GPU mode
	void Set_Regularization(int i_regular_, double lambda_);
	//	Set learning rate: CPU and GPU mode
	void Set_Learning_Rate(double learning_rate_);
	//	Set activation function of the whole neuron: CPU mode only
	bool Set_Activation_F(int activation_);
	//	Set activation function of a specific neuron: CPU mode only
	bool Set_Activation_F(int i_depth_, int i_neuron_, int activation_);
	//	Set type of the whole neuron: CPU mode only
	bool Set_Type(int i_depth_, int i_neuron_, int type_);
	//	Set the mini batch size: CPU mode only
	void Set_Mini_Batch_Size(int n_mini_batch_);

	/*	Functions related to get value and print to the screen	*/
	//	Print the general information of the MLP
	bool Print_Information();
	//	Print the status of the matrixes, vectors
	//	Matrixes: w, dw, m, v, Training data
	//	Vectors : y, phi, mu, activation, type, cost
	bool Print_Status(std::string input_);
	//	Get output y: CPU mode only
	//	Copy output y to the matrix y_
	bool Get_Output_y(Matrix<double> *y_);
	//	Get total cost: cost.A[n_out]
	double Get_Cost();

	/*	Functions save and load the MLP	*/
	//	Save the MLP
	//	path_: Path including file name
	bool Save_MLP(std::string path_);
	//	Load the MLP
	//	path_: Path including file name
	bool Load_MLP(std::string path_);

};


#endif // !MLP_H_INCLUDED
