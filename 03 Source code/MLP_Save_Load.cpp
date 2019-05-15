#include "MLP.h"

/**********************************************************************

	MLP public functions that prints

**********************************************************************/

using namespace std;

/***********************************************
	Save the MLP
***********************************************/
bool MLP::Save_MLP(string path_)
{
	int i = 0;
	int n_1 = 0, n_2 = 0;
	ofstream file_out;

	//	MLP should be constructed before save
	if (!isConstructed)
		return false;

	//	Open file
	file_out.open(path_, ios::out | ios::binary);
	//	In the case of file open failure
	if (!file_out.is_open())
		return false;

	//	Initialize transpose mode
	if (isTransposed)
		Transpose();

	//	Write information of the MLP
	//		MLP size
	file_out.write((char*)&n_in		, sizeof(n_in)		);
	file_out.write((char*)&n_out	, sizeof(n_out)		);
	file_out.write((char*)&n_neuron	, sizeof(n_neuron)	);
	file_out.write((char*)&n_depth	, sizeof(n_depth)	);

	//		MLP option
	file_out.write((char*)&i_method	, sizeof(i_method)	);
	file_out.write((char*)&i_cost	, sizeof(i_cost)	);
	file_out.write((char*)&i_regular, sizeof(i_regular)	);

	//		MLP parameters
	file_out.write((char*)&learning_rate, sizeof(learning_rate)	);
	file_out.write((char*)&lambda		, sizeof(lambda)		);
	file_out.write((char*)&beta1		, sizeof(beta1)			);
	file_out.write((char*)&beta2		, sizeof(beta2)			);
	file_out.write((char*)&eps			, sizeof(eps)			);

	//		MLP array
	//		Size of the arrays
	n_1 = n_neuron * (n_in + (n_depth - 1) * n_neuron + n_out);
	n_2 = n_in + n_depth * n_neuron + n_out;
	//		w
	for (i = 0; i < n_1; i++)
		file_out.write((char*)(w[0].A + i), sizeof(w[0].A[i]));
	//		mu
	for (i = 0; i < n_2; i++)
		file_out.write((char*)(mu[0].A + i), sizeof(mu[0].A[i]));
	//		activation
	for (i = 0; i < n_2; i++)
		file_out.write((char*)(activation[0].A + i), sizeof(activation[0].A[i]));
	//		type
	for (i = 0; i < n_2; i++)
		file_out.write((char*)(type[0].A + i), sizeof(type[0].A[i]));

	//	Close file
	file_out.close();

	return true;
}



/***********************************************
	Load the MLP
***********************************************/
bool MLP::Load_MLP(string path_)
{
	int i = 0;
	int n_1 = 0, n_2 = 0;
	ifstream file_in;

	if (isConstructed)
		Destruct();

	//	Open file
	file_in.open(path_, ios::in | ios::binary);
	//	In the case of file open failure
	if (!file_in.is_open())
		return false;

	//	Read information of the MLP
	//		MLP size: Behave like Set_Size
	file_in.read((char*)&n_in		, sizeof(n_in)		);
	file_in.read((char*)&n_out		, sizeof(n_out)		);
	file_in.read((char*)&n_neuron	, sizeof(n_neuron)	);
	file_in.read((char*)&n_depth	, sizeof(n_depth)	);

	//		Construct MLP based on MLP size
	Construct();

	//		MLP option
	file_in.read((char*)&i_method	, sizeof(i_method)	);
	file_in.read((char*)&i_cost		, sizeof(i_cost)	);
	file_in.read((char*)&i_regular	, sizeof(i_regular)	);

	//		MLP parameters
	file_in.read((char*)&learning_rate	, sizeof(learning_rate)	);
	file_in.read((char*)&lambda			, sizeof(lambda)		);
	file_in.read((char*)&beta1			, sizeof(beta1)			);
	file_in.read((char*)&beta2			, sizeof(beta2)			);
	file_in.read((char*)&eps			, sizeof(eps)			);

	//		MLP array
	//		Size of the arrays
	n_1 = n_neuron * (n_in + (n_depth - 1) * n_neuron + n_out);
	n_2 = n_in + n_depth * n_neuron + n_out;
	//		w
	for (i = 0; i < n_1; i++)
		file_in.read((char*)(w[0].A + i), sizeof(w[0].A[i]));
	//		mu
	for (i = 0; i < n_2; i++)
		file_in.read((char*)(mu[0].A + i), sizeof(mu[0].A[i]));
	//		activation
	for (i = 0; i < n_2; i++)
		file_in.read((char*)(activation[0].A + i), sizeof(activation[0].A[i]));
	//		type
	for (i = 0; i < n_2; i++)
		file_in.read((char*)(type[0].A + i), sizeof(type[0].A[i]));

	//	Close file
	file_in.close();

	return true;
}