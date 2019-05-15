#include "MLP.h"

/**********************************************************************

	MLP public functions that gets value

**********************************************************************/

using namespace std;

/***********************************************
	Get output y: CPU mode only
		Copy host y[n_depth + 1] to y_
***********************************************/
bool MLP::Get_Output_y(Matrix<double> *y_)
{
	//	Check if MLP is constructed
	if (!isConstructed)
	{
		cout << "MLP is not constructed" << endl;
		return false;
	}

	//	Check y_ dimension
	if (y_->m != n_out)
	{
		cout << "Dimension mis-match: Get_Output_y(Matrix_f*)" << endl;
		cout << "n_out: " << n_out << endl;
		cout << "y_ size: <" << y_->m << " X " << y_->n << ">" << endl;
		return false;
	}

	//	Copy memory - CPU mode
	y_->Set_Matrix(&y[n_depth + 1]);

	return true;
}



/***********************************************
	Get cost
***********************************************/
double MLP::Get_Cost()
{
	return cost.A[n_out];
}