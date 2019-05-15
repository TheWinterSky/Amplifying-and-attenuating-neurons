#include "MLP.h"

using namespace std;

int test_check();
int test_sinu();
int test_1D();
int test_2D();
int test_2D_Loop(int n_);
double Ackley(double a_, double b_, double c_, Matrix<double> *x_);
int Stabilize_Network(MLP *ANN, int n_data_, int n_iter_);

int main()
{
	srand((unsigned int)time(0));

	//test_check();
	//test_sinu();
	//test_spiral();
	//test_1D();
	//test_2D();
	//test_2D_Loop(100);
	

	return 0;
}



int test_check()
{
	MLP ANN;
	Matrix<double> xt, yt;
	int n = 0;

	xt.Load_Matrix("./Output/Ackley_2D_xt.mat");
	yt.Load_Matrix("./Output/Ackley_2D_yt.mat");
	//xt.Load_Matrix("./Output/1D_xt.mat");
	//yt.Load_Matrix("./Output/1D_yt.mat");
	n = xt.n;
	

	ANN.Load_MLP("./Output/Trained_10.mlp");
	ANN.Set_Training_Data(&xt, &yt);
	ANN.Print_Information();
	
	ANN.Print_Status("type");
	ANN.Print_Status("activation");

	{
		Matrix<double> y_;
		int i = 0, idx = 0;
		double cost = 0.0, cost_a = 0.0, cost_m = 0.0;
		y_.Construct(yt.m, 1);

		cost_a = 0.0;
		cost_m = 0.0;
		for (i = 0; i < n; i++)
		{
			idx = i;
			ANN.Set_Input_y(idx);
			ANN.Forward_CPU();
			ANN.Calculate_Cost_CPU(idx);
			ANN.Get_Output_y(&y_);
			cost = ANN.Get_Cost();

			//cout << i << "\t cost = " << cost << endl;
			//printf("%3d\t%e\t%f\t%f\n", i, cost, yt.A[idx], y_.A[0]);
			cost_a += cost;
			cost_m = fmax(cost_m, cost);
		}
		cost_a /= (double)n;
		printf("cost_average = %e\n", cost_a);
		printf("cost_max     = %e\n", cost_m);
	}

	return 0;
}



int test_sinu()
{
	MLP ANN;
	Matrix<double> xt, yt;
	int i = 0, j = 0, k = 0, epoch = 0;
	double tmp = 0.0;
	int idx = 0, n = 0;
	double cost = 0.0, cost_a = 0.0, cost_m = 0.0;

	/*{
		double x_, y_, pi;
		x_ = 0.0;
		y_ = 0.0;
		pi = acos(-1.0);

		n = 150;
		xt.Construct(1, n);
		yt.Construct(1, n);

		for (i = 0; i < n; i++)
		{
			x_ = ((double)rand() / (double)RAND_MAX);
			x_ = x_ * 2.0 * pi;

			xt.Set_Value(0, i, x_);
			yt.Set_Value(0, i, sin(3.0*x_) + sin(6.0*x_));
		}
		xt.Save_Matrix("./Output/sinu_xt.mat");
		yt.Save_Matrix("./Output/sinu_yt.mat");
	}*/

	xt.Load_Matrix("./Output/sinu_xt.mat");
	yt.Load_Matrix("./Output/sinu_yt.mat");
	n = xt.n;

	//ANN.Construct(1, 1, 8, 5);
	ANN.Construct(1, 1, 7, 4);
	ANN.Set_Regularization(2, 0.002);
	ANN.Set_Activation_F(6);

	/*for (i = 1; i < 5; i++)
	{
		ANN.Set_Type(i, 7, 2);
		ANN.Set_Type(i, 6, 3);
		ANN.Set_Type(i, 5, 3);
		ANN.Set_Type(i, 4, 2);

		//	Stabilize ANN
		Stabilize_Network(&ANN, n, 50);
	}*/
	ANN.Set_Training_Data(&xt, &yt);


	tmp = 0.001;
	cout << "Start" << endl;

	for (k = 0; k < 10; k++)
	{
		for (j = 0; j < 20; j++)
		{
			for (epoch = 0; epoch < 500; epoch++)
			{
				cost_a = 0.0;
				cost_m = 0.0;

				for (i = 0; i < n; i++)
				{
					idx = rand() % n;
					ANN.Train_CPU(idx);
					cost = ANN.Get_Cost();

					cost_a += cost;
					cost_m = fmax(cost_m, cost);
				}
				cost_a /= (double)n;
				printf("%2d%3d%5d - ", k, j, epoch);
				printf("Average cost: %e\tMaximum cost: %e\n", cost_a, cost_m);
			}
			tmp = tmp * 0.95;
			ANN.Set_Learning_Rate(tmp);
		}
	}

	ANN.Save_MLP("./Output/Trained.mlp");
	ANN.Print_Information();
	ANN.Print_Status("type");
	ANN.Print_Status("activation");

	/*ANN.Load_MLP("./Output/Trained.mlp");
	ANN.Set_Training_Data(&xt, &yt);*/
	for (i = 0; i < n; i++)
	{
		idx = rand() % n;
		ANN.Set_Input_y(idx);
		ANN.Forward_CPU();
		ANN.Calculate_Cost_CPU(idx);
		cost = ANN.Get_Cost();

		//cout << i << "\t cost = " << cost << endl;
		printf("%3d\t%e\n", i, cost);
	}

	ofstream file_out;
	file_out.open("./Output/data.dat");
	if (file_out.is_open())
	{
		file_out << "Variables = x, y";
		for (i = 0; i < n; i++)
			file_out << endl << xt.A[i] << "\t" << yt.A[i];
	}
	file_out.close();

	file_out.open("./Output/out_0.dat");
	if (file_out.is_open())
	{
		int resol = 2000;
		double xmax = 0.0;
		Matrix<double> x_, y_;
		double x_e = 0.0, y_e = 0.0;
		x_.Construct(xt.m, 1);
		y_.Construct(yt.m, 1);

		for (i = 0; i < n; i++)
			xmax = fmax(xmax, xt.A[i]);

		file_out << "Variables = x, y, y_e";;
		n = (int)(1.2 * xmax * (double)resol);

		for (i = 0; i < n; i++)
		{
			x_.A[0] = (double)i / (double)resol;
			ANN.Set_Input_y(&x_);
			ANN.Forward_CPU();
			ANN.Get_Output_y(&y_);

			x_e = x_.A[0];
			y_e = sin(3.0 * x_e) + sin(6.0 * x_e);

			file_out << endl << x_.A[0] << "\t" << y_.A[0] << "\t" << y_e;
		}
	}
	file_out.close();

	return 0;
}



int test_1D()
{
	MLP ANN;
	Matrix<double> xt, yt;
	int i = 0, j = 0, k = 0, epoch = 0;
	int idx = 0, n = 0;
	double x_end = 0.0;
	double tmp = 0.0, pi = 0.0;
	double cost = 0.0, cost_a = 0.0, cost_m = 0.0;
	pi = acos(-1.0);

	x_end = 2.0 * pi;
	/*n = 194;
	xt.Construct(1, n);
	yt.Construct(1, n);
	{
		double x_ = 0.0, y_ = 0.0;
		
		x_end = 2.0 * pi;
		for (i = 0; i < n; i++)
		{
			x_ = x_end * ((double)rand() / (double)RAND_MAX);
			y_ = log(x_ + 0.5) + 0.2*sin(x_) + 0.4*sin(2.0*x_) + 0.3*sin(3.0*x_) - 0.1*sin(5.0*x_) - 0.2*sin(7.0*x_) + 0.15*sin(20.0*x_);
			xt.Set_Value(0, i, x_);
			yt.Set_Value(0, i, y_);
		}
	}
	xt.Save_Matrix("./Output/1D_xt.mat");
	yt.Save_Matrix("./Output/1D_yt.mat");*/
	xt.Load_Matrix("./Output/1D_xt.mat");
	yt.Load_Matrix("./Output/1D_yt.mat");
	n = xt.n;


	//	Construct MLP
	ANN.Construct(1, 1, 10, 9);
	ANN.Set_Training_Data(&xt, &yt);
	ANN.Set_Regularization(2, 0.002);
	ANN.Set_Activation_F(5);

	//	Stabilize network
	Stabilize_Network(&ANN, n, 2500);


	for (k = 1; k < 6; k++)
	{
		ANN.Set_Type(k, 9, 2);
		ANN.Set_Type(k, 8, 2);
		ANN.Set_Type(k, 7, 2);
		ANN.Set_Type(k, 6, 2);
		ANN.Set_Type(k, 5, 3);

		//	Stabilize ANN
		Stabilize_Network(&ANN, n, 50);
	}

	//	Train
	cout << "Start" << endl;
	tmp = 0.001;
	ANN.Set_Learning_Rate(tmp);

	for (k = 0; k < 10; k++)
	{
		for (j = 0; j < 20; j++)
		{
			for (epoch = 0; epoch < 1000; epoch++)
			{
				cost_a = 0.0;
				cost_m = 0.0;

				for (i = 0; i < n; i++)
				{
					idx = rand() % n;
					ANN.Train_CPU(idx);
					cost = ANN.Get_Cost();

					cost_a += cost;
					cost_m = fmax(cost_m, cost);
				}
				cost_a /= (double)n;
				printf("%2d%3d%5d - ", k, j, epoch);
				printf("Average cost: %e\tMaximum cost: %e\n", cost_a, cost_m);
			}
			tmp = tmp * 0.975;
			ANN.Set_Learning_Rate(tmp);
		}
	}
	ANN.Save_MLP("./Output/Trained_0.mlp");
	/*ANN.Load_MLP("./Output/Trained_3.mlp");
	ANN.Set_Training_Data(&xt, &yt);*/

	ANN.Print_Information();
	ANN.Print_Status("activation");
	ANN.Print_Status("type");
	{
		Matrix<double> x_, y_;
		x_.Construct(xt.m, 1);
		y_.Construct(yt.m, 1);

		cost_a = 0.0;
		cost_m = 0.0;
		for (i = 0; i < n; i++)
		{
			idx = i;
			ANN.Set_Input_y(idx);
			ANN.Forward_CPU();
			ANN.Calculate_Cost_CPU(idx);
			ANN.Get_Output_y(&y_);
			cost = ANN.Get_Cost();

			//cout << i << "\t cost = " << cost << endl;
			printf("%3d\t%e\t%f\t%f\n", i, cost, yt.A[idx], y_.A[0]);
			cost_a += cost;
			cost_m = fmax(cost_m, cost);
		}
		cost_a /= (double)n;
		printf("cost_average = %e\n", cost_a);
		printf("cost_max     = %e\n", cost_m);
	}

	//	Training data
	ofstream file_out;
	file_out.open("./Output/data.dat");
	if (file_out.is_open())
	{
		file_out << "Variables = x, y, y_e";
		for (i = 0; i < n; i++)
		{
			idx = i * xt.m;
			file_out << endl;
			file_out << xt.A[idx + 0] << "\t";

			idx = i * yt.m;
			file_out << yt.A[idx + 0] << "\t";
			file_out << 0.0;
		}

	}
	file_out.close();


	//	MLP output
	file_out.open("./Output/out_0.dat");
	if (file_out.is_open())
	{
		int resol = 2000;
		Matrix<double> x_, y_;
		double x_e = 0.0, y_e = 0.0;
		x_.Construct(xt.m, 1);
		y_.Construct(yt.m, 1);

		file_out << "Variables = x, y, y_e";;
		n = (int)(x_end * (double)resol);

		for (i = 0; i < n; i++)
		{
			x_.A[0] = (double)i / (double)resol;
			ANN.Set_Input_y(&x_);
			ANN.Forward_CPU();
			ANN.Get_Output_y(&y_);

			x_e = x_.A[0];
			y_e = log(x_e + 0.5)
				+ 0.2*sin(1.0*x_e) + 0.4*sin(2.0*x_e)
				+ 0.3*sin(3.0*x_e) - 0.1*sin(5.0*x_e)
				- 0.2*sin(7.0*x_e) + 0.15*sin(20.0*x_e);

			file_out << endl << x_.A[0] << "\t" << y_.A[0] << "\t" << y_e;
		}
	}
	file_out.close();

	return 0;
}



int test_2D()
{
	MLP ANN;
	Matrix<double> xt, yt;
	int i = 0, j = 0, k = 0, epoch = 0;
	int idx = 0, n = 0;
	double cost = 0.0, cost_a = 0.0, cost_m = 0.0;
	double x_bias = 0.0, x_length = 0.0;
	double tmp = 0.0, pi = 0.0;
	pi = acos(-1.0);

	x_bias		= -3.0;
	x_length	= 6.0;

	/*n			= 300;
	xt.Construct(2, n);
	yt.Construct(1, n);
	{
		Matrix<double> x_;
		x_.Construct(2, 1);

		for (i = 0; i < n; i++)
		{
			x_.A[0] = x_bias + x_length * ((double)rand() / (double)RAND_MAX);
			x_.A[1] = x_bias + x_length * ((double)rand() / (double)RAND_MAX);
			
			xt.Set_Value(0, i, x_.A[0]);
			xt.Set_Value(1, i, x_.A[1]);
			yt.Set_Value(0, i, Ackley(20.0, 0.2, 2.0 * pi, &x_));
		}
	}
	xt.Save_Matrix("./Output/Ackley_2D_xt.mat");
	yt.Save_Matrix("./Output/Ackley_2D_yt.mat");*/
	xt.Load_Matrix("./Output/Ackley_2D_xt.mat");
	yt.Load_Matrix("./Output/Ackley_2D_yt.mat");
	n = xt.n;

	//	Construct MLP
	ANN.Construct(2, 1, 10, 6);
	ANN.Set_Training_Data(&xt, &yt);
	ANN.Set_Regularization(2, 0.002);
	ANN.Set_Activation_F(5);

	//	Stabilize network
	Stabilize_Network(&ANN, n, 2500);


	for (k = 2; k < 6; k++)
	{
		ANN.Set_Type(k, 9, 3);
		ANN.Set_Type(k, 8, 3);
		ANN.Set_Type(k, 7, 3);
		ANN.Set_Type(k, 6, 2);
		//ANN.Set_Type(k, 5, 3);

		//	Stabilize ANN
		Stabilize_Network(&ANN, n, 100);
	}

	//	Train
	cout << "Start" << endl;
	tmp = 0.001;
	ANN.Set_Learning_Rate(tmp);

	for (k = 0; k < 10; k++)
	{
		for (j = 0; j < 20; j++)
		{
			for (epoch = 0; epoch < 1000; epoch++)
			{
				cost_a = 0.0;
				cost_m = 0.0;

				for (i = 0; i < n; i++)
				{
					idx = rand() % n;
					ANN.Train_CPU(idx);
					cost = ANN.Get_Cost();

					cost_a += cost;
					cost_m = fmax(cost_m, cost);
				}
				cost_a /= (double)n;
				printf("%2d%3d%5d - ", k, j, epoch);
				printf("Average cost: %e\tMaximum cost: %e\n", cost_a, cost_m);
			}
			tmp = tmp * 0.95;
			ANN.Set_Learning_Rate(tmp);
		}
	}
	ANN.Save_MLP("./Output/Trained_0.mlp");
	/*ANN.Load_MLP("./Output/Trained_99.mlp");
	ANN.Set_Training_Data(&xt, &yt);*/

	ANN.Print_Information();
	ANN.Print_Status("activation");
	ANN.Print_Status("type");
	{
		Matrix<double> y_;
		y_.Construct(yt.m, 1);

		cost_a = 0.0;
		cost_m = 0.0;
		for (i = 0; i < n; i++)
		{
			idx = i;
			ANN.Set_Input_y(idx);
			ANN.Forward_CPU();
			ANN.Calculate_Cost_CPU(idx);
			ANN.Get_Output_y(&y_);
			cost = ANN.Get_Cost();

			//cout << i << "\t cost = " << cost << endl;
			//printf("%3d\t%e\t%f\t%f\n", i, cost, yt.A[idx], y_.A[0]);
			cost_a += cost;
			cost_m = fmax(cost_m, cost);
		}
		cost_a /= (double)n;
		printf("cost_average = %e\n", cost_a);
		printf("cost_max     = %e\n", cost_m);
	}

	//	Training data
	ofstream file_out;
	file_out.open("./Output/data.dat");
	if (file_out.is_open())
	{
		file_out << "Variables = x, y, z, z_e";
		for (i = 0; i < n; i++)
		{
			idx = i * xt.m;
			file_out << endl;
			for (j = 0; j < xt.m; j++)
				file_out << xt.A[idx + j] << "\t";

			idx = i * yt.m;
			file_out << yt.A[idx + 0] << "\t";
			file_out << 0.0;
		}
	}
	file_out.close();


	//	MLP output
	file_out.open("./Output/out_0.dat");
	if (file_out.is_open())
	{
		int resol = 50;
		//int resol = 120;
		Matrix<double> x_, y_;
		x_.Construct(xt.m, 1);
		y_.Construct(yt.m, 1);

		n = (int)(x_length * (double)resol);
		file_out << "Variables = x, y, z, z_e" << endl;
		file_out << "zone i = " << n << " j = " << n;

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				x_.A[0] = x_bias + (double)i / (double)resol;
				x_.A[1] = x_bias + (double)j / (double)resol;

				ANN.Set_Input_y(&x_);
				ANN.Forward_CPU();
				ANN.Get_Output_y(&y_);

				file_out << endl;
				file_out << x_.A[0] << "\t";
				file_out << x_.A[1] << "\t";
				file_out << y_.A[0] << '\t';
				file_out << Ackley(20.0, 0.2, 2.0 * pi, &x_);
			}
		}
	}
	file_out.close();

	return 0;
}



int test_2D_Loop(int n_)
{
	MLP ANN;
	Matrix<double> xt, yt;
	int i_ = 0;
	int i = 0, j = 0, k = 0, epoch = 0;
	int idx = 0, n = 0;
	double cost = 0.0, cost_a = 0.0, cost_m = 0.0, cost_mt = 999.0;
	double tmp = 0.0;

	xt.Load_Matrix("./Output/Ackley_2D_xt.mat");
	yt.Load_Matrix("./Output/Ackley_2D_yt.mat");
	n = xt.n;

	for (i_ = 0; i_ < n_; i_++)
	{
		//	Construct MLP
		ANN.Construct(2, 1, 10, 6);
		ANN.Set_Training_Data(&xt, &yt);
		ANN.Set_Regularization(2, 0.002);
		ANN.Set_Activation_F(5);

		//	Stabilize network
		Stabilize_Network(&ANN, n, 2500);

		for (k = 2; k < 6; k++)
		{
			ANN.Set_Type(k, 9, 3);
			ANN.Set_Type(k, 8, 3);
			ANN.Set_Type(k, 7, 3);
			ANN.Set_Type(k, 6, 2);
			//ANN.Set_Type(k, 5, 3);

			//	Stabilize ANN
			Stabilize_Network(&ANN, n, 100);
		}

		//	Train
		//cout << "Start" << endl;
		tmp = 0.001;
		ANN.Set_Learning_Rate(tmp);

		for (k = 0; k < 10; k++)
		{
			for (j = 0; j < 20; j++)
			{
				for (epoch = 0; epoch < 1000; epoch++)
				{
					cost_a = 0.0;
					cost_m = 0.0;

					for (i = 0; i < n; i++)
					{
						idx = rand() % n;
						ANN.Train_CPU(idx);
						cost = ANN.Get_Cost();

						cost_a += cost;
						cost_m = fmax(cost_m, cost);
					}
					cost_a /= (double)n;
					//printf("%2d%3d%5d - ", k, j, epoch);
					//printf("Average cost: %e\tMaximum cost: %e\n", cost_a, cost_m);
				}
				tmp = tmp * 0.95;
				ANN.Set_Learning_Rate(tmp);
			}
		}

		{
			cost_a = 0.0;
			cost_m = 0.0;
			for (i = 0; i < n; i++)
			{
				idx = i;
				ANN.Set_Input_y(idx);
				ANN.Forward_CPU();
				ANN.Calculate_Cost_CPU(idx);
				cost = ANN.Get_Cost();

				cost_a += cost;
				cost_m = fmax(cost_m, cost);
			}
			cost_a /= (double)n;
			printf("cost_average = %e\n", cost_a);
			printf("cost_max     = %e\n", cost_m);
		}

		if (cost_m < cost_mt)
		{
			cout << "Save current MLP: Previous cost_max: " << cost_mt << endl;
			cost_mt = cost_m;
			ANN.Save_MLP("./Output/Trained_99.mlp");
		}

		ANN.Destruct();
	}

	return 0;
}



double Ackley(double a_, double b_, double c_, Matrix<double> *x_)
{
	int i = 0, d = 0;
	double x = 0.0, tmp1 = 0.0, tmp2 = 0.0, y = 0.0;

	d = x_->m;
	for (i = 0; i < d; i++)
	{
		x = x_->A[i];
		tmp1 += x * x;
		tmp2 += cos(c_ * x);
	}
	tmp1 = -b_ * sqrt(tmp1 / (double)d);
	tmp2 = tmp2 / (double)d;

	y = - a_ * exp(tmp1) - exp(tmp2) + a_ + exp(1.0);

	return y;
}



int Stabilize_Network(MLP *ANN, int n_data_, int n_iter_)
{
	int i = 0, epoch = 0, idx = 0;
	double cost = 0.0, cost_a = 0.0, cost_m = 0.0;

	//	Stabilize network
	for (epoch = 0; epoch < n_iter_; epoch++)
	{
		cost_a = 0.0;
		cost_m = 0.0;

		for (i = 0; i < n_data_; i++)
		{
			idx = rand() % n_data_;
			ANN->Train_CPU(idx);
			cost = ANN->Get_Cost();

			cost_a += cost;
			cost_m = fmax(cost_m, cost);
		}
		cost_a /= (double)n_data_;
		printf("epoch %6d - ", epoch);
		printf("Average cost: %e\tMaximum cost: %e\n", cost_a, cost_m);
	}
	ANN->Initialize_Momentum();

	return 0;
}