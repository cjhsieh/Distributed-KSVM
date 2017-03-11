#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <mpi.h>
#include <vector>
#include "svm.h"
#include <omp.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: svm-train-mpi [options] training_set_file testing_set_file \n"
//	"Usage: svm-train-mpi [options] training_set_file testing_set_file [model_file]\n"
	"options:\n"
//	"-s svm_type : set type of SVM (default 0)\n"
//	"	0 -- C-SVC		(multi-class classification)\n"
//	"	1 -- nu-SVC		(multi-class classification)\n"
//	"	2 -- one-class SVM\n"
//	"	3 -- epsilon-SVR	(regression)\n"
//	"	4 -- nu-SVR		(regression)\n"
//	"-t kernel_type : set type of kernel function (default 2)\n"
//	"	0 -- linear: u'*v\n"
//	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
//	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
//	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
//	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
//	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
//	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
//	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
//	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.1)\n"
//	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
//	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
//	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
//	"-v n: n-fold cross validation mode\n"
//	"-M M : set the number of samples used for clustering (default 20000)\n"
	"-T T : set the maximum number of outer iterations (default 10)\n"
//	"-K K : set the number of clusters for initial alpha\n"
	"-A A : set the solver type (0: our method(default), 1: SGD)\n"
	"-R R : set the cluster (0: kmeans(default), 1:random)\n"
	"-F F : set the function type (0: SVM(default), 1:logistic regression)\n"
	"-N N : set the number of threads per machine (default: environment variable OMP_NUM_THREADS)\n"
	"-D D : D=0: not using divide-and-conquer (default), D=1: using divide-and-conquer\n"
	"-p p : print out the accuracy/objective function every p seconds (default p=10)\n"
//	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *testing_file_name, char *model_file_name);

//void read_problem(const char *filename, struct svm_problem &now_prob, struct svm_node *now_x_space, struct svm_parameter &now_param);
void read_problem(const char *filename);
void read_testing_problem(const char *filename_testing);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;

struct svm_node *x_space_test;

int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

int nearest_cluster(struct svm_node *px, int k, double *centers, vector<double> &center_normsq, double &value)
{
	double mindis = 100000000000.0;
	int minidx = -1;

	// square now (not necssary)
	struct svm_node *pxx = px;
	double sq = 0.0;
	while ( pxx->index != -1 )
	{
		sq += (pxx->value)*(pxx->value);
		pxx++;
	}

	for ( int j=0 ; j<k ; j++ )
	{
		double product = 0;
		int jd = j*prob.d;
		pxx = px;
		while (pxx->index != -1 )
		{
			product += centers[jd+pxx->index]*(pxx->value);
			++pxx;
		}
		double nowdis = center_normsq[j]-product*2+sq;
		if ( nowdis < mindis)
		{
			mindis = nowdis;
			minidx = j;
		}
	}
	value = mindis;
	return minidx;
}

void kmeans_single(struct svm_problem &prob, int k, int M, double *centers)
{
	int maxiter = 20;
//	centers.resize(k);
	vector<int> count(k);
	vector<int> idx(M);
	vector<double> center_normsq(k);

	// Initila
	for ( int i=0 ; i<k ; i++ )
	{
		int p = rand()%M;
		struct svm_node *px = prob.x[p];
		int di = i*prob.d;
		while (px->index != -1 )
		{
			centers[di+px->index] = px->value;
			++px;
		}
	}

	for ( int iter =0 ; iter < maxiter ; iter++ )
	{
		for ( int i=0 ; i<k ; i++ )
		{
			center_normsq[i] = 0;
			int di = i*prob.d;
			for ( int j=0 ; j<prob.d ; j++ )
				center_normsq[i] += centers[di+j]*centers[di+j];
		}
		
		double obj = 0;
		for ( int i=0 ; i<M ; i++ )
		{
			double value;
			idx[i] = nearest_cluster(prob.x[i], k, centers, center_normsq, value);
			obj += value;
		}
		
//		printf("Iter %d obj %lf\n", iter, obj);
		// Computer new centers
		for ( int i=0 ; i<k ; i++) {
			count[i] = 0;
			for ( int j=0, id=i*prob.d ; j<prob.d ; j++ )
				 centers[id+j] = 0;
		}
		for ( int i=0 ; i<M ; i++ )
		{
			int nowidx = idx[i];
			int nowidxd = nowidx*prob.d;
			struct svm_node *px = prob.x[i];
			while (px->index != -1 )
			{
				centers[nowidxd+px->index]  += px->value;
				++px;
			}
			count[nowidx] ++;
		}
		for ( int i=0 ; i<k ; i++ )
			for ( int j=0, id = i*prob.d ; j<prob.d ; j++ )
				centers[id+j] /= (double)count[i];
	}
}


static char* readline(FILE *input)
{
	int len;

//	printf("READLINE BEGIN\n");
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
//		printf("%d %d\n", max_line_len, len);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}

//	printf("READLINE END\n");
	return line;
}

int main(int argc, char **argv)
{
	int rank, size;
int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	    if (provided != MPI_THREAD_FUNNELED) {
			        fprintf(stderr, "Warning MPI did not provide MPI_THREAD_MULTIPLE\n");
					    }
//	printf("%d %d\n", MPI_THREAD_SINGLE, provided);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int k = size;
//	printf("rank %d size %d\n", rank, size);

	char input_file_name[1024];
	char model_file_name[1024];
	char testing_file_name[1024];
	const char *error_msg;
//	vector<vector<double> > centers;

	parse_command_line(argc, argv, input_file_name, testing_file_name, model_file_name);

	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);

	if (rank==0)
		printf("Testing file name: %s\n", testing_file_name); 
	read_testing_problem(testing_file_name);

//	read_problem(testing_file_name);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	// Initial centers
	double *centers = (double *)malloc(sizeof(double)*k*prob.d);
	for ( int i=0 ; i<k*prob.d ; i++ )
		centers[i] = 0;
	int M=20000; // number of samples for kmeans clustering
	if (M > prob.l)
		M = prob.l;
	double timetime = omp_get_wtime();
	// Clustering on root
	if ( rank==0 )
	{
		kmeans_single(prob, k, M, centers);
//		kmeans_single(prob, size, M, centers);
	}

	MPI_Bcast(centers, k*prob.d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Compute Ownership
	int *idx = (int *)malloc(sizeof(int)*prob.l);
	int *localidx = (int *)malloc(sizeof(int)*prob.l);
	int *intervals = (int *)malloc(sizeof(int)*(size+1));
	int *num_elements = (int *)malloc(sizeof(int)*size);
	int *displacement = (int *)malloc(sizeof(int)*size);
	int xx = prob.l/size;
	for ( int i=0 ; i<size ; i++)
		intervals[i] = i*xx;
	intervals[size] = prob.l;
	for ( int i=0 ; i<size ; i++ )
		num_elements[i] = (intervals[i+1] - intervals[i]);
	displacement[0] = 0;
	for ( int i=1 ; i<size ; i++ )
		displacement[i] = displacement[i-1]+num_elements[i-1];
	int mystart = intervals[rank];
	int myend = intervals[rank+1];

	vector<double> center_normsq(k);
	for ( int i=0 ; i<k ; i++ )
	{
		center_normsq[i] = 0;
		int di = i*prob.d;
		for ( int j=0 ; j<prob.d ; j++ )
			center_normsq[i] += centers[di+j]*centers[di+j];
	}

	double local_dis_sum = 0.0;
	for ( int i=mystart ; i<myend ; i++ )
	{
		double value = 0;
		localidx[i] = nearest_cluster(prob.x[i], k, centers, center_normsq, value);
		local_dis_sum += value;
		if (param.israndom == 1)
			localidx[i] = rand()%size;
	}

	MPI_Allgatherv(&(localidx[mystart]), num_elements[rank], MPI_INTEGER, idx, num_elements, displacement, MPI_INTEGER, MPI_COMM_WORLD);

	double global_dis_sum  =0.0;
	MPI_Reduce(&local_dis_sum, &global_dis_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

//	if ( rank == 0)
//	{
//		printf("Sum: %lf\n", global_dis_sum);
//	}


//	printf("clsutering time: %lf\n", omp_get_wtime()-timetime);

	MPI_Barrier(MPI_COMM_WORLD);

// Compute ownership for testing samples
	prob.idx_test = (int *)malloc(sizeof(int)*prob.l_test);
	for ( int i=0 ; i<prob.l_test ; i++ )
	{
		double value = 0;
		prob.idx_test[i] = nearest_cluster(prob.x_test[i], k, centers, center_normsq, value);
		if ( param.israndom == 1)
			prob.idx_test[i] = rand()%size;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	



	// Train
	model = svm_train_distributed(&prob,&param, size, idx);

	if ( rank==0 )
	{
	if(svm_save_model(model_file_name,model))
	{
		fprintf(stderr, "can't save model to file %s\n", model_file_name);
		exit(1);
	}
	}


	MPI_Finalize();
//	svm_free_and_destroy_model(&model);
	
//	svm_destroy_param(&param);

//	free(centers);
//	free(prob.y);
//	free(prob.x);
//	free(x_space);
//	free(line);

	return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *testing_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-1;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.maxiter = 10;
//	param.k_initial = 0;
	param.solver_type = 0;
	param.israndom = 0;
	param.function_type = 0;
	cross_validation = 0;
	param.DC = 0; 
	param.p = 10;
//	M = 20000;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
//			case 'M': 
//				M = atoi(argv[i]);
//				break;
			case 'T':
				param.maxiter = atoi(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
//			case 'K':
//				param.k_initial = atoi(argv[i]);
//				break;
			case 'A':
				param.solver_type = atoi(argv[i]);
				break;
			case 'D':
				param.DC = atoi(argv[i]);
				break;
			case 'R':
				param.israndom = atoi(argv[i]);
				break;
			case 'F':
				param.function_type = atoi(argv[i]);
				break;
			case 'N': 
				param.t = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames
	if (i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	i++;
	if (i>=argc)
		exit_with_help();
	strcpy(testing_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}

/*
void read_problem(const char *filename, struct svm_problem &now_prob, struct svm_node *now_x_space, struct svm_parameter &now_param)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	int dd = 0;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	now_prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++now_prob.l;
	}
	rewind(fp);

	now_prob.y = Malloc(double,prob.l);
	now_prob.x = Malloc(struct svm_node *,prob.l);
	now_x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		now_prob.x[i] = &now_x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		now_prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			now_x_space[j].index = (int) strtol(idx,&endptr,10);

			if (now_x_space[j].index > dd )
				dd = now_x_space[j].index;

			if(endptr == idx || errno != 0 || *endptr != '\0' || now_x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = now_x_space[j].index;

			errno = 0;
			now_x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		now_x_space[j++].index = -1;
	}

	now_prob.d = dd+1;

	if(now_param.gamma == 0 && max_index > 0)
		now_param.gamma = 1.0/max_index;

	if(now_param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (now_prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)now_prob.x[i][0].value <= 0 || (int)now_prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
*/



// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	int dd = 0;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
//		printf("%d\n", prob.l);
	}
	rewind(fp);

	if (rank==0)
		printf("In read_problem, prob.l=%ld, elements=%ld\n", prob.l, elements);
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);

			if (x_space[j].index > dd )
				dd = x_space[j].index;

			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	prob.d = dd+1;

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}


void read_testing_problem(const char *filename_testing)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename_testing,"r");
	char *endptr;
	char *idx, *val, *label;
	int dd = 0;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename_testing);
		exit(1);
	}

	prob.l_test = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l_test;
	}
	rewind(fp);

	prob.y_test = Malloc(double,prob.l_test);
	prob.x_test = Malloc(struct svm_node *,prob.l_test);
	x_space_test = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l_test;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x_test[i] = &x_space_test[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y_test[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space_test[j].index = (int) strtol(idx,&endptr,10);

//			if (x_space[j].index > dd )
//				dd = x_space[j].index;

			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space_test[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space_test[j].index;

			errno = 0;
			x_space_test[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			if ( x_space_test[j].index < prob.d)
				++j;
			else
			{
				printf("ERROR prob.d=%d, %d-th data, xindex = %d",prob.d, i, x_space_test[j].index );
			}
		}

//		if(inst_max_index > max_index)
//			max_index = inst_max_index;
		x_space_test[j++].index = -1;
	}

//	prob.d = dd+1;

//	if(param.gamma == 0 && max_index > 0)
//		param.gamma = 1.0/max_index;

/*	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}
*/
	fclose(fp);
}

