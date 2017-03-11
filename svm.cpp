#include<time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include <mpi.h>
#include <omp.h>
#include "svm.h"
int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
	public:
		Cache(int l,long int size);
		~Cache();

		// request data [0,len)
		// return some position p where [p,len) need to be filled
		// (p >= len if nothing needs to be filled)
		int get_data(const int index, Qfloat **data, int len);
		void swap_index(int i, int j);	
	private:
		int l;
		long int size;
		struct head_t
		{
			head_t *prev, *next;	// a circular list
			Qfloat *data;
			int len;		// data[0,len) is cached in this entry
		};

		head_t *head;
		head_t lru_head;
		void lru_delete(head_t *h);
		void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
	public:
		virtual Qfloat *get_Q(int column, int len) const = 0;
		virtual double *get_QD() const = 0;
		virtual void swap_index(int i, int j) const = 0;
		virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
	public:
		Kernel(int l, svm_node * const * x, const svm_parameter& param);
		virtual ~Kernel();

		static double k_function(const svm_node *x, const svm_node *y,
				const svm_parameter& param);
		virtual Qfloat *get_Q(int column, int len) const = 0;
		virtual double *get_QD() const = 0;
		virtual void swap_index(int i, int j) const	// no so const...
		{
			swap(x[i],x[j]);
			if(x_square) swap(x_square[i],x_square[j]);
		}
	protected:

		double (Kernel::*kernel_function)(int i, int j) const;

	private:
		const svm_node **x;
		double *x_square;

		// svm_parameter
		const int kernel_type;
		const int degree;
		const double gamma;
		const double coef0;

		static double dot(const svm_node *px, const svm_node *py);
		double kernel_linear(int i, int j) const
		{
			return dot(x[i],x[j]);
		}
		double kernel_poly(int i, int j) const
		{
			return powi(gamma*dot(x[i],x[j])+coef0,degree);
		}
		double kernel_rbf(int i, int j) const
		{
			return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
		}
		double kernel_sigmoid(int i, int j) const
		{
			return tanh(gamma*dot(x[i],x[j])+coef0);
		}
		double kernel_precomputed(int i, int j) const
		{
			return x[i][(int)(x[j][0].value)].value;
		}
};

	Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
	gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
		const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
			{
				double sum = 0;
				while(x->index != -1 && y->index !=-1)
				{
					if(x->index == y->index)
					{
						double d = x->value - y->value;
						sum += d*d;
						++x;
						++y;
					}
					else
					{
						if(x->index > y->index)
						{	
							sum += y->value * y->value;
							++y;
						}
						else
						{
							sum += x->value * x->value;
							++x;
						}
					}
				}

				while(x->index != -1)
				{
					sum += x->value * x->value;
					++x;
				}

				while(y->index != -1)
				{
					sum += y->value * y->value;
					++y;
				}

				return exp(-param.gamma*sum);
			}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}
//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
	public:
		SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
			:Kernel(prob.l, prob.x, param)
		{
			clone(y,y_,prob.l);
			cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
			QD = new double[prob.l];
			for(int i=0;i<prob.l;i++)
				QD[i] = (this->*kernel_function)(i,i);
		}

		Qfloat *get_Q(int i, int len) const
		{
			Qfloat *data;
			int start, j;
			if((start = cache->get_data(i,&data,len)) < len)
			{
#pragma omp parallel for private(j)
				for(j=start;j<len;j++)
				{
					data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
				}
			}
			return data;
		}

		double *get_QD() const
		{
			return QD;
		}

		void swap_index(int i, int j) const
		{
			cache->swap_index(i,j);
			Kernel::swap_index(i,j);
			swap(y[i],y[j]);
			swap(QD[i],QD[j]);
		}

		~SVC_Q()
		{
			delete[] y;
			delete cache;
			delete[] QD;
		}
	private:
		schar *y;
		Cache *cache;
		double *QD;
};

// A Distributed algorithm 
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, y, C
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver_Dis {
	public:
		Solver_Dis() {};
		virtual ~Solver_Dis() {};

		struct SolutionInfo {
			double obj;
			double upper_bound_p;
			double upper_bound_n;
		};

		void Solve_LR_SGD(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
				double C, double eps,
				//		   SolutionInfo* si, 
				int shrinking, int maxiter, int *intervals_, int *num_elements_, svm_problem *prob, const svm_parameter *param);



		void Solve_SGD(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
				double C, double eps,
				//		   SolutionInfo* si, 
				int shrinking, int maxiter, int *intervals_, int *num_elements_, svm_problem *prob, const svm_parameter *param);


		void Solve_LR(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
				double C, double eps,
				//		   SolutionInfo* si, 
				int shrinking, int maxiter, int *intervals_, int *num_elements_, svm_problem *prob, const svm_parameter *param);



		void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
				double C, double eps,
				//		   SolutionInfo* si, 
				int shrinking, int maxiter, int *intervals_, int *num_elements_, svm_problem *prob, const svm_parameter *param);
	protected:
		int active_size;
		schar *y;
		double *G;		// gradient of objective function
		double *alphag, *alphaH;
		enum { LOWER_BOUND, UPPER_BOUND, FREE };
		char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
		double *alpha;
		double *oldalpha;
		double *delta_alpha;
		double *local_delta_alpha;
		const QMatrix *Q;
		const double *QD;
		double eps;
		double Cp,Cn;
		double *p;
		int *active_set;
		double *G_old;
		double *G_delta;
		double *G_local_delta;
		int l;
		bool unshrink;	// XXX
		int *intervals;
		int *num_elements;

		double get_C(int i)
		{
			//		return C;
			return (y[i] > 0)? Cp : Cn;
		}
		void update_alpha_status(int i)
		{
			if(alpha[i] >= get_C(i))
				alpha_status[i] = UPPER_BOUND;
			else if(alpha[i] <= 0)
				alpha_status[i] = LOWER_BOUND;
			else alpha_status[i] = FREE;
		}
		bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
		bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
		bool is_free(int i) { return alpha_status[i] == FREE; }
		//	void swap_index(int i, int j);
		//	void reconstruct_gradient();
		virtual int select_working_set(int &i, int rank);
		virtual int select_working_set_LR(int &i, int rank);
		//	virtual void do_shrinking_nobias();
	private:
		//	bool be_shrunk(int i, double Gmax1, double Gmax2);	
		//	bool be_shrunk_nobias(int i, double Gmax1);	
};

void Solver_Dis::Solve_LR_SGD(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
		double C, double eps,
		int shrinking, int maxiter, int *intervals_, int *num_elements_, 
		svm_problem *prob, const svm_parameter *param
		)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("Rank %d Begin P-pack SVM\n", rank);

	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_, l);
	clone(y, y_,l);
	clone(alpha, alpha_, l);
	clone(oldalpha, alpha_, l);
	this->Cp = C;
	this->Cn = C;
	this->eps = eps;
	this->intervals = intervals_;
	this->num_elements = num_elements_;

	srand(0);

	int *owner = Malloc(int, l);
	for ( int i=0 ; i<l ; i++)
		owner[i] = rand()%size;

	int key_len = 0;
	int *key_list = Malloc(int, l);
	double *val_list = Malloc(double, l);
	int *hash_table = Malloc(int, l);
	for ( int i=0 ; i<l ; i++ )
		hash_table[i] = -1;

	int r = 100; 

	int *indlist = Malloc(int, l); // For random permutation
	for ( int i=0 ; i<l ; i++ )
		indlist[i] = i;

	// put the key into table..
	for ( int i=0 ; i<l ; i++ )
		if ( owner[i] == rank )
		{
			key_list[key_len] = i;
			val_list[key_len] = 0.001;
			hash_table[i] = key_len;
			key_len++;
		}

	int *nowindlist = Malloc(int, r);
	double *nowpredlist_local = Malloc(double, r);
	double *nowpredlist_global = Malloc(double, r);
	double *kvalue_table = Malloc(double, r*r);
	double *kvalue_table_local = Malloc(double, r);
	double *update_vals = Malloc(double, r);

	double s = 1.0;
	double norm = 0;
	double sigma = 1.0/(l*C);
	double stepsize = 0.25;

	double *ypred_test_local = Malloc(double, prob->l_test);
	double *ypred_test_global = Malloc(double, prob->l_test);
	maxiter = maxiter*r;

	double totaltime = 0.0;
	for ( int t=1 ; t<=maxiter/r ; t++ )
	{
		//		printf("Rank %d begin iteration %d\n", rank, t);
		double timebegin = omp_get_wtime();
		if ( rank == 0 )
		{
			// Randomly pick r samples
			for ( int i=0 ; i<r ; i++ )
			{
				int j = rand()%(l-i)+i;
				int tmp = indlist[j];
				indlist[j] = indlist[i];
				indlist[i] = tmp;
				nowindlist[i] = indlist[i];
			}
		}
		// Broadcast to workers
		MPI_Bcast(nowindlist, r, MPI_INT, 0, MPI_COMM_WORLD);

		// Compute y prediction
		for ( int i=0 ; i<r ; i++ )
		{
			int pi = nowindlist[i];
			nowpredlist_local[i] = 0;
			for ( int j=0 ; j<key_len ; j++ )
			{
				double kvalue = Kernel::k_function(prob->x[pi] , prob->x[key_list[j]] ,(*param));
				nowpredlist_local[i] += kvalue*val_list[j];
			}
		}

		// Gather PRediction (to root)
		MPI_Allreduce(nowpredlist_local, nowpredlist_global, r, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Compute Local kvalues
		for ( int k=0 ; k<r ; k++  )
			for ( int l=k ; l<r ; l++ )
			{
				kvalue_table[k*r+l] = Kernel::k_function(prob->x[nowindlist[k]], prob->x[nowindlist[l]], (*param));
			}

		// Solve
		for ( int k=0 ; k<r ; k++ )
			update_vals[k] = 0;
		for ( int k=0 ; k<r ; k++ )
		{
			int tp = t*r+k+1;
			s = (1-1.0/tp)*s;
			for ( int l=k+1 ; l<r ; l++ ) 
			{
				nowpredlist_global[l] = (1-1.0/tp)*nowpredlist_global[l];
			}
			double yk = y[nowindlist[k]];
			double ykp = nowpredlist_global[k];

			double aaa = -yk*(1.0/(1.0+exp(-yk*ykp))-1);
			//			if ( yk*ykp < 1.0 )
			//			{
			//				double step = stepsize*(yk/(sigma*tp));
			double step = stepsize*aaa/(sigma*tp);
			norm = norm + 2*ykp*step + step*step*kvalue_table[k*r+k];
			update_vals[k] = step/s;
			//				norm = norm + 2*yk*ykp/(sigma*tp) + (yk*yk)/(sigma*sigma*tp*tp)*kvalue_table[k*r+k];
			//				update_vals[k] = yk/(s*sigma*tp);
			/*				for ( int l=k+1 ; l<r ; l++ )
							nowpredlist_global[l] = nowpredlist_global[l] + step*kvalue_table[k*r+l];*/
			//					nowpredlist_global[l] = nowpredlist_global[l] + yk*kvalue_table[k*r+l]/(sigma*tp);
			if ( norm > 1.0/sigma )
			{
				double scale = sqrt(norm*sigma);
				s = s/scale;
				norm = 1/sigma;
				/*					for ( int l=k+1 ; l<r ; l++ ) 
									{
									nowpredlist_global[l] = nowpredlist_global[l]/scale;
									}*/
			}
			//			}
		}

		// Broadcast to processors
		MPI_Bcast(update_vals, r, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Update the key_val pairs
		for ( int k=0 ; k<r ; k++ )
		{
			int pi = nowindlist[k];
			if ( owner[pi] == rank )
			{
				int nowpos = hash_table[pi];
				if ( nowpos == -1 ) {
					// Create a new entry
					key_list[key_len] = pi;
					val_list[key_len] = update_vals[k];
					hash_table[pi] = key_len;
					key_len++;
				} else	{
					val_list[nowpos] += update_vals[k];
				}
			}
		}

		// Finished One interation
		MPI_Barrier(MPI_COMM_WORLD);
		totaltime = totaltime + (omp_get_wtime()-timebegin);

		if ( t%500 == 0) 
		{
			// Prediction
			for ( int i=0 ; i<prob->l_test ; i++ )
			{
				ypred_test_local[i] = 0;
				for ( int j=0 ; j<key_len ; j++ )
				{
					double kvalue = Kernel::k_function(prob->x_test[i] , prob->x[key_list[j]] ,(*param));
					ypred_test_local[i] += kvalue*val_list[j];
				}
			}

			MPI_Allreduce(ypred_test_local, ypred_test_global, prob->l_test, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			int correct = 0;
			for ( int i=0 ; i<prob->l_test ; i++ )
			{
				//			printf("%lf %lf\n", prob->y_test[i], ypred_test_global[i]);
				if ( prob->y_test[i] * ypred_test_global[i] >= 0 )
					correct++;
			}
			double accuracy = (double)correct/(double)prob->l_test;

			printf("Iteration %d Rank %d  Accuracy %lf Time %lf\n", t*r, rank,  accuracy, totaltime);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}




	delete[] owner;
	delete[] key_list;
	delete[] val_list;
	delete[] hash_table;
	delete[] indlist;
	delete[] nowindlist;
	delete[] nowpredlist_local;
	delete[] nowpredlist_global;
	delete[] kvalue_table;
	delete[] kvalue_table_local;
	delete[] update_vals;
	delete[] ypred_test_local;
	delete[] ypred_test_global;
}



void Solver_Dis::Solve_SGD(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
		double C, double eps,
		int shrinking, int maxiter, int *intervals_, int *num_elements_, 
		svm_problem *prob, const svm_parameter *param
		)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("Rank %d Begin P-pack SVM\n", rank);

	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_, l);
	clone(y, y_,l);
	clone(alpha, alpha_, l);
	clone(oldalpha, alpha_, l);
	this->Cp = C;
	this->Cn = C;
	this->eps = eps;
	this->intervals = intervals_;
	this->num_elements = num_elements_;

	srand(0);

	int *owner = Malloc(int, l);
	for ( int i=0 ; i<l ; i++)
		owner[i] = rand()%size;

	int key_len = 0;
	int *key_list = Malloc(int, l);
	double *val_list = Malloc(double, l);
	int *hash_table = Malloc(int, l);
	for ( int i=0 ; i<l ; i++ )
		hash_table[i] = -1;

	int r = 100; 

	int *indlist = Malloc(int, l); // For random permutation
	for ( int i=0 ; i<l ; i++ )
		indlist[i] = i;

	int *nowindlist = Malloc(int, r);
	double *nowpredlist_local = Malloc(double, r);
	double *nowpredlist_global = Malloc(double, r);
	double *kvalue_table = Malloc(double, r*r);
	double *kvalue_table_local = Malloc(double, r);
	double *update_vals = Malloc(double, r);

	double s = 1.0;
	double norm = 0;
	double sigma = 1.0/(l*C);

	double *ypred_test_local = Malloc(double, prob->l_test);
	double *ypred_test_global = Malloc(double, prob->l_test);
	maxiter = maxiter*r;

	double totaltime = 0.0;
	for ( int t=1 ; t<=maxiter/r ; t++ )
	{
		//		printf("Rank %d begin iteration %d\n", rank, t);
		double timebegin = omp_get_wtime();
		if ( rank == 0 )
		{
			// Randomly pick r samples
			for ( int i=0 ; i<r ; i++ )
			{
				int j = rand()%(l-i)+i;
				int tmp = indlist[j];
				indlist[j] = indlist[i];
				indlist[i] = tmp;
				nowindlist[i] = indlist[i];
			}
		}
		// Broadcast to workers
		MPI_Bcast(nowindlist, r, MPI_INT, 0, MPI_COMM_WORLD);

		// Compute y prediction
		for ( int i=0 ; i<r ; i++ )
		{
			int pi = nowindlist[i];
			nowpredlist_local[i] = 0;
			for ( int j=0 ; j<key_len ; j++ )
			{
				double kvalue = Kernel::k_function(prob->x[pi] , prob->x[key_list[j]] ,(*param));
				nowpredlist_local[i] += kvalue*val_list[j];
			}
		}

		// Gather PRediction (to root)
		MPI_Allreduce(nowpredlist_local, nowpredlist_global, r, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Compute Local kvalues
		for ( int k=0 ; k<r ; k++  )
			for ( int l=k ; l<r ; l++ )
			{
				kvalue_table[k*r+l] = Kernel::k_function(prob->x[nowindlist[k]], prob->x[nowindlist[l]], (*param));
			}

		// Solve
		for ( int k=0 ; k<r ; k++ )
			update_vals[k] = 0;
		for ( int k=0 ; k<r ; k++ )
		{
			int tp = t*r+k+1;
			s = (1-1.0/tp)*s;
			for ( int l=k+1 ; l<r ; l++ ) 
			{
				nowpredlist_global[l] = (1-1.0/tp)*nowpredlist_global[l];
			}
			double yk = y[nowindlist[k]];
			double ykp = nowpredlist_global[k];
			if ( yk*ykp < 1.0 )
			{
				double step = yk/(sigma*tp);
				norm = norm + 2*ykp*step + step*step*kvalue_table[k*r+k];
				update_vals[k] = step/s;
				//				norm = norm + 2*yk*ykp/(sigma*tp) + (yk*yk)/(sigma*sigma*tp*tp)*kvalue_table[k*r+k];
				//				update_vals[k] = yk/(s*sigma*tp);
				/*				for ( int l=k+1 ; l<r ; l++ )
								nowpredlist_global[l] = nowpredlist_global[l] + step*kvalue_table[k*r+l];*/
				//					nowpredlist_global[l] = nowpredlist_global[l] + yk*kvalue_table[k*r+l]/(sigma*tp);
				if ( norm > 1.0/sigma )
				{
					double scale = sqrt(norm*sigma);
					s = s/scale;
					norm = 1/sigma;
					/*					for ( int l=k+1 ; l<r ; l++ ) 
										{
										nowpredlist_global[l] = nowpredlist_global[l]/scale;
										}*/
				}
			}
		}

		// Broadcast to processors
		MPI_Bcast(update_vals, r, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Update the key_val pairs
		for ( int k=0 ; k<r ; k++ )
		{
			int pi = nowindlist[k];
			if ( owner[pi] == rank )
			{
				int nowpos = hash_table[pi];
				if ( nowpos == -1 ) {
					// Create a new entry
					key_list[key_len] = pi;
					val_list[key_len] = update_vals[k];
					hash_table[pi] = key_len;
					key_len++;
				} else	{
					val_list[nowpos] += update_vals[k];
				}
			}
		}

		// Finished One interation
		MPI_Barrier(MPI_COMM_WORLD);
		totaltime = totaltime + (omp_get_wtime()-timebegin);

		if ( t%500 == 0) 
		{
			// Prediction
			for ( int i=0 ; i<prob->l_test ; i++ )
			{
				ypred_test_local[i] = 0;
				for ( int j=0 ; j<key_len ; j++ )
				{
					double kvalue = Kernel::k_function(prob->x_test[i] , prob->x[key_list[j]] ,(*param));
					ypred_test_local[i] += kvalue*val_list[j];
				}
			}

			MPI_Allreduce(ypred_test_local, ypred_test_global, prob->l_test, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			int correct = 0;
			for ( int i=0 ; i<prob->l_test ; i++ )
				if ( prob->y_test[i] * ypred_test_global[i] >= 0 )
					correct++;
			double accuracy = (double)correct/(double)prob->l_test;

			printf("Iteration %d Rank %d  Accuracy %lf Time %lf\n", t*r, rank,  accuracy, totaltime);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}




	delete[] owner;
	delete[] key_list;
	delete[] val_list;
	delete[] hash_table;
	delete[] indlist;
	delete[] nowindlist;
	delete[] nowpredlist_local;
	delete[] nowpredlist_global;
	delete[] kvalue_table;
	delete[] kvalue_table_local;
	delete[] update_vals;
	delete[] ypred_test_local;
	delete[] ypred_test_global;
}


void Solver_Dis::Solve_LR(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
		double C, double eps,
		//		   SolutionInfo* si, 
		int shrinking, int maxiter, int *intervals_, int *num_elements_, 
		svm_problem *prob, const svm_parameter *param
		)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double totaltime = 0.0;
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_, l);
	clone(y, y_,l);
	clone(alpha, alpha_, l);
	clone(oldalpha, alpha_, l);
	this->Cp = C;
	this->Cn = C;
	this->eps = eps;
	this->intervals = intervals_;
	this->num_elements = num_elements_;
	//	unshrink = false;

	// Create Subproblem
	int mystart = intervals[rank];
	int myend = intervals[rank+1];
	int lsub = myend - mystart;


	printf("Rank %d size %d Kernel Logistic Regression\n", rank, lsub);

	for ( int i=0 ; i<l; i++ )
		alpha[i] = 0.001*C;


	// initialize gradient
	{
		alphag = new double[l];
		alphaH = new double[l];
		G = new double[l];
		G_old = new double[l];
		G_delta = new double[l];
		G_local_delta = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_old[i] = p[i];
			alphag[i] = log(alpha[i]/(C-alpha[i]));
			alphaH[i] = C/(C-alpha[i])/alpha[i];
		}
	}

	double ttgg = omp_get_wtime();
	for ( int i=0 ; i<l ; i++ ) {
		G_local_delta[i] = 0;
	}
	for ( int pi = mystart ; pi<myend ; pi++)
	{
		const Qfloat *Q_i = Q.get_Q(pi,l);
		for (int i=0 ; i<l ; i++ )
			G_local_delta[i] += Q_i[i]*alpha[pi];
	}

	// Configuration
	MPI_Barrier(MPI_COMM_WORLD);
	double ttg = omp_get_wtime();
	MPI_Allreduce(G_local_delta, G, l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	totaltime += (omp_get_wtime()-ttgg);
	double allreducetime = omp_get_wtime()-ttg;
	// Initial OBJ
	double nowobj = 0;
	for ( int i=0 ; i<l ; i++ )
		nowobj += alpha[i]*G[i];
	nowobj /=2;
	for ( int i=0 ; i<l ; i++ )
		nowobj += (alpha[i]*log(alpha[i]) + (C-alpha[i])*log(C-alpha[i]));

	printf("Rank %d Finished Initial Communicationb, Allreducetime: %lf secs. Initial OBJ: %lf\n", rank, allreducetime, nowobj);



	// optimization step

	double timeout = allreducetime *100;
	double firsttimeout = allreducetime *400;
	int miditer = 5000;
	int firstiter = 20000;
	delta_alpha = new double[l];
	local_delta_alpha = new double[myend - mystart];
	double timebegin = omp_get_wtime();
	double connection_time = 0.0;
	double innereps = 1e-2;


	printf("Rank %d begin inner loop\n", rank);
	for ( int iter=0 ; iter<maxiter ; iter++ )
	{
		MPI_Barrier(MPI_COMM_WORLD);
		double innertimebegin_loop = omp_get_wtime();
		// backup current alpha
		for (int i=0; i<l; i++ )
			oldalpha[i] = alpha[i];
		for ( int i=0; i<l ; i++ )
			G_old[i] = G[i];
		// Solve subproblems independently
		double innertimebegin = omp_get_wtime();

		printf("Rank %d HERE %d %d %d\n", rank, mystart, lsub, l);
		int inneriter = 0;
		//		miditer = 1;
		for ( inneriter = 0; inneriter < miditer ; inneriter++ )
		{
			int pi;
			select_working_set_LR(pi, rank);
			if ( pi < 0 )
				break;
			const Qfloat *Q_i = Q.get_Q(pi,l);
			double C_i = get_C(pi);
			double old_alpha_i = alpha[pi];
			double a = QD[pi];
			double b = G[pi];
			//			if ( quad_coef <= 0)
			//				quad_coef = TAU;

			double z = old_alpha_i;
			double gp = b + alphag[pi];

			double eta = 0.1;
			int inner_iter = 0;
			int max_inner_iter = 10;
			double epsnow = max(fabs(gp)*innereps, 1e-10);
			while (inner_iter<=max_inner_iter)
			{
				if (fabs(gp) < epsnow)
					break;
				double gpp = a+C/(C-z)/z;
				double tmpz = z-gp/gpp;
				if ( tmpz <= 0)
					z*=eta;
				else if ( tmpz >= C)
					z = C*(1-eta);
				else
					z = tmpz;
				gp = a*(z-old_alpha_i) + b + log(z/(C-z));
				inner_iter++;
			}

			alpha[pi] = z;
			alphag[pi] = log(z/(C-z));
			alphaH[pi] = C/(C-z)/z;
			double delta_alpha_i = z-old_alpha_i;

			for(int k=0;k<l;k++)
				G[k] += Q_i[k]*delta_alpha_i;

			if ( omp_get_wtime() - innertimebegin > timeout )
				break;
		}

		printf("Rank %d Finished Inner iter\n", rank);
		MPI_Barrier(MPI_COMM_WORLD);
		totaltime += (omp_get_wtime() - innertimebegin_loop);
		int correct = 0;
		int global_correct = 0;
		double accuracy_new = 0;
		/*
		   if ( (iter < 2) || (iter%30==0) ) {
		// Compute Testing error (use local + global alpha)
		//		int mystart = intervals[rank];
		//		int myend = intervals[rank+1];
		for ( int itest =0 ; itest < prob->l_test ; itest++ )
		if ( prob->idx_test[itest] == rank)
		{
		double pred_value = 0.0;
		for ( int i=0 ; i<l ; i++ ) {
		double nowalpha = 0.0;
		//					if ( (i >= mystart) && (i<myend)) 
		//						nowalpha = oldalpha[i] + delta_alpha[i];
		//					else
		//						nowalpha = oldalpha[i];
		nowalpha = alpha[i];
		if ( nowalpha != 0) {
		double kvalue = Kernel::k_function(prob->x_test[itest],prob->x[i],(*param));
		pred_value += kvalue*nowalpha*(prob->y[i]);
		}
		}
		if ( (prob->y_test[itest])*pred_value >= 0)
		correct++;
		}

		MPI_Allreduce(&correct, &global_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		accuracy_new = (double)global_correct/(double)prob->l_test;

		printf("Iteration %d Rank %d AccuracyNew %lf Time %lf Connection Time %lf Inneriter %d\n", iter, rank, accuracy_new, totaltime, connection_time, inneriter );
		}
		*/		
		// Restart calculating time
		innertimebegin_loop = omp_get_wtime();

		// Recompute G
		for (int i=mystart, j=0 ; i<myend ; i++, j++ )
		{
			double delta_alpha_i = alpha[i] - oldalpha[i];
			local_delta_alpha[j] = delta_alpha_i;
		}
		//		for ( int i=0 ; i<l ; i++ )
		//			printf("%lf,%lf,%lf,%lf   ", local_delta_alpha[i], G[i], alphag[i], alphaH[i]);
		//		printf("\n");

		printf("Rank %d time %lf finished all %d\n", rank, omp_get_wtime()-innertimebegin_loop, myend-mystart);

		double tta = omp_get_wtime();
		// Communicate G
		for ( int i=0 ; i<l ; i++ )
			G_local_delta[i] = G[i] - G_old[i];

		MPI_Allreduce(G_local_delta, G_delta, l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Communicate alpha
		//		for ( int i=mystart, j=0 ; i<myend ; i++, j++)
		//			local_delta_alpha[j] = alpha[i] - oldalpha[i];


		MPI_Allgatherv(local_delta_alpha, myend-mystart, MPI_DOUBLE, delta_alpha, num_elements, intervals, MPI_DOUBLE, MPI_COMM_WORLD);

		connection_time += (omp_get_wtime() - tta);


		double a = 0, b = 0, c=0;
		for ( int i=0 ; i<l; i++ ) {
			a += delta_alpha[i]*G_old[i];
			b += delta_alpha[i]*G_delta[i];
			c+= oldalpha[i]*G_old[i];
		}
		c*=0.5;
		double t = 1.0;
		int lineiter =0;
		int maxlineiter = 30;
		while (lineiter < maxlineiter)
		{
			double obj_local = 0.0;
			double obj_global;

			for ( int i=mystart ; i<myend ; i++ ) 
			{
				double newz = oldalpha[i]+t*delta_alpha[i];
				obj_local += newz*log(newz)+(C-newz)*log(C-newz);
			}

			MPI_Allreduce(&obj_local, &obj_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			obj_global += (a*t+0.5*t*t*b+c);
			if ( rank == 0)
				printf("Line iter %d, Obj %lf\n", lineiter, obj_global);
			if ( obj_global < nowobj )
			{
				nowobj = obj_global;
				break;
			}

			t/=2.0;
			lineiter++;
		}
		if ( lineiter == maxlineiter ){
			printf("ERROR: line search failed\n");
			break;
		}

		for ( int i=0 ; i<l ; i++ )
		{
			alpha[i] = oldalpha[i] + t*delta_alpha[i];
			alphag[i] = log(alpha[i]/(C-alpha[i]));
			alphaH[i] = C/(alpha[i])/(C-alpha[i]);
		}
		for ( int i=0 ; i<l ; i++ )
			G[i] = G_old[i] + t*G_delta[i];

		/*
		   for ( int i=0 ; i<l ; i++ ) {
		   G_local_delta[i] = 0;
		   }
		   for ( int pi = mystart ; pi<myend ; pi++)
		   {
		   const Qfloat *Q_i = Q.get_Q(pi,l);
		   for (int i=0 ; i<l ; i++ )
		   G_local_delta[i] += Q_i[i]*alpha[pi];
		   }

		// Configuration
		MPI_Barrier(MPI_COMM_WORLD);
		double ttg = omp_get_wtime();
		MPI_Allreduce(G_local_delta, G_delta, l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if ( rank == 0)
		{
		for ( int i=0 ; i<l ; i++ )
		printf("%lf=%lf ", G_delta[i], G[i]);
		printf("\n");
		}

*/

		totaltime += (omp_get_wtime() - innertimebegin_loop);
		// end of computation

		/*	
			double obj =0.0;
			for ( int i=0 ; i<l ; i++ )
			obj += alpha[i]*(G[i]+1.0);
			obj/=2.0;
			for ( int i=0 ; i<l ;i++ )
			obj -= alpha[i];
			*/	
		// Compute Testing Error (use global alpha)

		double accuracy = 0;
		if ( (iter < 4) || (iter%10==0) ) {
			MPI_Barrier(MPI_COMM_WORLD);
			correct = 0;
			global_correct = 0;
			for ( int itest = 0; itest < prob->l_test ; itest ++ ) 
				if ( prob->idx_test[itest] == rank)
				{
					double pred_value = 0.0;
					for(int i=0;i<l;i++) {
						if ( alpha[i] != 0) {
							double kvalue = Kernel::k_function(prob->x_test[itest],prob->x[i],(*param));
							pred_value += kvalue*alpha[i]*(prob->y[i]);
						}
					}
					if ( (prob->y_test[itest])*pred_value >= 0 )
						correct++;
				}

			MPI_Allreduce(&correct, &global_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			accuracy = (double)global_correct/(double)prob->l_test;
		}


		if ( (iter < 4) || (iter%10==0) ) {
			printf("Iteration %d Rank %d Obj %lf  Accuracy %lf AccuracyNew %lf Time %lf Connection Time %lf Inneriter %d\n", iter, rank, nowobj, accuracy, accuracy_new, totaltime, connection_time, inneriter );
		} else {
			printf("Iteration %d Rank %d Obj %lf Time %lf Connection Time %lf Inneriter %d\n", iter, rank, nowobj,  totaltime, connection_time, inneriter );
		}

	}

	MPI_Barrier(MPI_COMM_WORLD);

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[i] = alpha[i];
	}



	//	info("\noptimization finished");

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] oldalpha;
	//	delete[] alpha_status;
	//	delete[] active_set;
	delete[] G;
	delete[] G_old;
	delete[] G_delta;
}

int Solver_Dis::select_working_set_LR(int &out_i, int rank)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = 0;
	int Gmax_idx = -1;

	int mystart = intervals[rank];
	int myend = intervals[rank+1];
	for(int t=mystart;t<myend;t++)
	{
		double delta = fabs((G[t] +alphag[t])/(alphaH[t]+QD[t]));
		if ( delta > Gmax)
		{
			Gmax = delta;
			Gmax_idx = t;
		}
	}

	out_i = Gmax_idx;
	return 0;
}



void Solver_Dis::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_, double *alpha_, 
		double C, double eps,
		int shrinking, int maxiter, int *intervals_, int *num_elements_, 
		svm_problem *prob, const svm_parameter *param
		)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_, l);
	clone(y, y_,l);
	clone(alpha, alpha_, l);
	clone(oldalpha, alpha_, l);
	this->Cp = C;
	this->Cn = C;
	this->eps = eps;
	this->intervals = intervals_;
	this->num_elements = num_elements_;
	//	unshrink = false;

	// Create Subproblem
	int mystart = intervals[rank];
	int myend = intervals[rank+1];
	int lsub = myend - mystart;

	//	omp_set_num_threads(param->t);

	if ( rank == 0)
		printf("Using %d threads per machine\n", omp_get_max_threads());

//	printf("Rank %d size %d\n", rank, lsub);

	if (rank == 0 )
	{
		printf("Data per machine: ");
		for ( int ii=0 ; ii<size ; ii++ )
			printf("%d ", intervals[ii+1]-intervals[ii]);
		printf("\n");
	}

	//	Creat the subproblem for DC-SVM
	svm_node **xsub = Malloc(svm_node *, lsub);
	double *ysub = Malloc(double, lsub);
	for ( int i=mystart, ii=0 ; i<myend ; i++, ii++) {
		xsub[ii] = prob->x[i];
		ysub[ii] = prob->y[i];
	}
	svm_problem subprob;
	subprob.d = prob->d;
	subprob.l = lsub;
	subprob.alpha = Malloc(double, lsub);
	subprob.x = xsub;
	subprob.y = ysub; 
	for ( int i=0 ; i<lsub ; i++ )
		subprob.alpha[i] = 0;

	schar *ysub_schar = new schar[lsub];
	for ( int i=0 ; i<lsub ; i++ )
	{
		if ( ysub[i]>0 ) ysub_schar[i] = +1;
		else
			ysub_schar[i] = -1;
	}

	SVC_Q Qsub(subprob,*param,ysub_schar);


	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize gradient
	{
		G = new double[l];
		G_old = new double[l];
		G_delta = new double[l];
		G_local_delta = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_old[i] = p[i];
		}
	}

	for ( int i=0 ; i<l ; i++ ) {
		G_local_delta[i] = 0;
	}

	// Configuration
	MPI_Barrier(MPI_COMM_WORLD);
	double ttg = omp_get_wtime();
	MPI_Allreduce(G_local_delta, G_delta, l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double allreducetime = omp_get_wtime()-ttg;

	// optimization step

	double timeout = allreducetime *100;
	double firsttimeout = allreducetime *30000;
	//	int miditer =300;
	//	int miditer = 5000;
	//	int firstiter = 50000;
	int miditer = 1000;
	int firstiter = 50000;
	int active_size = l;
	//	int mystart = intervals[rank], myend = intervals[rank+1];
	delta_alpha = new double[l];
	local_delta_alpha = new double[myend - mystart];
	double timebegin = omp_get_wtime();
	double connection_time = 0.0;
	double totaltime = 0.0;

	double totaltime_reset=0.0;

	double global_epsilon=10000000;
//	printf("first iter: %d iters, %lf time\n", firstiter, firsttimeout);

	for ( int iter=0 ; (iter<maxiter) && (global_epsilon > param->eps); iter++ )
	{
		MPI_Barrier(MPI_COMM_WORLD);
		double innertimebegin_loop = omp_get_wtime();
		// backup current alpha
		for (int i=0; i<l; i++ )
			oldalpha[i] = alpha[i];
		for ( int i=0; i<l ; i++ )
			G_old[i] = G[i];
		// Solve subproblems independently
		double innertimebegin = omp_get_wtime();

		double local_epsilon = 0;
		int inneriter = 0;
		if ( (iter ==0) && (param->DC==1) )
		{
			for ( inneriter = 0; inneriter < firstiter; inneriter++ ) 
			{
				//				if ( inneriter%10==0)
				//					printf("rank %d inneriter %d time %lf\n", rank, inneriter, omp_get_wtime()-innertimebegin);
				int pi;
				select_working_set(pi, rank);
				if ( pi < 0 )
					break;
				const Qfloat *Q_i = Qsub.get_Q(pi-mystart,lsub);
				double C_i = get_C(pi);
				double old_alpha_i = alpha[pi];
				double quad_coef = QD[pi];
				if ( quad_coef <= 0)
					quad_coef = TAU;
				double delta = -G[pi]/quad_coef;
				local_epsilon = fabs(delta);
				alpha[pi] += delta;
				if (alpha[pi]<0)
					alpha[pi] = 0;
				if (alpha[pi] > C_i)
					alpha[pi] = C_i;

				double delta_alpha_i = alpha[pi] - old_alpha_i;

				for(int k=0;k<lsub;k++)
					G[k+mystart] += Q_i[k]*delta_alpha_i;

				if ( iter == 0)
				{
					if ( (omp_get_wtime() - innertimebegin > firsttimeout)  )
						break;	
				} 
			}
		}
		else
		{
			printf("Rank %d start iter %d\n", rank, iter);
			for ( inneriter = 0; inneriter < miditer ; inneriter++ )
			{
				int pi;
				select_working_set(pi, rank);
				if ( pi < 0 ) {
//					printf("Rank %d reach optimal\n", rank, pi);
					break;
				}
				const Qfloat *Q_i = Q.get_Q(pi,l);
				double C_i = get_C(pi);
				double old_alpha_i = alpha[pi];
				double quad_coef = QD[pi];
				if ( quad_coef <= 0)
					quad_coef = TAU;
				double delta = -G[pi]/quad_coef;
				local_epsilon = fabs(delta);
				alpha[pi] += delta;
				if (alpha[pi]<0)
					alpha[pi] = 0;
				if (alpha[pi] > C_i)
					alpha[pi] = C_i;

				double delta_alpha_i = alpha[pi] - old_alpha_i;

				for(int k=0;k<active_size;k++)
					G[k] += Q_i[k]*delta_alpha_i;

				if ( (omp_get_wtime() - innertimebegin > timeout)  )
					break;
			}
		}

		MPI_Allreduce(&local_epsilon, &global_epsilon, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


//		printf("Rank %d Iteration %d timeout %lf\n", rank, inneriter, timeout);
		MPI_Barrier(MPI_COMM_WORLD);
		totaltime += (omp_get_wtime() - innertimebegin_loop);
		int correct = 0;
		int global_correct = 0;
		double accuracy_new = 0;

		int is_active = 0;  // Record whether we have computed the latest accuracy
		double time_accuracy = param->p;  // print out every 10 seconds
		if ( (totaltime - totaltime_reset >=time_accuracy) || (iter==0 && param->DC==1) ) {
			is_active = 1;
			totaltime_reset = totaltime;
			double ttt = omp_get_wtime();

			accuracy_new = 0;
			
#pragma omp parallel for
			for ( int itest =0 ; itest < prob->l_test ; itest++ )
				if ( prob->idx_test[itest] == rank)
				{
					double pred_value = 0.0;
					for ( int i=0 ; i<l ; i++ ) {
						double nowalpha = 0.0;
						nowalpha = alpha[i];
						if ( nowalpha != 0) {
							double kvalue = Kernel::k_function(prob->x_test[itest],prob->x[i],(*param));
							double inc = kvalue*nowalpha*(prob->y[i]);
							pred_value += inc;
						}
					}
					if ( (prob->y_test[itest])*pred_value >= 0 ) {
#pragma omp atomic
						correct++;
					}
				}

			MPI_Allreduce(&correct, &global_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			accuracy_new = (double)global_correct/(double)prob->l_test;
		}

		if ( rank == 0 ) {
			if ( (param->DC==1) && (iter==0) ) 
				printf("Iteration %d Accuracy %lf Time %lf\n", iter, rank,  accuracy_new, totaltime);
		}


		// Restart calculating time
		innertimebegin_loop = omp_get_wtime();
		
		// Recompute G
		for (int i=mystart, j=0 ; i<myend ; i++, j++ )
		{
			double delta_alpha_i = alpha[i] - oldalpha[i];
			local_delta_alpha[j] = delta_alpha_i;
			if ( (iter == 0) && param->DC==1 ) {
				if ( delta_alpha_i != 0)
				{
					const Qfloat *Q_i = Q.get_Q(i,l);
					for(int k=0;k<mystart;k++)
						G[k] += Q_i[k]*delta_alpha_i;
					for ( int k=myend ; k<l ; k++ )
						G[k] += Q_i[k]*delta_alpha_i;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		double tta = omp_get_wtime();
		// Communicate G
		for ( int i=0 ; i<l ; i++ )
			G_local_delta[i] = G[i] - G_old[i];

		MPI_Allreduce(G_local_delta, G_delta, l, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Communicate alpha

		MPI_Allgatherv(local_delta_alpha, myend-mystart, MPI_DOUBLE, delta_alpha, num_elements, intervals, MPI_DOUBLE, MPI_COMM_WORLD);

		connection_time += (omp_get_wtime() - tta);


		double a = 0, b = 0;
		for ( int i=0 ; i<active_size; i++ ) {
			a += delta_alpha[i]*G_old[i];
			b += delta_alpha[i]*G_delta[i];
		}
		double t = -a/b; // step size
		if ( t>1)
			t=1;

//		if (rank == 0)
//			printf("step size = %lf a %lf b  %lf\n", t, a, b);

		for ( int i=0 ; i<l ; i++ )
			alpha[i] = oldalpha[i] + t*delta_alpha[i];
		for ( int i=0 ; i<l ; i++ )
			G[i] = G_old[i] + t*G_delta[i];


		totaltime += (omp_get_wtime() - innertimebegin_loop);
		// end of computation


		double obj =0.0;
		for ( int i=0 ; i<l ; i++ )
			obj += alpha[i]*(G[i]+1.0);
		obj/=2.0;
		for ( int i=0 ; i<l ;i++ )
			obj -= alpha[i];

		if ( rank == 0 ) {
			if ( is_active ) 
				printf("Iteration %d Obj %lf  Accuracy %lf Time %lf Epsilon %lf\n", iter, obj, accuracy_new, totaltime, global_epsilon);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[i] = alpha[i];
	}



	//	info("\noptimization finished");

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] oldalpha;
	delete[] alpha_status;
	//	delete[] active_set;
	delete[] G;
	delete[] G_old;
	delete[] G_delta;
}

int Solver_Dis::select_working_set(int &out_i, int rank)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmax = 0;
	int Gmax_idx = -1;

	int mystart = intervals[rank];
	int myend = intervals[rank+1];
	for(int t=mystart;t<myend;t++)
	{
		double newalpha = alpha[t] - G[t];
		if ( newalpha < 0)
			newalpha = 0;
		else if ( newalpha > Cp )
			newalpha = Cp;
		double tt = fabs(newalpha - alpha[t]);
		if ( tt > Gmax)
		{
			Gmax = tt;
			Gmax_idx = t;
		}
	}

	out_i = Gmax_idx;
	return 0;
}


		// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
		// Solves:
		//
		//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
		//
		//		y^T \alpha = \delta
		//		y_i = +1 or -1
		//		0 <= alpha_i <= Cp for y_i = 1
		//		0 <= alpha_i <= Cn for y_i = -1
		//
		// Given:
		//
		//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
		//	l is the size of vectors and matrices
		//	eps is the stopping tolerance
		//
		// solution will be put in \alpha, objective value will be put in obj
		//
		class Solver {
			public:
				Solver() {};
				virtual ~Solver() {};

				struct SolutionInfo {
					double initial_time;
					double obj;
					double rho;
					double upper_bound_p;
					double upper_bound_n;
					double r;	// for Solver_NU
				};

				void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
						double *alpha_, double Cp, double Cn, double eps,
						SolutionInfo* si, int shrinking);
			protected:
				int active_size;
				schar *y;
				double *G;		// gradient of objective function
				enum { LOWER_BOUND, UPPER_BOUND, FREE };
				char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
				double *alpha;
				const QMatrix *Q;
				const double *QD;
				double eps;
				double Cp,Cn;
				double *p;
				int *active_set;
				double *G_bar;		// gradient, if we treat free variables as 0
				int l;
				bool unshrink;	// XXX

				double get_C(int i)
				{
					return (y[i] > 0)? Cp : Cn;
				}
				void update_alpha_status(int i)
				{
					if(alpha[i] >= get_C(i))
						alpha_status[i] = UPPER_BOUND;
					else if(alpha[i] <= 0)
						alpha_status[i] = LOWER_BOUND;
					else alpha_status[i] = FREE;
				}
				bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
				bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
				bool is_free(int i) { return alpha_status[i] == FREE; }
				void swap_index(int i, int j);
				void reconstruct_gradient();
				virtual int select_working_set(int &i, int &j);
				virtual int select_working_set(int &i);
				virtual double calculate_rho();
				virtual void do_shrinking();
				virtual void do_shrinking_nobias();
			private:
				bool be_shrunk(int i, double Gmax1, double Gmax2);	
				bool be_shrunk_nobias(int i, double Gmax1);	
		};

	void Solver::swap_index(int i, int j)
	{
		Q->swap_index(i,j);
		swap(y[i],y[j]);
		swap(G[i],G[j]);
		swap(alpha_status[i],alpha_status[j]);
		swap(alpha[i],alpha[j]);
		swap(p[i],p[j]);
		swap(active_set[i],active_set[j]);
		swap(G_bar[i],G_bar[j]);
	}

	void Solver::reconstruct_gradient()
	{
		// reconstruct inactive elements of G from G_bar and free variables

		if(active_size == l) return;

		int i,j;
		int nr_free = 0;

		for(j=active_size;j<l;j++)
			G[j] = G_bar[j] + p[j];

		for(j=0;j<active_size;j++)
			if(is_free(j))
				nr_free++;

		if(2*nr_free < active_size)
			info("\nWARNING: using -h 0 may be faster\n");

		if (nr_free*l > 2*active_size*(l-active_size))
		{
			for(i=active_size;i<l;i++)
			{
				const Qfloat *Q_i = Q->get_Q(i,active_size);
				for(j=0;j<active_size;j++)
					if(is_free(j))
						G[i] += alpha[j] * Q_i[j];
			}
		}
		else
		{
			for(i=0;i<active_size;i++)
				if(is_free(i))
				{
					const Qfloat *Q_i = Q->get_Q(i,l);
					double alpha_i = alpha[i];
					for(j=active_size;j<l;j++)
						G[j] += alpha_i * Q_i[j];
				}
		}
	}

	// Modify here

	void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
			double *alpha_, double Cp, double Cn, double eps,
			SolutionInfo* si, int shrinking)
	{
		this->l = l;
		this->Q = &Q;
		QD=Q.get_QD();
		clone(p, p_,l);
		clone(y, y_,l);
		clone(alpha,alpha_,l);
		this->Cp = Cp;
		this->Cn = Cn;
		this->eps = eps;
		unshrink = false;

		double inittime = clock();

		// initialize alpha_status
		{
			alpha_status = new char[l];
			for(int i=0;i<l;i++)
				update_alpha_status(i);
		}

		// initialize active set (for shrinking)
		{
			active_set = new int[l];
			for(int i=0;i<l;i++)
				active_set[i] = i;
			active_size = l;
		}

		// initialize gradient
		{
			G = new double[l];
			G_bar = new double[l];
			int i;
			for(i=0;i<l;i++)
			{
				G[i] = p[i];
				G_bar[i] = 0;
			}
			for(i=0;i<l;i++)
				if(!is_lower_bound(i))
				{
					const Qfloat *Q_i = Q.get_Q(i,l);
					double alpha_i = alpha[i];
					int j;
					for(j=0;j<l;j++)
						G[j] += alpha_i*Q_i[j];
					if(is_upper_bound(i))
						for(j=0;j<l;j++)
							G_bar[j] += get_C(i) * Q_i[j];
				}
		}

		{
			double v = 0;
			int ii;
			for(ii=0;ii<l;ii++)
				v += alpha[ii] * (G[ii] + p[ii]);

			si->obj = v/2;
		}


		si->initial_time = (clock()-inittime)/CLOCKS_PER_SEC;
		// optimization step

		int iter = 0;
		int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
		int counter = min(l,1000)+1;

		max_iter = 300;
		printf("maxiter %d!!!\n", max_iter);
		double timebegin = omp_get_wtime();
		while(iter < max_iter)
		{
			// show progress and do shrinking

			if(--counter == 0)
			{
				counter = min(l,1000);
				if(shrinking) do_shrinking_nobias();
				info(".");
			}

			int i,j;
			if(select_working_set(i)!=0)
			{
				// reconstruct the whole gradient
				reconstruct_gradient();

				// reset active set size and check
				active_size = l;
				info("*");
				if(select_working_set(i)!=0)
					break;
				else
					counter = 1;	// do shrinking next iteration
			}

			++iter;

			// update alpha[i] and alpha[j], handle bounds carefully

			const Qfloat *Q_i = Q.get_Q(i,active_size);
			//		const Qfloat *Q_j = Q.get_Q(j,active_size);

			double C_i = get_C(i);
			//		double C_j = get_C(j);

			double old_alpha_i = alpha[i];
			//		double old_alpha_j = alpha[j];

			double quad_coef = QD[i];
			if ( quad_coef <= 0)
				quad_coef = TAU;
			double delta = -G[i]/quad_coef;
			alpha[i] += delta;
			if (alpha[i]<0)
				alpha[i] = 0;
			if (alpha[i] > C_i)
				alpha[i] = C_i;

			/*
			   if(y[i]!=y[j])
			   {
			   double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			   if (quad_coef <= 0)
			   quad_coef = TAU;
			   double delta = (-G[i]-G[j])/quad_coef;
			   double diff = alpha[i] - alpha[j];
			   alpha[i] += delta;
			   alpha[j] += delta;

			   if(diff > 0)
			   {
			   if(alpha[j] < 0)
			   {
			   alpha[j] = 0;
			   alpha[i] = diff;
			   }
			   }
			   else
			   {
			   if(alpha[i] < 0)
			   {
			   alpha[i] = 0;
			   alpha[j] = -diff;
			   }
			   }
			   if(diff > C_i - C_j)
			   {
			   if(alpha[i] > C_i)
			   {
			   alpha[i] = C_i;
			   alpha[j] = C_i - diff;
			   }
			   }
			   else
			   {
			   if(alpha[j] > C_j)
			   {
			   alpha[j] = C_j;
			   alpha[i] = C_j + diff;
			   }
			   }
			   }
			   else
			   {
			   double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			   if (quad_coef <= 0)
			   quad_coef = TAU;
			   double delta = (G[i]-G[j])/quad_coef;
			   double sum = alpha[i] + alpha[j];
			   alpha[i] -= delta;
			   alpha[j] += delta;

			   if(sum > C_i)
			   {
			   if(alpha[i] > C_i)
			   {
			   alpha[i] = C_i;
			   alpha[j] = sum - C_i;
			   }
			   }
			   else
			   {
			   if(alpha[j] < 0)
			   {
			   alpha[j] = 0;
			   alpha[i] = sum;
			   }
			   }
			   if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			   else
			   {
				   if(alpha[i] < 0)
				   {
					   alpha[i] = 0;
					   alpha[j] = sum;
				   }
			   }
		}
		*/

			// update G

			double delta_alpha_i = alpha[i] - old_alpha_i;
		//		double delta_alpha_j = alpha[j] - old_alpha_j;

		for(int k=0;k<active_size;k++)
		{
			//			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
			G[k] += Q_i[k]*delta_alpha_i;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			//			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			//			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}
			/*
			   if(uj != is_upper_bound(j))
			   {
			   Q_j = Q.get_Q(j,l);
			   if(uj)
			   for(k=0;k<l;k++)
			   G_bar[k] -= C_j * Q_j[k];
			   else
			   for(k=0;k<l;k++)
			   G_bar[k] += C_j * Q_j[k];
			   }*/
		}
		}

		if(iter >= max_iter)
		{
			if(active_size < l)
			{
				// reconstruct the whole gradient to calculate objective value
				reconstruct_gradient();
				active_size = l;
				info("*");
			}
			fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
		}

		// calculate rho

		si->rho = 0;
		//	si->rho = calculate_rho();

		// calculate objective value
		{
			double v = 0;
			int i;
			for(i=0;i<l;i++)
				v += alpha[i] * (G[i] + p[i]);
			si->obj = v/2;
		}

		// put back the solution
		{
			for(int i=0;i<l;i++)
				alpha_[active_set[i]] = alpha[i];
		}

		// juggle everything back
		/*{
		  for(int i=0;i<l;i++)
		  while(active_set[i] != i)
		  swap_index(i,active_set[i]);
		// or Q.swap_index(i,active_set[i]);
		}*/

		si->upper_bound_p = Cp;
		si->upper_bound_n = Cn;

		info("\noptimization finished, #iter = %d\n",iter);
		printf("Time: %lf\n", omp_get_wtime()-timebegin);

		delete[] p;
		delete[] y;
		delete[] alpha;
		delete[] alpha_status;
		delete[] active_set;
		delete[] G;
		delete[] G_bar;
	}

	int Solver::select_working_set(int &out_i)
	{
		// return i,j such that
		// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
		// j: minimizes the decrease of obj value
		//    (if quadratic coefficeint <= 0, replace it with tau)
		//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

		double Gmax = -INF;
		double Gmax2 = -INF;
		int Gmax_idx = -1;
		int Gmin_idx = -1;
		double obj_diff_min = INF;

		int t;
		for(t=0;t<active_size;t++)
		{
			if(!is_upper_bound(t))
			{
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
			}
			if(!is_lower_bound(t))
			{
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
			}
		}
		/*
		   int i = Gmax_idx;
		   const Qfloat *Q_i = NULL;
		   if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		   Q_i = Q->get_Q(i,active_size);

		   for(int j=0;j<active_size;j++)
		   {
		   if(y[j]==+1)
		   {
		   if (!is_lower_bound(j))
		   {
		   double grad_diff=Gmax+G[j];
		   if (G[j] >= Gmax2)
		   Gmax2 = G[j];
		   if (grad_diff > 0)
		   {
		   double obj_diff; 
		   double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
		   if (quad_coef > 0)
		   obj_diff = -(grad_diff*grad_diff)/quad_coef;
		   else
		   obj_diff = -(grad_diff*grad_diff)/TAU;

		   if (obj_diff <= obj_diff_min)
		   {
		   Gmin_idx=j;
		   obj_diff_min = obj_diff;
		   }
		   }
		   }
		   }
		   else
		   {
		   if (!is_upper_bound(j))
		   {
		   double grad_diff= Gmax-G[j];
		   if (-G[j] >= Gmax2)
		   Gmax2 = -G[j];
		   if (grad_diff > 0)
		   {
		   double obj_diff; 
		   double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
		   if (quad_coef > 0)
		   obj_diff = -(grad_diff*grad_diff)/quad_coef;
		   else
		   obj_diff = -(grad_diff*grad_diff)/TAU;

		   if (obj_diff <= obj_diff_min)
		   {
		   Gmin_idx=j;
		   obj_diff_min = obj_diff;
		   }
		   }
		   }
		   }
		   }
		   */
		//	printf("gmax: %lf eps: %lf\n", Gmax, eps);
		if(Gmax < eps)
			return 1;

		out_i = Gmax_idx;
		//	out_j = Gmin_idx;
		return 0;
	}


	// return 1 if already optimal, return 0 otherwise
	int Solver::select_working_set(int &out_i, int &out_j)
	{
		// return i,j such that
		// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
		// j: minimizes the decrease of obj value
		//    (if quadratic coefficeint <= 0, replace it with tau)
		//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

		double Gmax = -INF;
		double Gmax2 = -INF;
		int Gmax_idx = -1;
		int Gmin_idx = -1;
		double obj_diff_min = INF;

		for(int t=0;t<active_size;t++)
			if(y[t]==+1)	
			{
				if(!is_upper_bound(t))
					if(-G[t] >= Gmax)
					{
						Gmax = -G[t];
						Gmax_idx = t;
					}
			}
			else
			{
				if(!is_lower_bound(t))
					if(G[t] >= Gmax)
					{
						Gmax = G[t];
						Gmax_idx = t;
					}
			}

		int i = Gmax_idx;
		const Qfloat *Q_i = NULL;
		if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
			Q_i = Q->get_Q(i,active_size);

		for(int j=0;j<active_size;j++)
		{
			if(y[j]==+1)
			{
				if (!is_lower_bound(j))
				{
					double grad_diff=Gmax+G[j];
					if (G[j] >= Gmax2)
						Gmax2 = G[j];
					if (grad_diff > 0)
					{
						double obj_diff; 
						double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff)/TAU;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
			else
			{
				if (!is_upper_bound(j))
				{
					double grad_diff= Gmax-G[j];
					if (-G[j] >= Gmax2)
						Gmax2 = -G[j];
					if (grad_diff > 0)
					{
						double obj_diff; 
						double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff)/TAU;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
		}

		if(Gmax+Gmax2 < eps)
			return 1;

		out_i = Gmax_idx;
		out_j = Gmin_idx;
		return 0;
	}

	bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
	{
		if(is_upper_bound(i))
		{
			if(y[i]==+1)
				return(-G[i] > Gmax1);
			else
				return(-G[i] > Gmax2);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				return(G[i] > Gmax2);
			else	
				return(G[i] > Gmax1);
		}
		else
			return(false);
	}

	bool Solver::be_shrunk_nobias(int i, double Gmax1)
	{
		if(is_upper_bound(i))
		{
			return(-G[i] > Gmax1);
		}
		else if(is_lower_bound(i))
		{
			return(G[i] > Gmax1);
		}
		else
			return(false);
	}


	void Solver::do_shrinking_nobias()
	{
		int i;
		double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))	
		{
			if(-G[i] >= Gmax1)
				Gmax1 = -G[i];
		}
		if(!is_lower_bound(i))	
		{
			if(G[i] >= Gmax1)
				Gmax1 = G[i];
		}
	}

	if(unshrink == false && Gmax1 <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk_nobias(i, Gmax1))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk_nobias(active_size, Gmax1))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
	}


	void Solver::do_shrinking()
	{
		int i;
		double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
	}

	double Solver::calculate_rho()
	{
		double r;
		int nr_free = 0;
		double ub = INF, lb = -INF, sum_free = 0;
		for(int i=0;i<active_size;i++)
		{
			double yG = y[i]*G[i];

			if(is_upper_bound(i))
			{
				if(y[i]==-1)
					ub = min(ub,yG);
				else
					lb = max(lb,yG);
			}
			else if(is_lower_bound(i))
			{
				if(y[i]==+1)
					ub = min(ub,yG);
				else
					lb = max(lb,yG);
			}
			else
			{
				++nr_free;
				sum_free += yG;
			}
		}

		if(nr_free>0)
			r = sum_free/nr_free;
		else
			r = (ub+lb)/2;

		return r;
	}

	//
	// Solver for nu-svm classification and regression
	//
	// additional constraint: e^T \alpha = constant
	//
	class Solver_NU: public Solver
		{
			public:
				Solver_NU() {}
				void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
						double *alpha, double Cp, double Cn, double eps,
						SolutionInfo* si, int shrinking)
				{
					this->si = si;
					Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
				}
			private:
				SolutionInfo *si;
				int select_working_set(int &i, int &j);
				double calculate_rho();
				bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
				void do_shrinking();
		};

	// return 1 if already optimal, return 0 otherwise
	int Solver_NU::select_working_set(int &out_i, int &out_j)
	{
		// return i,j such that y_i = y_j and
		// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
		// j: minimizes the decrease of obj value
		//    (if quadratic coefficeint <= 0, replace it with tau)
		//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

		double Gmaxp = -INF;
		double Gmaxp2 = -INF;
		int Gmaxp_idx = -1;

		double Gmaxn = -INF;
		double Gmaxn2 = -INF;
		int Gmaxn_idx = -1;

		int Gmin_idx = -1;
		double obj_diff_min = INF;

		for(int t=0;t<active_size;t++)
			if(y[t]==+1)
			{
				if(!is_upper_bound(t))
					if(-G[t] >= Gmaxp)
					{
						Gmaxp = -G[t];
						Gmaxp_idx = t;
					}
			}
			else
			{
				if(!is_lower_bound(t))
					if(G[t] >= Gmaxn)
					{
						Gmaxn = G[t];
						Gmaxn_idx = t;
					}
			}

		int ip = Gmaxp_idx;
		int in = Gmaxn_idx;
		const Qfloat *Q_ip = NULL;
		const Qfloat *Q_in = NULL;
		if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
			Q_ip = Q->get_Q(ip,active_size);
		if(in != -1)
			Q_in = Q->get_Q(in,active_size);

		for(int j=0;j<active_size;j++)
		{
			if(y[j]==+1)
			{
				if (!is_lower_bound(j))	
				{
					double grad_diff=Gmaxp+G[j];
					if (G[j] >= Gmaxp2)
						Gmaxp2 = G[j];
					if (grad_diff > 0)
					{
						double obj_diff; 
						double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff)/TAU;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
			else
			{
				if (!is_upper_bound(j))
				{
					double grad_diff=Gmaxn-G[j];
					if (-G[j] >= Gmaxn2)
						Gmaxn2 = -G[j];
					if (grad_diff > 0)
					{
						double obj_diff; 
						double quad_coef = QD[in]+QD[j]-2*Q_in[j];
						if (quad_coef > 0)
							obj_diff = -(grad_diff*grad_diff)/quad_coef;
						else
							obj_diff = -(grad_diff*grad_diff)/TAU;

						if (obj_diff <= obj_diff_min)
						{
							Gmin_idx=j;
							obj_diff_min = obj_diff;
						}
					}
				}
			}
		}

		if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
			return 1;

		if (y[Gmin_idx] == +1)
			out_i = Gmaxp_idx;
		else
			out_i = Gmaxn_idx;
		out_j = Gmin_idx;

		return 0;
	}

	bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
	{
		if(is_upper_bound(i))
		{
			if(y[i]==+1)
				return(-G[i] > Gmax1);
			else	
				return(-G[i] > Gmax4);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				return(G[i] > Gmax2);
			else	
				return(G[i] > Gmax3);
		}
		else
			return(false);
	}

	void Solver_NU::do_shrinking()
	{
		double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
	}

	double Solver_NU::calculate_rho()
	{
		int nr_free1 = 0,nr_free2 = 0;
		double ub1 = INF, ub2 = INF;
		double lb1 = -INF, lb2 = -INF;
		double sum_free1 = 0, sum_free2 = 0;

		for(int i=0;i<active_size;i++)
		{
			if(y[i]==+1)
			{
				if(is_upper_bound(i))
					lb1 = max(lb1,G[i]);
				else if(is_lower_bound(i))
					ub1 = min(ub1,G[i]);
				else
				{
					++nr_free1;
					sum_free1 += G[i];
				}
			}
			else
			{
				if(is_upper_bound(i))
					lb2 = max(lb2,G[i]);
				else if(is_lower_bound(i))
					ub2 = min(ub2,G[i]);
				else
				{
					++nr_free2;
					sum_free2 += G[i];
				}
			}
		}

		double r1,r2;
		if(nr_free1 > 0)
			r1 = sum_free1/nr_free1;
		else
			r1 = (ub1+lb1)/2;

		if(nr_free2 > 0)
			r2 = sum_free2/nr_free2;
		else
			r2 = (ub2+lb2)/2;

		si->r = (r1+r2)/2;
		return (r1-r2)/2;
	}


	class ONE_CLASS_Q: public Kernel
		{
			public:
				ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
					:Kernel(prob.l, prob.x, param)
				{
					cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
					QD = new double[prob.l];
					for(int i=0;i<prob.l;i++)
						QD[i] = (this->*kernel_function)(i,i);
				}

				Qfloat *get_Q(int i, int len) const
				{
					Qfloat *data;
					int start, j;
					if((start = cache->get_data(i,&data,len)) < len)
					{
						for(j=start;j<len;j++)
							data[j] = (Qfloat)(this->*kernel_function)(i,j);
					}
					return data;
				}

				double *get_QD() const
				{
					return QD;
				}

				void swap_index(int i, int j) const
				{
					cache->swap_index(i,j);
					Kernel::swap_index(i,j);
					swap(QD[i],QD[j]);
				}

				~ONE_CLASS_Q()
				{
					delete cache;
					delete[] QD;
				}
			private:
				Cache *cache;
				double *QD;
		};

	class SVR_Q: public Kernel
		{ 
			public:
				SVR_Q(const svm_problem& prob, const svm_parameter& param)
					:Kernel(prob.l, prob.x, param)
				{
					l = prob.l;
					cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
					QD = new double[2*l];
					sign = new schar[2*l];
					index = new int[2*l];
					for(int k=0;k<l;k++)
					{
						sign[k] = 1;
						sign[k+l] = -1;
						index[k] = k;
						index[k+l] = k;
						QD[k] = (this->*kernel_function)(k,k);
						QD[k+l] = QD[k];
					}
					buffer[0] = new Qfloat[2*l];
					buffer[1] = new Qfloat[2*l];
					next_buffer = 0;
				}

				void swap_index(int i, int j) const
				{
					swap(sign[i],sign[j]);
					swap(index[i],index[j]);
					swap(QD[i],QD[j]);
				}

				Qfloat *get_Q(int i, int len) const
				{
					Qfloat *data;
					int j, real_i = index[i];
					if(cache->get_data(real_i,&data,l) < l)
					{
						for(j=0;j<l;j++)
							data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
					}

					// reorder and copy
					Qfloat *buf = buffer[next_buffer];
					next_buffer = 1 - next_buffer;
					schar si = sign[i];
					for(j=0;j<len;j++)
						buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
					return buf;
				}

				double *get_QD() const
				{
					return QD;
				}

				~SVR_Q()
				{
					delete cache;
					delete[] sign;
					delete[] index;
					delete[] buffer[0];
					delete[] buffer[1];
					delete[] QD;
				}
			private:
				int l;
				Cache *cache;
				schar *sign;
				int *index;
				mutable int next_buffer;
				Qfloat *buffer[2];
				double *QD;
		};

	//
	// construct and solve various formulations
	//
	static void solve_c_svc(
			const svm_problem *prob, const svm_parameter* param,
			double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
	{
		int l = prob->l;
		double *minus_ones = new double[l];
		schar *y = new schar[l];

		int i;


		for(i=0;i<l;i++)
		{
			if (prob->isalpha == 0)
				alpha[i] = 0;
			else
				alpha[i] = prob->alpha[i];
			minus_ones[i] = -1;
			if ( prob->ispp != 0)
				minus_ones[i] += prob->pp[i];
			if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
		}

		Solver s;
		s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
				alpha, Cp, Cn, param->eps, si, param->shrinking);

		double sum_alpha=0;
		for(i=0;i<l;i++)
			sum_alpha += alpha[i];

		if (Cp==Cn)
			info("nu = %f\n", sum_alpha/(Cp*prob->l));

		for(i=0;i<l;i++)
			alpha[i] *= y[i];

		delete[] minus_ones;
		delete[] y;
	}

	static void solve_nu_svc(
			const svm_problem *prob, const svm_parameter *param,
			double *alpha, Solver::SolutionInfo* si)
	{
		int i;
		int l = prob->l;
		double nu = param->nu;

		schar *y = new schar[l];

		for(i=0;i<l;i++)
			if(prob->y[i]>0)
				y[i] = +1;
			else
				y[i] = -1;

		double sum_pos = nu*l/2;
		double sum_neg = nu*l/2;

		for(i=0;i<l;i++)
			if(y[i] == +1)
			{
				alpha[i] = min(1.0,sum_pos);
				sum_pos -= alpha[i];
			}
			else
			{
				alpha[i] = min(1.0,sum_neg);
				sum_neg -= alpha[i];
			}

		double *zeros = new double[l];

		for(i=0;i<l;i++)
			zeros[i] = 0;

		Solver_NU s;
		s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
				alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
		double r = si->r;

		info("C = %f\n",1/r);

		for(i=0;i<l;i++)
			alpha[i] *= y[i]/r;

		si->rho /= r;
		si->obj /= (r*r);
		si->upper_bound_p = 1/r;
		si->upper_bound_n = 1/r;

		delete[] y;
		delete[] zeros;
	}

	static void solve_one_class(
			const svm_problem *prob, const svm_parameter *param,
			double *alpha, Solver::SolutionInfo* si)
	{
		int l = prob->l;
		double *zeros = new double[l];
		schar *ones = new schar[l];
		int i;

		int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

		for(i=0;i<n;i++)
			alpha[i] = 1;
		if(n<prob->l)
			alpha[n] = param->nu * prob->l - n;
		for(i=n+1;i<l;i++)
			alpha[i] = 0;

		for(i=0;i<l;i++)
		{
			zeros[i] = 0;
			ones[i] = 1;
		}

		Solver s;
		s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
				alpha, 1.0, 1.0, param->eps, si, param->shrinking);

		delete[] zeros;
		delete[] ones;
	}

	static void solve_epsilon_svr(
			const svm_problem *prob, const svm_parameter *param,
			double *alpha, Solver::SolutionInfo* si)
	{
		int l = prob->l;
		double *alpha2 = new double[2*l];
		double *linear_term = new double[2*l];
		schar *y = new schar[2*l];
		int i;

		for(i=0;i<l;i++)
		{
			alpha2[i] = 0;
			linear_term[i] = param->p - prob->y[i];
			y[i] = 1;

			alpha2[i+l] = 0;
			linear_term[i+l] = param->p + prob->y[i];
			y[i+l] = -1;
		}

		Solver s;
		s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
				alpha2, param->C, param->C, param->eps, si, param->shrinking);

		double sum_alpha = 0;
		for(i=0;i<l;i++)
		{
			alpha[i] = alpha2[i] - alpha2[i+l];
			sum_alpha += fabs(alpha[i]);
		}
		info("nu = %f\n",sum_alpha/(param->C*l));

		delete[] alpha2;
		delete[] linear_term;
		delete[] y;
	}

	static void solve_nu_svr(
			const svm_problem *prob, const svm_parameter *param,
			double *alpha, Solver::SolutionInfo* si)
	{
		int l = prob->l;
		double C = param->C;
		double *alpha2 = new double[2*l];
		double *linear_term = new double[2*l];
		schar *y = new schar[2*l];
		int i;

		double sum = C * param->nu * l / 2;
		for(i=0;i<l;i++)
		{
			alpha2[i] = alpha2[i+l] = min(sum,C);
			sum -= alpha2[i];

			linear_term[i] = - prob->y[i];
			y[i] = 1;

			linear_term[i+l] = prob->y[i];
			y[i+l] = -1;
		}

		Solver_NU s;
		s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
				alpha2, C, C, param->eps, si, param->shrinking);

		info("epsilon = %f\n",-si->r);

		for(i=0;i<l;i++)
			alpha[i] = alpha2[i] - alpha2[i+l];

		delete[] alpha2;
		delete[] linear_term;
		delete[] y;
	}

	//
	// decision_function
	//
	struct decision_function
	{
		double *alpha;
		double rho;
		double obj;
		double initial_time;
	};

	static decision_function svm_train_one(
			const svm_problem *prob, const svm_parameter *param,
			double Cp, double Cn)
	{
		double *alpha = Malloc(double,prob->l);
		Solver::SolutionInfo si;
		switch(param->svm_type)
		{
			case C_SVC:
				solve_c_svc(prob,param,alpha,&si,Cp,Cn);
				break;
			case NU_SVC:
				solve_nu_svc(prob,param,alpha,&si);
				break;
			case ONE_CLASS:
				solve_one_class(prob,param,alpha,&si);
				break;
			case EPSILON_SVR:
				solve_epsilon_svr(prob,param,alpha,&si);
				break;
			case NU_SVR:
				solve_nu_svr(prob,param,alpha,&si);
				break;
		}

		info("obj = %f, rho = %f\n",si.obj,si.rho);

		// output SVs

		int nSV = 0;
		int nBSV = 0;
		for(int i=0;i<prob->l;i++)
		{
			if(fabs(alpha[i]) > 0)
			{
				++nSV;
				if(prob->y[i] > 0)
				{
					if(fabs(alpha[i]) >= si.upper_bound_p)
						++nBSV;
				}
				else
				{
					if(fabs(alpha[i]) >= si.upper_bound_n)
						++nBSV;
				}
			}
		}

		info("nSV = %d, nBSV = %d\n",nSV,nBSV);

		decision_function f;
		f.alpha = alpha;
		f.rho = si.rho;
		f.obj = si.obj;
		f.initial_time = si.initial_time;
		return f;
	}

	// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
	static void sigmoid_train(
			int l, const double *dec_values, const double *labels, 
			double& A, double& B)
	{
		double prior1=0, prior0 = 0;
		int i;

		for (i=0;i<l;i++)
			if (labels[i] > 0) prior1+=1;
			else prior0+=1;

		int max_iter=100;	// Maximal number of iterations
		double min_step=1e-10;	// Minimal step taken in line search
		double sigma=1e-12;	// For numerically strict PD of Hessian
		double eps=1e-5;
		double hiTarget=(prior1+1.0)/(prior1+2.0);
		double loTarget=1/(prior0+2.0);
		double *t=Malloc(double,l);
		double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
		double newA,newB,newf,d1,d2;
		int iter; 

		// Initial Point and Initial Fun Value
		A=0.0; B=log((prior0+1.0)/(prior1+1.0));
		double fval = 0.0;

		for (i=0;i<l;i++)
		{
			if (labels[i]>0) t[i]=hiTarget;
			else t[i]=loTarget;
			fApB = dec_values[i]*A+B;
			if (fApB>=0)
				fval += t[i]*fApB + log(1+exp(-fApB));
			else
				fval += (t[i] - 1)*fApB +log(1+exp(fApB));
		}
		for (iter=0;iter<max_iter;iter++)
		{
			// Update Gradient and Hessian (use H' = H + sigma I)
			h11=sigma; // numerically ensures strict PD
			h22=sigma;
			h21=0.0;g1=0.0;g2=0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*A+B;
				if (fApB >= 0)
				{
					p=exp(-fApB)/(1.0+exp(-fApB));
					q=1.0/(1.0+exp(-fApB));
				}
				else
				{
					p=1.0/(1.0+exp(fApB));
					q=exp(fApB)/(1.0+exp(fApB));
				}
				d2=p*q;
				h11+=dec_values[i]*dec_values[i]*d2;
				h22+=d2;
				h21+=dec_values[i]*d2;
				d1=t[i]-p;
				g1+=dec_values[i]*d1;
				g2+=d1;
			}

			// Stopping Criteria
			if (fabs(g1)<eps && fabs(g2)<eps)
				break;

			// Finding Newton direction: -inv(H') * g
			det=h11*h22-h21*h21;
			dA=-(h22*g1 - h21 * g2) / det;
			dB=-(-h21*g1+ h11 * g2) / det;
			gd=g1*dA+g2*dB;


			stepsize = 1;		// Line Search
			while (stepsize >= min_step)
			{
				newA = A + stepsize * dA;
				newB = B + stepsize * dB;

				// New function value
				newf = 0.0;
				for (i=0;i<l;i++)
				{
					fApB = dec_values[i]*newA+newB;
					if (fApB >= 0)
						newf += t[i]*fApB + log(1+exp(-fApB));
					else
						newf += (t[i] - 1)*fApB +log(1+exp(fApB));
				}
				// Check sufficient decrease
				if (newf<fval+0.0001*stepsize*gd)
				{
					A=newA;B=newB;fval=newf;
					break;
				}
				else
					stepsize = stepsize / 2.0;
			}

			if (stepsize < min_step)
			{
				info("Line search fails in two-class probability estimates\n");
				break;
			}
		}

		if (iter>=max_iter)
			info("Reaching maximal iterations in two-class probability estimates\n");
		free(t);
	}

	static double sigmoid_predict(double decision_value, double A, double B)
	{
		double fApB = decision_value*A+B;
		// 1-p used later; avoid catastrophic cancellation
		if (fApB >= 0)
			return exp(-fApB)/(1.0+exp(-fApB));
		else
			return 1.0/(1+exp(fApB)) ;
	}

	// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
	static void multiclass_probability(int k, double **r, double *p)
	{
		int t,j;
		int iter = 0, max_iter=max(100,k);
		double **Q=Malloc(double *,k);
		double *Qp=Malloc(double,k);
		double pQp, eps=0.005/k;

		for (t=0;t<k;t++)
		{
			p[t]=1.0/k;  // Valid if k = 1
			Q[t]=Malloc(double,k);
			Q[t][t]=0;
			for (j=0;j<t;j++)
			{
				Q[t][t]+=r[j][t]*r[j][t];
				Q[t][j]=Q[j][t];
			}
			for (j=t+1;j<k;j++)
			{
				Q[t][t]+=r[j][t]*r[j][t];
				Q[t][j]=-r[j][t]*r[t][j];
			}
		}
		for (iter=0;iter<max_iter;iter++)
		{
			// stopping condition, recalculate QP,pQP for numerical accuracy
			pQp=0;
			for (t=0;t<k;t++)
			{
				Qp[t]=0;
				for (j=0;j<k;j++)
					Qp[t]+=Q[t][j]*p[j];
				pQp+=p[t]*Qp[t];
			}
			double max_error=0;
			for (t=0;t<k;t++)
			{
				double error=fabs(Qp[t]-pQp);
				if (error>max_error)
					max_error=error;
			}
			if (max_error<eps) break;

			for (t=0;t<k;t++)
			{
				double diff=(-Qp[t]+pQp)/Q[t][t];
				p[t]+=diff;
				pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
				for (j=0;j<k;j++)
				{
					Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
					p[j]/=(1+diff);
				}
			}
		}
		if (iter>=max_iter)
			info("Exceeds max_iter in multiclass_prob\n");
		for(t=0;t<k;t++) free(Q[t]);
		free(Q);
		free(Qp);
	}

	// Cross-validation decision values for probability estimates
	static void svm_binary_svc_probability(
			const svm_problem *prob, const svm_parameter *param,
			double Cp, double Cn, double& probA, double& probB)
	{
		int i;
		int nr_fold = 5;
		int *perm = Malloc(int,prob->l);
		double *dec_values = Malloc(double,prob->l);

		// random shuffle
		for(i=0;i<prob->l;i++) perm[i]=i;
		for(i=0;i<prob->l;i++)
		{
			int j = i+rand()%(prob->l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<nr_fold;i++)
		{
			int begin = i*prob->l/nr_fold;
			int end = (i+1)*prob->l/nr_fold;
			int j,k;
			struct svm_problem subprob;

			subprob.l = prob->l-(end-begin);
			subprob.x = Malloc(struct svm_node*,subprob.l);
			subprob.y = Malloc(double,subprob.l);

			if ( prob->isalpha == 0)
				subprob.isalpha = 0;
			else
			{
				subprob.isalpha = 1;
				subprob.alpha = Malloc(double, subprob.l);
			}

			k=0;
			for(j=0;j<begin;j++)
			{
				subprob.x[k] = prob->x[perm[j]];
				subprob.y[k] = prob->y[perm[j]];
				++k;
			}
			for(j=end;j<prob->l;j++)
			{
				subprob.x[k] = prob->x[perm[j]];
				subprob.y[k] = prob->y[perm[j]];
				if (subprob.isalpha == 1)
					subprob.alpha[k] = prob->alpha[perm[j]];
				++k;
			}
			int p_count=0,n_count=0;
			for(j=0;j<k;j++)
				if(subprob.y[j]>0)
					p_count++;
				else
					n_count++;

			if(p_count==0 && n_count==0)
				for(j=begin;j<end;j++)
					dec_values[perm[j]] = 0;
			else if(p_count > 0 && n_count == 0)
				for(j=begin;j<end;j++)
					dec_values[perm[j]] = 1;
			else if(p_count == 0 && n_count > 0)
				for(j=begin;j<end;j++)
					dec_values[perm[j]] = -1;
			else
			{
				svm_parameter subparam = *param;
				subparam.probability=0;
				subparam.C=1.0;
				subparam.nr_weight=2;
				subparam.weight_label = Malloc(int,2);
				subparam.weight = Malloc(double,2);
				subparam.weight_label[0]=+1;
				subparam.weight_label[1]=-1;
				subparam.weight[0]=Cp;
				subparam.weight[1]=Cn;
				struct svm_model *submodel = svm_train(&subprob,&subparam);
				for(j=begin;j<end;j++)
				{
					svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]])); 
					// ensure +1 -1 order; reason not using CV subroutine
					dec_values[perm[j]] *= submodel->label[0];
				}		
				svm_free_and_destroy_model(&submodel);
				svm_destroy_param(&subparam);
			}
			free(subprob.x);
			free(subprob.y);
		}		
		sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
		free(dec_values);
		free(perm);
	}

	// Return parameter of a Laplace distribution 
	static double svm_svr_probability(
			const svm_problem *prob, const svm_parameter *param)
	{
		int i;
		int nr_fold = 5;
		double *ymv = Malloc(double,prob->l);
		double mae = 0;

		svm_parameter newparam = *param;
		newparam.probability = 0;
		svm_cross_validation(prob,&newparam,nr_fold,ymv);
		for(i=0;i<prob->l;i++)
		{
			ymv[i]=prob->y[i]-ymv[i];
			mae += fabs(ymv[i]);
		}		
		mae /= prob->l;
		double std=sqrt(2*mae*mae);
		int count=0;
		mae=0;
		for(i=0;i<prob->l;i++)
			if (fabs(ymv[i]) > 5*std) 
				count=count+1;
			else 
				mae+=fabs(ymv[i]);
		mae /= (prob->l-count);
		info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
		free(ymv);
		return mae;
	}


	// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
	// perm, length l, must be allocated before calling this subroutine
	static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);
		int *data_label = Malloc(int,l);	
		int i;

		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
			{
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			}
			data_label[i] = j;
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+count[i-1];
		for(i=0;i<l;i++)
		{
			perm[start[data_label[i]]] = i;
			++start[data_label[i]];
		}
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+count[i-1];

		*nr_class_ret = nr_class;
		*label_ret = label;
		*start_ret = start;
		*count_ret = count;
		free(data_label);
	}

	//svm_model *svm_train_distributed(const svm_problem *prob, const svm_parameter *param, int k, int *idx)
	svm_model *svm_train_distributed(svm_problem *prob, const svm_parameter *param, int k, int *idx)
	{
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		svm_model *model = Malloc(svm_model,1);
		model->param = *param;
		model->free_sv = 0;	// XXX

		int l = prob->l;
		int nr_class=2;
		// y can only be 1 or -1

		// reorder	
		int *perm = Malloc(int,l);
		int *invperm = Malloc(int, l);
		int *intervals = (int *)malloc(sizeof(int)*(size+1));
		int *num_elements = (int *)malloc(sizeof(int)*size);

		for ( int i=0 ; i<size ; i++ )
			num_elements[i] = 0;
		for ( int i=0 ; i<l ; i++ )
			num_elements[idx[i]]++;
		intervals[0] = 0;
		for ( int i=1 ; i<=size ; i++ )
			intervals[i] = intervals[i-1] + num_elements[i-1];
		for ( int i=0 ; i<l ; i++ )
		{
			int nowidx = idx[i];
			invperm[i] = intervals[nowidx];
			perm[intervals[nowidx]] = i;
			intervals[nowidx]++;
		}
		for ( int i=0 ; i<size ; i++ )
			intervals[i] -= num_elements[i];
		/*
		   if ( rank == 0)
		   {
		   for ( int i=0 ; i<=size ; i++ )
		   printf("%d:%d ", i, intervals[i]);
		   printf("\n");
		   printf("idx: ");
		   for ( int i=0 ; i<l ; i++ )
		   printf("%d:%d ", i, idx[i]);
		   printf("\n");
		   printf("perm: ");
		   for ( int i=0 ; i<l ; i++ )
		   printf("%d:%d ", i, perm[i]);
		   printf("\n");
		   }
		   */
		MPI_Barrier(MPI_COMM_WORLD);

		svm_node **xnew = Malloc(svm_node *, l);
		for( int i=0;i<l;i++)
			xnew[i] = prob->x[perm[i]];
		double *ynew = Malloc(double, l);
		for ( int i=0 ; i<l ; i++ )
			ynew[i] = prob->y[perm[i]];
		svm_node **xtmp = prob->x;
		double *ytmp = prob->y;
		prob->x = xnew;
		prob->y = ynew;

		// Pass into solver
		double *alpha = Malloc(double,prob->l);

		double *minus_ones = new double[l];
		schar *y = new schar[l];

		for(int i=0;i<l;i++)
		{
			alpha[i] = 0;
			minus_ones[i] = -1;
			if(prob->y[i] > 0) y[i] = +1; else y[i] = -1; // allow 0/1 or -1/1
		}
		for ( int i=0 ; i<prob->l_test ; i++ )
			if ( prob->y_test[i] > 0) prob->y_test[i] = +1; else prob->y_test[i] = -1;

		if (rank==0) {
			printf("Parameter setting: C=%lf   gamma=%lf   ", param->C, param->gamma);
			if ( param->israndom == 0)
				printf("kmeans_clustering   ");
			else
				printf("random_partition   ");
			if ( param->DC==0)
				printf("no-Divide-and-conquer\n");
			else 
				printf("Divide-and-conquer\n");
		}

		Solver_Dis s;
		if ( param->function_type == 0 ) {
			if ( param->solver_type==1) {
				s.Solve_SGD(l, SVC_Q(*prob,*param,y), minus_ones, y,
						alpha, param->C, param->eps, 0, param->maxiter, intervals, num_elements, prob, param);
			} else {
				s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
						alpha, param->C, param->eps, 0, param->maxiter, intervals, num_elements, prob, param);
			}
		} else {
			if ( param->solver_type==1) {
				s.Solve_LR_SGD(l, SVC_Q(*prob,*param,y), minus_ones, y,
						alpha, param->C, param->eps, 0, param->maxiter, intervals, num_elements, prob, param);
			} else {
				s.Solve_LR(l, SVC_Q(*prob,*param,y), minus_ones, y,
						alpha, param->C, param->eps, 0, param->maxiter, intervals, num_elements, prob, param);
			}
		}

		//	if ( rank == 0 )
		{
			//	svm_model *model = Malloc(svm_model,1);
			model->param = *param;
			model->free_sv = 0;	// XXX


			model->nr_class = nr_class;

			model->label = Malloc(int,nr_class);
			model->label[0] = 1;
			model->label[1] = -1;
			//	for(i=0;i<nr_class;i++)
			//		model->label[i] = label[i];

			model->rho = Malloc(double,nr_class*(nr_class-1)/2);
			model->rho[0] = 0;
			//		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			//			model->rho[i] = f[i].rho;
			model->probA=NULL;
			model->probB=NULL;

			int total_sv = 0;
			int *nz_count = Malloc(int,nr_class);
			model->nSV = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
			{
				int nSV = 0;
				int nowy = 1;
				if ( i==1)
					nowy = -1;
				for ( int j=0 ; j<l ; j++ )
					if ( ynew[j] == nowy)
						if ( alpha[j] != 0)
						{
							++nSV;
							++total_sv;
						}
				/*		for(int j=0;j<count[i];j++)
						if(nonzero[start[i]+j])
						{	
						++nSV;
						++total_sv;
						}*/
				model->nSV[i] = nSV;
				nz_count[i] = nSV;
			}

			info("Total nSV = %d\n",total_sv);

			model->l = total_sv;
			model->SV = Malloc(svm_node *,total_sv);
			model->sv_indices = Malloc(int,total_sv);
			int p = 0;
			for(int i=0;i<l;i++)
				if(alpha[invperm[i]]!=0)
				{
					model->SV[p] = xtmp[i];
					//			model->SV[p] = xnew[i];
					model->sv_indices[p++] = perm[i] + 1;
				}

			/*	int *nz_start = Malloc(int,nr_class);
				nz_start[0] = 0;
				for(i=1;i<nr_class;i++)
				nz_start[i] = nz_start[i-1]+nz_count[i-1];
				*/
			model->sv_coef = Malloc(double *,nr_class-1);
			for(int i=0;i<nr_class-1;i++)
				model->sv_coef[i] = Malloc(double,total_sv);

			p = 0;
			for ( int jj=0 ; jj<l ; jj++ )
			{
				int j = invperm[jj];
				if (alpha[j] != 0)
					model->sv_coef[0][p++] = alpha[j]*ynew[j];
			}
			/*		for(i=0;i<nr_class;i++)
					for(int j=i+1;j<nr_class;j++)
					{
			// classifier (i,j): coefficients with
			// i are in sv_coef[j-1][nz_start[i]...],
			// j are in sv_coef[i][nz_start[j]...]

			int si = start[i];
			int sj = start[j];
			int ci = count[i];
			int cj = count[j];

			int q = nz_start[i];
			int k;
			for(k=0;k<ci;k++)
			if(nonzero[si+k])
			model->sv_coef[j-1][q++] = f[p].alpha[k];
			q = nz_start[j];
			for(k=0;k<cj;k++)
			if(nonzero[sj+k])
			model->sv_coef[i][q++] = f[p].alpha[ci+k];
			++p;
			}

			free(label);
			free(probA);
			free(probB);
			free(count);
			free(perm);
			free(start);
			free(x);
			free(weighted_C);
			free(nonzero);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
			free(f);
			free(nz_count);
			free(nz_start);
			*/
		}
		free(perm);

		return model;
	}


	//
	// Interface functions
	//
	svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
	{
		svm_model *model = Malloc(svm_model,1);
		model->param = *param;
		model->free_sv = 0;	// XXX

		if(param->svm_type == ONE_CLASS ||
				param->svm_type == EPSILON_SVR ||
				param->svm_type == NU_SVR)
		{
			// regression or one-class-svm
			model->nr_class = 2;
			model->label = NULL;
			model->nSV = NULL;
			model->probA = NULL; model->probB = NULL;
			model->sv_coef = Malloc(double *,1);

			if(param->probability && 
					(param->svm_type == EPSILON_SVR ||
					 param->svm_type == NU_SVR))
			{
				model->probA = Malloc(double,1);
				model->probA[0] = svm_svr_probability(prob,param);
			}

			decision_function f = svm_train_one(prob,param,0,0);
			model->rho = Malloc(double,1);
			model->rho[0] = f.rho;

			int nSV = 0;
			int i;
			for(i=0;i<prob->l;i++)
				if(fabs(f.alpha[i]) > 0) ++nSV;
			model->l = nSV;
			model->SV = Malloc(svm_node *,nSV);
			model->sv_coef[0] = Malloc(double,nSV);
			model->sv_indices = Malloc(int,nSV);
			int j = 0;
			for(i=0;i<prob->l;i++)
				if(fabs(f.alpha[i]) > 0)
				{
					model->SV[j] = prob->x[i];
					model->sv_coef[0][j] = f.alpha[i];
					model->sv_indices[j] = i+1;
					++j;
				}		

			free(f.alpha);
		}
		else
		{
			// classification
			int l = prob->l;
			int nr_class;
			int *label = NULL;
			int *start = NULL;
			int *count = NULL;
			int *perm = Malloc(int,l);

			// group training data of the same class
			svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
			if(nr_class == 1) 
				info("WARNING: training data in only one class. See README for details.\n");

			svm_node **x = Malloc(svm_node *,l);
			int i;
			for(i=0;i<l;i++)
				x[i] = prob->x[perm[i]];
			double *alpha;
			if ( prob->isalpha == 1)
			{
				alpha = Malloc(double, l);
				for ( i=0 ; i<l ; i++ )
					alpha[i] = prob->alpha[perm[i]];
			}
			double *pp;
			if ( prob->ispp == 1)
			{
				pp = Malloc(double, l);
				for ( i=0 ; i<l ; i++ )
					pp[i] = prob->pp[perm[i]];
			}

			// calculate weighted C

			double *weighted_C = Malloc(double, nr_class);
			for(i=0;i<nr_class;i++)
				weighted_C[i] = param->C;
			for(i=0;i<param->nr_weight;i++)
			{	
				int j;
				for(j=0;j<nr_class;j++)
					if(param->weight_label[i] == label[j])
						break;
				if(j == nr_class)
					fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
				else
					weighted_C[j] *= param->weight[i];
			}

			// train k*(k-1)/2 models

			bool *nonzero = Malloc(bool,l);
			for(i=0;i<l;i++)
				nonzero[i] = false;
			decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

			double *probA=NULL,*probB=NULL;
			if (param->probability)
			{
				probA=Malloc(double,nr_class*(nr_class-1)/2);
				probB=Malloc(double,nr_class*(nr_class-1)/2);
			}

			int p = 0;
			for(i=0;i<nr_class;i++)
				for(int j=i+1;j<nr_class;j++)
				{
					svm_problem sub_prob;
					int si = start[i], sj = start[j];
					int ci = count[i], cj = count[j];
					sub_prob.l = ci+cj;
					sub_prob.x = Malloc(svm_node *,sub_prob.l);
					sub_prob.y = Malloc(double,sub_prob.l);
					if (prob->isalpha == 0)
						sub_prob.isalpha = 0;
					else
					{
						sub_prob.isalpha = 1;
						sub_prob.alpha = Malloc(double, sub_prob.l);
					}
					if (prob->ispp == 0)
						sub_prob.ispp = 0;
					else
					{
						sub_prob.ispp = 1;
						sub_prob.pp = Malloc(double, sub_prob.l);
					}

					int k;
					for(k=0;k<ci;k++)
					{
						sub_prob.x[k] = x[si+k];
						sub_prob.y[k] = +1;
						if ( sub_prob.isalpha == 1)
							sub_prob.alpha[k] = alpha[si+k];
						if ( sub_prob.ispp == 1)
							sub_prob.pp[k] = pp[si+k];
					}
					for(k=0;k<cj;k++)
					{
						sub_prob.x[ci+k] = x[sj+k];
						sub_prob.y[ci+k] = -1;
						if ( sub_prob.isalpha == 1)
							sub_prob.alpha[ci+k] = alpha[sj+k];
						if ( sub_prob.ispp == 1)
							sub_prob.pp[ci+k] = pp[sj+k];
					}

					if(param->probability)
						svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

					f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
					model->obj = f[p].obj;
					model->initial_time = f[p].initial_time;
					for(k=0;k<ci;k++)
						if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
							nonzero[si+k] = true;
					for(k=0;k<cj;k++)
						if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
							nonzero[sj+k] = true;
					free(sub_prob.x);
					free(sub_prob.y);
					++p;
				}

			// build output

			model->nr_class = nr_class;

			model->label = Malloc(int,nr_class);
			for(i=0;i<nr_class;i++)
				model->label[i] = label[i];

			model->rho = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
				model->rho[i] = f[i].rho;

			if(param->probability)
			{
				model->probA = Malloc(double,nr_class*(nr_class-1)/2);
				model->probB = Malloc(double,nr_class*(nr_class-1)/2);
				for(i=0;i<nr_class*(nr_class-1)/2;i++)
				{
					model->probA[i] = probA[i];
					model->probB[i] = probB[i];
				}
			}
			else
			{
				model->probA=NULL;
				model->probB=NULL;
			}

			int total_sv = 0;
			int *nz_count = Malloc(int,nr_class);
			model->nSV = Malloc(int,nr_class);
			for(i=0;i<nr_class;i++)
			{
				int nSV = 0;
				for(int j=0;j<count[i];j++)
					if(nonzero[start[i]+j])
					{	
						++nSV;
						++total_sv;
					}
				model->nSV[i] = nSV;
				nz_count[i] = nSV;
			}

			info("Total nSV = %d\n",total_sv);

			model->l = total_sv;
			model->SV = Malloc(svm_node *,total_sv);
			model->sv_indices = Malloc(int,total_sv);
			p = 0;
			for(i=0;i<l;i++)
				if(nonzero[i])
				{
					model->SV[p] = x[i];
					model->sv_indices[p++] = perm[i] + 1;
				}

			int *nz_start = Malloc(int,nr_class);
			nz_start[0] = 0;
			for(i=1;i<nr_class;i++)
				nz_start[i] = nz_start[i-1]+nz_count[i-1];

			model->sv_coef = Malloc(double *,nr_class-1);
			for(i=0;i<nr_class-1;i++)
				model->sv_coef[i] = Malloc(double,total_sv);

			p = 0;
			for(i=0;i<nr_class;i++)
				for(int j=i+1;j<nr_class;j++)
				{
					// classifier (i,j): coefficients with
					// i are in sv_coef[j-1][nz_start[i]...],
					// j are in sv_coef[i][nz_start[j]...]

					int si = start[i];
					int sj = start[j];
					int ci = count[i];
					int cj = count[j];

					int q = nz_start[i];
					int k;
					for(k=0;k<ci;k++)
						if(nonzero[si+k])
							model->sv_coef[j-1][q++] = f[p].alpha[k];
					q = nz_start[j];
					for(k=0;k<cj;k++)
						if(nonzero[sj+k])
							model->sv_coef[i][q++] = f[p].alpha[ci+k];
					++p;
				}

			free(label);
			free(probA);
			free(probB);
			free(count);
			free(perm);
			free(start);
			free(x);
			free(weighted_C);
			free(nonzero);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
				free(f[i].alpha);
			free(f);
			free(nz_count);
			free(nz_start);
		}
		return model;
	}

	// Stratified cross validation
	void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
	{
		int i;
		int *fold_start = Malloc(int,nr_fold+1);
		int l = prob->l;
		int *perm = Malloc(int,l);
		int nr_class;

		// stratified cv may not give leave-one-out rate
		// Each class to l folds -> some folds may have zero elements
		if((param->svm_type == C_SVC ||
					param->svm_type == NU_SVC) && nr_fold < l)
		{
			int *start = NULL;
			int *label = NULL;
			int *count = NULL;
			svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

			// random shuffle and then data grouped by fold using the array perm
			int *fold_count = Malloc(int,nr_fold);
			int c;
			int *index = Malloc(int,l);
			for(i=0;i<l;i++)
				index[i]=perm[i];
			for (c=0; c<nr_class; c++) 
				for(i=0;i<count[c];i++)
				{
					int j = i+rand()%(count[c]-i);
					swap(index[start[c]+j],index[start[c]+i]);
				}
			for(i=0;i<nr_fold;i++)
			{
				fold_count[i] = 0;
				for (c=0; c<nr_class;c++)
					fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
			}
			fold_start[0]=0;
			for (i=1;i<=nr_fold;i++)
				fold_start[i] = fold_start[i-1]+fold_count[i-1];
			for (c=0; c<nr_class;c++)
				for(i=0;i<nr_fold;i++)
				{
					int begin = start[c]+i*count[c]/nr_fold;
					int end = start[c]+(i+1)*count[c]/nr_fold;
					for(int j=begin;j<end;j++)
					{
						perm[fold_start[i]] = index[j];
						fold_start[i]++;
					}
				}
			fold_start[0]=0;
			for (i=1;i<=nr_fold;i++)
				fold_start[i] = fold_start[i-1]+fold_count[i-1];
			free(start);	
			free(label);
			free(count);	
			free(index);
			free(fold_count);
		}
		else
		{
			for(i=0;i<l;i++) perm[i]=i;
			for(i=0;i<l;i++)
			{
				int j = i+rand()%(l-i);
				swap(perm[i],perm[j]);
			}
			for(i=0;i<=nr_fold;i++)
				fold_start[i]=i*l/nr_fold;
		}

		for(i=0;i<nr_fold;i++)
		{
			int begin = fold_start[i];
			int end = fold_start[i+1];
			int j,k;
			struct svm_problem subprob;

			subprob.l = l-(end-begin);
			subprob.x = Malloc(struct svm_node*,subprob.l);
			subprob.y = Malloc(double,subprob.l);

			k=0;
			for(j=0;j<begin;j++)
			{
				subprob.x[k] = prob->x[perm[j]];
				subprob.y[k] = prob->y[perm[j]];
				++k;
			}
			for(j=end;j<l;j++)
			{
				subprob.x[k] = prob->x[perm[j]];
				subprob.y[k] = prob->y[perm[j]];
				++k;
			}
			struct svm_model *submodel = svm_train(&subprob,param);
			if(param->probability && 
					(param->svm_type == C_SVC || param->svm_type == NU_SVC))
			{
				double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
				for(j=begin;j<end;j++)
					target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
				free(prob_estimates);			
			}
			else
				for(j=begin;j<end;j++)
					target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
			svm_free_and_destroy_model(&submodel);
			free(subprob.x);
			free(subprob.y);
		}		
		free(fold_start);
		free(perm);	
	}


	int svm_get_svm_type(const svm_model *model)
	{
		return model->param.svm_type;
	}

	int svm_get_nr_class(const svm_model *model)
	{
		return model->nr_class;
	}

	void svm_get_labels(const svm_model *model, int* label)
	{
		if (model->label != NULL)
			for(int i=0;i<model->nr_class;i++)
				label[i] = model->label[i];
	}

	void svm_get_sv_indices(const svm_model *model, int* indices)
	{
		if (model->sv_indices != NULL)
			for(int i=0;i<model->l;i++)
				indices[i] = model->sv_indices[i];
	}

	int svm_get_nr_sv(const svm_model *model)
	{
		return model->l;
	}

	double svm_get_svr_probability(const svm_model *model)
	{
		if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
				model->probA!=NULL)
			return model->probA[0];
		else
		{
			fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
			return 0;
		}
	}

	double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
	{
		int i;
		if(model->param.svm_type == ONE_CLASS ||
				model->param.svm_type == EPSILON_SVR ||
				model->param.svm_type == NU_SVR)
		{
			double *sv_coef = model->sv_coef[0];
			double sum = 0;
			for(i=0;i<model->l;i++)
				sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
			sum -= model->rho[0];
			*dec_values = sum;

			if(model->param.svm_type == ONE_CLASS)
				return (sum>0)?1:-1;
			else
				return sum;
		}
		else
		{
			int nr_class = model->nr_class;
			int l = model->l;

			double *kvalue = Malloc(double,l);
			//#pragma omp parallel for private(i)
			for(i=0;i<l;i++)
				kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

			int *start = Malloc(int,nr_class);
			start[0] = 0;
			for(i=1;i<nr_class;i++)
				start[i] = start[i-1]+model->nSV[i-1];

			int *vote = Malloc(int,nr_class);
			for(i=0;i<nr_class;i++)
				vote[i] = 0;

			int p=0;
			for(i=0;i<nr_class;i++)
				for(int j=i+1;j<nr_class;j++)
				{
					double sum = 0;
					int si = start[i];
					int sj = start[j];
					int ci = model->nSV[i];
					int cj = model->nSV[j];

					int k;
					double *coef1 = model->sv_coef[j-1];
					double *coef2 = model->sv_coef[i];
					for(k=0;k<ci;k++)
						sum += coef1[si+k] * kvalue[si+k];
					for(k=0;k<cj;k++)
						sum += coef2[sj+k] * kvalue[sj+k];
					sum -= model->rho[p];
					dec_values[p] = sum;

					if(dec_values[p] > 0)
						++vote[i];
					else
						++vote[j];
					p++;
				}

			int vote_max_idx = 0;
			for(i=1;i<nr_class;i++)
				if(vote[i] > vote[vote_max_idx])
					vote_max_idx = i;

			free(kvalue);
			free(start);
			free(vote);
			return model->label[vote_max_idx];
		}
	}

	double svm_predict(const svm_model *model, const svm_node *x)
	{
		int nr_class = model->nr_class;
		double *dec_values;
		if(model->param.svm_type == ONE_CLASS ||
				model->param.svm_type == EPSILON_SVR ||
				model->param.svm_type == NU_SVR)
			dec_values = Malloc(double, 1);
		else 
			dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		double pred_result = svm_predict_values(model, x, dec_values);
		free(dec_values);
		return pred_result;
	}

	double svm_predict_probability(
			const svm_model *model, const svm_node *x, double *prob_estimates)
	{
		if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
				model->probA!=NULL && model->probB!=NULL)
		{
			int i;
			int nr_class = model->nr_class;
			double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
			svm_predict_values(model, x, dec_values);

			double min_prob=1e-7;
			double **pairwise_prob=Malloc(double *,nr_class);
			for(i=0;i<nr_class;i++)
				pairwise_prob[i]=Malloc(double,nr_class);
			int k=0;
			for(i=0;i<nr_class;i++)
				for(int j=i+1;j<nr_class;j++)
				{
					pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
					pairwise_prob[j][i]=1-pairwise_prob[i][j];
					k++;
				}
			multiclass_probability(nr_class,pairwise_prob,prob_estimates);

			int prob_max_idx = 0;
			for(i=1;i<nr_class;i++)
				if(prob_estimates[i] > prob_estimates[prob_max_idx])
					prob_max_idx = i;
			for(i=0;i<nr_class;i++)
				free(pairwise_prob[i]);
			free(dec_values);
			free(pairwise_prob);	     
			return model->label[prob_max_idx];
		}
		else 
			return svm_predict(model, x);
	}

	static const char *svm_type_table[] =
	{
		"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
	};

	static const char *kernel_type_table[]=
	{
		"linear","polynomial","rbf","sigmoid","precomputed",NULL
	};

	int svm_save_model(const char *model_file_name, const svm_model *model)
	{
		FILE *fp = fopen(model_file_name,"w");
		if(fp==NULL) return -1;

		char *old_locale = strdup(setlocale(LC_ALL, NULL));
		setlocale(LC_ALL, "C");

		const svm_parameter& param = model->param;

		fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
		fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

		if(param.kernel_type == POLY)
			fprintf(fp,"degree %d\n", param.degree);

		if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
			fprintf(fp,"gamma %g\n", param.gamma);

		if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
			fprintf(fp,"coef0 %g\n", param.coef0);

		int nr_class = model->nr_class;
		int l = model->l;
		fprintf(fp, "nr_class %d\n", nr_class);
		fprintf(fp, "total_sv %d\n",l);

		{
			fprintf(fp, "rho");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fprintf(fp," %g",model->rho[i]);
			fprintf(fp, "\n");
		}

		if(model->label)
		{
			fprintf(fp, "label");
			for(int i=0;i<nr_class;i++)
				fprintf(fp," %d",model->label[i]);
			fprintf(fp, "\n");
		}

		if(model->probA) // regression has probA only
		{
			fprintf(fp, "probA");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fprintf(fp," %g",model->probA[i]);
			fprintf(fp, "\n");
		}
		if(model->probB)
		{
			fprintf(fp, "probB");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fprintf(fp," %g",model->probB[i]);
			fprintf(fp, "\n");
		}

		if(model->nSV)
		{
			fprintf(fp, "nr_sv");
			for(int i=0;i<nr_class;i++)
				fprintf(fp," %d",model->nSV[i]);
			fprintf(fp, "\n");
		}

		fprintf(fp, "SV\n");
		const double * const *sv_coef = model->sv_coef;
		const svm_node * const *SV = model->SV;

		for(int i=0;i<l;i++)
		{
			for(int j=0;j<nr_class-1;j++)
				fprintf(fp, "%.16g ",sv_coef[j][i]);

			const svm_node *p = SV[i];

			if(param.kernel_type == PRECOMPUTED)
				fprintf(fp,"0:%d ",(int)(p->value));
			else
				while(p->index != -1)
				{
					fprintf(fp,"%d:%.8g ",p->index,p->value);
					p++;
				}
			fprintf(fp, "\n");
		}

		setlocale(LC_ALL, old_locale);
		free(old_locale);

		if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
		else return 0;
	}

	static char *line = NULL;
	static int max_line_len;

	static char* readline(FILE *input)
	{
		int len;

		if(fgets(line,max_line_len,input) == NULL)
			return NULL;

		while(strrchr(line,'\n') == NULL)
		{
			max_line_len *= 2;
			line = (char *) realloc(line,max_line_len);
			len = (int) strlen(line);
			if(fgets(line+len,max_line_len-len,input) == NULL)
				break;
		}
		return line;
	}

	svm_model *svm_load_model(const char *model_file_name)
	{
		FILE *fp = fopen(model_file_name,"rb");
		if(fp==NULL) return NULL;

		char *old_locale = strdup(setlocale(LC_ALL, NULL));
		setlocale(LC_ALL, "C");

		// read parameters

		svm_model *model = Malloc(svm_model,1);
		svm_parameter& param = model->param;
		model->rho = NULL;
		model->probA = NULL;
		model->probB = NULL;
		model->label = NULL;
		model->nSV = NULL;

		char cmd[81];
		while(1)
		{
			fscanf(fp,"%80s",cmd);

			if(strcmp(cmd,"svm_type")==0)
			{
				fscanf(fp,"%80s",cmd);
				int i;
				for(i=0;svm_type_table[i];i++)
				{
					if(strcmp(svm_type_table[i],cmd)==0)
					{
						param.svm_type=i;
						break;
					}
				}
				if(svm_type_table[i] == NULL)
				{
					fprintf(stderr,"unknown svm type.\n");

					setlocale(LC_ALL, old_locale);
					free(old_locale);
					free(model->rho);
					free(model->label);
					free(model->nSV);
					free(model);
					return NULL;
				}
			}
			else if(strcmp(cmd,"kernel_type")==0)
			{		
				fscanf(fp,"%80s",cmd);
				int i;
				for(i=0;kernel_type_table[i];i++)
				{
					if(strcmp(kernel_type_table[i],cmd)==0)
					{
						param.kernel_type=i;
						break;
					}
				}
				if(kernel_type_table[i] == NULL)
				{
					fprintf(stderr,"unknown kernel function.\n");

					setlocale(LC_ALL, old_locale);
					free(old_locale);
					free(model->rho);
					free(model->label);
					free(model->nSV);
					free(model);
					return NULL;
				}
			}
			else if(strcmp(cmd,"degree")==0)
				fscanf(fp,"%d",&param.degree);
			else if(strcmp(cmd,"gamma")==0)
				fscanf(fp,"%lf",&param.gamma);
			else if(strcmp(cmd,"coef0")==0)
				fscanf(fp,"%lf",&param.coef0);
			else if(strcmp(cmd,"nr_class")==0)
				fscanf(fp,"%d",&model->nr_class);
			else if(strcmp(cmd,"total_sv")==0)
				fscanf(fp,"%d",&model->l);
			else if(strcmp(cmd,"rho")==0)
			{
				int n = model->nr_class * (model->nr_class-1)/2;
				model->rho = Malloc(double,n);
				for(int i=0;i<n;i++)
					fscanf(fp,"%lf",&model->rho[i]);
			}
			else if(strcmp(cmd,"label")==0)
			{
				int n = model->nr_class;
				model->label = Malloc(int,n);
				for(int i=0;i<n;i++)
					fscanf(fp,"%d",&model->label[i]);
			}
			else if(strcmp(cmd,"probA")==0)
			{
				int n = model->nr_class * (model->nr_class-1)/2;
				model->probA = Malloc(double,n);
				for(int i=0;i<n;i++)
					fscanf(fp,"%lf",&model->probA[i]);
			}
			else if(strcmp(cmd,"probB")==0)
			{
				int n = model->nr_class * (model->nr_class-1)/2;
				model->probB = Malloc(double,n);
				for(int i=0;i<n;i++)
					fscanf(fp,"%lf",&model->probB[i]);
			}
			else if(strcmp(cmd,"nr_sv")==0)
			{
				int n = model->nr_class;
				model->nSV = Malloc(int,n);
				for(int i=0;i<n;i++)
					fscanf(fp,"%d",&model->nSV[i]);
			}
			else if(strcmp(cmd,"SV")==0)
			{
				while(1)
				{
					int c = getc(fp);
					if(c==EOF || c=='\n') break;	
				}
				break;
			}
			else
			{
				fprintf(stderr,"unknown text in model file: [%s]\n",cmd);

				setlocale(LC_ALL, old_locale);
				free(old_locale);
				free(model->rho);
				free(model->label);
				free(model->nSV);
				free(model);
				return NULL;
			}
		}

		// read sv_coef and SV

		int elements = 0;
		long pos = ftell(fp);

		max_line_len = 1024;
		line = Malloc(char,max_line_len);
		char *p,*endptr,*idx,*val;

		while(readline(fp)!=NULL)
		{
			p = strtok(line,":");
			while(1)
			{
				p = strtok(NULL,":");
				if(p == NULL)
					break;
				++elements;
			}
		}
		elements += model->l;

		fseek(fp,pos,SEEK_SET);

		int m = model->nr_class - 1;
		int l = model->l;
		model->sv_coef = Malloc(double *,m);
		int i;
		for(i=0;i<m;i++)
			model->sv_coef[i] = Malloc(double,l);
		model->SV = Malloc(svm_node*,l);
		svm_node *x_space = NULL;
		if(l>0) x_space = Malloc(svm_node,elements);

		int j=0;
		for(i=0;i<l;i++)
		{
			readline(fp);
			model->SV[i] = &x_space[j];

			p = strtok(line, " \t");
			model->sv_coef[0][i] = strtod(p,&endptr);
			for(int k=1;k<m;k++)
			{
				p = strtok(NULL, " \t");
				model->sv_coef[k][i] = strtod(p,&endptr);
			}

			while(1)
			{
				idx = strtok(NULL, ":");
				val = strtok(NULL, " \t");

				if(val == NULL)
					break;
				x_space[j].index = (int) strtol(idx,&endptr,10);
				x_space[j].value = strtod(val,&endptr);

				++j;
			}
			x_space[j++].index = -1;
		}
		free(line);

		setlocale(LC_ALL, old_locale);
		free(old_locale);

		if (ferror(fp) != 0 || fclose(fp) != 0)
			return NULL;

		model->free_sv = 1;	// XXX
		return model;
	}

	void svm_free_model_content(svm_model* model_ptr)
	{
		if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
			free((void *)(model_ptr->SV[0]));
		if(model_ptr->sv_coef)
		{
			for(int i=0;i<model_ptr->nr_class-1;i++)
				free(model_ptr->sv_coef[i]);
		}

		free(model_ptr->SV);
		model_ptr->SV = NULL;

		free(model_ptr->sv_coef);
		model_ptr->sv_coef = NULL;

		free(model_ptr->rho);
		model_ptr->rho = NULL;

		free(model_ptr->label);
		model_ptr->label= NULL;

		free(model_ptr->probA);
		model_ptr->probA = NULL;

		free(model_ptr->probB);
		model_ptr->probB= NULL;

		free(model_ptr->nSV);
		model_ptr->nSV = NULL;
	}

	void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
	{
		if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
		{
			svm_free_model_content(*model_ptr_ptr);
			free(*model_ptr_ptr);
			*model_ptr_ptr = NULL;
		}
	}

	void svm_destroy_param(svm_parameter* param)
	{
		free(param->weight_label);
		free(param->weight);
	}

	const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
	{
		// svm_type

		int svm_type = param->svm_type;
		if(svm_type != C_SVC &&
				svm_type != NU_SVC &&
				svm_type != ONE_CLASS &&
				svm_type != EPSILON_SVR &&
				svm_type != NU_SVR)
			return "unknown svm type";

		// kernel_type, degree

		int kernel_type = param->kernel_type;
		if(kernel_type != LINEAR &&
				kernel_type != POLY &&
				kernel_type != RBF &&
				kernel_type != SIGMOID &&
				kernel_type != PRECOMPUTED)
			return "unknown kernel type";

		if(param->gamma < 0)
			return "gamma < 0";

		if(param->degree < 0)
			return "degree of polynomial kernel < 0";

		// cache_size,eps,C,nu,p,shrinking

		if(param->cache_size <= 0)
			return "cache_size <= 0";

		if(param->eps <= 0)
			return "eps <= 0";

		if(svm_type == C_SVC ||
				svm_type == EPSILON_SVR ||
				svm_type == NU_SVR)
			if(param->C <= 0)
				return "C <= 0";

		if(svm_type == NU_SVC ||
				svm_type == ONE_CLASS ||
				svm_type == NU_SVR)
			if(param->nu <= 0 || param->nu > 1)
				return "nu <= 0 or nu > 1";

		if(svm_type == EPSILON_SVR)
			if(param->p < 0)
				return "p < 0";

		if(param->shrinking != 0 &&
				param->shrinking != 1)
			return "shrinking != 0 and shrinking != 1";

		if(param->probability != 0 &&
				param->probability != 1)
			return "probability != 0 and probability != 1";

		if(param->probability == 1 &&
				svm_type == ONE_CLASS)
			return "one-class SVM probability output not supported yet";


		// check whether nu-svc is feasible

		if(svm_type == NU_SVC)
		{
			int l = prob->l;
			int max_nr_class = 16;
			int nr_class = 0;
			int *label = Malloc(int,max_nr_class);
			int *count = Malloc(int,max_nr_class);

			int i;
			for(i=0;i<l;i++)
			{
				int this_label = (int)prob->y[i];
				int j;
				for(j=0;j<nr_class;j++)
					if(this_label == label[j])
					{
						++count[j];
						break;
					}
				if(j == nr_class)
				{
					if(nr_class == max_nr_class)
					{
						max_nr_class *= 2;
						label = (int *)realloc(label,max_nr_class*sizeof(int));
						count = (int *)realloc(count,max_nr_class*sizeof(int));
					}
					label[nr_class] = this_label;
					count[nr_class] = 1;
					++nr_class;
				}
			}

			for(i=0;i<nr_class;i++)
			{
				int n1 = count[i];
				for(int j=i+1;j<nr_class;j++)
				{
					int n2 = count[j];
					if(param->nu*(n1+n2)/2 > min(n1,n2))
					{
						free(label);
						free(count);
						return "specified nu is infeasible";
					}
				}
			}
			free(label);
			free(count);
		}

		return NULL;
	}

	int svm_check_probability_model(const svm_model *model)
	{
		return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
				model->probA!=NULL && model->probB!=NULL) ||
			((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
			 model->probA!=NULL);
	}

	void svm_set_print_string_function(void (*print_func)(const char *))
	{
		if(print_func == NULL)
			svm_print_string = &print_string_stdout;
		else
			svm_print_string = print_func;
	}
