#include "mpi.h"
#include<stdio.h>
#include<math.h>
#define NELEMS(x) (sizeof(x)/sizeof(x[0]))

void output_mat(int myid, double* up, 
                int size_up, double* mid, int size_mid, 
                double* low, int size_low)
{
	printf("%d, up: ", myid);
	for(int i = 0; i< size_up; i++)
	{
		printf("%.2f ", up[i]);
	}
	printf("\n");
	printf("%d mid: ", myid);
	for(int i = 0; i< size_mid; i++)
	{
		printf("%.2f ", mid[i]);
	}
	printf("\n");
	printf("%d low: ", myid);
	for(int i = 0; i< size_low; i++)
	{
		printf("%.2f ", low[i]);
	}
	printf("\n");
}

void output_vec(int myid, double* vec, int len)
{
	printf("%d vec: ", myid);
	for(int i = 0; i< len; i++)
	{
		printf("%.15f ", vec[i]);
	}
	printf("\n");
}

void output_vec_tag(int myid, double* vec, int len, char* tag)
{
    printf("%s", tag);
    output_vec(myid, vec, len);
}

void output_single(int myid, double d)
{
	printf("%d single: ", myid);
	printf("%.15f\n", d);
}
int LU(double* up, double* mid, double* low, int m)
{
	for(int i = 0; i< m-1; i++)
	{
		low[i] = low[i]/mid[i];
		mid[i + 1] = mid[i + 1] - low[i]*up[i];
	}
	return 0;
}

void solve(double* up, double* mid, double* low, double* b, double* x, int m)
{
	// LUx = b
	// solve Lx' = b
	x[0] = b[0];
	for(int i = 0; i< m-1; i++)
	{
		x[i+1] = b[1 + i] - x[i] * low[i]; 
	}
	x[m-1] = x[m-1] / mid[m-1];
	for(int i = 1; i< m; i++)
	{
		x[m-1-i] = (x[m-1-i] - up[m-1-i] * x[m-i]) / mid[m-1-i];
	} 
}

void vec_mult_scal(double* vec, double scal, int m)
{
	for(int i = 0; i< m; i++)
	{
		vec[i] = vec[i] * scal;
	}
}

void vec_sub_vec(double* vec1, double* vec2, int m)
{	
	for(int i = 0; i< m; i++)
	{
		vec1[i] = vec1[i] - vec2[i];
	}
}

double cycle(double low, double mid, double up, 
			 double b, 
			 int myid, 
			 int lneighbour, 
			 int rneighbour, 
			 int nprocs, 
			 int pow2)
{
    if(lneighbour < 0 && rneighbour >= nprocs)
    {
        return b / mid;
    }

    if(myid % pow2 == pow2 - 1)
    {
        double vec[4];
        vec[0] = low;
        vec[1] = mid;
        vec[2] = up;
        vec[3] = b;
        double lmid[4];
        MPI_Recv(&lmid, 4, MPI_DOUBLE, lneighbour, 
                    0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double lmult = low / lmid[1];
        mid = mid - lmult * lmid[2];
        b = b - lmult * lmid[3];
        low = -lmult * lmid[0];
        if(rneighbour < nprocs)
        {
            double rmid[4];
            MPI_Recv(&rmid, 4, MPI_DOUBLE, rneighbour, 
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double rmult = up / rmid[1];
            mid = mid - rmult * rmid[0];
            b = b - rmult * rmid[3];
            up = -rmult * rmid[2];
        }
        double x = cycle(low, mid, up, b, myid, myid - pow2, 
                        myid + pow2, nprocs, pow2 * 2);
        // now we just send out x
        //double x = 0;
        MPI_Request req1;
        MPI_Request req2;
        MPI_Isend(&x, 1, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, &req1);
        if(rneighbour < nprocs)
        {
            double xp = x;
            MPI_Isend(&xp, 1, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &req2);
            MPI_Wait(&req2, MPI_STATUS_IGNORE); 
        }
        MPI_Wait(&req1, MPI_STATUS_IGNORE); 
        return x;
    }
    else
    {
        // send mid, up to myid + pow2 / 2 send mid, low to myid - pow2 / 2;

        double lmid_s[4];
        lmid_s[0] = low;
        lmid_s[1] = mid;
        lmid_s[2] = up;
        lmid_s[3] = b;
        double rmid_s[4];
        rmid_s[0] = low;
        rmid_s[1] = mid;
        rmid_s[2] = up;
        rmid_s[3] = b;
        MPI_Request req1;
        MPI_Request req2;
        if(lneighbour >= 0)
        {
            MPI_Isend(&rmid_s, 4, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, &req1);
        }
        if(rneighbour < nprocs)
        {
            MPI_Isend(&lmid_s, 4, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &req2);
            MPI_Wait(&req2, MPI_STATUS_IGNORE); 
        }
        
        if(lneighbour >= 0)
        {
            MPI_Wait(&req1, MPI_STATUS_IGNORE); 
        }
        if(lneighbour >= 0)
        {
            double xl;
            MPI_Recv(&xl, 1, MPI_DOUBLE, lneighbour, 
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            b = b - low * xl;
        }
        if(rneighbour < nprocs)
        {
            double xr;
            MPI_Recv(&xr, 1, MPI_DOUBLE, rneighbour, 
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            b = b - up * xr;
        }
        return b / mid;

    }

}


double id(double x)
{
    return x;
}


double sq(double x)
{
    return x*x;
}


double basis(double x)
{
    if(x >= 0 && x <= 1)
    {
        return 0.5 * sq(x);
    }
    else if(x > 1 && x <= 2)
    {
        return -sq(x-1) + x-1 + 0.5;
    }
    else if(x > 2 && x <= 3)
    {
        return 0.5 * sq(3-x);
    }
    else
    {
        return 0;
    }
}


double scl_basis(double i, double h, double a, double x)
{
    double in = (x-a) / h + 2 - i;
    //printf("%.4f, %.2f, %.2f, %.2f\n", in, a, h, x);
    return basis(in);
}


void generate_b(int myid, int nprocs, 
                int q, double a, double b, 
                int n, double func(double), 
                double* b1, double* b2)
{
    // we get the subinterval that belongs to me
    double h = (b-a) / n;
    double n_per_proc = (n + 2) / nprocs;
    if(myid == 0)
    {
        b1[0] = func(a) / 4;
        for(int i= 0; i< n_per_proc - 2; i++)
        {
            b1[i + 1] = func((double)(a + i * h + h / 2));
        }
        *b2 = func((double)(a +  (n_per_proc - 2) * h + h / 2));
    }
    else if(myid == nprocs - 1)
    {
        for(int i=0; i< n_per_proc - 1; i++)
        {
            b1[i] = func((double) (a + ((n_per_proc -1 ) + 
                    (myid - 1) * n_per_proc + i) * h + h / 2));
        }
        *b2 = func(b) / 4;
    }
    else
    {
        
        for(int i=0; i< n_per_proc - 1; i++)
        {
            b1[i] = func((double)(a + ((n_per_proc -1 ) + 
                    (myid - 1) * n_per_proc + i) * h + h / 2));
        }
        *b2 = func((double)(a + (n_per_proc -1 + (myid - 1) * n_per_proc + n_per_proc - 1) 
                                * h + h/2));
    }
}


void generate_mat(int myid, 
				int nprocs, 
				int q, 
				double* low, 
				double* mid, 
				double* up, 
				double* A2, 
				double* C1, 
				double* A3, 
				double* A4, 
				double* B1)
{
    for(int i = 0; i< q-1; i++)
    {
        if(i < q-2)
        {
            low[i] = (double) 1 / 8;
            up[i] = (double)1 / 8;
        }
        mid[i] = (double)3/4;
        A2[i] = 0;
        C1[i] = 0;
    }
    A2[q-2] = (double)1 / 8;
    C1[0] = (double)1 / 8;
    *A3 = (double)1 / 8;
    *B1 = (double)1 / 8;
    *A4 = (double)3/4;
    if(myid == 0)
    {
        mid[0] = (double)1 / 8;
    }
    else if(myid == nprocs - 1)
    {
        *A4 = (double)1 / 8;
    }
}


double* partition_2(int myid, 
					int nprocs, 
					int lneighbour, 
					int rneighbour, 
					int q, 
					double* up1, 
					double* mid1, 
					double* low1, 
					double* A2, 
					double *C1, 
					double* b11, 
					double b12, 
					double A3, 
					double A4, 
					double B1, 
					double* output)
{
	double B2 = 0;
    double C2 = 0;   
	//solve(up, mid, low, b, x, 3);
	//output_vec(x, 4);
	LU(up1, mid1, low1, q-1);
	solve(up1, mid1, low1, b11, b11, q-1);
    //output_vec_tag(myid, b11, q-1, "b ");
	solve(up1, mid1, low1, A2, A2, q-1);
    //output_vec_tag(myid, A2, q-1, "A2 ");
    if(myid > 0)
    {
        solve(up1, mid1, low1, C1, C1, q-1);
        //output_vec_tag(myid, C1, q-1, "C1 ");
    }
    double buf[3];
    double s_buf[3];
    MPI_Status istatus; MPI_Request req;

    if(myid > 0 && myid < nprocs - 1)
    {
        s_buf[0] = C1[0];
        s_buf[1] = A2[0];
        s_buf[2] = b11[0];
        // we want to send C2[0], A2[0], b11[0] to previous processor
        MPI_Isend(s_buf, 3, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(buf, 3, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &istatus);
	    A4 = A4 - A3 * A2[q-2] - B1 * buf[0];	
    	B2 = -B1 * buf[1];
        C2 = -A3 * C1[q-2];
	    b12 = b12 - A3 * b11[q-2] - B1 * buf[2];
        MPI_Wait(&req, &istatus);
    }
    else if(myid == 0)
    {
        // we dont need to send anything
        // no C matrix
        MPI_Recv(buf, 3, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &istatus);
	    A4 = A4 - A3 * A2[q-2] - B1 * buf[0];	
    	B2 = -B1 * buf[1];
	    b12 = b12 - A3 * b11[q-2] - B1 * buf[2];
        
    }
    else if(myid == nprocs - 1)
    {
        // we don't need to receive anything
        // no B matrix
        s_buf[0] = C1[0];
        s_buf[1] = A2[0];
        s_buf[2] = b11[0];
        // we want to send C2[0], A2[0], b21[0] to previous processor
        MPI_Isend(s_buf, 3, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, &req);
	    A4 = A4 - A3 * A2[q-2];	
        C2 = -A3 * C1[q-2];
	    b12 = b12 - A3 * b11[q-2];

        MPI_Wait(&req, &istatus);
    }
    if(myid == 0)
    {
        C2 = 0;
    }
    if(myid == nprocs - 1)
    {
        B2 = 0;
    }

	double x12 = cycle(C2, A4, B2, b12, myid, lneighbour, rneighbour, nprocs, 2);

    if(myid == 0)
    {
        double xsend = x12;
        MPI_Request req;
        MPI_Isend(&xsend, 1, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &req);
	    vec_mult_scal(A2, x12, q-1);
	    vec_sub_vec(b11, A2, q-1);
            
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    if(myid == nprocs - 1)
    {
        double x22;
        MPI_Recv(&x22, 1, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
	    vec_mult_scal(A2, x12, q-1);
	    vec_sub_vec(b11, A2, q-1);
        vec_mult_scal(C1, x22, q-1);
        vec_sub_vec(b11, C1, q-1);
    }
    if(myid != 0 && myid != nprocs - 1)
    {
        double xsend = x12;
        double x22;
        MPI_Isend(&xsend, 1, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(&x22, 1, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
	    vec_mult_scal(A2, x12, q-1);
	    vec_sub_vec(b11, A2, q-1);
        vec_mult_scal(C1, x22, q-1);
        vec_sub_vec(b11, C1, q-1);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    output[q-1] = x12;
    for(int i = 0; i< q-1; i++)
    {
        output[i] = b11[i];
    }
    return 0;
}


double eval_basis(int myid, int q, int i, double h, double a, double x, double* coeffs)
{
    int first_i = myid * q;
    if(myid == 0)
    {
        first_i = 0;    
    }
    //printf("%.3f\n", x);
    //output_single(myid, scl_basis(i, h ,a, x));
    return coeffs[i - first_i] * scl_basis(i, h, a, x) + 
            coeffs[i+1 - first_i] * scl_basis(i+1, h, a, x) + 
            coeffs[i+2 - first_i] * scl_basis(i+2, h, a, x);
//double scl_basis(double i, double h, double a, double x)
}


void eval_basis_vec(int myid, int nprocs, 
                    int q, double h, 
                    double a, double b, 
                    double* x, double* coeffs)
{
    int left_i;
    double left;
    if(myid == 0)
    {
        left_i = 0;
        left = a;
        for(int i= 0; i< q; i++)
        {
            for(int j = 0; j<3; j++)
            {

                x[i*3 + j] = eval_basis(myid, q, left_i + i, h, a, 
                                        left + i * h + j * h / 3, coeffs);
            }
        }
    }
    else if(myid == nprocs - 1)
    {
        left_i = myid * q;
        left = a + (double)(myid * q) * h;
        for(int i=0; i< q - 2; i++)
        {
            for(int j = 0; j<3; j++)
            {
                
                double y = left + i * h + j * h / 3;
                y = eval_basis(myid, q, left_i + i, h, a, y, coeffs);
                x[i*3 + j] = y;
                //printf("%.2f\n", y);
            }
        }
        x[3 * (q -2)] = eval_basis(myid, q, left_i + q - 3, h, a, b, coeffs);
    }
    else
    {
        left_i = myid * q;
        left = a + (double)(myid * q) * h;
        for(int i=0; i< q; i++)
        {
            for(int j = 0; j<3; j++)
            {

                x[i*3 + j] = eval_basis(myid, q, left_i + i, h, a, 
                                        left + i * h + j * h / 3, coeffs);
            }
        }
    }
}

void eval_func(int myid, int nprocs, 
                int q, double h, 
                double a, double b, 
                double* x, double func(double))
{
    int left_i;
    double left;
    
    if(myid == 0)
    {
        left_i = 0;
        left = a;
        for(int i= 0; i< q; i++)
        {
            for(int j = 0; j<3; j++)
            {

                x[i*3 + j] = func(left + i * h + j * h / 3);
            }
        }
    }
    else if(myid == nprocs - 1)
    {
        left_i = myid * q;
        left = a + (double)(myid * q) * h;
        for(int i=0; i< q - 2; i++)
        {
            for(int j = 0; j<3; j++)
            {
                
                double y = left + i * h + j * h / 3;
                y = func(y);
                x[i*3 + j] = y;
            }
        }
        x[3 * (q -2)] = func(b);
    }
    else
    {
        left_i = myid * q;
        left = a + (double)(myid * q) * h;
        for(int i=0; i< q; i++)
        {
            for(int j = 0; j<3; j++)
            {

                x[i*3 + j] = func(left + i * h + j * h / 3);
            }
        }
    }
}


double get_max_diff(double* spline, double* fvals, double* errs_nodes, int num)
{
    double max = 0;
    double errs_n = 0;
    for(int i = 0; i< num; i++)
    {
        if(i % 3 == 0)
        {
            errs_n = fmax(errs_n, fabs(spline[i] - fvals[i]));
        }
        max = fmax(max, fabs(spline[i] - fvals[i]));
    }
    *errs_nodes = errs_n;
    return max;
}

double global_max(double* send_buf, double* recv_buf, int buf_dim, int rank, int size)
{
    int d = (int)round(log2(size));
    unsigned int mask = 1;
    MPI_Status istatus;
    MPI_Request req;
    double comm_buf[buf_dim];
    comm_buf[0] = send_buf[0];
    for(int i = 0; i<d; i++){
        int addr = rank ^ mask;
        MPI_Isend(comm_buf, buf_dim, MPI_DOUBLE, addr, rank, MPI_COMM_WORLD, &req);
        MPI_Recv(recv_buf, buf_dim, MPI_DOUBLE, addr, addr, MPI_COMM_WORLD, &istatus);
        MPI_Wait(&req, &istatus);
        mask <<= 1;
        comm_buf[0] = fmax(comm_buf[0], recv_buf[0]);
    }
    recv_buf[0] = comm_buf[0];
}

int main(int argc, char** argv) 
{	
    int myid, nprocs, lneighbour, rneighbour;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lneighbour = myid - 1;
    rneighbour = myid + 1;
    double a = 0;
    double b = 1;//12 * acos(-1.0);
    double num[6] = {30, 62, 126, 254, 510, 1022}; //62, 126, 254, 510, 1022};
    double errs[6];
    double n_errs[6];
    for(int i = 0; i< 6; i++)
    {
	    int q = (num[i] + 2) / nprocs;
	    double up[q-2];
	    double mid[q-1];
	    double low[q-2];
	    double A2[q-1];
        double C1[q-1];
        double c[q];
	    double A3;
	    double A4;
	    double B1;
        generate_mat(myid, nprocs, q, low, mid, up, A2, C1, &A3, &A4, &B1);

	    double b1[q-1];
	    double b2;

        generate_b(myid, nprocs, q, a, b, num[i], sq, b1, &b2);
        partition_2(myid, nprocs, lneighbour, rneighbour, 
                    q, up, mid, low, A2, C1, b1, b2, A3, A4, B1, c);
        MPI_Request req; MPI_Status istatus;
        double s_buf[2];
        double buf[2];
        double c_extended[q + 2]; 
        if(myid > 0 && myid < nprocs - 1)
        {
            s_buf[0] = c[0];
            s_buf[1] = c[1];
            MPI_Isend(s_buf, 2, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, &req);
            MPI_Recv(buf, 2, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &istatus);
            for(int i = 0; i< q; i++)
            {
                c_extended[i] = c[i];
            }
            c_extended[q] = buf[0];
            c_extended[q+1] = buf[1];
            MPI_Wait(&req, &istatus);
        }
        else if(myid == 0)
        {
            MPI_Recv(buf, 2, MPI_DOUBLE, rneighbour, 0, MPI_COMM_WORLD, &istatus);
            for(int i = 0; i< q; i++)
            {
                c_extended[i] = c[i];
            }
            c_extended[q] = buf[0];
            c_extended[q+1] = buf[1];
        }
        else if(myid == nprocs - 1)
        {
            s_buf[0] = c[0];
            s_buf[1] = c[1];
            MPI_Isend(s_buf, 2, MPI_DOUBLE, lneighbour, 0, MPI_COMM_WORLD, &req);
            for(int i = 0; i< q; i++)
            {
                c_extended[i] = c[i];
            }
            MPI_Wait(&req, &istatus);
        }

        double h = (b-a) / num[i];
        double left;
        double left_i;
        if(myid == 0)
        {
            left_i = 0;
            left = a;
        }
        else
        {
            left_i = myid * q;
            left = a + (myid * q) * h;
        }
        double x = eval_basis(myid, q, left_i, h, a, left + h, c);
        double out[3 * q];
        eval_basis_vec(myid, nprocs, q, h, a, b, out, c_extended);
        double fvals[3 * q];
        eval_func(myid, nprocs, q, h, a, b, fvals, sq);
        double err;
        double n_err;
        if(myid != nprocs -1)
        {
            err = get_max_diff(out, fvals, &n_err, 3 * q);
        }
        else
        {
            err = get_max_diff(out, fvals, &n_err, 3 * (q-2) + 1);
        }
        double max_err;
        double max_n_err;
        global_max(&err, &max_err, 1, myid, nprocs);
        global_max(&n_err, &max_n_err, 1, myid, nprocs);
        errs[i] = max_err;
        n_errs[i] = max_n_err;
    }
    output_vec(myid, errs, 6);
    output_vec(myid, n_errs, 6);
    MPI_Finalize();    
	return 0;
}
