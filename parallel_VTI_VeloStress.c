#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "malloc.h"
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif


#define PI 3.1415926
#define R 0.000001  //reflection coefficient, range from 0.07 to 0.09 is suitable.
#define B 600.0   //Cosine-type absorbing boundary RC(reflection coefficient).

void **alloc2 (size_t n1, size_t n2, size_t size);
void *alloc1 (size_t n1, size_t size);
void Zero1D(float *BUF1, int N1);
void Zero2D(float **BUF2, int N1, int N2);
void free1 (void *p);
void free2 (void **p);
void create_model(float **vp, float **vs, float **rho, int Nx, int Nz);
void extmodel(float **ext_model, float **init_model, int Nx, int Nz, int pml);
float maxvelocity(float **v, int Nx, int Nz);

void main()
{

	int ix, iz, it, nsx, nsz, x, z, n1, N;
	int Nx0, Nx, Nz0, Nz, Nt, pml, snapshot;
	float dx, dz, dt, vp0, stable, limit;
	float t, t0, fm, coeffi_x1, coeffi_x2, coeffi_z1, coeffi_z2;
	float D1, D2, D3, D4, D5, D6, Y1, Y2, Y3, Y4, a0, a1, a2, a3, a4, a5;
	float **TXX, **TZZ, **TXZ, **TXX_x, **TXX_z, **TZZ_x, **TZZ_z, **TXZ_x, **TXZ_z;
	float **Record_vx, **Record_vz;
	float **vx, **vz, **vx_x, **vx_z, **vz_x, **vz_z;
	float *Source, *ddx, *ddz;
	float **lambda, **mu, **rho, **vp, **vs;
	float **lambda_ini, **mu_ini, **rho_ini, **vp_ini, **vs_ini;
	float data[32767], *data0;
	float c11, c13, c33, c44, epsilon, delta, f;
	
	omp_set_num_threads(8);

	#ifdef _OPENMP
    	/* Testing for OpenMP */
    	float start_time, end_time;
    	#endif

	FILE *fp, *fpa, *fpr;
  
/*	2N=4: a0=-2.166666; a1=1.125; a2=-0.04166667; a3=0.0; a4=0.0; a5=0.0;
	2N=6: a0=-2.22291668; a1=1.171875;a2=-0.06510416; a3=0.0046875; a4=0.0; a5=0.0;
	2N=8: a0=-2.250818337;a1=1.196289; a2=-0.0797526; a3=0.009570313; a4=-0.0006975447; a5=0.0;
	2N=10: a0=-2.242516633;a1=1.2112427; a2=-0.08972168; a3=0.01384277; a4=-0.00176566; a5=0.0001186795;*/
	
	float a[6] = {-2.242516633, 1.2112427, -0.08972168, 0.01384277, -0.00176566, 0.0001186795};
	//a0=-2.242516633;a1=1.2112427; a2=-0.08972168; a3=0.01384277; a4=-0.00176566; a5=0.0001186795;

	N = 5;     // Orderrrrrrr 
	dx = 5.0;         // step size
	dz = 5.0;         
	dt = 0.0005;        //sampling interval 
	fm = 35;           //mian-frequency 
	Nx0 = 1001;
	Nz0 = 1001;
	Nt = 4001;
	pml=100;

	snapshot = Nt-10;

	coeffi_x1 = 0.0;
	coeffi_x2 = 0.0;
	coeffi_z1 = 0.0;
	coeffi_z2 = 0.0;
	Y1 = 0.0; Y2 = 0.0; Y3 = 0.0; Y4 = 0.0;
	D1 = 0.0; D2 = 0.0; D3 = 0.0; D4 = 0.0; D5 = 0.0; D6=0.0;

	Nx = Nx0+2*pml;
	Nz = Nz0+2*pml;
	nsx = (int)(Nx/2);      
	nsz = pml+5;        // location of source

	epsilon = 0.0;
	delta = 0.0;
	c11 = 0.0; c13 = 0.0; c33 = 0.0; c44 = 0.0; f = 0.0; 


	#ifdef _OPENMP
    	start_time = omp_get_wtime();
    	#endif
    	
/*********************Dynamic Memory Allocation******************/
	Source = (float *)alloc1(Nt, sizeof(float));
	ddx = (float *)alloc1(Nx, sizeof(float));
	ddz = (float *)alloc1(Nz, sizeof(float));

	Record_vx = (float **)alloc2(Nt, Nx, sizeof(float));
	Record_vz = (float **)alloc2(Nt, Nx, sizeof(float));
	
	vx = (float **)alloc2(Nz, Nx, sizeof(float));
	vz = (float **)alloc2(Nz, Nx, sizeof(float));

	vx_x = (float **)alloc2(Nz, Nx, sizeof(float));
	vx_z = (float **)alloc2(Nz, Nx, sizeof(float));
	vz_x = (float **)alloc2(Nz, Nx,sizeof(float));
	vz_z = (float **)alloc2(Nz, Nx, sizeof(float));

	TXX = (float **)alloc2(Nz, Nx, sizeof(float));
	TXX_x = (float **)alloc2(Nz, Nx,sizeof(float));
	TXX_z = (float **)alloc2(Nz, Nx, sizeof(float));

	TZZ = (float **)alloc2(Nz, Nx, sizeof(float));
	TZZ_x = (float **)alloc2(Nz, Nx, sizeof(float));
	TZZ_z = (float **)alloc2(Nz, Nx, sizeof(float));

	TXZ = (float **)alloc2(Nz, Nx,sizeof(float));
	TXZ_x = (float **)alloc2(Nz, Nx, sizeof(float));
	TXZ_z = (float **)alloc2(Nz, Nx, sizeof(float));

	lambda = (float **)alloc2(Nz, Nx, sizeof(float));
	mu = (float **)alloc2(Nz, Nx, sizeof(float));
	rho = (float **)alloc2(Nz, Nx, sizeof(float));
	vp = (float **)alloc2(Nz, Nx,sizeof(float));
	vs = (float **)alloc2(Nz, Nx,sizeof(float));

	vp_ini = (float **)alloc2(Nz0, Nx0, sizeof(float));
	vs_ini = (float **)alloc2(Nz0, Nx0, sizeof(float));
	lambda_ini = (float **)alloc2(Nz0, Nx0, sizeof(float));
	mu_ini = (float **)alloc2(Nz0, Nx0, sizeof(float));
	rho_ini = (float **)alloc2(Nz0, Nx0, sizeof(float));

	data0 = (float *)alloc1(Nz, sizeof(float));
/************************Initial Value****************************/ 

	Zero1D(Source,Nt);

	Zero1D(ddx,Nx);
	Zero1D(ddz,Nz);

	Zero2D(Record_vx,Nx,Nt);
	Zero2D(Record_vz,Nx,Nt);

	Zero2D(vx,Nx, Nz);
	Zero2D(vx_x,Nx, Nz);
	Zero2D(vx_z,Nx, Nz);

	Zero2D(vz,Nx, Nz);
	Zero2D(vz_x,Nx, Nz);
	Zero2D(vz_z,Nx, Nz);

	Zero2D(TXX,Nx, Nz);
	Zero2D(TXX_x,Nx, Nz);
	Zero2D(TXX_z,Nx, Nz);

	Zero2D(TZZ,Nx, Nz);
	Zero2D(TZZ_x,Nx, Nz);
	Zero2D(TZZ_z,Nx, Nz);

	Zero2D(TXZ,Nx, Nz);
	Zero2D(TXZ_x,Nx, Nz);
	Zero2D(TXZ_z,Nx, Nz);

	Zero2D(vp_ini,Nx0, Nz0);
	Zero2D(vs_ini,Nx0, Nz0);
	Zero2D(lambda_ini,Nx0, Nz0);
	Zero2D(mu_ini,Nx0, Nz0);
	Zero2D(rho_ini,Nx0, Nz0);

	Zero2D(vp,Nx, Nz);
	Zero2D(vs,Nx, Nz);
	Zero2D(lambda,Nx, Nz);
	Zero2D(mu,Nx, Nz);
	Zero2D(rho,Nx, Nz);

	Zero1D(data0, Nz);

	create_model(vp_ini, vs_ini, rho_ini, Nx0, Nz0);

	extmodel(vp, vp_ini, Nx0, Nz0, pml);
	extmodel(vs, vs_ini, Nx0, Nz0, pml);
	extmodel(rho, rho_ini, Nx0, Nz0, pml);

	for(ix=0; ix<Nx; ix++)
	{
		for(iz=0; iz<Nz; iz++)
    		{
			mu[ix][iz] = rho[ix][iz]*vs[ix][iz]*vs[ix][iz];
			lambda[ix][iz] = rho[ix][iz]*vp[ix][iz]*vp[ix][iz]-2.0*mu[ix][iz];
		}
	}

	// Find the maximum vp_velocity
	vp0 = maxvelocity(vp, Nx, Nz);
	printf("The vp0 = %f\n", vp0);

  	free2((void **)lambda_ini);
  	free2((void **)mu_ini);
  	free2((void **)rho_ini);
  	free2((void **)vp_ini);
  	free2((void **)vs_ini);
/*************************source function**************************/    
	fp=fopen("Source.txt","w"); 
	for(it=0; it<Nt; it++)
	{
		t = (it-(int)(1.0/fm/dt))*dt;       //move main-frequency to right axis 
		Source[it] = (1-2*PI*PI*fm*fm*t*t)*exp(-PI*PI*fm*fm*t*t);
		fprintf(fp,"%f\n",Source[it]);
	}
	fclose(fp);
 
/************************PML abosorbing boundary *******************/
/*******************Exponential Absorbing Boundary******************/	
/*
	for(ix=0; ix<Nx; ix++)
	{
		if(ix<=pml)
		{
			x = pml-ix;
			ddx[ix]=-(3.0*vp0/2.0/pml)*log(R)*(x*x)/(pml*pml);	
		}
		else if(ix>=Nx-pml)
		{
			x = ix-Nx+pml;  
			ddx[ix]=-(3.0*vp0/2.0/pml)*log(R)*(x*x)/(pml*pml);
		}
	}


	for(iz=0; iz<Nz; iz++)
	{			
		if(iz<=pml)
		{
			z = pml-iz;  
			ddz[iz]=-(3.0*vp0/2.0/pml)*log(R)*(z*z)/(pml*pml);
    		}
    		else if(iz>Nz-pml)
		{
      			z = iz-Nz+pml;
      			ddz[iz]=-(3.0*vp0/2.0/pml)*log(R)*(z*z)/(pml*pml);
    		}
	}
*/
/*******************Cosine-type Absorbing Boundary*******************/
  	for(ix=0;ix<=Nx;ix++)
	{
		if(ix<=pml)
		{
			x = ix;
			ddx[ix]=B*(1.0-sin(PI*(x)/2.0/pml));
		}
		else if (ix>=Nx-pml)
		{
			x = Nx-ix; 
		        ddx[ix]=B*(1.0-sin(PI*(x)/2.0/pml));
		}
	}
	for(iz=0;iz<=Nz;iz++)
	{
		if(iz<=pml)
		{
			z = iz;
			ddz[iz]=B*(1.0-sin(PI*(z)/2.0/pml));
		}
		else if (iz>=Nz-pml)
		{
			z = Nz-iz;
			ddz[iz]=B*(1.0-sin(PI*(z)/2.0/pml));
		}
	}
   
/*for(ix=0;ix<6;ix++){
	printf("a = %f\n",a[ix]);
}*/

/***********************Numerical Calculation************************/
	float Z;
	

	//fp = fopen("wavefieldall_vx.dat","wb");
  	for(it=0; it<Nt; it++)
  	{	
		stable = dt*vp0*(sqrt(1.0/dx/dx+1.0/dz/dz));
		limit = a[1]+a[2]+a[3]+a[4]+a[5];

		if(stable>=limit){
			printf("      NONSTABLE !!!!\nPLEASE RESET PARAMETERS !!!!\n");
			break;
		}

   		if(it%100==0||it==Nt-1) printf("it = %d,  propagation time = %f\n", it, it*dt);
      
  		#ifdef _OPENMP
        #pragma omp parallel for default(none) \
        	private(ix, iz, f, c33, c44, c11, c13, coeffi_x1, coeffi_x2, coeffi_z1, coeffi_z2, D1, D2, D5, D6, Z ) \
            shared(a,Nx, Nz, N, dt, dx, dz, ddx, ddz, vs, vp, rho, epsilon, delta, vx, vz, TXX_x, TXX_z, TZZ_x, TZZ_z, TXZ_x, TXZ_z, TXX, TZZ, TXZ)
        #endif
 		for(ix=N; ix<Nx-N; ix++)     //Simulation
    		{
			for(iz=N; iz<Nz-N; iz++)
    			{
				f = 1-(vs[ix][iz]*vs[ix][iz])/(vp[ix][iz]*vp[ix][iz]);
				c33 = rho[ix][iz]*vp[ix][iz]*vp[ix][iz];
				c44 = rho[ix][iz]*vs[ix][iz]*vs[ix][iz];
				c11 = (2*epsilon+1)*c33;
				
				Z = 2*delta*c33/(c33-c44);
				
				//c13 = c33*sqrt(f*(f+2*delta))-c44;    //Exact expression
				c13 = -c44+(c33-c44)*(1+0.5*Z/(1+0.25*Z));		//1-th Pade approximation
				//c13 = -c44+(c33-c44)*(1+delta*c33/(c33-c44));		//R.Bloot et al. 2013 - weakly anisotropic VTI media
				
						
				coeffi_x1 = (1.0-0.5*dt*ddx[ix])/(1.0+0.5*dt*ddx[ix]);  
				coeffi_x2 = dt/(1+0.5*dt*ddx[ix]);
				coeffi_z1 = (1.0-0.5*dt*ddz[iz])/(1.0+0.5*dt*ddz[iz]);
				coeffi_z2 = dt/(1+0.5*dt*ddz[iz]);

				D1 = (a[1]*(vx[ix+1][iz]-vx[ix][iz])+a[2]*(vx[ix+2][iz]-vx[ix-1][iz])+a[3]*(vx[ix+3][iz]-vx[ix-2][iz])+a[4]*(vx[ix+4][iz]-vx[ix-3][iz])+a[5]*(vx[ix+5][iz]-vx[ix-4][iz]))/dx;

				D2 = (a[1]*(vz[ix][iz]-vz[ix][iz-1])+a[2]*(vz[ix][iz+1]-vz[ix][iz-2])+a[3]*(vz[ix][iz+2]-vz[ix][iz-3])+a[4]*(vz[ix][iz+3]-vz[ix][iz-4])+a[5]*(vz[ix][iz+4]-vz[ix][iz-5]))/dz;

			//	D3 = (a[1]*(vx[ix+1][iz]-vx[ix][iz])+a[2]*(vx[ix+2][iz]-vx[ix-1][iz])+a[3]*(vx[ix+3][iz]-vx[ix-2][iz])+a[4]*(vx[ix+4][iz]-vx[ix-3][iz])+a[5]*(vx[ix+5][iz]-vx[ix-4][iz]))/dx;

			//	D4 = (a[1]*(vz[ix][iz]-vz[ix][iz-1])+a[2]*(vz[ix][iz+1]-vz[ix][iz-2])+a[3]*(vz[ix][iz+2]-vz[ix][iz-3])+a[4]*(vz[ix][iz+3]-vz[ix][iz-4])+a[5]*(vz[ix][iz+4]-vz[ix][iz-5]))/dz;

				D5 = (a[1]*(vx[ix][iz+1]-vx[ix][iz])+a[2]*(vx[ix][iz+2]-vx[ix][iz-1])+a[3]*(vx[ix][iz+3]-vx[ix][iz-2])+a[4]*(vx[ix][iz+4]-vx[ix][iz-3])+a[5]*(vx[ix][iz+5]-vx[ix][iz-4]))/dz;

				D6 = (a[1]*(vz[ix][iz]-vz[ix-1][iz])+a[2]*(vz[ix+1][iz]-vz[ix-2][iz])+a[3]*(vz[ix+2][iz]-vz[ix-3][iz])+a[4]*(vz[ix+3][iz]-vz[ix-4][iz])+a[5]*(vz[ix+4][iz]-vz[ix-5][iz]))/dx;


			
				TXX_x[ix][iz] = coeffi_x1*TXX_x[ix][iz]+coeffi_x2*c11*D1;
				
				TXX_z[ix][iz] = coeffi_z1*TXX_z[ix][iz]+coeffi_z2*c13*D2;	

				TZZ_x[ix][iz] = coeffi_x1*TZZ_x[ix][iz]+coeffi_x2*c13*D1;

				TZZ_z[ix][iz] = coeffi_z1*TZZ_z[ix][iz]+coeffi_z2*c33*D2;

				TXZ_x[ix][iz] = coeffi_x1*TXZ_x[ix][iz]+coeffi_x2*c44*D6;

				TXZ_z[ix][iz] = coeffi_z1*TXZ_z[ix][iz]+coeffi_z2*c44*D5;

				TXX[ix][iz] = TXX_x[ix][iz]+TXX_z[ix][iz];

				TZZ[ix][iz] = TZZ_x[ix][iz]+TZZ_z[ix][iz];

				TXZ[ix][iz] = TXZ_x[ix][iz]+TXZ_z[ix][iz];	
				
			}
   		}
		#ifdef _OPENMP
        #pragma omp parallel for default(none) \
        	private(ix, iz, coeffi_x1, coeffi_x2, coeffi_z1, coeffi_z2, Y1, Y2, Y3, Y4 )\
            shared(a,Nx, Nz, N, dt, dx, dz, ddx, ddz, rho, vx_x, vx_z, vz_x, vz_z, vx, vz, TXX, TZZ, TXZ)
        #endif
  		for(ix=N; ix<Nx-N; ix++)
        	{
		  	for(iz=N; iz<Nz-N; iz++)
	    		{
				coeffi_x1 = (1.0-0.5*dt*ddx[ix])/(1.0+0.5*dt*ddx[ix]);
				coeffi_x2 = dt/(1+0.5*dt*ddx[ix]);
				coeffi_z1 = (1.0-0.5*dt*ddz[iz])/(1.0+0.5*dt*ddz[iz]);
				coeffi_z2 = dt/(1+0.5*dt*ddz[iz]);
				

				Y1 = (a[1]*(TXX[ix][iz]-TXX[ix-1][iz])+a[2]*(TXX[ix+1][iz]-TXX[ix-2][iz])+a[3]*(TXX[ix+2][iz]-TXX[ix-3][iz])+a[4]*(TXX[ix+3][iz]-TXX[ix-4][iz])+a[5]*(TXX[ix+4][iz]-TXX[ix-5][iz]))/dx;
				
				Y2 = (a[1]*(TXZ[ix][iz]-TXZ[ix][iz-1])+a[2]*(TXZ[ix][iz+1]-TXZ[ix][iz-2])+a[3]*(TXZ[ix][iz+2]-TXZ[ix][iz-3])+a[4]*(TXZ[ix][iz+3]-TXZ[ix][iz-4])+a[5]*(TXZ[ix][iz+4]-TXZ[ix][iz-5]))/dz;

				Y3 = (a[1]*(TXZ[ix+1][iz]-TXZ[ix][iz])+a[2]*(TXZ[ix+2][iz]-TXZ[ix-1][iz])+a[3]*(TXZ[ix+3][iz]-TXZ[ix-2][iz])+a[4]*(TXZ[ix+4][iz]-TXZ[ix-3][iz])+a[5]*(TXZ[ix+5][iz]-TXZ[ix-4][iz]))/dx;

				Y4 = (a[1]*(TZZ[ix][iz+1]-TZZ[ix][iz])+a[2]*(TZZ[ix][iz+2]-TZZ[ix][iz-1])+a[3]*(TZZ[ix][iz+3]-TZZ[ix][iz-2])+a[4]*(TZZ[ix][iz+4]-TZZ[ix][iz-3])+a[5]*(TZZ[ix][iz+5]-TZZ[ix][iz-4]))/dz;
	
				vx_x[ix][iz] = coeffi_x1*vx_x[ix][iz]+coeffi_x2*(1.0/rho[ix][iz])*Y1;

				vx_z[ix][iz] = coeffi_z1*vx_z[ix][iz]+coeffi_z2*(1.0/rho[ix][iz])*Y2;

				vz_x[ix][iz] = coeffi_x1*vz_x[ix][iz]+coeffi_x2*(1.0/rho[ix][iz])*Y3;

				vz_z[ix][iz] = coeffi_z1*vz_z[ix][iz]+coeffi_z2*(1.0/rho[ix][iz])*Y4;
			
				vx[ix][iz] = vx_x[ix][iz]+vx_z[ix][iz];

				vz[ix][iz] = vz_x[ix][iz]+vz_z[ix][iz];
				
	    		}
    		}   
/************************Source Loading**************************/
		//TXX_x[nsx][nsz] = TXX_x[nsx][nsz]+Source[it];      // location of source
        	//TXX_z[nsx][nsz] = TXX_z[nsx][nsz]+Source[it]; 
		//vx[nsx][nsz] = vx[nsx][nsz]+Source[it]; 
        	vz[nsx][nsz] = vz[nsx][nsz]+Source[it]; 
        	 
        	 
	      	for(ix=0; ix<Nx0; ix++)
	    	{
		    	Record_vx[ix+pml][it] = vx[ix+pml][pml+5];   //Record 
	    	}
	    	
	    	
		for(iz=0; iz<Nz0; iz++)
	    	{
		    	Record_vz[iz+pml][it] = vz[Nz0+pml+5][iz+pml];   //Record 
	    	}

/***************************snapshot****************************/
		if(it==snapshot)
	    	{
		  	fpa = fopen("P_vx_snapshot.dat","wb");
		  	for(ix=0; ix<Nx0; ix++)
	    	    	{
				for(iz=0; iz<Nz0; iz++){
		      			data[iz] = vx[ix+pml][iz+pml];	
				}
				fwrite(data, sizeof(float), Nz0, fpa);
	    	    	}
		  	fclose(fpa);

		  	fpa = fopen("P_vz_snapshot.dat","wb");
		  	for(ix=0; ix<Nx0; ix++)
	    	    	{
				for(iz=0; iz<Nz0; iz++){
					data[iz] = vz[ix+pml][iz+pml];
				}
		      		fwrite(data, sizeof(float), Nz0, fpa);
	    	    	}
		  	fclose(fpa);

	    	}
		
		// wavefield all
		/*for(ix=0; ix<Nx0; ix++){
			for(iz=0; iz<Nz0; iz++){
				data0[iz] = vx[ix+pml][iz+pml];	
			}
			fwrite(data0, sizeof(float), Nz0, fp);
		}*/
	}
	//fclose(fp);
/*****************************record****************************/
  	fpr = fopen("P_Record_vx.dat","wb");
  	for(ix=0; ix<Nx0; ix++)
    	{	
		for(it=0; it<Nt; it++){
			data[it] = Record_vx[ix+pml][it];
		}
      		fwrite(data, sizeof(float), Nt, fpr);
    	}
  	fclose(fpr);

  	fpr = fopen("P_Record_vz.dat","wb");
  	for(iz=0; iz<Nz0; iz++)
    	{
      		for(it=0; it<Nt; it++){
			data[it] = Record_vz[iz+pml][it];
		}
      		fwrite(data, sizeof(float), Nt, fpr);
    	}
  	fclose(fpr);

	free1((void *)Source);
	free1((void *)ddx);
	free1((void *)ddz);

  	free2((void **)Record_vx);
  	free2((void **)Record_vz);

  	free2((void **)vx);
  	free2((void **)vx_x);
  	free2((void **)vx_z);

  	free2((void **)vz);
  	free2((void **)vz_x);
  	free2((void **)vz_z);

  	free2((void **)TXX);
  	free2((void **)TXX_x);
  	free2((void **)TXX_z);

  	free2((void **)TZZ);
  	free2((void **)TZZ_x);
  	free2((void **)TZZ_z);

  	free2((void **)TXZ);
  	free2((void **)TXZ_x);
  	free2((void **)TXZ_z);

  	free2((void **)lambda);
  	free2((void **)mu);
  	free2((void **)rho);
  	free2((void **)vp);
  	free2((void **)vs);

  	free1((void *)data0);
  	
  	#ifdef _OPENMP
        end_time = omp_get_wtime();
        printf("Totally %f(s).\n",end_time - start_time);
    #endif
}


/* allocate a 2-d array */
/*void **alloc2 (int n1, int n2, int size)*/
void **alloc2 (size_t n1, size_t n2, size_t size)
{
        int i2;
        void **p;

        if ((p=(void**)malloc(n2*sizeof(void*)))==NULL)
                return NULL;
        if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
                free(p);
                return NULL;
        }
        for (i2=0; i2<n2; i2++)
                p[i2] = (char*)p[0]+size*n1*i2;
        return p;
}

/* free a 2-d array */
void free2 (void **p)
{
        free(p[0]);
        free(p);
}

/* allocate a 1-d array */
void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=(void *)malloc(n1*size))==NULL)
		return NULL;
	return p;
}

/* free a 1-d array */
void free1 (void *p)
{
	free(p);
}

void Zero2D(float **BUF2, int N1, int N2)
{
	int i,j;
	for(i=0; i<N1; i++) {
	    for(j=0; j<N2; j++) {
	        BUF2[i][j] = 0.0;
	    }
	}
	
}

void Zero1D(float *BUF1, int N1)
{
	int i,j;
	for(i=0; i<N1; i++) {
	    BUF1[i] = 0.0;
	}
	
}


//void create_model(float **vp, float **vs, float **rho, float **lambda, float **mu, int N1, int N2)
void create_model(float **vp, float **vs, float **rho, int Nx, int Nz)
{
	int ix,iz;
	for (ix=0; ix<Nx; ix++)
		for (iz=0; iz<Nz; iz++)
		{
			if(iz<(int)(Nz/20)){
				vp[ix][iz]=3000.0;
				vs[ix][iz]=1230.0;
				rho[ix][iz]=2.665;
			}
			else if(iz<(int)(3*Nz/20) && iz>=(int)(Nz/20))
			{
				vp[ix][iz]=3200.0;
				vs[ix][iz]=1580.0;
				rho[ix][iz]=2.782;
			}
			else if(iz<(int)(5*Nz/20) && iz>=(int)(3*Nz/20))
			{
				vp[ix][iz]=3300.0;
				vs[ix][iz]=1640.0;
				rho[ix][iz]=2.811;
			}
			else if(iz<(int)(8*Nz/20) && iz>=(int)(5*Nz/20))
			{
				vp[ix][iz]=3600.0;
				vs[ix][iz]=1770.0;
				rho[ix][iz]=2.897;
			}
			else 
			{	
				vp[ix][iz]=4000.0;
				vs[ix][iz]=2440.0;
				rho[ix][iz]=3.302;
			}
		}
}

//将模型扩边,用于 PML
void extmodel(float **ext_model, float **init_model,int Nx,int Nz,int pml)
{
	float **p;
	int ix,iz;
	int Nx2=Nx+2*pml;
	int Nz2=Nz+2*pml;
	p = ext_model;

	for (ix=pml; ix<Nx+pml; ix++){
		for (iz=0; iz<pml; iz++){
			ext_model[ix][iz]=init_model[ix-pml][0]; //  1
		}
	}
	for (ix=pml; ix<Nx+pml; ix++){
		for (iz=Nz+pml; iz<Nz2; iz++){
			ext_model[ix][iz]=init_model[ix-pml][Nz-1];  //  2
		}
	}
	for (ix=Nx+pml; ix<Nx2; ix++){
		for (iz=pml; iz<pml+Nz; iz++){
			ext_model[ix][iz]=init_model[Nx-1][iz-pml];  //  3
		}
	}
	for(ix=0; ix<pml; ix++){
		for(iz=pml; iz<Nz+pml; iz++){
			ext_model[ix][iz]=init_model[0][iz-pml];  //  4
		}
	}
	for(ix=0; ix<pml; ix++){
		for(iz=0; iz<pml; iz++){
			ext_model[ix][iz]=init_model[0][0];  //  5
		}
	}
	for(ix=0; ix<pml; ix++){
		for(iz=Nz+pml; iz<Nz2; iz++){
			ext_model[ix][iz]=init_model[0][Nz-1];  //  6
		}
	}
	for (ix=Nx+pml; ix<Nx2; ix++){
		for (iz=0; iz<pml; iz++){
			ext_model[ix][iz]=init_model[Nx-1][0];  //  7
		}
	}
	for (ix=Nx+pml; ix<Nx2; ix++){
		for (iz=Nz+pml; iz<Nz2; iz++){
			ext_model[ix][iz]=init_model[Nx-1][Nz-1];  //  8
		}
	}
	for (ix=pml; ix<Nx+pml; ix++){
		for (iz=pml; iz<Nz+pml; iz++){
			ext_model[ix][iz]=init_model[ix-pml][iz-pml];  //  9
		}
	}

}

float maxvelocity(float **v, int Nx, int Nz)
{
	int ix, iz;
	float vmax;
	
	vmax = 0.0;
	
	for(ix=0; ix<Nx; ix++) {
		for(iz=0; iz<Nz; iz++) {
			if(v[ix][iz]>vmax) {
				vmax = v[ix][iz];
			}
		}
	}
	return vmax;
}
			



