//////////////////////////////////////////////////////////////////////////////////
//DispROMS_GPU                                                                    //
//Copyright (C) 2013 Bosserelle                                                 //
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <netcdf.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>


#include <math.h>

// includes, GL
#include <GL/glew.h> 
#include <GL/glut.h>

#include <cuda_gl_interop.h>
#include <vector_types.h>

#define pi 3.14159265


#include "DispROMS_kernel.cu"

char ncfile[256];
char ncoutfile[256];

int nxiu,nxiv,netau,netav,nl,nt; 

float *Uo,*Un;
float *Vo,*Vn;
float *Uo_g,*Un_g,*Ux_g;
float *Vo_g,*Vn_g,*Vx_g;
float *Umask, *Vmask, *Umask_g, *Vmask_g;

float *Nincel,*cNincel,*cTincel;
float *Nincel_g,*cNincel_g,*cTincel_g;

float *distXU, *distYU, *distXV, *distYV;

float *lat_u,*lon_u,*lat_v,*lon_v;

int hdstep,hdstart,hdend;
int lev;

float hddt;
int stp,outstep,nextoutstep,outtype;


float *xp,*yp,*zp,*tp;
float *xl,*yl;
float *xp_g,*yp_g,*zp_g,*tp_g;
float *xl_g,*yl_g;

//particle properties
int npart,backswitch;
float dt,Eh,Ev,minrwdepth;

int GPUDEV=0;

int SEED = 777;
float * d_Rand; //GPU random number
curandGenerator_t gen;

cudaError CUDerr;

cudaArray* Ux_gp;
cudaArray* Vx_gp;
cudaArray* distXU_gp;
cudaArray* distYU_gp;
cudaArray* distXV_gp;
cudaArray* distYV_gp;
cudaArray* lon_ugp;
cudaArray* lon_vgp;
cudaArray* lat_ugp;
cudaArray* lat_vgp;

cudaChannelFormatDesc channelDescU = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescV = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescdXU = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescdXV = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescdYU = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescdYV = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesclonu = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesclonv = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesclatu = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesclatv = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);


/////////////////////////////////
//Define external functions
////////////////////////////////
void writexyz(float * x, float * y, float * z,float *t,float *xl,float *yl,  int npart,char outfile[]);
void readgridsize(char ncfile[],int &nxiu,int &nxiv,int &netau,int &netav,int &nl,int &nt);
void readHDstep(char ncfile[],int nxiu,int nxiv,int netau,int netav,int nl,int nt, int hdstep,int lev,float *&Uo,float *&Vo);
void readlatlon(char ncfile[],int nxiu,int nxiv,int netau,int netav,float *&lat_u,float *&lon_u,float *&lat_v,float *&lon_v);
void readUVmask(char ncfile[],int nxiu,int nxiv,int netau,int netav,float *&Uo,float *&Vo);

void creatncfile(char outfile[], int nx,int ny,float *xval, float *yval,float totaltime,float *Nincel,float *cNincel,float *cTincel);
void writestep2nc(char outfile[], int nx,int ny,float totaltime,float *Nincel,float *cNincel,float * cTincel);

template <class T> const T& min (const T& a, const T& b);
template <class T> const T& max (const T& a, const T& b);
template <class T> const T& round(const T& a);




void CUDA_CHECK(cudaError CUDerr)
{    


    if( cudaSuccess != CUDerr) {                                             

        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \

                __FILE__, __LINE__, cudaGetErrorString( CUDerr) );     

        exit(EXIT_FAILURE);

    }
}


void runCuda(void)
{
	float data_nu=netau*nxiu;
	float data_nv=netav*nxiv; 
	dim3 blockDimHD(16, 1, 1);
	dim3 gridDimHD(ceil(max(netau,netav)*max(nxiu,nxiv) / (float)blockDimHD.x), 1, 1);
		
	if(stp*dt>=hddt*(hdstep-hdstart+1))//Not sure about the +1
	{
	  //Read next step
	  
	  hdstep++;
	  
	  int steptoread=hdstep;
	  
	  if (backswitch>0)
	  {
		  steptoread=hdend-hdstep;
	  }
	  readHDstep(ncfile,nxiu,nxiv,netau,netav,nl,nt,steptoread,lev,Un,Vn);
		
	   NextHDstep<<<gridDimHD, blockDimHD, 0>>>(data_nu,Uo_g,Un_g);
	   CUDA_CHECK( cudaThreadSynchronize() );
	   NextHDstep<<<gridDimHD, blockDimHD, 0>>>(data_nv,Vo_g,Vn_g);
	   CUDA_CHECK( cudaThreadSynchronize() );
	   
	  CUDA_CHECK( cudaMemcpy(Un_g, Un, data_nu*sizeof(float ), cudaMemcpyHostToDevice) );
	  CUDA_CHECK( cudaMemcpy(Vn_g, Vn, data_nv*sizeof(float ), cudaMemcpyHostToDevice) );
	  
    }

	ResetNincel<<<gridDimHD, blockDimHD, 0>>>(data_nu,Nincel_g);
	CUDA_CHECK( cudaThreadSynchronize() );
    
    int interpstep=hdstep-hdstart+1;
    HD_interp<<<gridDimHD, blockDimHD, 0>>>(data_nu,stp,backswitch,interpstep,dt,hddt/*,Umask_g*/,Uo_g,Un_g,Ux_g);
	CUDA_CHECK( cudaThreadSynchronize() );
	
	HD_interp<<<gridDimHD, blockDimHD, 0>>>(data_nv,stp,backswitch,interpstep,dt,hddt/*,Vmask_g*/,Vo_g,Vn_g,Vx_g);
	CUDA_CHECK( cudaThreadSynchronize() );
	  
	CUDA_CHECK( cudaMemcpyToArray( Ux_gp, 0, 0, Ux_g, data_nu* sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK( cudaMemcpyToArray( Vx_gp, 0, 0, Vx_g, data_nv* sizeof(float), cudaMemcpyDeviceToDevice));
	  
	//Generate some random numbers
	// Set seed 
	//curandSetPseudoRandomGeneratorSeed(gen, SEED);
	// Generate n floats on device 
	curandGenerateUniform(gen, d_Rand, npart);

		
	//run the model
	int nbblocks=npart/256;
	dim3 blockDim(256, 1, 1);
	dim3 gridDim(npart / blockDim.x, 1, 1);
	
	//Calculate particle step

	updatepartpos<<<gridDim, blockDim, 0>>>(npart,dt,Eh,d_Rand,xl_g, yl_g,zp_g,tp_g);
	CUDA_CHECK( cudaThreadSynchronize() );
	
	ij2lonlat<<<gridDim, blockDim, 0>>>(npart,xl_g,yl_g,xp_g,yp_g);
	CUDA_CHECK( cudaThreadSynchronize() );   
	
	CalcNincel<<<gridDim, blockDim, 0>>>(npart,nxiu,netau,xl_g, yl_g,tp_g,Nincel_g,cNincel_g,cTincel_g);
	CUDA_CHECK( cudaThreadSynchronize() );
		
}


int main(int argc, char **argv)
{
	 //////////////////////////////////////////////////////
    /////             Read Operational file           /////
    //////////////////////////////////////////////////////
 
	char opfile[]="Disp_G3.dat";

	
	//char hdfile[256];
	char seedfile[256];


	float xcenter;
    float ycenter;
    float LL;
    float HH;
	


	///////////////////////////////////////////////////////////
	//Read Operational file
	///////////////////////////////////////////////////////////

	FILE * fop;
    fop=fopen(opfile,"r");
	
	fscanf(fop,"%*s %s\t%*s",&ncfile);
	fscanf(fop,"%d\t%*s",&GPUDEV);
	fscanf(fop,"%f\t%*s",&hddt);
	fscanf(fop,"%d\t%*s",&lev);
	fscanf(fop,"%d,%d\t%*s",&hdstart,&hdend);
    fscanf(fop,"%u\t%*s",&npart);
    fscanf(fop,"%d\t%*s",&backswitch);
    fscanf(fop,"%f\t%*s",&dt);
    fscanf(fop,"%f\t%*s",&Eh);
    fscanf(fop,"%f\t%*s",&Ev);
	fscanf(fop,"%f\t%*s",&minrwdepth);
	fscanf(fop,"%f,%f\t%*s",&xcenter,&ycenter);
    fscanf(fop,"%f,%f\t%*s",&LL,&HH);
	fscanf(fop,"%s\t%*s",&seedfile);
	fscanf(fop,"%d\t%*s",&outtype);
	fscanf(fop,"%d\t%*s",&outstep);
	fscanf(fop,"%s\t%*s",&ncoutfile);
	fclose(fop);
	
	printf(" ncfile:%s\n Hddt:%f\t lev:%d\n Hdstart:%d \t Hdstop:%d\n npart:%d\n dt:%f\n Eh:%f\t Ev:%f\n Mindepth:%f\n Xcenter:%f\t Ycenter:%f\n LL:%f\t HH:%f\n Seed file:%s\n",ncfile,hddt,lev,hdstart,hdend,npart,dt,Eh,Ev,minrwdepth,xcenter,ycenter,LL,HH,seedfile);

	//read the dimentions of grid, levels and time 
	printf("Read nc file dimensions... ");
	readgridsize(ncfile,nxiu,nxiv,netau,netav,nl,nt);
	printf("...done\n");


	///////////////////////////
	//Allocate Memory on CPU //
	///////////////////////////
	printf("Allocate CPU memory... ");
	//Vel ARRAYS
	Uo= (float *)malloc(nxiu*netau*sizeof(float ));
	Un= (float *)malloc(nxiu*netau*sizeof(float ));
	Vo= (float *)malloc(nxiv*netav*sizeof(float ));
	Vn= (float *)malloc(nxiv*netav*sizeof(float ));
	Umask= (float *)malloc(nxiu*netau*sizeof(float ));
	Vmask= (float *)malloc(nxiv*netav*sizeof(float ));
		
	//Lat and Long for each array
	lat_u= (float *)malloc(nxiu*netau*sizeof(float ));
	lon_u= (float *)malloc(nxiu*netau*sizeof(float ));
	lat_v= (float *)malloc(nxiv*netav*sizeof(float ));
	lon_v= (float *)malloc(nxiv*netav*sizeof(float ));
	
	//Distance arrays
	distXU=(float *)malloc(nxiu*netau*sizeof(float ));
	distYU=(float *)malloc(nxiu*netau*sizeof(float ));
	distXV=(float *)malloc(nxiv*netav*sizeof(float ));
	distYV=(float *)malloc(nxiv*netav*sizeof(float ));

	//particles
	xp = (float *)malloc(npart*sizeof(float));
	yp = (float *)malloc(npart*sizeof(float));
	zp = (float *)malloc(npart*sizeof(float));
	tp = (float *)malloc(npart*sizeof(float));
	xl = (float *)malloc(npart*sizeof(float));
	yl = (float *)malloc(npart*sizeof(float));


		//Nincel
	Nincel= (float *)malloc(nxiu*netau*sizeof(float ));
	cNincel= (float *)malloc(nxiu*netau*sizeof(float ));
	cTincel= (float *)malloc(nxiu*netau*sizeof(float ));


	for (int i=0; i<nxiu; i++)
	{
		for (int j=0; j<netau; j++)
		{
			Nincel[i+j*nxiu]=0.0f;
		}
	}


	printf("...done\n");
	
	
	
	//read lat and lon
	printf("Read lat lon array... ");
	readlatlon(ncfile,nxiu,nxiv,netau,netav,lat_u,lon_u,lat_v,lon_v);
	printf(" ...Calculate distance array...");
	
	 
	float R = 6372797.560856;
	//Calculate the distance in meter between each cells for u grid 
	for (int i=0; i<nxiu-1; i++)
	{
		for (int j=0; j<netau; j++)
		{
			//calc distance between each i using haversine formula
			float dlat=(lat_u[(i+1)+j*nxiu]-lat_u[i+j*nxiu])*pi/180.0f;
			float lat1=lat_u[i+j*nxiu]*pi/180.0f;
			float lat2=lat_u[(i+1)+j*nxiu]*pi/180.0f;
			float dlon=(lon_u[(i+1)+j*nxiu]-lon_u[i+j*nxiu])*pi/180.0f;
			
			float a=sin(dlat/2)*sin(dlat/2)+cos(lat1)*cos(lat2)*sin(dlon/2)*sin(dlon/2);
			float c=2*atan2f(sqrtf(a),sqrtf(1-a));
			distXU[i+j*nxiu]=c*R;					
					
		}
	}
	for (int i=0; i<nxiu; i++)
	{
		for (int j=0; j<netau-1; j++)
		{
			//calc distance between each j using haversine formula
			float dlat=(lat_u[i+(j+1)*nxiu]-lat_u[i+j*nxiu])*pi/180.0f;
			float lat1=lat_u[i+j*nxiu]*pi/180.0f;
			float lat2=lat_u[i+(j+1)*nxiu]*pi/180.0f;
			float dlon=(lon_u[i+(j+1)*nxiu]-lon_u[i+j*nxiu])*pi/180.0f;
			
			float a=sin(dlat/2)*sin(dlat/2)+cos(lat1)*cos(lat2)*sin(dlon/2)*sin(dlon/2);
			float c=2*atan2f(sqrtf(a),sqrtf(1-a));
			distYU[i+j*nxiu]=c*R;					
			
		}
	}
		
	//fill in boundaries
	for (int j=0; j<netau; j++)
	{
		//
		distXU[nxiu-1+j*nxiu]=distXU[nxiu-2+j*nxiu];
	}
	for (int i=0; i<nxiu; i++)
	{
		//
		distYU[i+(netau-1)*nxiu]=distYU[i+(netau-2)*nxiu];
	}
	
	//Vdirection
	for (int i=0; i<nxiu-1; i++)
	{
		for (int j=0; j<netau; j++)
		{
			//calc distance between each i using haversine formula
			float dlat=(lat_v[(i+1)+j*nxiu]-lat_v[i+j*nxiu])*pi/180.0f;
			float lat1=lat_v[i+j*nxiu]*pi/180.0f;
			float lat2=lat_v[(i+1)+j*nxiu]*pi/180.0f;
			float dlon=(lon_v[(i+1)+j*nxiu]-lon_v[i+j*nxiu])*pi/180.0f;
			
			float a=sin(dlat/2)*sin(dlat/2)+cos(lat1)*cos(lat2)*sin(dlon/2)*sin(dlon/2);
			float c=2*atan2f(sqrtf(a),sqrtf(1-a));
			distXV[i+j*nxiu]=c*R;					
					
		}
	}
	for (int i=0; i<nxiu; i++)
	{
		for (int j=0; j<netau-1; j++)
		{
			//calc distance between each j using haversine formula
			float dlat=(lat_v[i+(j+1)*nxiu]-lat_v[i+j*nxiu])*pi/180.0f;
			float lat1=lat_v[i+j*nxiu]*pi/180.0f;
			float lat2=lat_v[i+(j+1)*nxiu]*pi/180.0f;
			float dlon=(lon_v[i+(j+1)*nxiu]-lon_v[i+j*nxiu])*pi/180.0f;
			
			float a=sin(dlat/2)*sin(dlat/2)+cos(lat1)*cos(lat2)*sin(dlon/2)*sin(dlon/2);
			float c=2*atan2f(sqrtf(a),sqrtf(1-a));
			distYV[i+j*nxiu]=c*R;					
			
		}
	}
		
	//fill in boundaries
	for (int j=0; j<netau; j++)
	{
		//
		distXV[nxiu-1+j*nxiu]=distXV[nxiu-2+j*nxiu];
	}
	for (int i=0; i<nxiu; i++)
	{
		//
		distYV[i+(netau-1)*nxiu]=distYV[i+(netau-2)*nxiu];
	}
	
	printf("...done\n");

	//Calculate first HD step
	//outstep=10;
	stp=0;//hdstart*hddt/dt;
	hdstep=hdstart;
	nextoutstep=outstep+stp;
	//printf("HD step:%d\n ",hdstep);
	if (hdend==0)
	{
		hdend=nt-1;
	}

	int steptoread=hdstep;
	  
	  if (backswitch>0)
	  {
		  steptoread=hdend-hdstep;
	  }
	//////////////////////////////
	//Read first step in Hd model
	///////////////////////////////
	//Read U and V mask
	//readUVmask(ncfile,nxiu,nxiv,netau,netav,Umask,Vmask);

	//printf("Read Hd model first step... ");
	readHDstep(ncfile,nxiu,nxiv,netau,netav,nl,nt,steptoread,lev,Uo,Vo);
	//printf("...done\n");

	//////////////////////////////
	// Init Particle position
	//////////////////////////////
	
	char noseedfile[] = "seed";
	  if (strcmp(seedfile,noseedfile)!=0)
	  {
		  printf("...reading seed file.\n");
		  FILE * fsd;
		  int nseedpos;

		  //read input data: 
		  fsd=fopen(seedfile,"r");
		  fscanf(fsd,"%u",&nseedpos);

		  for(int ppos=0; ppos<min(nseedpos,npart); ppos++)
		  {
			  fscanf(fsd,"%f %f %f %f %f %f",&xp[ppos],&yp[ppos],&zp[ppos],&tp[ppos],&xl[ppos],&yl[ppos]);
		  }


		  if (nseedpos<npart)
		  {
			  printf("WARNING there are less seed positions in file than particles seed position will be repeated");
			  for (int rppos=0; rppos<(npart-nseedpos);rppos++)
			  {
				  xp[nseedpos+rppos]=xp[rppos];
				  yp[nseedpos+rppos]=yp[rppos];
				  zp[nseedpos+rppos]=zp[rppos];
				  tp[nseedpos+rppos]=tp[rppos];
				  xl[nseedpos+rppos]=xl[rppos];
				  yl[nseedpos+rppos]=yl[rppos];			
			  }


		  }

		fclose(fsd);

		}
	  else
		  {

			printf("Generating particle initial position in CPU mem...");
   
			//Generating input data on CPU
 
			//Set initial position for particle
			float dlat=0.1;
			float dlon=0.1;
			float a,c,d;
			float R=6371000;
			float dlatrad,dlonrad,ycenterrad;
			dlatrad=dlat*pi/180;
			dlonrad=dlon*pi/180;
			ycenterrad=ycenter*pi/180;
	
	
			a=pow(sin(dlatrad/2),2)+cos(ycenterrad)*cos(ycenterrad+dlatrad)*pow(sin(dlonrad/2),2);
			c=2*atan2(sqrt(a),sqrt(1-a));
			d=R*c;
	
	
	
			LL=LL*dlon/d;
			HH=HH*dlat/d;
			float ddx=sqrt(LL*HH/npart);
	
			for(int i = 0; i < npart; i++)
			{
				// initialize random seed: 
		   
				float dist;
				int minki=0;
				int minkj=0;
				int test;
				float mindist=100000;
				
		   
		
				zp[i] = 0.5f;
				tp[i] = 0.0f;
             
				xp[i]=(xcenter-LL/2)+ddx*(i - floor(((float)i)/round(LL/ddx))*round(LL/ddx));
				//printf("xp[%d]=%f\n",i,xp[i]);
				yp[i]=(ycenter-HH/2)+ddx*(floor((float)i/(LL/ddx)));
				//printf("yp[%d]=%f\n",i,yp[i]);
				
				//
				for (int kj=0;kj<netau-1;kj++)
				{
					for (int ki=0; ki<nxiu-1;ki++)
					{
						dist=sqrt((lon_u[ki+kj*nxiu]-xp[i])*(lon_u[ki+kj*nxiu]-xp[i])+(lat_u[ki+kj*nxiu]-yp[i])*(lat_u[ki+kj*nxiu]-yp[i]));
						if (dist<mindist)
						{
							mindist=dist;
							minki=ki;
							minkj=kj;
						}
					}
				}
				//Yes I know pretty lazy stuff it is for quick seeding if you want the proper stuff do a seed file...
				xp[i]=lon_u[minki+minkj*nxiu];
				yp[i]=lat_u[minki+minkj*nxiu];
				xl[i]=minki;
				yl[i]=minkj;
				//xlv[i]=minki+0.5;in i and j coordinate it is jut off by half the node
				//ylu[i]=minkj-0.5;
				
			
           
			}
	
		}

	 printf(" ...done\n");


	/////////////////////////////
	//Prepare GPU
	////////////////////////////
	// Init GPU data
	int GPUDEVICE=GPUDEV;
	CUDA_CHECK(cudaSetDevice(GPUDEVICE));
	
	
	
	//CUT_DEVICE_INIT(argc, argv);
	
	/////////////////////////////////////
	// ALLOCATE GPU MEMORY
	/////////////////////////////////
	printf("Allocating GPU memory... ");

	float DATA_SZ=npart*sizeof(float);
			
	CUDA_CHECK(cudaMalloc((void **)&xp_g, DATA_SZ));
	CUDA_CHECK(cudaMalloc((void **)&yp_g, DATA_SZ));
	CUDA_CHECK(cudaMalloc((void **)&zp_g, DATA_SZ));
	CUDA_CHECK(cudaMalloc((void **)&tp_g, DATA_SZ));
	CUDA_CHECK(cudaMalloc((void **)&xl_g, DATA_SZ));
	CUDA_CHECK(cudaMalloc((void **)&yl_g, DATA_SZ));
	
	CUDA_CHECK(cudaMalloc((void **)&Uo_g, netau*nxiu* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&Un_g, netau*nxiu* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&Ux_g, netau*nxiu* sizeof(float)));
	
	CUDA_CHECK(cudaMalloc((void **)&Vo_g, netav*nxiv* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&Vn_g, netav*nxiv* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&Vx_g, netav*nxiv* sizeof(float)));

	CUDA_CHECK(cudaMalloc((void **)&Nincel_g, netau*nxiu* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&cNincel_g, netau*nxiu* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&cTincel_g, netau*nxiu* sizeof(float)));
	
	CUDA_CHECK(cudaMalloc((void **)&Umask_g, netau*nxiu* sizeof(float)));
	CUDA_CHECK(cudaMalloc((void **)&Vmask_g, netav*nxiv* sizeof(float)));
	
	printf(" ...done\n");
	
	printf("Transfert vectors to GPU memory... ");
	CUDA_CHECK( cudaMemcpy(Uo_g, Uo, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(Un_g, Uo, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(Ux_g, Uo, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaMemcpy(Vo_g, Vo, netav*nxiv*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(Vn_g, Vo, netav*nxiv*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(Vx_g, Vo, netav*nxiv*sizeof(float ), cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaMemcpy(Umask_g, Umask, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(Vmask_g, Vmask, netav*nxiv*sizeof(float ), cudaMemcpyHostToDevice) );

	CUDA_CHECK( cudaMemcpy(Nincel_g, Nincel, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(cNincel_g, Nincel, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(cTincel_g, Nincel, netau*nxiu*sizeof(float ), cudaMemcpyHostToDevice) );
	
	CUDA_CHECK( cudaMemcpy(xp_g, xp, npart*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(yp_g, yp, npart*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(zp_g, zp, npart*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(tp_g, tp, npart*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(xl_g, xl, npart*sizeof(float ), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(yl_g, yl, npart*sizeof(float ), cudaMemcpyHostToDevice) );
	
	// Loading random number generator
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	
	CUDA_CHECK( cudaMalloc((void **)&d_Rand,npart*sizeof(float)) );
	
	printf(" ...done\n");
	
	printf("Create textures on GPU memory... ");
	// Copy velocity arrays
	CUDA_CHECK( cudaMallocArray( &Ux_gp, &channelDescU, nxiu,netau  ));
    CUDA_CHECK( cudaMallocArray( &Vx_gp, &channelDescV, nxiv,netav  ));

	CUDA_CHECK( cudaMemcpyToArray( Ux_gp, 0, 0, Uo, netau*nxiu* sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpyToArray( Vx_gp, 0, 0, Vo, netav*nxiv* sizeof(float), cudaMemcpyHostToDevice));

	texU.addressMode[0] = cudaAddressModeWrap;
    texU.addressMode[1] = cudaAddressModeWrap;
    texU.filterMode = cudaFilterModeLinear;
    texU.normalized = false;
       
 
    CUDA_CHECK( cudaBindTextureToArray( texU, Ux_gp, channelDescU));
 
    texV.addressMode[0] = cudaAddressModeWrap;
    texV.addressMode[1] = cudaAddressModeWrap;
    texV.filterMode = cudaFilterModeLinear;
    texV.normalized = false;
 
    CUDA_CHECK( cudaBindTextureToArray( texV, Vx_gp, channelDescV));
    
    CUDA_CHECK( cudaMallocArray( &distXU_gp, &channelDescdXU, nxiu,netau  ));
	//CUDA_CHECK( cudaMallocArray( &distXV_gp, &channelDescdXV, netav, nxiv ));
	//CUDA_CHECK( cudaMallocArray( &distYU_gp, &channelDescdYU, netau, nxiu ));
	CUDA_CHECK( cudaMallocArray( &distYV_gp, &channelDescdYV, nxiv,netav ));
	
	CUDA_CHECK( cudaMallocArray( &lon_ugp, &channelDesclonu, nxiu,netau  ));
	CUDA_CHECK( cudaMallocArray( &lat_ugp, &channelDesclatu, nxiu, netau ));
	//CUDA_CHECK( cudaMallocArray( &lon_vgp, &channelDesclonv, netav, nxiv ));
	//CUDA_CHECK( cudaMallocArray( &lat_vgp, &channelDesclatv, netav, nxiv ));
		
	CUDA_CHECK( cudaMemcpyToArray(distXU_gp, 0, 0, distXU, netau*nxiu* sizeof(float), cudaMemcpyHostToDevice));
	//CUDA_CHECK( cudaMemcpyToArray(distYU_gp, 0, 0, distYU, netau*nxiu* sizeof(float), cudaMemcpyHostToDevice));
	//CUDA_CHECK( cudaMemcpyToArray(distXV_gp, 0, 0, distXV, netav*nxiv* sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK( cudaMemcpyToArray(distYV_gp, 0, 0, distYV, netav*nxiv* sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK( cudaMemcpyToArray(lon_ugp, 0, 0, lon_u, netau*nxiu* sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK( cudaMemcpyToArray(lat_ugp, 0, 0, lat_u, netau*nxiu* sizeof(float), cudaMemcpyHostToDevice));
	//CUDA_CHECK( cudaMemcpyToArray(lon_vgp, 0, 0, lon_v, netav*nxiv* sizeof(float), cudaMemcpyHostToDevice));
	//CUDA_CHECK( cudaMemcpyToArray(lat_vgp, 0, 0, lat_v, netav*nxiv* sizeof(float), cudaMemcpyHostToDevice));
	
	texlonu.addressMode[0] = cudaAddressModeWrap;
    texlonu.addressMode[1] = cudaAddressModeWrap;
    texlonu.filterMode = cudaFilterModeLinear;
    texlonu.normalized = false;
    
    CUDA_CHECK( cudaBindTextureToArray( texlonu, lon_ugp, channelDesclonu));
    
    texlatu.addressMode[0] = cudaAddressModeWrap;
    texlatu.addressMode[1] = cudaAddressModeWrap;
    texlatu.filterMode = cudaFilterModeLinear;
    texlatu.normalized = false;
    
    CUDA_CHECK( cudaBindTextureToArray( texlatu, lat_ugp, channelDesclatu));
    
    texdXU.addressMode[0] = cudaAddressModeWrap;
    texdXU.addressMode[1] = cudaAddressModeWrap;
    texdXU.filterMode = cudaFilterModeLinear;
    texdXU.normalized = false;
    
    CUDA_CHECK( cudaBindTextureToArray( texdXU, distXU_gp, channelDescdXU));
    
    texdYV.addressMode[0] = cudaAddressModeWrap;
    texdYV.addressMode[1] = cudaAddressModeWrap;
    texdYV.filterMode = cudaFilterModeLinear;
    texdYV.normalized = false;
    
    CUDA_CHECK( cudaBindTextureToArray( texdYV, distYV_gp, channelDescdYV));
    
    printf(" ...done\n");
    
    
    //int nbblocks=npart/256;
	//dim3 blockDim(256, 1, 1);
	//dim3 gridDim(npart / blockDim.x, 1, 1);
    
    //ij2lonlat<<<gridDim, blockDim, 0>>>(npart,xl_g,yl_g,xp_g,yp_g);
	//CUDA_CHECK( cudaThreadSynchronize() );
	//CUDA_CHECK( cudaMemcpy(xp, xp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
	//CUDA_CHECK( cudaMemcpy(yp, yp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
	
    char fileoutn[15];
    sprintf (fileoutn, "Part_%d.xyz", stp);
    writexyz(xp,yp,zp,tp,xl,yl,npart,fileoutn);

	creatncfile(ncoutfile,nxiu,netau,lon_u,lat_u,stp*dt,Nincel,Nincel,Nincel);
    
    printf("Running Model...\n");
    
    //Run the model without the GL stuff
	while (stp*dt<=hddt*hdend)
	{
		runCuda();
		
		if (stp==nextoutstep)
	 {
		 char fileoutn[15];
		 nextoutstep=nextoutstep+outstep;
		 switch (outtype)
		 {
		 case 1:
			 
			 
			 sprintf (fileoutn, "Part_%d.xyz", stp);
	 
			 //Get the results to plot.
			 CUDA_CHECK( cudaMemcpy(xp, xp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(yp, yp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 //CUDA_CHECK( cudaMemcpy(zp, zp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(tp, tp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(xl, xl_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(yl, yl_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 //printf("saving Part_%d.xyz file", stp);
			 writexyz(xp,yp,zp,tp,xl,yl,npart,fileoutn);
			 break;

		 case 2:

			 CUDA_CHECK( cudaMemcpy(Nincel, Nincel_g, nxiu*netau*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(cNincel, cNincel_g, nxiu*netau*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(cTincel, cTincel_g, nxiu*netau*sizeof(float), cudaMemcpyDeviceToHost) );
	 
			 writestep2nc(ncoutfile,nxiu,netau,stp*dt,Nincel,cNincel,cTincel);
			 break;

		 case 3:

			 sprintf (fileoutn, "Part_%d.xyz", stp);
	 
			 //Get the results to plot.
			 CUDA_CHECK( cudaMemcpy(xp, xp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(yp, yp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 //CUDA_CHECK( cudaMemcpy(zp, zp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(tp, tp_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(xl, xl_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(yl, yl_g, npart*sizeof(float), cudaMemcpyDeviceToHost) );
			 //printf("saving Part_%d.xyz file", stp);
			 writexyz(xp,yp,zp,tp,xl,yl,npart,fileoutn);


			 CUDA_CHECK( cudaMemcpy(Nincel, Nincel_g, nxiu*netau*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(cNincel, cNincel_g, nxiu*netau*sizeof(float), cudaMemcpyDeviceToHost) );
			 CUDA_CHECK( cudaMemcpy(cTincel, cTincel_g, nxiu*netau*sizeof(float), cudaMemcpyDeviceToHost) );
	 
			 writestep2nc(ncoutfile,nxiu,netau,stp*dt,Nincel,cNincel,cTincel);
			 break;
		 }

	 
	 }
	 
	 stp++;
    }
    
    
    
    
	
	cudaThreadExit();
	 

}
