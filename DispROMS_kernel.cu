// declare texture reference for 2D float texture

texture<float, 2, cudaReadModeElementType> texU;
texture<float, 2, cudaReadModeElementType> texV;
texture<float, 2, cudaReadModeElementType> texlonu;
texture<float, 2, cudaReadModeElementType> texlatu;
texture<float, 2, cudaReadModeElementType> texdXU;
texture<float, 2, cudaReadModeElementType> texdYV;



__global__ void HD_interp(int nnode,int stp,int backswitch, int nhdstp, float dt, float hddt/*,float *Umask*/,float * Uold,float * Unew, float * UU)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tx =threadIdx.x;

	__shared__ float Uxo[16];
	__shared__ float Uxn[16];
	__shared__ float Ums[16];
	
	
	float fac=1.0;
	/*Ums[tx]=Umask[ix];*/
	
	
	if (backswitch>0)
	{
		fac=-1.0f;
	}
	
	
	if (ix<nnode)
	{
		Uxo[tx]=fac*Uold[ix]/**Ums[tx]*/;
		Uxn[tx]=fac*Unew[ix]/**Ums[tx]*/;
		
		UU[ix]=Uxo[tx]+(stp*dt-hddt*nhdstp)*(Uxn[tx]-Uxo[tx])/hddt;
	}
}


__global__ void NextHDstep(int nnode, float * Uold,float * Unew)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	
	
	if (ix<nnode )
	{
		Uold[ix]=Unew[ix];
	}
}

__global__ void updatepartpos(int npart,float dt,float Eh,float * dd_rand,float *xx, float *yy,float *zz,float *tt)	 
{
	int i = blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
	
	float Ux=0.0f;
	float Vx=0.0f;
	
	float distu, distv;
	
	
	float xxx,yyy,ttt;
	xxx=xx[i];
	yyy=yy[i];
	ttt=tt[i];
		
	if(ttt>=0.0f)
	{
	//Interpolate wter depth, Uvel Vvel at the particle position
     
    Ux=tex2D(texU, xxx, yyy);
    Vx=tex2D(texV, xxx+0.5, yyy-0.5);// U and V don't have the same coordinates but in the number of nodes it is just off by half a grid node in both dimension
    distu=tex2D(texdXU, xxx, yyy);
    distv=tex2D(texdYV, xxx+0.5, yyy-0.5);
    
  	xx[i]=xxx+(Ux*dt+(dd_rand[i]*2-1)*sqrtf(6*Eh*dt))/distu;
	yy[i]=yyy+(Vx*dt+(dd_rand[npart-i]*2-1)*sqrtf(6*Eh*dt))/distv;
	}
	tt[i]=ttt+dt;
	
    
}

__global__ void ij2lonlat(int npart, float * xx, float *yy, float *xp, float *yp)
{
	int i = blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
	float lon;
	float lat;
	float xxx,yyy;
	xxx=xx[i];
	yyy=yy[i];
	
	lon=tex2D(texlonu, xxx, yyy);
    lat=tex2D(texlatu, xxx, yyy);
    
    xp[i]=lon;
    yp[i]=lat;
	
	//
	
} 
