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
#include <string>
#include <math.h>
#include <fstream>
#include <netcdf.h>


// Define Global variables



void handle_error(int status) {
     if (status != NC_NOERR) {
        fprintf(stderr, "%s\n", nc_strerror(status));
        exit(-1);
        }
     }

template <class T> const T& max (const T& a, const T& b) {
  return (a<b)?b:a;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> const T& min (const T& a, const T& b) {
  return !(b<a)?a:b;     // or: return !comp(b,a)?a:b; for version (2)
}

template <class T> const T& round(const T& a)
  {
  return floor( a + 0.5 );
  }

void readgridsize(char ncfile[],int &nxiu,int &nxiv,int &netau,int &netav,int &nl,int &nt)
{
	//read the dimentions of grid, levels and time 
	int status;
	int ncid,ndims,nvars,ngatts,unlimdimid;
	int bath_id,lon_id,lat_id;
	int nx_id,ny_id;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";

	size_t nx_u,ny_u,nx_v,ny_v,nlev,ntime;
	//Open NC file
	status =nc_open(ncfile,0,&ncid);
	if (status != NC_NOERR) handle_error(status);


	// Inquire number of var, dim,attributs and unlimited dimentions
	status = nc_inq(ncid, &ndims, &nvars, &ngatts, &unlimdimid);
	//printf("nvars=%d\n",nvars);

	status = nc_inq_dimid(ncid, "xi_u", &nx_id);
	status = nc_inq_dimlen(ncid, nx_id, &nx_u);
	

	status = nc_inq_dimid(ncid, "eta_u", &ny_id);
	status = nc_inq_dimlen(ncid, ny_id, &ny_u);

	status = nc_inq_dimid(ncid, "xi_v", &nx_id);
	status = nc_inq_dimlen(ncid, nx_id, &nx_v);
	

	status = nc_inq_dimid(ncid, "eta_v", &ny_id);
	status = nc_inq_dimlen(ncid, ny_id, &ny_v);

	status = nc_inq_dimid(ncid, "s_rho", &ny_id);
	status = nc_inq_dimlen(ncid, ny_id, &nlev);

	status = nc_inq_dimid(ncid, "ocean_time", &ny_id);
	status = nc_inq_dimlen(ncid, ny_id, &ntime);

	netau=ny_u;
	netav=ny_v;
	nxiu=nx_u;
	nxiv=nx_v;
	nl=nlev;
	nt=ntime;
	printf("\nncfile:%s\n",ncfile);
	printf("nxiu=%d\t netau=%d\n",nx_u,ny_u);
	printf("nxiv=%d\t netav=%d\n",nx_v,ny_v);
	printf("n level=%d\t n times=%d\n",nlev,ntime);
	
	status = nc_close(ncid);


}
void readlatlon(char ncfile[],int nxiu,int nxiv,int netau,int netav,float *&lat_u,float *&lon_u,float *&lat_v,float *&lon_v)
{
	int status;
    int ncid;
	int uu_id,vv_id;

	size_t start[]={0,0};
	size_t countlu[]={netau,nxiu};
	size_t countlv[]={netav,nxiv};

	//Open NC file
	status =nc_open(ncfile,0,&ncid);
	if (status != NC_NOERR) handle_error(status);

	// U lat and lon 
	status = nc_inq_varid (ncid, "lat_u", &uu_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid,uu_id,start,countlu,lat_u);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid (ncid, "lon_u", &uu_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid,uu_id,start,countlu,lon_u);
	if (status != NC_NOERR) handle_error(status);

	// U lat and lon 
	status = nc_inq_varid (ncid, "lat_v", &vv_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid,vv_id,start,countlv,lat_v);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid (ncid, "lon_v", &vv_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid,vv_id,start,countlv,lon_v);
	if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);

}
void readHDstep(char ncfile[],int nxiu,int nxiv,int netau,int netav,int nl,int nt, int hdstep,int lev,float *&Uo,float *&Vo)
{
	//
	int status;
    int ncid;
	int ndims,nvars,ngatts,unlimdimid;
	int uu_id,vv_id;
	
	printf("Reading HD step: %d ...",hdstep);
	size_t startl[]={hdstep-1,lev,0,0};
	size_t countlu[]={1,1,netau,nxiu};
	size_t countlv[]={1,1,netav,nxiv};
	

	static ptrdiff_t stridel[]={1,1,1,1};

	//Open NC file
	status =nc_open(ncfile,0,&ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid (ncid, "u", &uu_id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "v", &vv_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid,uu_id,startl,countlu,Uo);
	if (status != NC_NOERR) handle_error(status);
	status = nc_get_vara_float(ncid,vv_id,startl,countlv,Vo);
	if (status != NC_NOERR) handle_error(status);

	//Set land flag to 0.0m/s to allow particle to stick to the coast
	
	for (int i=0; i<nxiu; i++)
	{
		for (int j=0; j<netau; j++)
		{
			if (Uo[i+j*nxiu]>10.0f)
			{
				Uo[i+j*nxiu]=0.0f;
			}
		}
	}
	
	for (int i=0; i<nxiv; i++)
	{
		for (int j=0; j<netav; j++)
		{
			if (Vo[i+j*nxiv]>10.0f)
			{
				Vo[i+j*nxiv]=0.0f;
			}
		}
	}
	
	
	
	//printf("U[0][0]=%f\n",Uo[0]);
	
	status = nc_close(ncid);
	printf("...done\n");
}

void readUVmask(char ncfile[],int nxiu,int nxiv,int netau,int netav,float *&Uo,float *&Vo)
{
	//
	int status;
    int ncid;
	int ndims,nvars,ngatts,unlimdimid;
	int uu_id,vv_id;
	
	printf("Reading U and V Mask...");
	size_t startl[]={0,0};
	size_t countlu[]={netau,nxiu};
	size_t countlv[]={netav,nxiv};
	

	static ptrdiff_t stridel[]={1,1};

	//Open NC file
	status =nc_open(ncfile,0,&ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid (ncid, "mask_u", &uu_id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid (ncid, "mask_v", &vv_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid,uu_id,startl,countlu,Uo);
	if (status != NC_NOERR) handle_error(status);
	status = nc_get_vara_float(ncid,vv_id,startl,countlv,Vo);
	if (status != NC_NOERR) handle_error(status);

		
	status = nc_close(ncid);
	printf("...done\n");
}

