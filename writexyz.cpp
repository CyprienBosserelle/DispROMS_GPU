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

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
using namespace std;
//(xp,yp,zp,tp,xl,yl, npart,fileoutn)
void writexyz(float * x, float * y, float * z,float *t,float *xl,float *yl,  int npart,char outfile[])
{
	FILE * ofile;
	ofile= fopen (outfile,"w");

	for (int i=1; i<=npart; i++)
	{
		if(t[i-1]>=0.0f)// Do not output particle that haven't been released yet
		{
			fprintf(ofile,"%f\t%f\t%f\t%f\t%f\t%f\n",x[i-1],y[i-1],z[i-1],t[i-1],xl[i-1],yl[i-1]);
		}
	}
	fclose (ofile);
}
void creatncfile(char outfile[], int nx,int ny,float *xval, float *yval,float totaltime,float *Nincel,float *cNincel,float *cTincel)
{               
	int status;
   	int ncid,xx_dim,yy_dim,time_dim,p_dim;
	size_t nxx,nyy,nnpart;
	int  var_dimids[3], var2_dimids[2];
	
	int Nin_id,cNin_id,cTin_id,time_id,xx_id,yy_id;
	
	nxx=nx;
	nyy=ny;
	//nnpart=npart;

	static size_t start[] = {0, 0, 0}; // start at first value 
    	static size_t count[] = {1, ny, nx};
		//static size_t pstart[] = {0, 0}; // start at first value 
   // 	static size_t pcount[] = {1, npart};
	static size_t tst[]={0};
	static size_t xstart[] = {0, 0}; // start at first value 
    	static size_t xcount[] = {ny, nx};
	
	static size_t ystart[] = {0, 0}; // start at first value 
    	static size_t ycount[] = {ny, nx};
	




	//create the netcdf dataset
	status = nc_create(outfile, NC_NOCLOBBER, &ncid);
	
	//Define dimensions: Name and length
	
	status = nc_def_dim(ncid, "xx", nxx, &xx_dim);
	status = nc_def_dim(ncid, "yy", nyy, &yy_dim);
	//status = nc_def_dim(ncid, "npart",nnpart,&p_dim);
	status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	int tdim[]={time_dim};
	int xdim[]={xx_dim};
	int ydim[]={yy_dim};
	//int pdim[2];
	//pdim[0]=time_dim;
	//pdim[1]=p_dim;
	//define variables: Name, Type,...
	var_dimids[0] = time_dim;
    var_dimids[1] = yy_dim;
    var_dimids[2] = xx_dim;
	
	var2_dimids[0] = yy_dim;
    var2_dimids[1] = xx_dim;

	
   	status = nc_def_var (ncid, "time", NC_FLOAT,1,tdim, &time_id);
	status = nc_def_var (ncid, "x", NC_FLOAT,2,var2_dimids, &xx_id);
	status = nc_def_var (ncid, "y", NC_FLOAT,2,var2_dimids, &yy_id);
	//status = nc_def_var (ncid, "xxp", NC_FLOAT,2,pdim, &xxp_id);
	//status = nc_def_var (ncid, "yyp", NC_FLOAT,2,pdim, &yyp_id);
	
   	status = nc_def_var (ncid, "Nincel", NC_FLOAT, 3, var_dimids, &Nin_id);
    status = nc_def_var (ncid, "cNincel", NC_FLOAT, 3, var_dimids, &cNin_id);
    status = nc_def_var (ncid, "cTincel", NC_FLOAT, 3, var_dimids, &cTin_id);
    
	//put attriute: assign attibute values
	//nc_put_att

	//End definitions: leave define mode
	status = nc_enddef(ncid);  

	//Provide values for variables
	status = nc_put_var1_float(ncid, time_id,tst,&totaltime);
	status = nc_put_vara_float(ncid, xx_id,xstart,xcount, xval);
	status = nc_put_vara_float(ncid, yy_id,ystart,ycount, yval);
	//status = nc_put_vara_float(ncid, xxp_id,pstart,pcount, xxp);
	//status = nc_put_vara_float(ncid, yyp_id,pstart,pcount, yyp);
		
	
	status = nc_put_vara_float(ncid, Nin_id, start, count, Nincel);
	status = nc_put_vara_float(ncid, cNin_id, start, count, cNincel);
	status = nc_put_vara_float(ncid, cTin_id, start, count, cTincel);
	

	//close and save new file
	status = nc_close(ncid);  
}

void writestep2nc(char outfile[], int nx,int ny,float totaltime,float *Nincel,float *cNincel,float * cTincel)
{
	int status;
   	int ncid,time_dim,recid;
	size_t nxx,nyy;
	int Nincel_id,cNincel_id,cTincel_id,time_id;
	static size_t start[] = {0, 0, 0}; // start at first value 
    	static size_t count[] = {1, ny, nx};
 	//static size_t pstart[] = {0, 0}; // start at first value 
    //	static size_t pcount[] = {1, npart};
	static size_t tst[]={0};

	nxx=nx;
	nyy=ny;

	
	static size_t nrec;
	status = nc_open(outfile, NC_WRITE, &ncid);
	
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen  (ncid, recid, &nrec);
	printf("nrec=%d\n",nrec);

	//read file for variable ids
	status = nc_inq_varid(ncid, "time", &time_id);
	
	status = nc_inq_varid(ncid, "Nincel", &Nincel_id);
	status = nc_inq_varid(ncid, "cNincel", &cNincel_id);
	status = nc_inq_varid(ncid, "cTincel", &cTincel_id);
	
	
	
	start[0] = nrec;
	//pstart[0] = nrec;    
	tst[0]=nrec;

	//Provide values for variables
	status = nc_put_var1_float(ncid, time_id,tst,&totaltime);
	
	status = nc_put_vara_float(ncid, Nincel_id, start, count, Nincel);
	status = nc_put_vara_float(ncid, cNincel_id, start, count, cNincel);
	status = nc_put_vara_float(ncid, cTincel_id, start, count, cTincel);
	
	

	//close and save
	status = nc_close(ncid);


}
