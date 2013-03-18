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
