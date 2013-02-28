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
