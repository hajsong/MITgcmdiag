from MITgcmutils import rdmds
import numpy as np

def loadgrid(gridname,region=None,varname=None):
    """
    [Function]
    grd=loadgrid(gridname,region=None,varname=None):

    [Description]
    Read grid files and load them into the 'grd' object
    
    [Inputs]
    gridname : an identification string. 
               Once 'gridname' is given, 'dirGrid' will be assigned. 
               A user must define 'dirGrid' beforehand. Please see the code.
    region   : A list defining the boundary for the grid.
               [x0,x1,y0,y1] (default is [0,nx,0,ny])
    varname  : A list of strings for the grid data to load

    [Output]
    grd      : An object containing grid data.
               
    """

    if gridname=="dimes50":
        dirGrid="/Users/hajsong/School/DIMES/DIMES_OCCA/Data/"
    elif gridname=="arctic4":
        dirGrid="/Users/hajsong/School/Arctic/4km/Data/"
    elif gridname=="arctic36":
        dirGrid="/Users/hajsong/School/Arctic/36km/Data/"
    elif gridname=="dimes1deg":
        dirGrid="/Volumes/BigLacie/DIMES_OCCA/dimes_occa_dic_1deg/"
    elif gridname=="so_box":
        dirGrid="/Users/hajsong/MITgcm_contrib/hajsong/MITgcmdiag/Data/";

    if varname is None:
        varname=['XC','YC','RAC','DXC','DYC','hFacC','hFacW','hFacS','Depth',\
                 'RC','RF','DRC','DRF','XG','YG','RAZ','DXG','DYG'];

    class grd(object):
        for iv,vname in enumerate(varname):
            if region is None:
                exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'")');
                tmpvar=tmpvar.squeeze();
                exec(varname[iv]+'=tmpvar')
            else:
                if vname is 'RC' or vname is 'RF' or vname is 'DRC' or vname is 'DRF':
                    exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'")');
                else: 
                    exec('tmpvar=rdmds("'+dirGrid+varname[iv]+'",region='+str(region)+')');
                tmpvar=tmpvar.squeeze();
                exec(varname[iv]+'=tmpvar')
            if vname=='hFacC':
                mskC=hFacC.copy()
                mskC[mskC==0]=np.nan
                mskC[np.isfinite(mskC)]=1.
            if vname=='hFacW':
                mskW=hFacW.copy()
                mskW[mskW==0]=np.nan
                mskW[np.isfinite(mskW)]=1.
            if vname=='hFacS':
                mskS=hFacS.copy()
                mskS[mskS==0]=np.nan
                mskS[np.isfinite(mskS)]=1.
            del tmpvar
    del grd.iv,grd.vname

    return grd
