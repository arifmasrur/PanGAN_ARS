from osgeo import gdal, gdalconst, gdal_array # type: ignore

## shape of x = (x, y, c)

def write(x, tiff_file, ref_file, gdal_type: int = gdalconst.GDT_Float32, predictor: int = 3):
    """
    USAGE: write_tiff(array, tiff_file, ref_file)
    Use predictor=3 for float types and predictor=2 for integer types.
    """
    gtiff_flags = [ 'COMPRESS=ZSTD', # also LZW and DEFLATE works well
    'ZSTD_LEVEL=9', # should be between 1-22, and 22 is highest compression.
    # 9 is default and gets essentially the same compression-rate
    'PREDICTOR=%d' % predictor, # default is 1, use 2 for ints, and 3 for floats
    'TILED=YES', # so that we can read sub-arrays efficiently
    'BIGTIFF=YES' # in case resulting file is >4GB
    ]
    if x.ndim==3:
       nx, ny, nbands = x.shape
    elif x.ndim==2:
        nx, ny = x.shape
        nbands = 1
    else:
        assert(x.ndim==2 or x.ndim==3)
        
    if not os.path.exists(ref_file):
        raise(FileNotFoundError("<%s> doesn't exist" % ref_file))
    ds = gdal.Open(ref_file)
    if (ds.RasterYSize!=nx) and (ds.RasterXSize!=ny):
        print("Size mismatch between reference file and input array")
        print("x: %s, ref_file: %d, %d" %(x.shape, ds.RasterYSize, ds.RasterXSize))

    outDrv = gdal.GetDriverByName('GTiff')
    out = outDrv.Create(tiff_file, ny, nx, nbands, gdal_type, gtiff_flags )
    out.SetProjection(ds.GetProjection())
    out.SetGeoTransform(ds.GetGeoTransform())
    if x.ndim==3:
        for i in range(nbands):
            out.GetRasterBand(i+1).WriteArray(x[:,:,i])
    else:
        out.GetRasterBand(1).WriteArray(x)
        out.FlushCache()
        
    del out # guarantee the flush
    del ds
