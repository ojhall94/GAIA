import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tempfile
import subprocess
import shutil
import gzip

from astroquery.simbad import Simbad
from astropy.table import Table
from tqdm import tqdm
tqdm.monitor_interval = 0

import barbershop
import glob
import os
import time
import sys

'''A script that downloads TGAS data for me,
totally stolen off Jo Bovys gaia_tools.'''

def get_TGAS(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + 'Gaia_TGAS') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'Gaia_TGAS')

    #Building local and remote paths
    local = [os.path.join(dirloc, 'Gaia_TGAS', 'TgasSource_000-000-%03i.csv.gz' % ii)
                for ii in range(16)]
    remote = [os.path.join('http://cdn.gea.esac.esa.int','Gaia','tgas_source','csv',
                'TgasSource_000-000-%03i.csv.gz' % ii) for ii in range(16)]

    #Download the data
    for localpath, remotepath in list(zip(local, remote)):
        #Check data doesnt already exist
        if not os.path.isfile(localpath):
            download(localpath, remotepath, verbose=True)
        else:
            sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
            sys.stdout.flush()

def get_KIC(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + 'KIC') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'KIC')

    #Building local and remote paths
    localpath = os.path.join(dirloc, 'KIC', 'kic.txt.gz')
    remotepath = os.path.join('http://archive.stsci.edu/pub/kepler/catalogs/','kic.txt.gz')

    #Check data doesnt already exist
    if not os.path.isfile(localpath):
        download(localpath, remotepath, verbose=True)
    else:
        sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
        sys.stdout.flush()

def get_2MASSxTGAS(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + '2MASSxTGAS') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'2MASSxTGAS')

    localpath = os.path.join(dirloc, '2MASSxTGAS', 'tgas-matched-2mass.fits.gz')
    remotepath = 'http://portal.nersc.gov/project/cosmo/temp/dstn/gaia/tgas-matched-2mass.fits.gz'

    if not os.path.isfile(localpath):
        download(localpath, remotepath, verbose=True)
    else:
        sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
        sys.stdout.flush()

def get_DR2(dirloc):
    #Check the directory exists
    if os.path.isdir(dirloc + 'Gaia_DR2') == False:
        print('Building storage directories...')
        os.makedirs(dirloc+'Gaia_DR2')

    #Building local and remote paths
    local = [os.path.join(dirloc, 'Gaia_DR2', 'DR2_000-000-%03i.csv' % ii)
                for ii in range(16)]
    remote = [os.path.join('http://cdn.gea.esac.esa.int','Gaia','DR2','csv',
                'DR2_000-000-%03i.csv.gz' % ii) for ii in range(16)]

    #Download the data
    for localpath, remotepath in list(zip(local, remote)):
        if not os.path.isfile(localpath):
            download(localpath, remotepath, verbose=True)
        else:
            sys.stdout.write('File %s already exists locally ... \n' % (os.path.basename(localpath)))
            sys.stdout.flush()

def download(localpath, remotepath, verbose=False):
    sys.stdout.write('\r'+"Downloading file %s ...\r" % (os.path.basename(remotepath)))
    sys.stdout.flush()

    downloading = True
    interrupted = False
    #Create a temporary file
    file, tmp_savefilename = tempfile.mkstemp()
    os.close(file)
    ntries = 1

    while downloading:
        try:
            cmd = ['wget','%s' % remotepath,
                    '-O', '%s' % tmp_savefilename,
                    '--read-timeout=10',
                    '--tries=3']
            if not verbose:
                cmd.append('-q')

            subprocess.check_call(cmd) #Run command
            shutil.move(tmp_savefilename, localpath) #Move temp file to local
            downloading=False   #End download
            if interrupted:
                raise KeyboardInterrupt

        #Check for keyboardinterrupted or repeated attempts at failed download
        except subprocess.CalledProcessError as e:
            if not downloading: #Assume Keyboardinterrupt
                raise
            elif ntries > 2:
                raise IOError('File %s does not appear to exist on the server ...' % (os.path.basename(remotepath)))
            elif not 'exit status 4' in str(e):
                interrupted = True
            os.remove(tmp_savefilename)

        #Check for OSError due to missing wget
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                raise OSError("Automagically downloading catalogs requires the wget program; please install wget and try again...")
            else:
                raise OSError("Not quite sure whats gone wrong here...")

        #Just make sure the temp file is removed
        finally:
            if os.path.exists(tmp_savefilename):
                os.remove(tmp_savefilename)

        #Up the ntries
        ntries += 1
    sys.stdout.write('\r'+"Download Complete"+'\r')
    sys.stdout.flush()

    return None

def unzip(sfile, sep = ',', columns=None, check_info=False):
    '''A simple function that returns a pandas dataframe from a gzipped .csv
    file, and optionally returns chose columns '''
    file = gzip.open(sfile, 'rb')

    if '.csv' in os.path.basename(sfile):
        if check_info:
            print(pd.read_csv(file, sep=sep).info())

        if columns == None:
            return pd.read_csv(file, sep=sep)
        else:
            return pd.read_csv(file, sep=sep, usecols=columns)

    elif '.fits' in os.path.basename(sfile):
        dat = Table.read(sfile, format='fits')
        if check_info:
            print(dat.info())
        return dat.to_pandas()[columns]

    elif '.txt' in os.path.basename(sfile):
        if check_info:
            print(pd.read_csv(file, sep=sep).info())

        if columns == None:
            return pd.read_csv(file, sep=sep)
        else:
            return pd.read_csv(file, sep=sep, usecols=columns)

    else:
        print('Filetype not recognised (currently only .fits, .csv and .txt supported)')

class query_simbad_oids():
    def __init__(self):
        self.timer = 0.
        self.queries = 0
        self.reset = False

    def __call__(self, object_name):

        if self.queries == 0:
            self.t1 = time.time()

        if self.queries == 6:
            wait = True
            while wait:
                sys.stdout.write('\rtick tock tick tock\r')
                sys.stdout.flush()
                if time.time() - self.t1 > 1.2:
                    wait = False
                    self.reset = True

        #Make the query
        results_table = Simbad.query_objectids(object_name)
        self.queries += 1

        if self.reset:
            self.queries = 0
            self.reset = False

        try:
            return pd.DataFrame(np.array(results_table))

        except ValueError:
            return 'noentry'

def collect_oids(df):
    df['2MASS'] = np.nan
    df['KIC'] = np.nan

    #Looping over each entry in TGAS
    query = query_simbad_oids()

    for idx in tqdm(range(len(df))):
        object_name = 'Gaia DR1 '+str(df.source_id[idx])
        # time.sleep(1)
        oids = query(object_name)

        try:
            #Match 2MASS and KIC IDs to the gaia dr1 id
            for odx in range(len(oids)):
                oid = oids.ID.loc[odx]  #Pull out the individual string
                if "2MASS" in oid:      #Assign 2MASS ID if it exists
                    df['2MASS'].loc[idx] = oid[6:]
                if "KIC" in oid:        #Assign KIC ID if it exists
                    df['KIC'][idx] = oid[4:]

        except AttributeError:
            print(oids)
            pass

    return df

if __name__ == "__main__":
    dirloc = '/home/oliver/PhD/Catalogues/'
    get_TGAS(dirloc)
    get_KIC(dirloc)

    columns = ['hip','tycho2_id','source_id','parallax','parallax_error','phot_g_mean_mag']

    files = glob.glob(dirloc+'Gaia_TGAS/*.csv.gz')
    df0 = unzip(files[0], sep=',', columns=columns)
    df1 = unzip(files[1], sep=',', columns=columns)
    df2 = unzip(files[2], sep=',', columns=columns)
    df3 = unzip(files[3], sep=',', columns=columns)
    df4 = unzip(files[4], sep=',', columns=columns)
    df5 = unzip(files[5], sep=',', columns=columns)
    df6 = unzip(files[6], sep=',', columns=columns)
    df7 = unzip(files[7], sep=',', columns=columns)
    df8 = unzip(files[8], sep=',', columns=columns)
    df9 = unzip(files[9], sep=',', columns=columns)
    df10 = unzip(files[10], sep=',', columns=columns)
    df11 = unzip(files[11], sep=',', columns=columns)
    df12 = unzip(files[12], sep=',', columns=columns)
    df13 = unzip(files[13], sep=',', columns=columns)
    df14 = unzip(files[14], sep=',', columns=columns)
    df15 = unzip(files[15], sep=',', columns=columns)

    tgas = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15])
    tgas = tgas.reset_index()

    outloc = os.path.join(dirloc, 'Gaia_TGAS', 'tgas.csv.gz')
    tgas.to_csv(outloc, compression='gzip')


    sys.exit()

    df = collect_oids(df)   #Grab 2MASS and KIC oids from SIMBAD

    #http://archive.stsci.edu/kepler/kic10/help/quickcol.html
    kk = os.path.join(dirloc, 'KIC', 'kic.txt.gz')
    columns = ['kic_kepler_id']
    kic = unzip(kk, sep='|', columns=columns)


    get_2MASSxTGAS(dirloc)
    tt = os.path.join(dirloc, '2MASSxTGAS','tgas-matched-2mass.fits.gz')
    columns = ['ra', 'dec']
    twomass = unzip(tt, columns=columns, check_info=True)
